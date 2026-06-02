#!/usr/bin/env python3
"""
policy_executor.py
==================
Runs a trained DP3 checkpoint on the real robot.

Pipeline (20 Hz DP3 loop):
  1. Read ZED depth + camera_info → reconstruct point cloud (with Z filter)
  2. Read robot1/robot2/gripper1/gripper2 joint states
  3. Run DP3 inference → action trajectory (n_action_steps × 14)
  4. Execute actions one-by-one at 20 Hz via RTDE servoJ (interpolated to
     RTDE_HZ for smooth motion)

Robot layout:
  robot1 @ 192.168.1.10  ←  6 arm joints + Robotiq gripper (finger_joint)
  robot2 @ 192.168.1.20  ←  6 arm joints + Robotiq gripper (finger_joint)

Gripper convention (matching training data):
  0.0 = fully open, 1.0 = fully closed
  Robotiq normalised command = position × 255 (0–255 register)

Usage:
    python3 policy_executor.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --robot1_ip  192.168.1.10 \
        --robot2_ip  192.168.1.20 \
        --hz         20 \
        --rtde_hz    125
"""

import argparse
import collections
import threading
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo

import rtde_control
import rtde_receive
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

# ── config ───────────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent / "z_filter_config.yaml"

ROBOT1_IP = "192.168.1.10"
ROBOT2_IP = "192.168.1.20"

# Only camera topics come from ROS2 — robot/gripper state is read via RTDE
DEPTH_TOPIC    = "/zed/zed_node/depth/depth_registered"
CAM_INFO_TOPIC = "/zed/zed_node/depth/camera_info"
RGB_TOPIC      = "/zed/zed_node/rgb/color/rect/image"

# ── point cloud helpers ───────────────────────────────────────────────────────

def load_z_filter():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        return float(cfg["z_min"]), float(cfg["z_max"])
    print("[WARN] z_filter_config.yaml not found — using defaults (0.1, 1.5)")
    return 0.1, 1.5


def depth_to_pointcloud(depth_msg: Image, cam_msg: CameraInfo, rgb_msg: Image,
                         z_min: float, z_max: float,
                         n_points: int = 1024) -> np.ndarray:
    """Reconstruct XYZRGB point cloud from depth + RGB + camera_info, matching training pipeline."""
    h, w = depth_msg.height, depth_msg.width
    depth = np.frombuffer(bytes(depth_msg.data), dtype=np.float32).reshape(h, w)

    fx = cam_msg.k[0]; fy = cam_msg.k[4]
    cx = cam_msg.k[2]; cy = cam_msg.k[5]

    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32)
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy

    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    valid = np.isfinite(pts[:, 2]) & (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    pts = pts[valid]

    # Extract per-pixel RGB aligned to the depth image
    raw = np.frombuffer(bytes(rgb_msg.data), dtype=np.uint8)
    enc = rgb_msg.encoding
    if enc in ("bgra8", "rgba8"):
        img = raw.reshape(h, w, 4)
        if enc == "bgra8":
            r_ch, g_ch, b_ch = img[:, :, 2], img[:, :, 1], img[:, :, 0]
        else:
            r_ch, g_ch, b_ch = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    else:  # rgb8
        img = raw.reshape(h, w, 3)
        r_ch, g_ch, b_ch = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    r = r_ch.reshape(-1)[valid].astype(np.float32) / 255.0
    g = g_ch.reshape(-1)[valid].astype(np.float32) / 255.0
    b = b_ch.reshape(-1)[valid].astype(np.float32) / 255.0
    pts_rgb = np.column_stack([pts, r, g, b])

    return fps_or_pad(pts_rgb, n_points)


# GPU FPS via pytorch3d — used in obs_worker thread via a dedicated CUDA stream
# so it doesn't serialize with the inference_worker's default stream.
try:
    import pytorch3d.ops as _p3d_ops
    _HAVE_P3D = True
except ImportError:
    _HAVE_P3D = False

_obs_cuda_stream = None   # lazily created on first GPU call


def fps_or_pad(pts: np.ndarray, n: int) -> np.ndarray:
    """
    Downsample pts (N, C) to (n, C) via FPS, or pad if N < n.

    Preferred path: pytorch3d GPU FPS on a dedicated CUDA stream (~2–10 ms).
    Fallback:       random pre-subsample to max(n, 8192) then CPU numpy FPS.
    """
    global _obs_cuda_stream
    n_features = pts.shape[1]

    if len(pts) == 0:
        return np.zeros((n, n_features), dtype=np.float32)
    if len(pts) <= n:
        pad = np.zeros((n - len(pts), n_features), dtype=np.float32)
        return np.vstack([pts, pad]).astype(np.float32)

    # ── GPU path ──────────────────────────────────────────────────────────────
    if _HAVE_P3D and torch.cuda.is_available():
        if _obs_cuda_stream is None:
            _obs_cuda_stream = torch.cuda.Stream()
        # Random pre-subsample to keep GPU transfer small (max 2× n or 16384)
        n_pre = min(len(pts), max(n * 2, 16384))
        if len(pts) > n_pre:
            idx_pre = np.random.choice(len(pts), n_pre, replace=False)
            pts_pre = pts[idx_pre]
        else:
            pts_pre = pts
        with torch.cuda.stream(_obs_cuda_stream):
            pts_t = torch.from_numpy(pts_pre).float().unsqueeze(0).cuda()
            _, sampled_idx = _p3d_ops.sample_farthest_points(pts_t[..., :3], K=n)
            result = pts_t[0, sampled_idx[0]].cpu().numpy()
        _obs_cuda_stream.synchronize()
        return result.astype(np.float32)

    # ── CPU fallback: random pre-subsample → numpy FPS ────────────────────────
    n_pre = min(len(pts), max(n, 8192))
    if len(pts) > n_pre:
        idx_pre = np.random.choice(len(pts), n_pre, replace=False)
        pts = pts[idx_pre]
    xyz   = pts[:, :3]
    idx   = np.zeros(n, dtype=np.int64)
    dists = np.full(len(pts), np.inf)
    cur   = 0
    for i in range(n):
        idx[i] = cur
        d = np.sum((xyz - xyz[cur]) ** 2, axis=1)
        dists = np.minimum(dists, d)
        cur = int(np.argmax(dists))
    return pts[idx].astype(np.float32)


def _read_n_points_from_cfg(cfg) -> int:
    """
    Extract the training n_points from a checkpoint config.
    Tries cfg.n_points first (new checkpoints), then falls back to reading the
    stored point_cloud shape from shape_meta (any checkpoint).
    Returns 1024 if neither is present (old checkpoint compatibility).
    """
    from omegaconf import OmegaConf
    try:
        val = OmegaConf.select(cfg, "n_points")
        if val is not None:
            return int(val)
    except Exception:
        pass
    try:
        shape = cfg.task.shape_meta.obs.point_cloud.shape
        return int(shape[0])
    except Exception:
        pass
    try:
        shape = cfg.shape_meta.obs.point_cloud.shape
        return int(shape[0])
    except Exception:
        pass
    return 1024   # safe fallback for checkpoints predating this change


# ── ROS2 sensor node ──────────────────────────────────────────────────────────

class CameraNode(Node):
    """Subscribes to ZED depth + RGB + camera_info, matching the training pipeline exactly."""

    def __init__(self):
        super().__init__("dp3_camera_node")
        be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.depth_msg    = None
        self.cam_info_msg = None
        self.rgb_msg      = None

        self.create_subscription(Image,      DEPTH_TOPIC,    self._depth_cb,   be)
        self.create_subscription(CameraInfo, CAM_INFO_TOPIC, self._caminfo_cb, be)
        self.create_subscription(Image,      RGB_TOPIC,      self._rgb_cb,     be)

    def _depth_cb(self, m):   self.depth_msg    = m
    def _caminfo_cb(self, m): self.cam_info_msg = m
    def _rgb_cb(self, m):     self.rgb_msg      = m

    def ready(self) -> bool:
        return (self.depth_msg is not None
                and self.cam_info_msg is not None
                and self.rgb_msg is not None)


# ── RTDE robot interface ──────────────────────────────────────────────────────

class BimanualRTDE:
    """Connects to both UR10e arms and Robotiq grippers via RTDE."""

    LOOKAHEAD = 0.1    # servoJ lookahead seconds
    GAIN      = 300    # servoJ controller gain
    ACC       = 1.0    # rad/s²
    VEL       = 1.0    # rad/s

    def __init__(self, robot1_ip: str, robot2_ip: str, rtde_hz: int):
        self.dt = 1.0 / rtde_hz

        print(f"Connecting to robot1 @ {robot1_ip} ...")
        self.rc1 = RTDEControl(robot1_ip)
        self.rr1 = RTDEReceive(robot1_ip)

        print(f"Connecting to robot2 @ {robot2_ip} ...")
        self.rc2 = RTDEControl(robot2_ip)
        self.rr2 = RTDEReceive(robot2_ip)

        # Gripper positions tracked locally — we send all commands so we
        # always know the last commanded value (matches training data init:
        # gripper1 always closed=1.0, gripper2 starts open=0.0)
        self._g1_pos = 1.0
        self._g2_pos = 0.0

        print("RTDE connected.")

    def get_state(self) -> np.ndarray:
        """Read 14-D state [r1(6), g1(1), r2(6), g2(1)].
        Arm joints from RTDE; gripper positions from last commanded value."""
        r1 = np.array(self.rr1.getActualQ(), dtype=np.float32)   # (6,)
        r2 = np.array(self.rr2.getActualQ(), dtype=np.float32)   # (6,)
        g1 = np.array([self._g1_pos], dtype=np.float32)          # (1,)
        g2 = np.array([self._g2_pos], dtype=np.float32)          # (1,)
        return np.concatenate([r1, g1, r2, g2])                   # (14,)

    def get_joints(self):
        """Return current (q1, q2) arm joint positions."""
        return (np.array(self.rr1.getActualQ()),
                np.array(self.rr2.getActualQ()))

    def servoJ_step(self, q1_target: np.ndarray, q2_target: np.ndarray):
        """Send one servoJ command to both arms simultaneously."""
        self.rc1.servoJ(q1_target.tolist(), self.VEL, self.ACC, self.dt, self.LOOKAHEAD, self.GAIN)
        self.rc2.servoJ(q2_target.tolist(), self.VEL, self.ACC, self.dt, self.LOOKAHEAD, self.GAIN)

    def set_gripper(self, which: int, position_01: float, min_change: float = 0.05):
        """Send gripper command non-blocking via daemon thread.
        Skipped if change < min_change to avoid blocking on every loop.
        which: 1 = robot1 gripper, 2 = robot2 gripper
        Uses sendCustomScript (port 30002) — works alongside External Control URCap.
        """
        cur = self._g1_pos if which == 1 else self._g2_pos
        if abs(position_01 - cur) < min_change:
            return   # no meaningful change — skip to avoid blocking

        pos_byte = int(np.clip(position_01, 0.0, 1.0) * 255)
        print(f"[GRIPPER] gripper{which}: {cur:.2f}→{position_01:.2f}  (pos_byte={pos_byte})")
        # Plain URScript — rq_set_pos is a Robotiq URCap built-in.
        # sendCustomScript sends to port 30002 and executes immediately,
        # which works alongside External Control (unlike sendCustomScriptFunction).
        script = f"def grip():\n  rq_set_pos({pos_byte})\nend\ngrip()\n"
        rc     = self.rc1 if which == 1 else self.rc2

        # Fire-and-forget — never block the control loop
        def _send():
            try:
                ok = rc.sendCustomScript(script)
                if not ok:
                    print(f"[WARN] gripper{which} sendCustomScript returned False "
                          f"(pos={pos_byte})")
            except Exception as e:
                print(f"[WARN] gripper{which} command failed: {e}")

        t = threading.Thread(target=_send, daemon=True)
        t.start()

        if which == 1:
            self._g1_pos = float(np.clip(position_01, 0.0, 1.0))
        else:
            self._g2_pos = float(np.clip(position_01, 0.0, 1.0))

    def stop(self):
        self.rc1.servoStop()
        self.rc2.servoStop()
        self.rc1.stopScript()
        self.rc2.stopScript()

    def disconnect(self):
        self.stop()
        self.rc1.disconnect()
        self.rc2.disconnect()
        self.rr1.disconnect()
        self.rr2.disconnect()


# ── interpolation ─────────────────────────────────────────────────────────────

def interpolate_waypoints(q_from: np.ndarray, q_to: np.ndarray,
                           n_steps: int) -> list[np.ndarray]:
    """Linear interpolation from q_from to q_to in n_steps."""
    return [q_from + (q_to - q_from) * (i + 1) / n_steps
            for i in range(n_steps)]


# ── main execution loop ───────────────────────────────────────────────────────

def run(args):
    # ── load checkpoint ──
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    payload = torch.load(args.checkpoint, map_location="cpu")
    cfg     = payload["cfg"]
    policy  = hydra.utils.instantiate(cfg.policy)

    # Prefer EMA weights — they average over training and generalise better
    state_dicts = payload["state_dicts"]
    if "ema_model" in state_dicts:
        policy.load_state_dict(state_dicts["ema_model"])
        print("  Loaded EMA weights")
    else:
        policy.load_state_dict(state_dicts["model"])
        print("  Loaded model weights (no EMA found)")

    policy.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)

    # Read n_points from the checkpoint so inference matches training exactly
    n_points = _read_n_points_from_cfg(cfg)
    print(f"Policy loaded on {device}  |  n_points={n_points}")

    z_min, z_max = load_z_filter()

    # ── ROS2 — camera only ──
    rclpy.init()
    camera = CameraNode()

    print("Waiting for ZED depth + RGB + camera_info ...")
    import threading
    spin_thread = threading.Thread(target=rclpy.spin, args=(camera,), daemon=True)
    spin_thread.start()
    while not camera.ready():
        time.sleep(0.05)
    print("Camera ready.")

    # ── RTDE — arms + grippers ──
    robots = BimanualRTDE(args.robot1_ip, args.robot2_ip, args.rtde_hz)
    interp_steps = max(1, args.rtde_hz // args.hz)

    # ── Reduce diffusion steps for speed ─────────────────────────────────────
    if args.infer_steps != policy.num_inference_steps:
        policy.num_inference_steps = args.infer_steps
        print(f"Inference steps: {args.infer_steps}")

    # ── Warm up GPU ───────────────────────────────────────────────────────────
    dummy_pc  = torch.zeros(1, 2, n_points, 6).to(device)
    dummy_pos = torch.zeros(1, 2, 14).to(device)
    for _ in range(3):
        t_w = time.time()
        with torch.no_grad():
            policy.predict_action({"point_cloud": dummy_pc, "agent_pos": dummy_pos})
    infer_ms = (time.time() - t_w) * 1000
    print(f"Inference: {infer_ms:.0f}ms  ({1000/infer_ms:.1f} Hz)\n")

    # ── Background obs thread: samples at exactly 20 Hz ──────────────────────
    # Inference takes ~66ms but training obs were 50ms apart.
    # A background thread keeps sampling at 20Hz so the policy always sees
    # obs history with correct 50ms spacing regardless of inference speed.
    obs_lock   = threading.Lock()
    obs_ring   = collections.deque(maxlen=2)   # always holds last 2 obs
    stop_obs   = threading.Event()

    def obs_worker():
        interval = 1.0 / args.hz   # 50ms
        while not stop_obs.is_set():
            t0 = time.time()
            pc    = depth_to_pointcloud(camera.depth_msg, camera.cam_info_msg,
                                        camera.rgb_msg, z_min, z_max, n_points)
            state = robots.get_state()
            with obs_lock:
                obs_ring.append((pc, state))
            elapsed = time.time() - t0
            wait = interval - elapsed
            if wait > 0:
                time.sleep(wait)

    obs_thread = threading.Thread(target=obs_worker, daemon=True)

    # ── Kickstart: nudge robot2 wrist to seed nonzero velocity ───────────────
    # Training data had static frames filtered out, so the policy has never
    # seen obs[t-1]==obs[t]. At inference the robot starts static — an OOD
    # input. The nudge seeds a small velocity to match the training distribution.
    # Skip with --no_kickstart if demos included motion from frame 1.
    obs_thread.start()
    time.sleep(0.12)   # wait for 2 obs at 50ms spacing to fill the ring

    if not args.no_kickstart:
        print("Kickstart: nudging robot2 wrist +0.08 rad ...")
        q1_cur, q2_cur = robots.get_joints()
        q2_nudge = q2_cur.copy()
        q2_nudge[5] += 0.08   # wrist3 nudge — visible velocity signal
        for q2_wp in interpolate_waypoints(q2_cur, q2_nudge, interp_steps):
            robots.servoJ_step(q1_cur, q2_wp)
            time.sleep(1.0 / args.rtde_hz)
        time.sleep(0.06)   # let obs thread capture post-nudge state

        with obs_lock:
            vel_seed = np.abs(obs_ring[-1][1] - obs_ring[0][1]).max() if len(obs_ring) == 2 else 0
        print(f"Kickstart done — velocity seeded: {vel_seed:.4f} rad\n")
    else:
        print("Kickstart disabled (--no_kickstart). Waiting for obs ring to fill ...")
        time.sleep(0.15)
        print("Obs ring ready.\n")

    # ── Async inference thread ────────────────────────────────────────────────
    # Inference (65ms) and execution (n_action_steps×48ms) run in parallel.
    # Training used 20Hz — each action step = 50ms. With n_action_steps=2:
    #   execution = 2×48ms = 96ms  >  inference = 65ms
    # → inference finishes before execution ends → zero stop-start gaps.
    import queue as _queue
    action_queue  = _queue.Queue(maxsize=2)
    stop_infer    = threading.Event()

    R2_MIN = np.array([-2.673, -1.945, -2.697,  0.504, -4.545, -1.937])
    R2_MAX = np.array([-1.767, -1.296, -1.905,  1.008, -3.725, -1.385])
    MARGIN = 0.15

    def inference_worker():
        while not stop_infer.is_set():
            with obs_lock:
                if len(obs_ring) < 2:
                    time.sleep(0.01)
                    continue
                obs_a, obs_b = list(obs_ring)

            state_now = obs_b[1]
            pc_np    = np.stack([obs_a[0], obs_b[0]], axis=0)
            state_np = np.stack([obs_a[1], obs_b[1]], axis=0)
            pc_t    = torch.from_numpy(pc_np).float().unsqueeze(0).to(device)
            state_t = torch.from_numpy(state_np).float().unsqueeze(0).to(device)

            t_inf = time.time()
            with torch.no_grad():
                result = policy.predict_action({"point_cloud": pc_t, "agent_pos": state_t})
            infer_ms = (time.time() - t_inf) * 1000
            actions  = result["action"].squeeze(0).cpu().numpy()

            vel       = np.abs(obs_b[1] - obs_a[1]).max()
            raw_delta = actions[0] - state_now
            print(f"  infer={infer_ms:4.0f}ms  vel={vel:.4f}  "
                  f"r1={np.abs(raw_delta[0:6]).max():.4f}  "
                  f"r2={np.abs(raw_delta[7:13]).max():.4f}  "
                  f"g2={state_now[13]:.2f}→{actions[0,13]:.2f}")

            # Seed gripper absolute positions from current state before walking through
            # the action chunk. Each step propagates the position forward so the
            # full chunk produces monotonically advancing gripper targets, preventing
            # the min_change deadband in set_gripper() from gating all but the first step.
            g1_abs = float(state_now[6])
            g2_abs = float(state_now[13])

            for step_i in range(len(actions)):
                if args.lock_robot1:
                    # Freeze robot1 — use for single-arm testing
                    actions[step_i, 0:6] = state_now[0:6]
                else:
                    # Robot1: delta clamp only (add hard joint limits here once known)
                    for j in range(6):
                        delta = np.clip(actions[step_i, j] - state_now[j],
                                        -args.max_step, args.max_step)
                        actions[step_i, j] = state_now[j] + delta * args.action_scale

                # Robot2: delta clamp + hard joint bounds from recorded workspace
                for j in range(6):
                    idx   = 7 + j
                    delta = np.clip(actions[step_i, idx] - state_now[idx],
                                    -args.max_step, args.max_step)
                    target = state_now[idx] + delta * args.action_scale
                    actions[step_i, idx] = np.clip(
                        target, R2_MIN[j] - MARGIN, R2_MAX[j] + MARGIN)

                # Grippers: the policy outputs joint deltas (same space as training data).
                # Accumulate each step's predicted delta onto the running absolute position
                # so the execution loop receives absolute [0,1] targets, not raw deltas.
                g1_d   = np.clip(float(actions[step_i, 6]),  -args.max_step, args.max_step)
                g1_abs = float(np.clip(g1_abs + g1_d, 0.0, 1.0))
                actions[step_i, 6] = g1_abs

                g2_d   = np.clip(float(actions[step_i, 13]), -args.max_step, args.max_step)
                g2_abs = float(np.clip(g2_abs + g2_d, 0.0, 1.0))
                actions[step_i, 13] = g2_abs

            try:
                action_queue.put(actions, timeout=0.1)
            except _queue.Full:
                pass   # execution is behind — drop stale action

    infer_thread = threading.Thread(target=inference_worker, daemon=True)
    infer_thread.start()

    # Prime the queue — wait for first action before moving
    print("Waiting for first inference result ...")
    first_actions = action_queue.get()
    print("First action ready — starting execution.\n")

    try:
        print(f"=== DP3 Execution  (async inference + {args.n_action_steps} steps, Ctrl+C to stop) ===\n")
        loop_count  = 0
        actions     = first_actions

        while True:
            # 4. Execute current action chunk
            for step_idx in range(args.n_action_steps):
                act       = actions[step_idx]
                q1_target = act[0:6]
                g1_target = act[6]
                q2_target = act[7:13]
                g2_target = act[13]

                q1_cur, q2_cur = robots.get_joints()
                for q1_wp, q2_wp in zip(
                        interpolate_waypoints(q1_cur, q1_target, interp_steps),
                        interpolate_waypoints(q2_cur, q2_target, interp_steps)):
                    t_s = time.time()
                    robots.servoJ_step(q1_wp, q2_wp)
                    wait = (1.0 / args.rtde_hz) - (time.time() - t_s)
                    if wait > 0:
                        time.sleep(wait)

                # Gripper control via rq_set_pos URScript (fire-and-forget thread).
                # Requires Robotiq URCap installed on the UR controller.
                # If this causes External Control conflicts, disable with --no_gripper.
                if not args.no_gripper:
                    robots.set_gripper(1, float(g1_target))
                    robots.set_gripper(2, float(g2_target))

            # 5. Get next action (inference thread should have it ready)
            try:
                actions = action_queue.get(timeout=0.5)
            except _queue.Empty:
                print("[WARN] inference too slow — holding last action")

            loop_count += 1

    except KeyboardInterrupt:
        print("\nStopping ...")

    finally:
        stop_infer.set()
        stop_obs.set()
        infer_thread.join(timeout=1.0)
        obs_thread.join(timeout=1.0)
        robots.disconnect()
        rclpy.shutdown()
        print("Stopped.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--robot1_ip",  default=ROBOT1_IP)
    parser.add_argument("--robot2_ip",  default=ROBOT2_IP)
    parser.add_argument("--hz",             type=int, default=20,  help="DP3 inference rate")
    parser.add_argument("--rtde_hz",        type=int, default=125, help="RTDE servoJ rate")
    parser.add_argument("--n_action_steps", type=int, default=2,
                        help="Action steps per inference chunk (default 2: 2×48ms=96ms > 65ms inference → zero gap)")
    parser.add_argument("--infer_steps",    type=int,   default=5,
                        help="Diffusion denoising steps at inference (default 5, trained with 10)")
    parser.add_argument("--action_scale",  type=float, default=1.0,
                        help="Amplify predicted delta on robot2 only (default 1.0 = no amplification).")
    parser.add_argument("--max_step",      type=float, default=0.05,
                        help="Max joint delta per step in rad before scaling (safety clamp, default 0.05)")
    parser.add_argument("--no_kickstart",  action="store_true",
                        help="Skip the wrist nudge. Use if training demos started from rest (static start is in-distribution).")
    parser.add_argument("--lock_robot1",   action="store_true",
                        help="Freeze robot1 joints — only robot2 moves. Use for single-arm testing.")
    parser.add_argument("--no_gripper",    action="store_true",
                        help="Disable gripper commands. Use if rq_set_pos conflicts with External Control URCap.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

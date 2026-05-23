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

N_POINTS = 1024

# ── point cloud helpers ───────────────────────────────────────────────────────

def load_z_filter():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        return float(cfg["z_min"]), float(cfg["z_max"])
    print("[WARN] z_filter_config.yaml not found — using defaults (0.1, 1.5)")
    return 0.1, 1.5


def depth_to_pointcloud(depth_msg: Image, cam_msg: CameraInfo,
                         z_min: float, z_max: float,
                         n_points: int = N_POINTS) -> np.ndarray:
    """Reconstruct point cloud from depth + camera_info, matching training pipeline."""
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
    return fps_or_pad(pts[valid], n_points)


FPS_PRESAMPLE = 8192   # random pre-subsample before FPS — keeps each frame <0.1s


def fps_or_pad(pts: np.ndarray, n: int) -> np.ndarray:
    if len(pts) == 0:
        return np.zeros((n, 3), dtype=np.float32)
    if len(pts) <= n:
        pad = np.zeros((n - len(pts), 3), dtype=np.float32)
        return np.vstack([pts, pad]).astype(np.float32)
    # Random pre-subsample so FPS runs on at most FPS_PRESAMPLE points
    if len(pts) > FPS_PRESAMPLE:
        idx_pre = np.random.choice(len(pts), FPS_PRESAMPLE, replace=False)
        pts = pts[idx_pre]
    if len(pts) <= n:
        pad = np.zeros((n - len(pts), 3), dtype=np.float32)
        return np.vstack([pts, pad]).astype(np.float32)
    idx  = np.zeros(n, dtype=np.int64)
    dists = np.full(len(pts), np.inf)
    cur  = 0
    for i in range(n):
        idx[i] = cur
        d = np.sum((pts - pts[cur]) ** 2, axis=1)
        dists = np.minimum(dists, d)
        cur = int(np.argmax(dists))
    return pts[idx].astype(np.float32)


# ── ROS2 sensor node ──────────────────────────────────────────────────────────

class CameraNode(Node):
    """Subscribes to ZED depth + camera_info, matching the training pipeline exactly."""

    def __init__(self):
        super().__init__("dp3_camera_node")
        be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.depth_msg    = None
        self.cam_info_msg = None

        self.create_subscription(Image,      DEPTH_TOPIC,    self._depth_cb,   be)
        self.create_subscription(CameraInfo, CAM_INFO_TOPIC, self._caminfo_cb, be)

    def _depth_cb(self, m):   self.depth_msg    = m
    def _caminfo_cb(self, m): self.cam_info_msg = m

    def ready(self) -> bool:
        return self.depth_msg is not None and self.cam_info_msg is not None


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

        # Gripper positions tracked in [0, 1] matching training zarr convention.
        # Init: gripper1 closed, gripper2 open — matches typical demo start.
        self._g1_pos = 1.0
        self._g2_pos = 0.0

        print("RTDE connected.")

    def get_state(self) -> np.ndarray:
        """Read 14-D state [r1(6), g1(1), r2(6), g2(1)].
        Arm joints from RTDE; gripper in [0, 1] matching training zarr convention."""
        r1 = np.array(self.rr1.getActualQ(), dtype=np.float32)   # (6,)
        r2 = np.array(self.rr2.getActualQ(), dtype=np.float32)   # (6,)
        g1 = np.array([self._g1_pos], dtype=np.float32)          # (1,) in [0, 1]
        g2 = np.array([self._g2_pos], dtype=np.float32)          # (1,) in [0, 1]
        return np.concatenate([r1, g1, r2, g2])                   # (14,)

    def get_joints(self):
        """Return current (q1, q2) arm joint positions."""
        return (np.array(self.rr1.getActualQ()),
                np.array(self.rr2.getActualQ()))

    def servoJ_step(self, q1_target: np.ndarray, q2_target: np.ndarray):
        """Send one servoJ command to both arms simultaneously."""
        self.rc1.servoJ(q1_target.tolist(), self.VEL, self.ACC, self.dt, self.LOOKAHEAD, self.GAIN)
        self.rc2.servoJ(q2_target.tolist(), self.VEL, self.ACC, self.dt, self.LOOKAHEAD, self.GAIN)

    def set_gripper(self, which: int, position_01: float, min_change: float = 0.02):
        """Send gripper command non-blocking via daemon thread.
        position_01: policy output in [0, 1] matching training zarr convention.
        min_change lowered to 0.02 — diffusion outputs gradual trajectories;
            0.05 filtered out nearly every incremental close step.
        """
        cur = self._g1_pos if which == 1 else self._g2_pos
        if abs(position_01 - cur) < min_change:
            return

        pos_byte = int(np.clip(position_01, 0.0, 1.0) * 255)
        script = f"def set_gripper():\n  rq_set_pos({pos_byte})\nend\n"
        rc = self.rc1 if which == 1 else self.rc2

        def _send():
            try:
                rc.sendCustomScriptFunction("set_gripper", script)
            except Exception:
                pass

        threading.Thread(target=_send, daemon=True).start()

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
    print(f"Policy loaded on {device}")

    z_min, z_max = load_z_filter()

    # ── ROS2 — camera only ──
    rclpy.init()
    camera = CameraNode()

    print("Waiting for ZED depth + camera_info ...")
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
    dummy_pc  = torch.zeros(1, 2, N_POINTS, 3).to(device)
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
            pc    = depth_to_pointcloud(camera.depth_msg, camera.cam_info_msg, z_min, z_max)
            state = robots.get_state()
            with obs_lock:
                obs_ring.append((pc, state))
            elapsed = time.time() - t0
            wait = interval - elapsed
            if wait > 0:
                time.sleep(wait)

    obs_thread = threading.Thread(target=obs_worker, daemon=True)

    # ── Kickstart: nudge robot2 wrist to seed nonzero velocity ───────────────
    # Policy needs obs[t-1] != obs[t] to exit the static attractor.
    # Nudge size matches typical teleop start velocity in training demos.
    print("Kickstart: recording pre-nudge state ...")
    obs_thread.start()
    time.sleep(0.12)   # wait for 2 obs at 50ms spacing to fill the ring

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
                  f"r2={np.abs(raw_delta[7:13]).max():.4f}  "
                  f"g2={state_now[13]:.2f}→{actions[0,13]:.2f}")

            # Safety clamp — only robot2 moves, robot1 (arm + gripper) locked.
            # 0:6 = arm joints, 6 = gripper1 — both must be frozen.
            for step_i in range(len(actions)):
                actions[step_i, 0:7] = state_now[0:7]
                for j in range(6):
                    idx   = 7 + j
                    delta = np.clip(actions[step_i, idx] - state_now[idx],
                                    -args.max_step, args.max_step)
                    target = state_now[idx] + delta * args.action_scale
                    actions[step_i, idx] = np.clip(
                        target, R2_MIN[j] - MARGIN, R2_MAX[j] + MARGIN)

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

                robots.set_gripper(1, g1_target)
                robots.set_gripper(2, g2_target)

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
    parser.add_argument("--n_action_steps", type=int, default=8,
                        help="Must match n_action_steps used during training (default 8). "
                             "Lower values skip late trajectory steps (e.g. gripper close).")
    parser.add_argument("--infer_steps",    type=int,   default=5,
                        help="Diffusion denoising steps at inference (default 5, trained with 10)")
    parser.add_argument("--action_scale",  type=float, default=1.0,
                        help="Amplify predicted delta on robot2 only (default 1.0 = no amplification).")
    parser.add_argument("--max_step",      type=float, default=0.05,
                        help="Max joint delta per step in rad before scaling (safety clamp, default 0.05)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

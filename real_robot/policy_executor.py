#!/usr/bin/env python3
"""
real_robot/policy_executor.py
==============================
Real-robot DP3 policy execution.

Architecture mirrors 3D-Diffusion-Policy/policy_executor.py:
  - ROS2 camera node in a background thread (depth + rgb + cam_info)
  - obs_worker thread: samples observation ring at obs_hz
  - inference_worker thread: runs DDIM policy async
  - Main control loop: executes action chunks via RTDE servoJ

Robot state (arm joints + gripper) is read directly via RTDE — no ROS2
joint state subscriptions required.

Run:
  python3 real_robot/policy_executor.py \
    --config   real_robot/real_config.yaml \
    --checkpoint /path/to/epoch=XXXX-val_loss=X.ckpt \
    --n_action_steps 8 --infer_steps 10 --no_kickstart
"""

import argparse
import collections
import queue as _queue
import sys
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

sys.path.insert(0, str(Path(__file__).parent.parent / "3D-Diffusion-Policy"))
from train import TrainDP3Workspace


# ---------------------------------------------------------------------------
# Point cloud helpers — GPU FPS preferred (pytorch3d), CPU fallback
# ---------------------------------------------------------------------------

try:
    import pytorch3d.ops as _p3d_ops
    _HAVE_P3D = True
except ImportError:
    _HAVE_P3D = False

_obs_cuda_stream = None


def fps_or_pad(pts: np.ndarray, n: int) -> np.ndarray:
    """Downsample (N,C) to (n,C) via FPS on GPU if available, else CPU."""
    global _obs_cuda_stream
    n_feat = pts.shape[1]
    if len(pts) == 0:
        return np.zeros((n, n_feat), dtype=np.float32)
    if len(pts) <= n:
        pad = np.zeros((n - len(pts), n_feat), dtype=np.float32)
        return np.vstack([pts, pad]).astype(np.float32)

    if _HAVE_P3D and torch.cuda.is_available():
        if _obs_cuda_stream is None:
            _obs_cuda_stream = torch.cuda.Stream()
        n_pre = min(len(pts), max(n * 2, 16384))
        if len(pts) > n_pre:
            idx_pre = np.random.choice(len(pts), n_pre, replace=False)
            pts_pre = pts[idx_pre]
        else:
            pts_pre = pts
        with torch.cuda.stream(_obs_cuda_stream):
            pts_t = torch.from_numpy(pts_pre).float().unsqueeze(0).cuda()
            _, idx = _p3d_ops.sample_farthest_points(pts_t[..., :3], K=n)
            result = pts_t[0, idx[0]].cpu().numpy()
        _obs_cuda_stream.synchronize()
        return result.astype(np.float32)

    # CPU fallback
    n_pre = min(len(pts), max(n, 8192))
    if len(pts) > n_pre:
        pts = pts[np.random.choice(len(pts), n_pre, replace=False)]
    xyz = pts[:, :3]
    sel = np.zeros(n, dtype=np.int64)
    d   = np.full(len(pts), np.inf)
    cur = 0
    for i in range(n):
        sel[i] = cur
        nd  = np.sum((xyz - xyz[cur]) ** 2, axis=1)
        d   = np.minimum(d, nd)
        cur = int(np.argmax(d))
    return pts[sel].astype(np.float32)


def depth_to_pointcloud(depth_msg: Image, cam_msg: CameraInfo, rgb_msg: Image,
                         ws: dict, n_points: int) -> np.ndarray:
    """Reconstruct XYZRGB point cloud — same pipeline as training."""
    h, w = depth_msg.height, depth_msg.width
    enc_d = depth_msg.encoding
    raw_d = bytes(depth_msg.data)
    if enc_d == "32FC1":
        depth = np.frombuffer(raw_d, dtype=np.float32).reshape(h, w)
    elif enc_d == "16UC1":
        depth = np.frombuffer(raw_d, dtype=np.uint16).reshape(h, w).astype(np.float32) / 1000.0
    else:
        raise ValueError(f"Unsupported depth encoding: {enc_d!r}")

    fx, fy = cam_msg.k[0], cam_msg.k[4]
    cx, cy = cam_msg.k[2], cam_msg.k[5]
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    valid = (np.isfinite(pts[:, 2])
             & (pts[:, 0] >= ws["x_min"]) & (pts[:, 0] <= ws["x_max"])
             & (pts[:, 1] >= ws["y_min"]) & (pts[:, 1] <= ws["y_max"])
             & (pts[:, 2] >= ws["z_min"]) & (pts[:, 2] <= ws["z_max"]))
    pts = pts[valid]

    raw_rgb = np.frombuffer(bytes(rgb_msg.data), dtype=np.uint8)
    enc = rgb_msg.encoding
    if enc in ("bgra8", "rgba8"):
        img = raw_rgb.reshape(h, w, 4)
        r_ch = img[:, :, 2 if enc == "bgra8" else 0]
        g_ch = img[:, :, 1]
        b_ch = img[:, :, 0 if enc == "bgra8" else 2]
    else:
        img = raw_rgb.reshape(h, w, 3)
        r_ch, g_ch, b_ch = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    r = r_ch.reshape(-1)[valid].astype(np.float32) / 255.0
    g = g_ch.reshape(-1)[valid].astype(np.float32) / 255.0
    b = b_ch.reshape(-1)[valid].astype(np.float32) / 255.0
    xyzrgb = np.column_stack([pts, r, g, b])
    return fps_or_pad(xyzrgb, n_points)


# ---------------------------------------------------------------------------
# ROS2 camera node — spun in background thread, RTDE handles robot state
# ---------------------------------------------------------------------------

class CameraNode(Node):
    def __init__(self, topics: dict):
        super().__init__("dp3_camera_node")
        be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.depth_msg    = None
        self.cam_info_msg = None
        self.rgb_msg      = None
        self.create_subscription(Image,      topics["depth"],    self._cb_d,  be)
        self.create_subscription(CameraInfo, topics["cam_info"], self._cb_ci, be)
        self.create_subscription(Image,      topics["rgb"],      self._cb_rgb, be)

    def _cb_d(self,  m): self.depth_msg    = m
    def _cb_ci(self, m): self.cam_info_msg = m
    def _cb_rgb(self,m): self.rgb_msg      = m

    def ready(self) -> bool:
        return (self.depth_msg is not None
                and self.cam_info_msg is not None
                and self.rgb_msg is not None)


# ---------------------------------------------------------------------------
# BimanualRTDE — direct RTDE connection for both arms + grippers
# (identical to 3D-Diffusion-Policy/policy_executor.py)
# ---------------------------------------------------------------------------

class BimanualRTDE:
    LOOKAHEAD = 0.1
    GAIN      = 300
    ACC       = 1.0
    VEL       = 1.0

    def __init__(self, robot1_ip: str, robot2_ip: str, rtde_hz: int):
        self.dt = 1.0 / rtde_hz
        print(f"Connecting to robot1 @ {robot1_ip} ...")
        self.rc1 = RTDEControl(robot1_ip)
        self.rr1 = RTDEReceive(robot1_ip)
        print(f"Connecting to robot2 @ {robot2_ip} ...")
        self.rc2 = RTDEControl(robot2_ip)
        self.rr2 = RTDEReceive(robot2_ip)
        self._g1_pos = 1.0   # gripper1 starts closed (matches training init)
        self._g2_pos = 0.0   # gripper2 starts open
        print("RTDE connected.")

    def get_state(self) -> np.ndarray:
        """14-D [r1(6), g1(1), r2(6), g2(1)] — arms from RTDE, grippers local."""
        r1 = np.array(self.rr1.getActualQ(), dtype=np.float32)
        r2 = np.array(self.rr2.getActualQ(), dtype=np.float32)
        g1 = np.array([self._g1_pos], dtype=np.float32)
        g2 = np.array([self._g2_pos], dtype=np.float32)
        return np.concatenate([r1, g1, r2, g2])

    def get_joints(self):
        return (np.array(self.rr1.getActualQ()),
                np.array(self.rr2.getActualQ()))

    def servoJ_step(self, q1: np.ndarray, q2: np.ndarray):
        self.rc1.servoJ(q1.tolist(), self.VEL, self.ACC, self.dt, self.LOOKAHEAD, self.GAIN)
        self.rc2.servoJ(q2.tolist(), self.VEL, self.ACC, self.dt, self.LOOKAHEAD, self.GAIN)

    def set_gripper(self, which: int, position_01: float, min_change: float = 0.05):
        cur = self._g1_pos if which == 1 else self._g2_pos
        if abs(position_01 - cur) < min_change:
            return
        pos_byte = int(np.clip(position_01, 0.0, 1.0) * 255)
        print(f"[GRIPPER] gripper{which}: {cur:.2f}→{position_01:.2f}  (byte={pos_byte})")
        script = f"def grip():\n  rq_set_pos({pos_byte})\nend\ngrip()\n"
        rc = self.rc1 if which == 1 else self.rc2
        def _send():
            try:
                ok = rc.sendCustomScript(script)
                if not ok:
                    print(f"[WARN] gripper{which} sendCustomScript returned False")
            except Exception as e:
                print(f"[WARN] gripper{which} command failed: {e}")
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
        self.rc1.disconnect(); self.rc2.disconnect()
        self.rr1.disconnect(); self.rr2.disconnect()


def interpolate_waypoints(q_from: np.ndarray, q_to: np.ndarray, n: int):
    return [q_from + (q_to - q_from) * (i + 1) / n for i in range(n)]


# ---------------------------------------------------------------------------
# Policy loader
# ---------------------------------------------------------------------------

def _read_n_points(cfg) -> int:
    from omegaconf import OmegaConf
    for path in ("n_points",):
        try:
            v = OmegaConf.select(cfg, path)
            if v is not None:
                return int(v)
        except Exception:
            pass
    for path in ("task.shape_meta.obs.point_cloud.shape",
                 "shape_meta.obs.point_cloud.shape"):
        try:
            s = OmegaConf.select(cfg, path)
            if s is not None:
                return int(s[0])
        except Exception:
            pass
    return 1024


def load_policy(checkpoint_path: str, inference_steps: int):
    payload = torch.load(checkpoint_path, map_location="cpu")
    cfg = payload["cfg"]
    cfg.policy.num_inference_steps = inference_steps
    ws = TrainDP3Workspace(cfg)
    ws.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = ws.ema_model if cfg.training.use_ema else ws.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device).eval()
    return policy, device, _read_n_points(cfg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         default=str(Path(__file__).parent / "real_config.yaml"))
    parser.add_argument("--checkpoint",     default=None)
    parser.add_argument("--robot1_ip",      default=None)
    parser.add_argument("--robot2_ip",      default=None)
    parser.add_argument("--hz",             type=float, default=None, help="Obs/control Hz")
    parser.add_argument("--rtde_hz",        type=int,   default=None, help="RTDE servo Hz")
    parser.add_argument("--n_action_steps", type=int,   default=None)
    parser.add_argument("--infer_steps",    type=int,   default=None)
    parser.add_argument("--max_step",       type=float, default=0.05, help="Max joint delta per step")
    parser.add_argument("--no_kickstart",   action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    robot1_ip     = args.robot1_ip  or cfg["robots"]["robot1"]["ip"]
    robot2_ip     = args.robot2_ip  or cfg["robots"]["robot2"]["ip"]
    hz            = args.hz         or cfg["policy"]["obs_hz"]
    rtde_hz       = args.rtde_hz    or cfg["policy"]["rtde_hz"]
    n_action_steps= args.n_action_steps or cfg["policy"]["n_action_steps"]
    infer_steps   = args.infer_steps    or cfg["policy"]["inference_steps"]
    ckpt          = args.checkpoint     or cfg["policy"]["checkpoint_path"]
    ws            = cfg["workspace"]
    max_step      = args.max_step
    interp_steps  = max(1, rtde_hz // hz)

    print(f"Loading policy from: {ckpt}")
    policy, device, n_points = load_policy(ckpt, infer_steps)
    print(f"Policy loaded on {device}  |  n_points={n_points}")
    print(f"hz={hz}  rtde_hz={rtde_hz}  n_action_steps={n_action_steps}"
          f"  infer_steps={infer_steps}  interp_steps={interp_steps}")

    # ── ROS2 camera ──
    rclpy.init()
    camera = CameraNode(cfg["topics"])
    spin_thread = threading.Thread(target=rclpy.spin, args=(camera,), daemon=True)
    spin_thread.start()
    print("Waiting for ZED depth + RGB + camera_info ...")
    while not camera.ready():
        time.sleep(0.05)
    print("Camera ready.")

    # ── RTDE robot connections ──
    robots = BimanualRTDE(robot1_ip, robot2_ip, rtde_hz)

    # ── GPU warm-up ──
    dummy_pc  = torch.zeros(1, 2, n_points, 6).to(device)
    dummy_pos = torch.zeros(1, 2, 14).to(device)
    for _ in range(3):
        t_w = time.time()
        with torch.no_grad():
            policy.predict_action({"point_cloud": dummy_pc, "agent_pos": dummy_pos})
    infer_ms = (time.time() - t_w) * 1000
    print(f"Inference: {infer_ms:.0f} ms  ({1000/infer_ms:.1f} Hz)\n")

    # ── obs_worker thread: samples at exactly hz ──
    obs_lock = threading.Lock()
    obs_ring = collections.deque(maxlen=2)
    stop_obs = threading.Event()

    def obs_worker():
        interval = 1.0 / hz
        while not stop_obs.is_set():
            t0 = time.time()
            try:
                pc    = depth_to_pointcloud(camera.depth_msg, camera.cam_info_msg,
                                             camera.rgb_msg, ws, n_points)
                state = robots.get_state()
                with obs_lock:
                    obs_ring.append((pc, state))
            except Exception as e:
                print(f"[obs_worker] {e}", flush=True)
            elapsed = time.time() - t0
            wait = interval - elapsed
            if wait > 0:
                time.sleep(wait)

    obs_thread = threading.Thread(target=obs_worker, daemon=True)
    obs_thread.start()
    time.sleep(0.15)   # fill obs ring (2 frames @ 50 ms spacing)

    # ── Optional kickstart ──
    if not args.no_kickstart:
        print("Kickstart: nudging robot2 wrist +0.08 rad ...")
        q1_cur, q2_cur = robots.get_joints()
        q2_nudge = q2_cur.copy(); q2_nudge[5] += 0.08
        for q1_wp, q2_wp in zip(interpolate_waypoints(q1_cur, q1_cur, interp_steps),
                                 interpolate_waypoints(q2_cur, q2_nudge, interp_steps)):
            t_s = time.time()
            robots.servoJ_step(q1_wp, q2_wp)
            wait = (1.0 / rtde_hz) - (time.time() - t_s)
            if wait > 0:
                time.sleep(wait)
        time.sleep(0.06)
        print("Kickstart done.\n")
    else:
        print("Kickstart disabled.\n")

    # ── inference_worker thread ──
    action_queue = _queue.Queue(maxsize=2)
    stop_infer   = threading.Event()

    LATCH_COUNT    = 5
    g1_latch_count = 0
    g2_latch_count = 0
    g1_latched     = False
    g2_latched     = False
    g1_abs_carry   = float(robots._g1_pos)
    g2_abs_carry   = float(robots._g2_pos)

    def inference_worker():
        nonlocal g1_abs_carry, g2_abs_carry
        nonlocal g1_latch_count, g2_latch_count, g1_latched, g2_latched
        while not stop_infer.is_set():
            with obs_lock:
                if len(obs_ring) < 2:
                    time.sleep(0.01)
                    continue
                obs_a, obs_b = list(obs_ring)

            state_now = obs_b[1]
            pc_t    = torch.from_numpy(
                          np.stack([obs_a[0], obs_b[0]], axis=0)
                      ).float().unsqueeze(0).to(device)
            state_t = torch.from_numpy(
                          np.stack([obs_a[1], obs_b[1]], axis=0)
                      ).float().unsqueeze(0).to(device)

            t_inf = time.time()
            with torch.no_grad():
                result = policy.predict_action({"point_cloud": pc_t, "agent_pos": state_t})
            infer_ms = (time.time() - t_inf) * 1000
            actions  = result["action"].squeeze(0).cpu().numpy()   # (n_action_steps, 14)

            raw_d = actions[0].copy()
            print(
                f"  infer={infer_ms:4.0f}ms"
                f"  r1_Δmax={np.abs(raw_d[0:6]).max():.4f}"
                f"  r2_Δmax={np.abs(raw_d[7:13]).max():.4f}"
                f"  g1_Δ={raw_d[6]:.3f}  g2_Δ={raw_d[13]:.3f}",
                flush=True,
            )

            # Accumulate deltas → absolute targets; apply gripper latch
            q1_base = state_now[0:6].copy().astype(np.float64)
            q2_base = state_now[7:13].copy().astype(np.float64)
            g1_abs  = g1_abs_carry
            g2_abs  = g2_abs_carry

            for i in range(len(actions)):
                for j in range(6):
                    d = np.clip(float(actions[i, j]), -max_step, max_step)
                    actions[i, j] = q1_base[j] + d
                q1_base = actions[i, 0:6].copy()

                for j in range(6):
                    d = np.clip(float(actions[i, 7 + j]), -max_step, max_step)
                    actions[i, 7 + j] = q2_base[j] + d
                q2_base = actions[i, 7:13].copy()

                g1_d  = np.clip(float(actions[i, 6]),  -max_step, max_step)
                g1_abs = float(np.clip(g1_abs + g1_d, 0.0, 1.0))
                if g1_abs >= 0.99:
                    g1_latch_count += 1
                if g1_latch_count >= LATCH_COUNT:
                    g1_latched = True
                if g1_latched:
                    g1_abs = 1.0
                actions[i, 6] = g1_abs

                g2_d  = np.clip(float(actions[i, 13]), -max_step, max_step)
                g2_abs = float(np.clip(g2_abs + g2_d, 0.0, 1.0))
                if g2_abs >= 0.99:
                    g2_latch_count += 1
                if g2_latch_count >= LATCH_COUNT:
                    g2_latched = True
                if g2_latched:
                    g2_abs = 1.0
                actions[i, 13] = g2_abs

            g1_abs_carry = g1_abs
            g2_abs_carry = g2_abs

            try:
                action_queue.put(actions, timeout=0.1)
            except _queue.Full:
                pass   # execution behind — drop stale chunk

    infer_thread = threading.Thread(target=inference_worker, daemon=True)
    infer_thread.start()

    print("Waiting for first inference result ...")
    first_actions = action_queue.get()
    print("First action ready — starting execution.\n")

    # ── Main control loop ──
    try:
        print(f"=== DP3 Execution  ({n_action_steps} steps/chunk @ {hz} Hz, Ctrl+C to stop) ===\n")
        actions = first_actions
        while True:
            for step_idx in range(n_action_steps):
                act       = actions[step_idx]
                q1_target = act[0:6]
                q2_target = act[7:13]
                g1_target = float(act[6])
                g2_target = float(act[13])

                q1_cur, q2_cur = robots.get_joints()
                for q1_wp, q2_wp in zip(
                        interpolate_waypoints(q1_cur, q1_target, interp_steps),
                        interpolate_waypoints(q2_cur, q2_target, interp_steps)):
                    t_s = time.time()
                    robots.servoJ_step(q1_wp, q2_wp)
                    wait = (1.0 / rtde_hz) - (time.time() - t_s)
                    if wait > 0:
                        time.sleep(wait)

                robots.set_gripper(1, g1_target)
                robots.set_gripper(2, g2_target)

            try:
                actions = action_queue.get(timeout=0.5)
            except _queue.Empty:
                print("[WARN] inference too slow — holding last action")

    except KeyboardInterrupt:
        print("\nStopping ...")
    finally:
        stop_infer.set()
        stop_obs.set()
        infer_thread.join(timeout=1.0)
        obs_thread.join(timeout=1.0)
        robots.disconnect()
        camera.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

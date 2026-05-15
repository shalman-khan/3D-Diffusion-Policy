#!/usr/bin/env python3
"""
check_pc_alignment.py
=====================
Quantitatively compares training point clouds vs live ZED to detect:
  1. Camera intrinsics mismatch (K matrix changed)
  2. Axis flip or rotation (coordinate frame changed)
  3. Scale / translation offset
  4. Depth encoding difference

Usage:
    python3 check_pc_alignment.py
"""

import sqlite3, sys, time, threading
from pathlib import Path

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo

ZARR_PATH   = "/home/rosi/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real_cable_pull.zarr"
BAGS_DIR    = Path("/home/rosi/ROSBAG_NEW")
CONFIG_PATH = Path(__file__).parent / "z_filter_config.yaml"

# ── helpers ───────────────────────────────────────────────────────────────────

def load_z_filter():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f: cfg = yaml.safe_load(f)
        return float(cfg.get("z_min", 0.0)), float(cfg.get("z_max", 9999.0))
    return 0.0, 9999.0


def reconstruct_xyz(depth_data: bytes, h: int, w: int, K) -> np.ndarray:
    depth = np.frombuffer(depth_data, dtype=np.float32).reshape(h, w)
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32)
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return pts[np.isfinite(pts[:,2]) & (pts[:,2] > 0)]


def stats(pts: np.ndarray, label: str):
    print(f"\n  {label}  ({len(pts):,} valid pts)")
    for i, ax in enumerate("XYZ"):
        print(f"    {ax}: [{pts[:,i].min():8.3f}, {pts[:,i].max():8.3f}]  "
              f"mean={pts[:,i].mean():7.3f}  std={pts[:,i].std():6.3f}")
    print(f"    centroid: ({pts[:,0].mean():.3f}, {pts[:,1].mean():.3f}, {pts[:,2].mean():.3f})")


# ── Step 1: load intrinsics from training bag ─────────────────────────────────

def load_bag_frame():
    bag_dir  = sorted(BAGS_DIR.glob("session_*"))[0]
    db3      = list(bag_dir.glob("*.db3"))[0]
    conn     = sqlite3.connect(str(db3))
    cur      = conn.cursor()

    def get(topic, MsgType, offset=5):
        cur.execute("SELECT id FROM topics WHERE name=?", (topic,))
        row = cur.fetchone()
        if not row: return None
        tid = row[0]
        cur.execute(f"SELECT data FROM messages WHERE topic_id=? LIMIT 1 OFFSET {offset}", (tid,))
        raw = cur.fetchone()
        return deserialize_message(bytes(raw[0]), MsgType) if raw else None

    depth_msg = get("/zed/zed_node/depth/depth_registered", Image)
    cam_msg   = get("/zed/zed_node/depth/camera_info",      CameraInfo)
    conn.close()
    return depth_msg, cam_msg


# ── Step 2: live frame ────────────────────────────────────────────────────────

class Grabber(Node):
    def __init__(self):
        super().__init__("pc_align_check")
        be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        durability=DurabilityPolicy.VOLATILE,
                        history=HistoryPolicy.KEEP_LAST, depth=1)
        self.depth = None; self.cam = None
        self.create_subscription(Image,      "/zed/zed_node/depth/depth_registered", lambda m: setattr(self, 'depth', m), be)
        self.create_subscription(CameraInfo, "/zed/zed_node/depth/camera_info",      lambda m: setattr(self, 'cam', m),   be)
    def ready(self): return self.depth and self.cam


def load_live_frame():
    rclpy.init()
    node = Grabber()
    t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t.start()
    print("Waiting for live ZED frame ...")
    for _ in range(100):
        if node.ready(): break
        time.sleep(0.05)
    if not node.ready():
        print("ERROR: no ZED data"); rclpy.shutdown(); sys.exit(1)
    d, c = node.depth, node.cam
    rclpy.shutdown()
    return d, c


# ── Main comparison ───────────────────────────────────────────────────────────

def main():
    SEP = "=" * 60

    # Training bag
    print("Loading training bag frame ...")
    bag_depth, bag_cam = load_bag_frame()
    # Live
    live_depth, live_cam = load_live_frame()

    # ── 1. Intrinsics comparison ──────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  1. CAMERA INTRINSICS (K matrix)")
    print(SEP)
    bk = bag_cam.k;  lk = live_cam.k
    fields = [("fx", 0), ("fy", 4), ("cx", 2), ("cy", 5)]
    all_match = True
    for name, idx in fields:
        diff  = abs(bk[idx] - lk[idx])
        match = diff < 1.0
        if not match: all_match = False
        print(f"  {name:4s}  train={bk[idx]:8.3f}  live={lk[idx]:8.3f}  "
              f"diff={diff:6.3f}  {'✓' if match else '✗ MISMATCH'}")
    print(f"  Resolution  train={bag_cam.width}×{bag_cam.height}  "
          f"live={live_cam.width}×{live_cam.height}  "
          f"{'✓' if bag_cam.width==live_cam.width else '✗ MISMATCH'}")
    print(f"  {'✓ Intrinsics match' if all_match else '✗ INTRINSICS DIFFER — point clouds are in different scales'}")

    # ── 2. Depth encoding ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  2. DEPTH ENCODING")
    print(SEP)
    print(f"  Training encoding : {bag_depth.encoding}")
    print(f"  Live encoding     : {live_depth.encoding}")
    enc_ok = bag_depth.encoding == live_depth.encoding
    print(f"  {'✓ Match' if enc_ok else '✗ ENCODING DIFFERS — units may be wrong (mm vs m?)'}")

    # ── 3. Point cloud stats ──────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  3. POINT CLOUD SPATIAL STATS  (raw, no Z filter)")
    print(SEP)
    bag_pts  = reconstruct_xyz(bytes(bag_depth.data),  bag_depth.height,  bag_depth.width,  bk)
    live_pts = reconstruct_xyz(bytes(live_depth.data), live_depth.height, live_depth.width, lk)
    stats(bag_pts,  "Training bag")
    stats(live_pts, "Live ZED    ")

    # Centroid difference
    bc = bag_pts.mean(axis=0); lc = live_pts.mean(axis=0)
    offset = lc - bc
    print(f"\n  Centroid offset (live - train): "
          f"X={offset[0]:+.3f}m  Y={offset[1]:+.3f}m  Z={offset[2]:+.3f}m")
    big_offset = np.abs(offset).max() > 0.3
    print(f"  {'⚠ LARGE OFFSET — camera moved or scene changed' if big_offset else '✓ Centroids close'}")

    # ── 4. Axis flip check ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  4. AXIS FLIP / SIGN CHECK")
    print(SEP)
    for i, ax in enumerate("XYZ"):
        bs = np.sign(bag_pts[:,i].mean()); ls = np.sign(live_pts[:,i].mean())
        match = bs == ls
        print(f"  {ax} mean sign:  train={bs:+.0f}  live={ls:+.0f}  "
              f"{'✓' if match else '✗ AXIS FLIPPED — point cloud is mirrored'}")

    # ── 5. Compare with zarr (FPS-sampled) vs raw reconstruction ─────────────
    print(f"\n{SEP}")
    print("  5. ZARR STORED vs RAW RECONSTRUCTION (training)")
    print(SEP)
    from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
    buf = ReplayBuffer.copy_from_path(ZARR_PATH)
    zarr_pc = buf['point_cloud'][0]           # first frame, (1024, 3)
    zarr_valid = zarr_pc[(zarr_pc != 0).any(axis=1)]
    print(f"  Zarr ep0 step0: {len(zarr_valid)} non-zero points")
    zc = zarr_valid.mean(axis=0)
    print(f"  Zarr centroid : X={zc[0]:.3f}  Y={zc[1]:.3f}  Z={zc[2]:.3f}")
    rc = bag_pts.mean(axis=0) if len(bag_pts) > 0 else np.zeros(3)
    # The zarr centroid should be close to the bag raw reconstruction centroid
    diff = np.abs(zc - rc[:3]).max()
    print(f"  Max diff zarr vs raw: {diff:.4f}m  "
          f"{'✓ Consistent' if diff < 0.5 else '⚠ Check FPS sampling'}")

    # ── 6. Z filter check ────────────────────────────────────────────────────
    z_min, z_max = load_z_filter()
    print(f"\n{SEP}")
    print("  6. Z FILTER APPLIED")
    print(SEP)
    print(f"  z_min={z_min}  z_max={z_max}")
    live_filt = live_pts[(live_pts[:,2] >= z_min) & (live_pts[:,2] <= z_max)]
    bag_filt  = bag_pts[ (bag_pts[:,2]  >= z_min) & (bag_pts[:,2]  <= z_max)]
    print(f"  After filter — train: {len(bag_filt):,}  live: {len(live_filt):,}")
    if len(live_filt) < 10000:
        print(f"  ⚠ Very few points after filter — z_max may be too tight for current scene")
    else:
        print(f"  ✓ Sufficient points for FPS to 1024")

    # ── 7. Observation timing mismatch ───────────────────────────────────────
    print(f"\n{SEP}")
    print("  7. OBSERVATION TIMING  (most likely cause of policy not moving)")
    print(SEP)
    train_hz        = 20.0
    train_obs_gap   = 1.0 / train_hz * 1000   # 50ms between consecutive obs

    infer_ms_5step  = 66.0    # measured: --infer_steps 5
    infer_ms_10step = 127.0   # measured: --infer_steps 10
    exec_ms_1step   = 48.0    # 1 action step × 6 RTDE × 8ms

    deploy_obs_gap_5  = infer_ms_5step  + exec_ms_1step   # 114ms
    deploy_obs_gap_10 = infer_ms_10step + exec_ms_1step   # 175ms

    print(f"  Training:   consecutive obs spaced {train_obs_gap:.0f}ms  ({train_hz:.0f} Hz)")
    print(f"  Deploy (5 infer steps):  obs spaced {deploy_obs_gap_5:.0f}ms  "
          f"({1000/deploy_obs_gap_5:.1f} Hz)  "
          f"→ {deploy_obs_gap_5/train_obs_gap:.1f}× SLOWER than training")
    print(f"  Deploy (10 infer steps): obs spaced {deploy_obs_gap_10:.0f}ms  "
          f"({1000/deploy_obs_gap_10:.1f} Hz)  "
          f"→ {deploy_obs_gap_10/train_obs_gap:.1f}× SLOWER than training")
    print()
    print("  ⚠ The model was trained with obs 50ms apart.")
    print("  ⚠ At deployment, obs are 114ms+ apart because inference takes 66ms.")
    print("  ⚠ The policy reads the obs history velocity as much smaller than actual.")
    print()
    print("  FIX: Run executor with --n_action_steps 3 and decouple obs rate from")
    print("  inference. With 3 action steps × 48ms = 144ms execution, then re-infer.")
    print("  OR: maintain obs history at fixed 20Hz independent of inference rate.")
    print()

    # Compute joint state velocity (delta) between consecutive zarr frames
    print(f"  Training action delta magnitudes (zarr ep0, first 10 steps):")
    buf = ReplayBuffer.copy_from_path(ZARR_PATH)
    states = buf['state'][:20]
    for i in range(min(10, len(states)-1)):
        delta = np.abs(states[i+1] - states[i])
        print(f"    step {i:2d}→{i+1:2d}:  max_delta={delta.max():.4f} rad  "
              f"r2_pan_delta={delta[7]:.4f}")

    print(f"\n{SEP}")
    print("  SUMMARY")
    print(SEP)
    print("  Run:  python3 check_pc_alignment.py")
    print("  and look for ✗ markers above to find the root cause.")

if __name__ == "__main__":
    main()

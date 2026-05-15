#!/usr/bin/env python3
"""
rosbag_to_zarr.py
=================
Converts all rosbags in a folder to a single DP3-compatible zarr dataset.

For each bag (= one episode) at 20 Hz:
  - Reconstructs point cloud from depth + camera_info
  - Applies Z filter (from z_filter_config.yaml)
  - FPS-downsamples to 1024 points
  - Extracts joint states: [robot1(6), gripper1(1), robot2(6), gripper2(1)] = 14-D
  - Action = absolute joint positions at the NEXT timestep

Zarr keys written:
  data/state       (T, 14)   current joint positions
  data/action      (T, 14)   joint positions at t+1 (absolute)
  data/point_cloud (T, 1024, 3)

Usage:
    python3 rosbag_to_zarr.py \
        --bags_dir /home/rosi/rosbags_96221 \
        --output   /home/rosi/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real_cable_pull.zarr \
        --hz       20 \
        --n_points 1024
"""

import argparse
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, CameraInfo, JointState

from diffusion_policy_3d.common.replay_buffer import ReplayBuffer

# ── constants ────────────────────────────────────────────────────────────────

ROBOT1_TOPIC   = "/robot1/joint_states"
ROBOT2_TOPIC   = "/robot2/joint_states"
GRIPPER1_TOPIC = "/gripper1/joint_states"
GRIPPER2_TOPIC = "/gripper2/joint_states"
DEPTH_TOPIC    = "/zed/zed_node/depth/depth_registered"
CAM_INFO_TOPIC = "/zed/zed_node/depth/camera_info"
RGB_TOPIC      = "/zed/zed_node/rgb/color/rect/image"   # not used for PC but kept for reference

CONFIG_PATH = Path(__file__).parent / "z_filter_config.yaml"


# ── utility functions ────────────────────────────────────────────────────────

def load_z_filter():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        z_min = float(cfg.get("z_min", 0.1))
        z_max = float(cfg.get("z_max", 1.5))
    else:
        print("[WARN] z_filter_config.yaml not found — using defaults (0.1, 1.5).")
        print("       Run z_filter_gui.py first to configure the filter.")
        z_min, z_max = 0.1, 1.5
    print(f"[z_filter] z_min={z_min:.3f}m  z_max={z_max:.3f}m")
    return z_min, z_max


def read_bag_messages(db3_path: Path):
    """Return dict: topic_name → list of (timestamp_ns, deserialized_msg)."""
    conn = sqlite3.connect(str(db3_path))
    cur  = conn.cursor()

    cur.execute("SELECT id, name, type FROM topics")
    topic_map = {row[0]: (row[1], row[2]) for row in cur.fetchall()}

    topic_interest = {
        ROBOT1_TOPIC, ROBOT2_TOPIC,
        GRIPPER1_TOPIC, GRIPPER2_TOPIC,
        DEPTH_TOPIC, CAM_INFO_TOPIC,
    }

    msg_type_map = {
        ROBOT1_TOPIC:   JointState,
        ROBOT2_TOPIC:   JointState,
        GRIPPER1_TOPIC: JointState,
        GRIPPER2_TOPIC: JointState,
        DEPTH_TOPIC:    Image,
        CAM_INFO_TOPIC: CameraInfo,
    }

    data = {t: [] for t in topic_interest}

    cur.execute("SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp ASC")
    for topic_id, ts_ns, raw in cur.fetchall():
        if topic_id not in topic_map:
            continue
        name, _ = topic_map[topic_id]
        if name not in topic_interest:
            continue
        msg = deserialize_message(bytes(raw), msg_type_map[name])
        data[name].append((ts_ns, msg))

    conn.close()
    return data


def nearest_msg(messages, query_ns):
    """Binary-search for the message with timestamp closest to query_ns."""
    if not messages:
        return None
    timestamps = [m[0] for m in messages]
    idx = np.searchsorted(timestamps, query_ns)
    if idx == 0:
        return messages[0][1]
    if idx >= len(messages):
        return messages[-1][1]
    before = messages[idx - 1]
    after  = messages[idx]
    if abs(after[0] - query_ns) < abs(before[0] - query_ns):
        return after[1]
    return before[1]


def depth_to_pointcloud(depth_msg: Image, cam_msg: CameraInfo,
                         z_min: float, z_max: float, n_points: int):
    """Reconstruct point cloud, apply Z filter, FPS-downsample to n_points."""
    h, w = depth_msg.height, depth_msg.width
    depth = np.frombuffer(bytes(depth_msg.data), dtype=np.float32).reshape(h, w)

    fx = cam_msg.k[0]; fy = cam_msg.k[4]
    cx = cam_msg.k[2]; cy = cam_msg.k[5]

    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32)
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy

    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    valid = (np.isfinite(pts[:, 2]) & (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max))
    pts = pts[valid]

    return fps_or_pad(pts, n_points)


FPS_PRESAMPLE = 8192   # random subsample before FPS — keeps FPS fast


def fps_or_pad(pts: np.ndarray, n: int) -> np.ndarray:
    """
    Downsample to n points, zero-pad if fewer than n points.

    Strategy:
      1. Random subsample to FPS_PRESAMPLE (fast, reduces O(N^2) FPS cost)
      2. FPS on the small set to get exactly n well-spread points
    """
    if len(pts) == 0:
        return np.zeros((n, 3), dtype=np.float32)

    if len(pts) <= n:
        pad = np.zeros((n - len(pts), 3), dtype=np.float32)
        return np.vstack([pts, pad]).astype(np.float32)

    # Step 1: random pre-subsample so FPS runs on at most FPS_PRESAMPLE points
    if len(pts) > FPS_PRESAMPLE:
        idx_pre = np.random.choice(len(pts), FPS_PRESAMPLE, replace=False)
        pts = pts[idx_pre]

    if len(pts) <= n:
        pad = np.zeros((n - len(pts), 3), dtype=np.float32)
        return np.vstack([pts, pad]).astype(np.float32)

    # Step 2: FPS on the pre-sampled set
    idx = np.zeros(n, dtype=np.int64)
    dists = np.full(len(pts), np.inf)
    cur = 0
    for i in range(n):
        idx[i] = cur
        d = np.sum((pts - pts[cur]) ** 2, axis=1)
        dists = np.minimum(dists, d)
        cur = int(np.argmax(dists))
    return pts[idx].astype(np.float32)


def extract_joint_state(msg: JointState) -> np.ndarray:
    """Return joint positions as float32 array."""
    return np.array(msg.position, dtype=np.float32)


def process_bag(bag_dir: Path, z_min: float, z_max: float,
                hz: int, n_points: int):
    """
    Process one bag → returns dict with keys 'state', 'action', 'point_cloud'.
    Returns None if the bag is too short or missing required topics.
    """
    db3 = list(bag_dir.glob("*.db3"))
    if not db3:
        print(f"  [SKIP] No .db3 file in {bag_dir.name}")
        return None

    print(f"  Reading {bag_dir.name} ...")
    msgs = read_bag_messages(db3[0])

    for topic in [ROBOT1_TOPIC, ROBOT2_TOPIC, GRIPPER1_TOPIC, GRIPPER2_TOPIC,
                  DEPTH_TOPIC, CAM_INFO_TOPIC]:
        if not msgs[topic]:
            print(f"  [SKIP] Missing topic {topic} in {bag_dir.name}")
            return None

    # Build 20 Hz timeline between first and last robot joint timestamp
    all_ts = [m[0] for m in msgs[ROBOT1_TOPIC]]
    t_start = all_ts[0]
    t_end   = all_ts[-1]
    step_ns = int(1e9 / hz)
    timeline = list(range(t_start, t_end, step_ns))

    if len(timeline) < 4:
        print(f"  [SKIP] Too short: {bag_dir.name} ({len(timeline)} frames)")
        return None

    states      = []
    pointclouds = []

    # Cache camera intrinsics (stable across bag)
    cam_info = nearest_msg(msgs[CAM_INFO_TOPIC], t_start)

    # Maximum allowed staleness for joint state lookup.
    # Smart recorder skips joint states during pauses — those ticks get stale
    # repeated data from nearest_msg, recreating the static attractor.
    # Skip any tick where the nearest joint state is > 1.5 frames old (75ms).
    MAX_STATE_AGE_NS = int(75e6)   # 75ms = 1.5 × 50ms frames

    skipped_stale = 0
    for ts in timeline:
        # Find nearest joint state timestamps — skip if too stale (pause period)
        r2_msgs_ts = [m[0] for m in msgs[ROBOT2_TOPIC]]
        r2_idx = np.searchsorted(r2_msgs_ts, ts)
        r2_idx = min(max(r2_idx, 0), len(r2_msgs_ts) - 1)
        r2_nearest_ts = r2_msgs_ts[r2_idx]
        if abs(r2_nearest_ts - ts) > MAX_STATE_AGE_NS:
            skipped_stale += 1
            continue   # pause period — no joint state recorded, skip this tick

        r1  = nearest_msg(msgs[ROBOT1_TOPIC],   ts)
        r2  = nearest_msg(msgs[ROBOT2_TOPIC],   ts)
        g1  = nearest_msg(msgs[GRIPPER1_TOPIC], ts)
        g2  = nearest_msg(msgs[GRIPPER2_TOPIC], ts)
        dep = nearest_msg(msgs[DEPTH_TOPIC],    ts)

        r1_q = extract_joint_state(r1)   # (6,)
        r2_q = extract_joint_state(r2)   # (6,)
        g1_q = np.array([g1.position[0]], dtype=np.float32)  # (1,)
        g2_q = np.array([g2.position[0]], dtype=np.float32)  # (1,)

        state = np.concatenate([r1_q, g1_q, r2_q, g2_q])  # (14,)
        states.append(state)

        pc = depth_to_pointcloud(dep, cam_info, z_min, z_max, n_points)
        pointclouds.append(pc)

    if skipped_stale:
        print(f"  Skipped {skipped_stale} stale ticks (pause periods with no joint state)")

    states      = np.array(states,      dtype=np.float32)  # (T, 14)
    pointclouds = np.array(pointclouds, dtype=np.float32)  # (T, 1024, 3)

    # Action = absolute joint state at t+1 (last step repeats)
    actions = np.concatenate([states[1:], states[-1:]], axis=0)  # (T, 14)

    print(f"  → {len(states)} frames  | PC shape: {pointclouds.shape}")
    return {
        "state":       states,
        "action":      actions,
        "point_cloud": pointclouds,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def process_bag_worker(args_tuple):
    """Top-level function for ProcessPoolExecutor (must be picklable)."""
    bag_dir, z_min, z_max, hz, n_points = args_tuple
    # Each worker needs its own rclpy init
    rclpy.init(args=None)
    try:
        result = process_bag(Path(bag_dir), z_min, z_max, hz, n_points)
    finally:
        rclpy.shutdown()
    return str(bag_dir), result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bags_dir", default="/home/rosi/rosbags_96221",
                        type=Path)
    parser.add_argument("--output",
                        default="/home/rosi/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real_cable_pull.zarr",
                        type=Path)
    parser.add_argument("--hz",       type=int, default=20)
    parser.add_argument("--n_points", type=int, default=1024)
    parser.add_argument("--workers",  type=int, default=8,
                        help="Parallel worker processes (default: 8)")
    args = parser.parse_args()

    z_min, z_max = load_z_filter()

    bag_dirs = sorted(args.bags_dir.glob("session_*"))
    print(f"\nFound {len(bag_dirs)} bags in {args.bags_dir}")
    print(f"Workers: {args.workers}  |  FPS pre-sample: {FPS_PRESAMPLE} pts\n")

    if args.output.exists():
        import shutil
        shutil.rmtree(args.output)
        print(f"Removed existing zarr at {args.output}")

    buffer = ReplayBuffer.create_from_path(str(args.output), mode="a")

    worker_args = [
        (str(d), z_min, z_max, args.hz, args.n_points)
        for d in bag_dirs
    ]

    episodes = {}   # bag_dir_str → episode dict (preserves order)
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_bag_worker, a): a[0] for a in worker_args}
        pbar = tqdm(as_completed(futures), total=len(futures),
                    desc="Converting bags", unit="bag")
        for fut in pbar:
            bag_path, episode = fut.result()
            episodes[bag_path] = episode

    # Write episodes in sorted order so episode_ends are deterministic
    n_ok = 0
    for d in bag_dirs:
        ep = episodes.get(str(d))
        if ep is None:
            continue
        buffer.add_episode(ep)
        n_ok += 1

    elapsed = time.time() - t0
    print(f"\n✓ Done in {elapsed:.1f}s")
    print(f"  Episodes : {n_ok}/{len(bag_dirs)}")
    print(f"  Steps    : {buffer.n_steps}")
    print(f"  Zarr     : {args.output}")


if __name__ == "__main__":
    main()


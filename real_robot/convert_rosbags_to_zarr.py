#!/usr/bin/env python3
"""
convert_rosbags_to_zarr.py
==========================
Converts real-robot ROS2 bags (trial2–trial8) into a single zarr dataset
compatible with the DP3 RobosuiteDataset format.

Each trial has two bags:
  camera/trial{n}_camera/   — /camera/camera/depth/color/points  (~15 Hz)
  Trajectory_Gripper/trial{n}_robot/ — /robot1/joint_states, /robot2/joint_states,
                                        /gripper1/joint_states, /gripper2/joint_states

Processing per trial:
  1. Read all messages from both bags via rosbag2_py
  2. Synchronise to a 20 Hz timeline using nearest-neighbour per-topic
  3. Parse PointCloud2 → XYZRGB numpy array, crop workspace, FPS downsample to 1024
  4. Build agent_pos (14-dim) and action (joint delta, 14-dim)
  5. Append episode to zarr

Usage:
  python convert_rosbags_to_zarr.py [--config real_config.yaml]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import zarr
import yaml
from numcodecs import Blosc

# ROS2 bag reading
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def open_bag_reader(bag_path: str):
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def read_all_messages(bag_path: str, topic_names: list) -> dict:
    """
    Returns {topic: [(timestamp_ns, msg), ...]} for all requested topics.
    """
    reader = open_bag_reader(bag_path)

    # Build type map
    topic_types = {}
    for info in reader.get_all_topics_and_types():
        if info.name in topic_names:
            topic_types[info.name] = get_message(info.type)

    # Filter to only requested topics
    storage_filter = rosbag2_py.StorageFilter(topics=topic_names)
    reader.set_filter(storage_filter)

    result = {t: [] for t in topic_names}
    while reader.has_next():
        topic, data, ts = reader.read_next()
        if topic in topic_types:
            msg = deserialize_message(data, topic_types[topic])
            result[topic].append((ts, msg))

    return result


def parse_pointcloud2_xyzrgb(msg) -> np.ndarray:
    """
    Parse PointCloud2 message (XYZ + RGB, point_step=20) into (N,6) float32.
    Layout: x(0,f32) y(4,f32) z(8,f32) pad(12,u32) rgb(16,f32-packed-uint32)
    RGB packed as uint32: bits [23:16]=R [15:8]=G [7:0]=B, normalised to [0,1].
    """
    dt = np.dtype([
        ("x",   np.float32),
        ("y",   np.float32),
        ("z",   np.float32),
        ("pad", np.uint32),
        ("rgb", np.float32),
    ])
    arr = np.frombuffer(bytes(msg.data), dtype=dt)
    xyz = np.column_stack([arr["x"], arr["y"], arr["z"]])
    rgb_int = arr["rgb"].view(np.uint32)
    r = ((rgb_int >> 16) & 0xFF).astype(np.float32) / 255.0
    g = ((rgb_int >> 8)  & 0xFF).astype(np.float32) / 255.0
    b = ( rgb_int        & 0xFF).astype(np.float32) / 255.0
    return np.column_stack([xyz, r, g, b])


def crop_workspace(xyz: np.ndarray, ws: dict) -> np.ndarray:
    mask = (
        (xyz[:, 0] >= ws["x_min"]) & (xyz[:, 0] <= ws["x_max"]) &
        (xyz[:, 1] >= ws["y_min"]) & (xyz[:, 1] <= ws["y_max"]) &
        (xyz[:, 2] >= ws["z_min"]) & (xyz[:, 2] <= ws["z_max"])
    )
    # Also remove NaN / inf
    valid = np.all(np.isfinite(xyz), axis=1)
    return xyz[mask & valid]


def fps_numpy(points: np.ndarray, n_samples: int, pre_subsample: int = 8192) -> np.ndarray:
    """
    Farthest Point Sampling (CPU, numpy). Distances computed on XYZ only;
    all feature channels (e.g. RGB) are preserved in the output.
    Pre-subsamples randomly to `pre_subsample` points before FPS to keep
    runtime manageable on dense real-world point clouds (~150k points).
    """
    n_features = points.shape[1]
    n = len(points)
    if n == 0:
        return np.zeros((n_samples, n_features), dtype=np.float32)
    if n <= n_samples:
        idx = np.random.choice(n, n_samples, replace=True)
        return points[idx].astype(np.float32)

    # Random pre-subsample to keep FPS tractable
    if n > pre_subsample:
        idx = np.random.choice(n, pre_subsample, replace=False)
        points = points[idx]
        n = pre_subsample

    xyz = points[:, :3].astype(np.float64)   # use only XYZ for spatial distances
    selected = np.zeros(n_samples, dtype=np.int64)
    distances = np.full(n, np.inf, dtype=np.float64)
    selected[0] = np.random.randint(n)
    for i in range(1, n_samples):
        last = xyz[selected[i - 1]]
        d = np.sum((xyz - last) ** 2, axis=1)
        distances = np.minimum(distances, d)
        selected[i] = np.argmax(distances)
    return points[selected].astype(np.float32)


def extract_joint_positions(msg, joint_names: list) -> np.ndarray:
    """Extract positions in canonical order from a JointState message."""
    name_to_pos = dict(zip(msg.name, msg.position))
    out = []
    for name in joint_names:
        if name not in name_to_pos:
            raise KeyError(f"Joint '{name}' not found in message. Available: {list(name_to_pos.keys())}")
        out.append(name_to_pos[name])
    return np.array(out, dtype=np.float32)


def nearest_message(messages: list, query_ts: int, tolerance_ns: int):
    """
    Find the message with timestamp closest to query_ts within tolerance.
    messages: [(timestamp_ns, msg), ...]
    Returns msg or None if none within tolerance.
    """
    if not messages:
        return None
    timestamps = np.array([ts for ts, _ in messages], dtype=np.int64)
    diffs = np.abs(timestamps - query_ts)
    idx = np.argmin(diffs)
    if diffs[idx] <= tolerance_ns:
        return messages[idx][1]
    return None


def build_20hz_timeline(start_ns: int, end_ns: int, hz: int = 20) -> np.ndarray:
    step_ns = int(1e9 / hz)
    return np.arange(start_ns, end_ns, step_ns, dtype=np.int64)


# ---------------------------------------------------------------------------
# Per-trial processing
# ---------------------------------------------------------------------------

def process_trial(trial_n: int, cfg: dict) -> tuple:
    """
    Returns (point_clouds, agent_pos, actions) as numpy arrays,
    or (None, None, None) on failure.
    """
    base = Path(cfg["rosbag"]["base_dir"])
    cam_bag = str(base / cfg["rosbag"]["camera_subdir"] / f"trial{trial_n}_camera")
    rob_bag = str(base / cfg["rosbag"]["robot_subdir"] / f"trial{trial_n}_robot")

    print(f"\n[Trial {trial_n}] Reading camera bag: {cam_bag}")
    cam_msgs = read_all_messages(
        cam_bag,
        [cfg["topics"]["point_cloud"]]
    )

    print(f"[Trial {trial_n}] Reading robot bag: {rob_bag}")
    rob_msgs = read_all_messages(
        rob_bag,
        [
            cfg["topics"]["robot1_joints"],
            cfg["topics"]["robot2_joints"],
            cfg["topics"]["gripper1_joints"],
            cfg["topics"]["gripper2_joints"],
        ]
    )

    pc_msgs     = cam_msgs[cfg["topics"]["point_cloud"]]
    r1_msgs     = rob_msgs[cfg["topics"]["robot1_joints"]]
    r2_msgs     = rob_msgs[cfg["topics"]["robot2_joints"]]
    g1_msgs     = rob_msgs[cfg["topics"]["gripper1_joints"]]
    g2_msgs     = rob_msgs[cfg["topics"]["gripper2_joints"]]

    print(f"[Trial {trial_n}] Messages — PC:{len(pc_msgs)} R1:{len(r1_msgs)} R2:{len(r2_msgs)} G1:{len(g1_msgs)} G2:{len(g2_msgs)}")

    if not pc_msgs or not r1_msgs or not r2_msgs:
        print(f"[Trial {trial_n}] SKIP — missing messages")
        return None, None, None

    # Determine overlapping time range
    all_starts = [
        pc_msgs[0][0], r1_msgs[0][0], r2_msgs[0][0],
        g1_msgs[0][0] if g1_msgs else pc_msgs[0][0],
        g2_msgs[0][0] if g2_msgs else pc_msgs[0][0],
    ]
    all_ends = [
        pc_msgs[-1][0], r1_msgs[-1][0], r2_msgs[-1][0],
        g1_msgs[-1][0] if g1_msgs else pc_msgs[-1][0],
        g2_msgs[-1][0] if g2_msgs else pc_msgs[-1][0],
    ]
    t_start = max(all_starts)
    t_end   = min(all_ends)

    timeline = build_20hz_timeline(t_start, t_end, cfg["data"]["target_hz"])
    tol_ns   = int(cfg["data"]["sync_tolerance_ms"] * 1e6)

    jn  = cfg["joint_names"]
    ws  = cfg["workspace"]
    n_pts = cfg["data"]["n_points"]
    max_jd = cfg["action"]["max_joint_delta"]
    max_gd = cfg["action"]["max_gripper_delta"]

    point_clouds = []
    agent_pos_list = []

    print(f"[Trial {trial_n}] Processing {len(timeline)} frames at {cfg['data']['target_hz']}Hz ...")

    for ts in timeline:
        pc_msg = nearest_message(pc_msgs, ts, tol_ns)
        r1_msg = nearest_message(r1_msgs, ts, tol_ns)
        r2_msg = nearest_message(r2_msgs, ts, tol_ns)
        g1_msg = nearest_message(g1_msgs, ts, tol_ns)
        g2_msg = nearest_message(g2_msgs, ts, tol_ns)

        if pc_msg is None or r1_msg is None or r2_msg is None:
            continue

        # Point cloud
        xyzrgb = parse_pointcloud2_xyzrgb(pc_msg)
        xyzrgb = crop_workspace(xyzrgb, ws)
        pc     = fps_numpy(xyzrgb, n_pts)  # (1024, 6)

        # Joint positions in canonical order
        r1_pos = extract_joint_positions(r1_msg, jn["robot1"])   # (6,)
        r2_pos = extract_joint_positions(r2_msg, jn["robot2"])   # (6,)

        g1_pos = np.array([g1_msg.position[0]], dtype=np.float32) if g1_msg else np.zeros(1, dtype=np.float32)
        g2_pos = np.array([g2_msg.position[0]], dtype=np.float32) if g2_msg else np.zeros(1, dtype=np.float32)

        # agent_pos: [r1(6), g1(1), r2(6), g2(1)] = 14-dim
        ap = np.concatenate([r1_pos, g1_pos, r2_pos, g2_pos])  # (14,)

        point_clouds.append(pc)
        agent_pos_list.append(ap)

    if len(point_clouds) < 10:
        print(f"[Trial {trial_n}] SKIP — only {len(point_clouds)} valid frames")
        return None, None, None

    point_clouds = np.stack(point_clouds, axis=0).astype(np.float32)   # (T, 1024, 6)
    agent_pos    = np.stack(agent_pos_list, axis=0).astype(np.float32)  # (T, 14)

    # Compute actions as joint delta: action[t] = state[t+1] - state[t]
    # Last action repeats the second-to-last (episode end padding)
    actions = np.diff(agent_pos, axis=0)  # (T-1, 14)
    actions = np.concatenate([actions, actions[[-1]]], axis=0)  # (T, 14)

    # Clip to configured max deltas
    actions[:, :6]  = np.clip(actions[:, :6],  -max_jd, max_jd)   # robot1 joints
    actions[:, 6]   = np.clip(actions[:, 6],   -max_gd, max_gd)   # gripper1
    actions[:, 7:13] = np.clip(actions[:, 7:13], -max_jd, max_jd) # robot2 joints
    actions[:, 13]  = np.clip(actions[:, 13],  -max_gd, max_gd)   # gripper2

    print(f"[Trial {trial_n}] Done — {len(point_clouds)} frames, pc shape: {point_clouds.shape}")
    return point_clouds, agent_pos, actions


# ---------------------------------------------------------------------------
# Zarr writer
# ---------------------------------------------------------------------------

def write_zarr(zarr_path: str, episodes: list):
    """
    episodes: list of (point_clouds, agent_pos, actions) tuples
    Zarr structure:
      data/
        point_cloud   (total_steps, 1024, 6)  — XYZRGB, RGB normalised [0,1]
        state         (total_steps, 14)
        action        (total_steps, 14)
      meta/
        episode_ends  (n_episodes,)  — cumulative end indices
    """
    compressor = Blosc(cname="lz4", clevel=5)
    store = zarr.open(zarr_path, mode="w")

    total_steps = sum(len(pc) for pc, _, _ in episodes)
    n_ep = len(episodes)
    n_features = episodes[0][0].shape[-1]  # 6 for XYZRGB

    pc_arr  = store.zeros("data/point_cloud", shape=(total_steps, 1024, n_features), dtype="f4", chunks=(1, 1024, n_features), compressor=compressor)
    st_arr  = store.zeros("data/state",       shape=(total_steps, 14),      dtype="f4", chunks=(1, 14),      compressor=compressor)
    ac_arr  = store.zeros("data/action",      shape=(total_steps, 14),      dtype="f4", chunks=(1, 14),      compressor=compressor)
    ep_ends = store.zeros("meta/episode_ends", shape=(n_ep,), dtype="i8")

    cursor = 0
    for i, (pc, state, action) in enumerate(episodes):
        n = len(pc)
        pc_arr[cursor:cursor+n]  = pc
        st_arr[cursor:cursor+n]  = state
        ac_arr[cursor:cursor+n]  = action
        cursor += n
        ep_ends[i] = cursor

    print(f"\nZarr written to {zarr_path}")
    print(f"  Episodes:    {n_ep}")
    print(f"  Total steps: {total_steps}")
    print(f"  point_cloud: {pc_arr.shape}")
    print(f"  state:       {st_arr.shape}")
    print(f"  action:      {ac_arr.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "real_config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    trials = cfg["rosbag"]["trials"]

    episodes = []
    for trial_n in trials:
        pc, state, action = process_trial(trial_n, cfg)
        if pc is not None:
            episodes.append((pc, state, action))

    if not episodes:
        print("ERROR: no valid episodes processed. Check config paths and workspace bounds.")
        sys.exit(1)

    write_zarr(cfg["data"]["output_zarr"], episodes)


if __name__ == "__main__":
    main()

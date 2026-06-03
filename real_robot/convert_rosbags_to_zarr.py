#!/usr/bin/env python3
"""
convert_rosbags_to_zarr.py
==========================
Converts real-robot ROS2 bags into a single zarr dataset compatible with the
DP3 RobosuiteDataset format.

Supports two bag layouts:
  Legacy (split bags):
    camera/trial{n}_camera/   — /camera/camera/depth/color/points  (~15 Hz)
    Trajectory_Gripper/trial{n}_robot/ — joint_states topics
  Current (combined ZED bags, e.g. 28May):
    session_YYYYMMDD_HHMMSS_filtered/ — all topics in one bag:
      /zed/zed_node/depth/depth_registered  (32FC1 depth image)
      /zed/zed_node/depth/camera_info
      /zed/zed_node/rgb/color/rect/image
      /robot1/joint_states, /robot2/joint_states
      /gripper1/joint_states, /gripper2/joint_states

Processing per session:
  1. Read all messages from the combined bag via rosbag2_py
  2. Synchronise to a 20 Hz timeline using nearest-neighbour per-topic
  3. Reconstruct XYZRGB point cloud from depth + camera_info + RGB, apply z-filter
     and workspace crop, FPS downsample to 1024 pts
  4. Build agent_pos (14-dim) and action (joint delta, 14-dim)
  5. Append episode to zarr

Usage:
  python convert_rosbags_to_zarr.py [--config real_config.yaml]
"""

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
# GPU FPS (pytorch3d) — thread-safe via per-thread CUDA streams
# Falls back to per-frame CPU FPS if pytorch3d / CUDA not available.
# ---------------------------------------------------------------------------
try:
    from pytorch3d.ops import sample_farthest_points as _p3d_sfp
    import torch as _torch
    _HAVE_GPU_FPS = _torch.cuda.is_available()
    if _HAVE_GPU_FPS:
        print("[converter] pytorch3d GPU FPS available — fast mode enabled.")
    else:
        print("[converter] pytorch3d found but no CUDA — using CPU FPS.")
except ImportError:
    _HAVE_GPU_FPS = False
    print("[converter] pytorch3d not found — using CPU FPS (slow for n_points>=4096).")

_tls = threading.local()   # per-thread CUDA stream


def _fps_batch(crops: list, n_pts: int) -> list:
    """
    Batch FPS for a list of variable-size (N_i, C) numpy arrays.

    GPU path  — all frames in one pytorch3d CUDA call (~2 s for 200 frames).
    CPU path  — per-frame fps_numpy fallback.
    Thread-safe: each thread gets its own CUDA stream via _tls.

    n_pre (pre-subsample budget) scales with n_pts so coverage is always ≥2×:
      n_pts  512 → n_pre  8192  (16× margin)
      n_pts 1024 → n_pre  8192  ( 8× margin)
      n_pts 2048 → n_pre  8192  ( 4× margin)
      n_pts 4096 → n_pre  8192  ( 2× margin)
      n_pts 8192 → n_pre 16384  ( 2× margin)
    """
    if not crops:
        return []

    C     = crops[0].shape[1]
    n_pre = min(max(n_pts * 2, 8192), 32768)

    if not _HAVE_GPU_FPS:
        return [fps_numpy(c, n_pts) for c in crops]

    # Initialise per-thread CUDA stream on first use
    if not hasattr(_tls, 'stream'):
        _tls.stream = _torch.cuda.Stream()

    results       = [None] * len(crops)
    batch_i       = []
    batch_pts_np  = []
    batch_lens    = []

    for i, crop in enumerate(crops):
        n = len(crop)
        if n == 0 or n <= n_pts:
            # Edge case: too few points — CPU pad (rare)
            results[i] = fps_numpy(crop, n_pts)
            continue

        # Random pre-subsample to n_pre
        if n > n_pre:
            idx = np.random.choice(n, n_pre, replace=False)
            pre = crop[idx].astype(np.float32)
        else:
            pre = crop.astype(np.float32)

        real_len = len(pre)
        # Pad to n_pre so all frames have the same shape in the GPU batch
        if real_len < n_pre:
            pad = np.zeros((n_pre - real_len, C), dtype=np.float32)
            pre = np.vstack([pre, pad])

        batch_i.append(i)
        batch_pts_np.append(pre)
        batch_lens.append(real_len)

    if batch_pts_np:
        pts_np = np.stack(batch_pts_np)          # (B, n_pre, C)
        with _torch.cuda.stream(_tls.stream):
            pts_t  = _torch.from_numpy(pts_np).cuda()
            lens_t = _torch.tensor(batch_lens, dtype=_torch.int64).cuda()
            # FPS on XYZ only; returns (B, n_pts) indices
            _, idx_t = _p3d_sfp(pts_t[..., :3], lengths=lens_t, K=n_pts)
            # Gather all C channels
            sampled = pts_t.gather(
                1, idx_t.unsqueeze(-1).expand(-1, -1, C)
            ).cpu().numpy()                       # (B, n_pts, C)
        _tls.stream.synchronize()

        for j, orig_i in enumerate(batch_i):
            results[orig_i] = sampled[j]

    return results

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


def load_z_filter(config_path: Path = None) -> tuple:
    """Load z_min/z_max from z_filter_config.yaml (generated by z_filter_gui.py)."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "3D-Diffusion-Policy" / "z_filter_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            d = yaml.safe_load(f)
        return float(d["z_min"]), float(d["z_max"])
    return 0.4, 1.1   # safe defaults


def depth_to_pointcloud_xyzrgb(depth_msg, cam_info_msg, rgb_msg,
                                z_min: float, z_max: float) -> np.ndarray:
    """
    Reconstruct an (N, 6) XYZRGB point cloud from ZED depth + camera_info + RGB.
    Matches the pipeline used in policy_executor.py at inference time.
    """
    from sensor_msgs.msg import Image as ImageMsg
    h, w = depth_msg.height, depth_msg.width
    depth = np.frombuffer(bytes(depth_msg.data), dtype=np.float32).reshape(h, w)

    fx = cam_info_msg.k[0]; fy = cam_info_msg.k[4]
    cx = cam_info_msg.k[2]; cy = cam_info_msg.k[5]

    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32)
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy

    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    valid = np.isfinite(pts[:, 2]) & (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    pts = pts[valid]

    raw = np.frombuffer(bytes(rgb_msg.data), dtype=np.uint8)
    enc = rgb_msg.encoding
    if enc in ("bgra8", "rgba8"):
        img = raw.reshape(h, w, 4)
        r_ch = img[:, :, 2] if enc == "bgra8" else img[:, :, 0]
        g_ch = img[:, :, 1]
        b_ch = img[:, :, 0] if enc == "bgra8" else img[:, :, 2]
    else:  # rgb8 / bgr8
        img = raw.reshape(h, w, 3)
        r_ch = img[:, :, 2] if enc == "bgr8" else img[:, :, 0]
        g_ch = img[:, :, 1]
        b_ch = img[:, :, 0] if enc == "bgr8" else img[:, :, 2]

    r = r_ch.reshape(-1)[valid].astype(np.float32) / 255.0
    g = g_ch.reshape(-1)[valid].astype(np.float32) / 255.0
    b = b_ch.reshape(-1)[valid].astype(np.float32) / 255.0
    return np.column_stack([pts, r, g, b])


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

def process_session(session_dir: Path, cfg: dict,
                    z_min: float, z_max: float) -> tuple:
    """
    Process one combined ZED bag session.
    Returns (point_clouds, agent_pos, actions) or (None, None, None) on failure.

    Expected topics inside the bag:
      cfg["topics"]["depth"]          — 32FC1 depth image
      cfg["topics"]["cam_info"]       — camera intrinsics
      cfg["topics"]["rgb"]            — colour image
      cfg["topics"]["robot1_joints"], ["robot2_joints"],
      cfg["topics"]["gripper1_joints"], ["gripper2_joints"]
    """
    label = session_dir.name
    t = cfg["topics"]

    all_topics = [
        t["depth"], t["cam_info"], t["rgb"],
        t["robot1_joints"], t["robot2_joints"],
        t["gripper1_joints"], t["gripper2_joints"],
    ]

    print(f"\n[{label}] Reading bag ...")
    msgs = read_all_messages(str(session_dir), all_topics)

    depth_msgs    = msgs[t["depth"]]
    caminfo_msgs  = msgs[t["cam_info"]]
    rgb_msgs      = msgs[t["rgb"]]
    r1_msgs       = msgs[t["robot1_joints"]]
    r2_msgs       = msgs[t["robot2_joints"]]
    g1_msgs       = msgs[t["gripper1_joints"]]
    g2_msgs       = msgs[t["gripper2_joints"]]

    print(f"[{label}] Messages — depth:{len(depth_msgs)} rgb:{len(rgb_msgs)} "
          f"R1:{len(r1_msgs)} R2:{len(r2_msgs)} G1:{len(g1_msgs)} G2:{len(g2_msgs)}")

    if not depth_msgs or not caminfo_msgs or not rgb_msgs or not r1_msgs or not r2_msgs:
        print(f"[{label}] SKIP — missing essential topics")
        return None, None, None

    # Overlapping time window across all streams
    streams_start = [depth_msgs[0][0], r1_msgs[0][0], r2_msgs[0][0]]
    streams_end   = [depth_msgs[-1][0], r1_msgs[-1][0], r2_msgs[-1][0]]
    if g1_msgs:
        streams_start.append(g1_msgs[0][0]);  streams_end.append(g1_msgs[-1][0])
    if g2_msgs:
        streams_start.append(g2_msgs[0][0]);  streams_end.append(g2_msgs[-1][0])

    t_start = max(streams_start)
    t_end   = min(streams_end)
    if t_start >= t_end:
        print(f"[{label}] SKIP — no overlapping time window")
        return None, None, None

    timeline = build_20hz_timeline(t_start, t_end, cfg["data"]["target_hz"])
    tol_ns   = int(cfg["data"]["sync_tolerance_ms"] * 1e6)

    jn    = cfg["joint_names"]
    ws    = cfg["workspace"]
    n_pts = cfg["data"]["n_points"]
    max_jd = cfg["action"]["max_joint_delta"]
    max_gd = cfg["action"]["max_gripper_delta"]

    # Use the most recent camera_info (intrinsics don't change mid-session)
    cam_info_msg = caminfo_msgs[-1][1]

    raw_crops      = []   # (N_i, 6) after z-filter + workspace crop, before FPS
    agent_pos_list = []

    t_cpu = time.time()
    print(f"[{label}] Processing {len(timeline)} frames at {cfg['data']['target_hz']}Hz ...")

    for ts in timeline:
        depth_msg = nearest_message(depth_msgs, ts, tol_ns)
        rgb_msg   = nearest_message(rgb_msgs,   ts, tol_ns)
        r1_msg    = nearest_message(r1_msgs,    ts, tol_ns)
        r2_msg    = nearest_message(r2_msgs,    ts, tol_ns)
        g1_msg    = nearest_message(g1_msgs,    ts, tol_ns)
        g2_msg    = nearest_message(g2_msgs,    ts, tol_ns)

        if depth_msg is None or rgb_msg is None or r1_msg is None or r2_msg is None:
            continue

        # Reconstruct + crop (no FPS yet — collected for batch GPU FPS below)
        xyzrgb = depth_to_pointcloud_xyzrgb(depth_msg, cam_info_msg, rgb_msg, z_min, z_max)
        xyzrgb = crop_workspace(xyzrgb, ws)
        raw_crops.append(xyzrgb)

        # Joint positions in canonical order
        r1_pos = extract_joint_positions(r1_msg, jn["robot1"])
        r2_pos = extract_joint_positions(r2_msg, jn["robot2"])
        g1_pos = np.array([g1_msg.position[0]], dtype=np.float32) if g1_msg else np.zeros(1, dtype=np.float32)
        g2_pos = np.array([g2_msg.position[0]], dtype=np.float32) if g2_msg else np.zeros(1, dtype=np.float32)
        agent_pos_list.append(np.concatenate([r1_pos, g1_pos, r2_pos, g2_pos]))

    if len(raw_crops) < 10:
        print(f"[{label}] SKIP — only {len(raw_crops)} valid frames")
        return None, None, None

    cpu_ms = (time.time() - t_cpu) * 1000

    # Batch GPU FPS for all frames in this session at once
    t_fps = time.time()
    pc_list = _fps_batch(raw_crops, n_pts)
    fps_ms  = (time.time() - t_fps) * 1000
    fps_tag = "GPU" if _HAVE_GPU_FPS else "CPU"
    print(f"[{label}]  cpu={cpu_ms:.0f}ms  fps({fps_tag})={fps_ms:.0f}ms  "
          f"({len(raw_crops)} frames × {n_pts} pts)")

    point_clouds = np.stack(pc_list,           axis=0).astype(np.float32)   # (T, n_pts, 6)
    agent_pos    = np.stack(agent_pos_list,    axis=0).astype(np.float32)  # (T, 14)

    actions = np.diff(agent_pos, axis=0)
    actions = np.concatenate([actions, actions[[-1]]], axis=0)   # (T, 14)
    actions[:, :6]   = np.clip(actions[:, :6],   -max_jd, max_jd)
    actions[:, 6]    = np.clip(actions[:, 6],    -max_gd, max_gd)
    actions[:, 7:13] = np.clip(actions[:, 7:13], -max_jd, max_jd)
    actions[:, 13]   = np.clip(actions[:, 13],   -max_gd, max_gd)

    print(f"[{label}] Done — {len(point_clouds)} frames")
    return point_clouds, agent_pos, actions


# ---------------------------------------------------------------------------
# Zarr writer
# ---------------------------------------------------------------------------

def write_zarr(zarr_path: str, episodes: list):
    """
    episodes: list of (point_clouds, agent_pos, actions) tuples
    Zarr structure:
      data/
        point_cloud   (total_steps, n_pts, 6)  — XYZRGB, RGB normalised [0,1]
        state         (total_steps, 14)
        action        (total_steps, 14)
      meta/
        episode_ends  (n_episodes,)  — cumulative end indices
    """
    compressor = Blosc(cname="lz4", clevel=5)
    store = zarr.open(zarr_path, mode="w")

    total_steps = sum(len(pc) for pc, _, _ in episodes)
    n_ep        = len(episodes)
    n_pts_stored = episodes[0][0].shape[1]   # read from actual data, not hardcoded
    n_features   = episodes[0][0].shape[2]   # 6 for XYZRGB

    pc_arr  = store.zeros("data/point_cloud", shape=(total_steps, n_pts_stored, n_features), dtype="f4", chunks=(1, n_pts_stored, n_features), compressor=compressor)
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
    parser.add_argument("--config",  default=str(Path(__file__).parent / "real_config.yaml"))
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel session workers (default 1). "
                             "2–4 recommended when GPU FPS is available. "
                             "Each worker reads its bag and runs GPU FPS concurrently "
                             "via separate per-thread CUDA streams.")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    base_dir = Path(cfg["rosbag"]["base_dir"])
    sessions = cfg["rosbag"]["sessions"]

    z_min, z_max = load_z_filter()
    print(f"Z-filter: z_min={z_min}  z_max={z_max}")
    print(f"Workers:  {args.workers}  |  GPU FPS: {_HAVE_GPU_FPS}\n")

    # Build list of valid session paths
    session_dirs = []
    for sname in sessions:
        sd = base_dir / sname
        if sd.exists():
            session_dirs.append(sd)
        else:
            print(f"[SKIP] {sd} not found")

    t_total = time.time()
    episodes = []
    lock     = threading.Lock()

    def _run_session(session_dir):
        pc, state, action = process_session(session_dir, cfg, z_min, z_max)
        if pc is not None:
            with lock:
                episodes.append((pc, state, action))

    if args.workers <= 1:
        for sd in session_dirs:
            _run_session(sd)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_session, sd): sd.name for sd in session_dirs}
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc:
                    print(f"[ERROR] {futures[fut]}: {exc}")

    elapsed = time.time() - t_total
    print(f"\nAll sessions done in {elapsed:.0f}s  ({elapsed/60:.1f} min)  "
          f"— {len(episodes)} valid episodes")

    if not episodes:
        print("ERROR: no valid episodes processed. Check config paths and workspace bounds.")
        sys.exit(1)

    write_zarr(cfg["data"]["output_zarr"], episodes)


if __name__ == "__main__":
    main()

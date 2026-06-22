from typing import Dict, Optional
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import zarr
from numcodecs import Blosc

from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset


# ---------------------------------------------------------------------------
# 3-D point cloud augmentation helpers
# ---------------------------------------------------------------------------

def _random_yaw_matrix() -> np.ndarray:
    """Random rotation matrix around the Z (gravity) axis, uniform in [-15°, +15°]."""
    angle = np.random.uniform(-np.pi / 12, np.pi / 12)   # ±15 deg
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def augment_point_cloud_sequence(point_cloud: np.ndarray) -> np.ndarray:
    """
    Apply randomised augmentations to a (T, N, C) point cloud sequence.

    Augmentation order (must not be reordered — see inline notes):
      1. Yaw rotation  — before scale/jitter so noise is not amplified by R
      2. XY translation — same SE(3) step; point-cloud-only (actions are joint
                          deltas in joint space, not Cartesian poses, so no
                          action transform is required or correct)
      3. Uniform scale ±10%
      4. XYZ jitter σ=0.005 m — after scale so magnitude stays sensor-relative
      5. RGB brightness jitter ±10% (channels 3-5 if C≥6)
      6. Random dropout 5% — last, so FPS/padding downstream sees full density

    Returns augmented copy; does NOT modify the input array.
    """
    pc = point_cloud.copy()
    T, N, C = pc.shape

    # 1 & 2 — Yaw rotation + XY translation (point cloud only)
    R = _random_yaw_matrix()                                   # (3, 3)
    t = np.array([np.random.uniform(-0.03, 0.03),              # ±3 cm X
                  np.random.uniform(-0.03, 0.03),              # ±3 cm Y
                  0.0], dtype=np.float32)                      # no Z shift
    pc[..., :3] = pc[..., :3] @ R.T + t                       # broadcast over T, N

    # 3 — Uniform scale ±10%
    scale = np.float32(np.random.uniform(0.90, 1.10))
    pc[..., :3] *= scale

    # 4 — XYZ jitter
    pc[..., :3] += (np.random.randn(T, N, 3) * 0.005).astype(np.float32)

    # 5 — RGB brightness jitter (only when colour channels present)
    if C >= 6:
        brightness = np.float32(np.random.uniform(0.90, 1.10))
        pc[..., 3:6] = np.clip(pc[..., 3:6] * brightness, 0.0, 1.0)

    # 6 — 5% point dropout (zero-out selected points)
    dropout_mask = np.random.rand(T, N) < 0.05
    pc[dropout_mask] = 0.0

    return pc


# ---------------------------------------------------------------------------
# FPS helpers
# ---------------------------------------------------------------------------

def fps_hybrid_numpy(points: np.ndarray, n_out: int,
                     presample_ratio: float = 1.5) -> np.ndarray:
    """
    Hybrid FPS: random pre-subsample to ceil(presample_ratio * n_out) then greedy FPS.
    Only used as a fallback — normal training goes through the on-disk cache.
    """
    n_in, C = points.shape
    if n_in == n_out:
        return points.astype(np.float32)
    if n_in < n_out:
        extra = np.random.choice(n_in, n_out - n_in, replace=True)
        return np.vstack([points, points[extra]]).astype(np.float32)

    n_pre = min(n_in, max(n_out, int(np.ceil(presample_ratio * n_out))))
    if n_pre < n_in:
        pre_idx = np.random.choice(n_in, n_pre, replace=False)
        pts = points[pre_idx]
    else:
        pts = points

    n    = len(pts)
    xyz  = pts[:, :3].astype(np.float64)
    sel  = np.zeros(n_out, dtype=np.int64)
    dist = np.full(n, np.inf, dtype=np.float64)
    sel[0] = np.random.randint(n)
    for i in range(1, n_out):
        d = np.sum((xyz - xyz[sel[i - 1]]) ** 2, axis=1)
        np.minimum(dist, d, out=dist)
        sel[i] = int(np.argmax(dist))
    return pts[sel].astype(np.float32)


# ---------------------------------------------------------------------------
# On-disk FPS cache (Option B)
# ---------------------------------------------------------------------------

def _peek_stored_n_points(zarr_path: str) -> int:
    """Read n_points from zarr metadata only — no data loaded."""
    zarray = Path(zarr_path) / "data" / "point_cloud" / ".zarray"
    if zarray.exists():
        with open(zarray) as f:
            return int(json.load(f)["shape"][1])
    # Fallback: open the store (still no data copy)
    z = zarr.open(zarr_path, "r")
    return int(z["data/point_cloud"].shape[1])


def _fps_cache_path(zarr_path: str, n_points: int) -> str:
    """
    Derive the sidecar cache path.
    /path/to/real_cable_pull_28may.zarr  →  /path/to/real_cable_pull_28may_4096.zarr
    """
    p = Path(zarr_path)
    stem = p.name[:-5] if p.name.endswith(".zarr") else p.name
    return str(p.parent / f"{stem}_{n_points}.zarr")


def _build_fps_cache(master_path: str, cache_path: str, n_points: int):
    """
    GPU batch FPS on the master zarr's point clouds, write a sidecar zarr.
    Runs once; subsequent training loads the sidecar directly (zero FPS overhead).

    Uses pytorch3d GPU FPS in batches of BATCH_SIZE frames.
    Falls back to per-frame cpu fps_hybrid_numpy if pytorch3d/CUDA unavailable.
    """
    try:
        from pytorch3d.ops import sample_farthest_points as _sfp
        import torch as _torch
        _use_gpu = _torch.cuda.is_available()
    except ImportError:
        _use_gpu = False

    master     = zarr.open(master_path, "r")
    total      = int(master["data/point_cloud"].shape[0])
    stored_n   = int(master["data/point_cloud"].shape[1])
    C          = int(master["data/point_cloud"].shape[2])
    n_ep       = int(master["meta/episode_ends"].shape[0])
    method     = "GPU (pytorch3d)" if _use_gpu else "CPU (numpy)"

    print(f"\n[FPS Cache] Building {n_points}-pt cache from {stored_n}-pt master")
    print(f"  Frames  : {total}")
    print(f"  Method  : {method}")
    print(f"  Output  : {cache_path}")
    print(f"  (This runs once — subsequent training loads the cache directly)\n")

    pc_out     = np.empty((total, n_points, C), dtype=np.float32)
    BATCH      = 256   # frames per GPU call (~50 MB GPU mem per batch at 8192 stored)
    t0         = time.time()

    for start in range(0, total, BATCH):
        end   = min(start + BATCH, total)
        chunk = master["data/point_cloud"][start:end]   # (B, stored_n, C)

        if _use_gpu:
            with _torch.no_grad():
                pts_t   = _torch.from_numpy(chunk).float().cuda()          # (B, N, C)
                _, idx  = _sfp(pts_t[..., :3], K=n_points)                 # (B, K)
                sampled = pts_t.gather(
                    1, idx.unsqueeze(-1).expand(-1, -1, C)
                ).cpu().numpy()                                              # (B, K, C)
        else:
            sampled = np.stack([fps_hybrid_numpy(chunk[i], n_points)
                                for i in range(len(chunk))])

        pc_out[start:end] = sampled

        if (start // BATCH) % 10 == 0 or end == total:
            elapsed = time.time() - t0
            pct     = 100 * end / total
            eta     = elapsed / max(end, 1) * (total - end)
            print(f"  {end:>6}/{total}  ({pct:5.1f}%)  "
                  f"elapsed={elapsed:5.0f}s  eta={eta:5.0f}s")

    # Write cache zarr with downsampled point cloud + same state/action/episode_ends
    compressor = Blosc(cname="lz4", clevel=5)
    cache      = zarr.open(cache_path, mode="w")

    pc_arr = cache.zeros("data/point_cloud",
                         shape=(total, n_points, C), dtype="f4",
                         chunks=(1, n_points, C), compressor=compressor)
    st_arr = cache.zeros("data/state",
                         shape=master["data/state"].shape, dtype="f4",
                         chunks=(1, 14), compressor=compressor)
    ac_arr = cache.zeros("data/action",
                         shape=master["data/action"].shape, dtype="f4",
                         chunks=(1, 14), compressor=compressor)
    ep_arr = cache.zeros("meta/episode_ends",
                         shape=(n_ep,), dtype="i8")

    pc_arr[:] = pc_out
    st_arr[:] = master["data/state"][:]
    ac_arr[:] = master["data/action"][:]
    ep_arr[:] = master["meta/episode_ends"][:]

    total_s = time.time() - t0
    print(f"\n[FPS Cache] Done in {total_s:.0f}s — {cache_path}\n")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RobosuiteDataset(BaseDataset):
    def __init__(self,
                 zarr_path,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None,
                 n_points: Optional[int] = None,
                 ):
        super().__init__()

        n_points_int = int(n_points) if n_points is not None else None

        # ── Resolve which zarr to load ────────────────────────────────────────
        # If n_points < stored_n_points: use (or build) a sidecar cache zarr
        # so __getitem__ never has to run FPS — it just loads pre-downsampled data.
        load_path = zarr_path
        if n_points_int is not None:
            peek = _peek_stored_n_points(zarr_path)
            if n_points_int < peek:
                cache = _fps_cache_path(zarr_path, n_points_int)
                if not Path(cache).exists():
                    _build_fps_cache(zarr_path, cache, n_points_int)
                else:
                    print(f"[RobosuiteDataset] FPS cache found — loading directly "
                          f"({n_points_int} pts): {cache}")
                load_path = cache
            elif n_points_int > peek:
                print(f"[RobosuiteDataset] WARNING: n_points={n_points_int} > "
                      f"stored={peek}. Points will be padded with repeats. "
                      f"Re-run zarr conversion with a higher n_points.")

        # ── Load zarr (cache or master) ───────────────────────────────────────
        self.replay_buffer = ReplayBuffer.copy_from_path(
            load_path, keys=["state", "action", "point_cloud"])

        self.stored_n_points: int = self.replay_buffer["point_cloud"].shape[1]
        self.n_points: int = n_points_int if n_points_int is not None \
            else self.stored_n_points

        print(f"[RobosuiteDataset] loaded={self.stored_n_points} pts  "
              f"training n_points={self.n_points}  "
              f"total_steps={self.replay_buffer['point_cloud'].shape[0]}")

        # ── Train / val split ─────────────────────────────────────────────────
        val_mask   = get_val_mask(n_episodes=self.replay_buffer.n_episodes,
                                  val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        if max_train_episodes is not None:
            train_mask[max_train_episodes:] = False

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)

        self.train_mask  = train_mask
        self.horizon     = horizon
        self.pad_before  = pad_before
        self.pad_after   = pad_after
        self.augment     = True

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        val_set.augment    = False
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action":      self.replay_buffer["action"],
            "agent_pos":   self.replay_buffer["state"],
            "point_cloud": self.replay_buffer["point_cloud"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _augment_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        return augment_point_cloud_sequence(point_cloud)

    def _sample_to_data(self, sample):
        agent_pos   = sample["state"].astype(np.float32)
        point_cloud = sample["point_cloud"].astype(np.float32)   # (T, n_points, C)
        action      = sample["action"].astype(np.float32)

        # FPS is only reached when n_points > stored (padding case) or if the
        # cache was bypassed.  Normal training hits this branch never.
        if self.n_points != self.stored_n_points:
            T, _, C = point_cloud.shape
            pc_out = np.empty((T, self.n_points, C), dtype=np.float32)
            for t in range(T):
                pc_out[t] = fps_hybrid_numpy(point_cloud[t], self.n_points)
            point_cloud = pc_out

        if self.augment:
            point_cloud = self._augment_point_cloud(point_cloud)

        return {
            "obs": {
                "point_cloud": point_cloud,
                "agent_pos":   agent_pos,
            },
            "action": action,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data   = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)

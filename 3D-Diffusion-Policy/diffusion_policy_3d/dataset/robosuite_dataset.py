from typing import Dict, Optional
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset


def fps_hybrid_numpy(points: np.ndarray, n_out: int,
                     presample_ratio: float = 1.5) -> np.ndarray:
    """
    Hybrid FPS: random pre-subsample to ceil(presample_ratio * n_out), then
    greedy Farthest Point Sampling to n_out.

    This is far cheaper than running FPS on the full stored cloud because the
    O(n_in * n_out) FPS cost is replaced by O(n_pre * n_out) where
    n_pre = ceil(presample_ratio * n_out) << n_in.

    Approximate wall-clock times on a modern CPU (presample_ratio=1.5):
      8192 → 4096 :  ~0.25 s   (n_pre=6144, 4096 FPS iters)
      8192 → 2048 :  ~0.06 s
      8192 → 1024 :  ~0.015 s
      8192 → 512  :  ~0.004 s

    For n_out=4096 with num_workers=8 this adds ~0.25 s per sample in the
    DataLoader worker.  If that bottlenecks training, reduce num_workers or
    set n_points <= 2048.
    """
    n_in, C = points.shape

    if n_in == n_out:
        return points.astype(np.float32)

    # Too few points: pad by repeating random existing points
    if n_in < n_out:
        extra = np.random.choice(n_in, n_out - n_in, replace=True)
        return np.vstack([points, points[extra]]).astype(np.float32)

    # Random pre-subsample to a manageable size before FPS
    n_pre = min(n_in, max(n_out, int(np.ceil(presample_ratio * n_out))))
    if n_pre < n_in:
        pre_idx = np.random.choice(n_in, n_pre, replace=False)
        pts = points[pre_idx]
    else:
        pts = points

    n = len(pts)
    xyz  = pts[:, :3].astype(np.float64)
    sel  = np.zeros(n_out, dtype=np.int64)
    dist = np.full(n, np.inf, dtype=np.float64)
    sel[0] = np.random.randint(n)
    for i in range(1, n_out):
        d = np.sum((xyz - xyz[sel[i - 1]]) ** 2, axis=1)
        np.minimum(dist, d, out=dist)
        sel[i] = int(np.argmax(dist))
    return pts[sel].astype(np.float32)


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

        # 1. Load the Zarr file we created in the conversion script
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud'])

        # Stored point count (fixed at zarr creation time)
        self.stored_n_points: int = self.replay_buffer['point_cloud'].shape[1]

        # Target point count for training (may be <= stored_n_points)
        if n_points is None:
            self.n_points = self.stored_n_points
        else:
            self.n_points = int(n_points)
            if self.n_points > self.stored_n_points:
                print(
                    f"[RobosuiteDataset] WARNING: n_points={self.n_points} > "
                    f"stored={self.stored_n_points}. "
                    f"Points will be padded with repeats. "
                    f"Re-run zarr conversion with a higher n_points to avoid this."
                )
        print(f"[RobosuiteDataset] stored={self.stored_n_points}  "
              f"training n_points={self.n_points}")

        # 2. Split train and validation datasets
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask

        if max_train_episodes is not None:
            train_mask[max_train_episodes:] = False

        # 3. Initialize the sequence sampler
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.augment = True

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
        val_set.augment = False
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        """
        DP3 normalizes the inputs (state and action) to [-1, 1] before passing 
        them to the diffusion model. Point clouds are usually centered separately.
        """
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'],
            'point_cloud': self.replay_buffer['point_cloud']  # <--- ADD THIS LINE!
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _augment_point_cloud(self, point_cloud):
        T, N, C = point_cloud.shape
        # XYZ: 10mm spatial jitter
        point_cloud[..., :3] += np.random.randn(T, N, 3).astype(np.float32) * 0.01
        # RGB: separate colour jitter (if present)
        if C > 3:
            point_cloud[..., 3:] += np.random.randn(T, N, C - 3).astype(np.float32) * 0.02
            point_cloud[..., 3:] = np.clip(point_cloud[..., 3:], 0.0, 1.0)
        # Random point dropout: 25% of points zeroed per timestep
        dropout_mask = np.random.rand(T, N) < 0.25
        point_cloud[dropout_mask] = 0.0
        # Random uniform scale: ±5% on XYZ only
        scale = np.float32(np.random.uniform(0.95, 1.05))
        point_cloud[..., :3] *= scale
        return point_cloud

    def _sample_to_data(self, sample):
        """
        Maps the raw sample dict to the policy format.
        'state' → 'agent_pos'; point cloud is FPS-downsampled to self.n_points.
        FPS is applied before augmentation so augmentation runs on the smaller cloud.
        """
        agent_pos   = sample['state'].astype(np.float32)
        point_cloud = sample['point_cloud'].astype(np.float32)  # (T, stored_N, C)
        action      = sample['action'].astype(np.float32)

        # FPS downsample each timestep independently
        if self.n_points != self.stored_n_points:
            T, _, C = point_cloud.shape
            pc_out = np.empty((T, self.n_points, C), dtype=np.float32)
            for t in range(T):
                pc_out[t] = fps_hybrid_numpy(point_cloud[t], self.n_points)
            point_cloud = pc_out

        if self.augment:
            point_cloud = self._augment_point_cloud(point_cloud)

        data = {
            'obs': {
                'point_cloud': point_cloud,
                'agent_pos': agent_pos,
            },
            'action': action,
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
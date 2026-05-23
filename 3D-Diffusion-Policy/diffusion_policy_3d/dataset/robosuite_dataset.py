from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset

class RobosuiteDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        super().__init__()
        
        # 1. Load the Zarr file we created in the conversion script
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud'])
            
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
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        val_set.augment = False
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        # Per-axis XYZ scaling (last_n_dims=1) stretches geometry (sphere → ellipsoid),
        # confusing PointNet. PointNet's internal LayerNorm handles meter-scale natively.
        normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _augment_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        point_cloud: (T, N, C) where C=3 (XYZ) or C=6 (XYZRGB).
        A single consistent transform is applied across the T observation window.
        """
        T, N, C = point_cloud.shape
        result = point_cloud.copy()
        xyz = result[..., :3]

        # Gaussian jitter: 5mm std
        xyz += np.random.randn(T, N, 3).astype(np.float32) * 0.005

        # Random yaw (Z-axis) rotation — single angle for the whole T-step window.
        # Assumes camera Z ≈ vertical (valid for downward-facing ZED).
        theta = np.random.uniform(-np.pi, np.pi)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0.0],
                      [s,  c, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        xyz[:] = xyz @ R.T

        # Translation jitter: 2cm — accounts for scene-to-scene position drift
        xyz += np.random.randn(3).astype(np.float32) * 0.02

        # Uniform scale: ±5%
        xyz *= np.float32(np.random.uniform(0.95, 1.05))

        # Point dropout via resampling (not zeroing).
        # Zeroing sends dropped points to the origin → after normalization they become
        # Z-outliers (e.g. -1.5 for typical depth range) that corrupt max-pool.
        for t in range(T):
            dropout_mask = np.random.rand(N) < 0.15
            n_drop = int(dropout_mask.sum())
            if n_drop > 0:
                valid_idx = np.where(~dropout_mask)[0]
                if len(valid_idx) > 0:
                    resample_idx = np.random.choice(valid_idx, size=n_drop, replace=True)
                    result[t, dropout_mask] = result[t, resample_idx]

        return result

    def _sample_to_data(self, sample):
        """
        Maps the raw sample dictionary to the format expected by the policy.
        Notice how 'state' from the Zarr is mapped to 'agent_pos'.
        """
        agent_pos = sample['state'].astype(np.float32)
        point_cloud = sample['point_cloud'].astype(np.float32)
        action = sample['action'].astype(np.float32)

        if self.augment:
            point_cloud = self._augment_point_cloud(point_cloud)

        data = {
            'obs': {
                'point_cloud': point_cloud,
                'agent_pos': agent_pos,
            },
            'action': action
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
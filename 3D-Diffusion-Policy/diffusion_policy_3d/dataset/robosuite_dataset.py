from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
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

    def _sample_to_data(self, sample):
        """
        Maps the raw sample dictionary to the format expected by the policy.
        Notice how 'state' from the Zarr is mapped to 'agent_pos'.
        """
        agent_pos = sample['state'].astype(np.float32) 
        point_cloud = sample['point_cloud'].astype(np.float32)
        action = sample['action'].astype(np.float32)

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
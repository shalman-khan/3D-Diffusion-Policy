"""
Evaluation runner for TwoArmLift with dual-arm joint-space control.

Creates the TwoArmLift environment with two Pandas in opposed configuration,
birdview RGB-D camera, and joint position control. Runs the policy in
closed-loop and reports success rate and reward.
"""

import sys
import os
# Ensure the full robosuite installation (with camera_utils) is on the path
# before the stub package installed in this conda env is used.
_ROBOSUITE_ROOT = "/home/rosi/robosuite"
if _ROBOSUITE_ROOT not in sys.path:
    sys.path.insert(0, _ROBOSUITE_ROOT)

import wandb
import numpy as np
import torch
import collections
import tqdm
import mujoco

import robosuite as suite
import robosuite.utils.camera_utils as camera_utils
from diffusion_policy_3d.env_runner.base_runner import BaseRunner

try:
    from pytorch3d.ops import sample_farthest_points
except ImportError:
    sample_farthest_points = None

# ─────────────── Camera & Point Cloud Config ─────────────────

CAMERA_NAME = "birdview"
CAMERA_HEIGHT = 256
CAMERA_WIDTH = 256
NUM_POINTS = 1024

# Joint position controller config (must match data collection).
# Arm name ("right") must be directly under "body_parts" when passing
# a raw dict — NOT nested inside an "arms" sub-key.
JOINT_POS_CONTROLLER_CONFIG = {
    "type": "BASIC",
    "body_parts": {
        "right": {
            "type": "JOINT_POSITION",
            "input_max": 1,
            "input_min": -1,
            "output_max": 0.05,
            "output_min": -0.05,
            "kp": 50,
            "damping_ratio": 1,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300],
            "damping_ratio_limits": [0, 10],
            "qpos_limits": None,
            "interpolation": None,
            "ramp_ratio": 0.2,
            "gripper": {"type": "GRIP"},
        }
    },
}


def depth_to_pointcloud(depth_img, camera_intrinsics, camera_extrinsics):
    depth_img = np.squeeze(depth_img)
    h, w = depth_img.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.flatten()
    y = y.flatten()
    depth = depth_img.flatten()

    valid = depth > 0
    x, y, depth = x[valid], y[valid], depth[valid]

    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]

    z_cam = depth
    x_cam = (x - cx) * z_cam / fx
    y_cam = (y - cy) * z_cam / fy

    points_cam = np.vstack((x_cam, y_cam, z_cam, np.ones_like(z_cam)))
    cam_to_world = np.linalg.inv(camera_extrinsics)
    points_world = (cam_to_world @ points_cam)[:3, :].T

    return points_world


def downsample_pc(point_cloud, num_points=1024):
    if len(point_cloud) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)

    if len(point_cloud) < num_points:
        pad_size = num_points - len(point_cloud)
        pad_idx = np.random.choice(len(point_cloud), pad_size, replace=True)
        point_cloud = np.vstack([point_cloud, point_cloud[pad_idx]])

    if sample_farthest_points is not None:
        pc_tensor = torch.tensor(point_cloud, dtype=torch.float32).unsqueeze(0)
        if torch.cuda.is_available():
            pc_tensor = pc_tensor.cuda()
        sampled_pc, _ = sample_farthest_points(pc_tensor, K=num_points)
        return sampled_pc.squeeze(0).cpu().numpy()
    else:
        idx = np.random.choice(len(point_cloud), num_points, replace=False)
        return point_cloud[idx]


class RobosuiteTwoArmLiftRunner(BaseRunner):
    def __init__(self, output_dir, n_train=20, max_steps=400,
                 n_obs_steps=2, n_action_steps=8, fps=20,
                 task_name="TwoArmLift", **kwargs):
        super().__init__(output_dir)
        self.output_dir = output_dir
        self.n_train = n_train
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.task_name = task_name

        # Create the dual-arm environment for evaluation
        self.env = suite.make(
            env_name=self.task_name,
            robots=["UR10e", "UR10e"],
            env_configuration="opposed",
            controller_configs=[JOINT_POS_CONTROLLER_CONFIG, JOINT_POS_CONTROLLER_CONFIG],
            gripper_types="Robotiq140Gripper",
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            use_object_obs=True,
            camera_names=CAMERA_NAME,
            camera_depths=True,
            camera_heights=CAMERA_HEIGHT,
            camera_widths=CAMERA_WIDTH,
            control_freq=fps,
            ignore_done=True,
            reward_shaping=True,
        )

    def _extract_obs(self, obs):
        """Extract point cloud and dual-arm proprioceptive state."""
        # Point cloud from birdview depth camera
        depth_key = f"{CAMERA_NAME}_depth"
        depth_map = obs[depth_key]
        depth_map = np.clip(depth_map, 0.0, 1.0)
        real_depth = camera_utils.get_real_depth_map(self.env.sim, depth_map)
        intrinsics = camera_utils.get_camera_intrinsic_matrix(
            self.env.sim, CAMERA_NAME, CAMERA_HEIGHT, CAMERA_WIDTH
        )
        extrinsics = camera_utils.get_camera_extrinsic_matrix(self.env.sim, CAMERA_NAME)

        pc = depth_to_pointcloud(real_depth, intrinsics, extrinsics)
        pc = downsample_pc(pc, NUM_POINTS)

        # Dual-arm proprioception [24]:
        #   robot0_joint_pos(6) + robot0_gripper_qpos(6) + robot1_joint_pos(6) + robot1_gripper_qpos(6)
        agent_pos = np.concatenate([
            obs['robot0_joint_pos'],       # 6
            obs['robot0_gripper_qpos'],    # 6
            obs['robot1_joint_pos'],       # 6
            obs['robot1_gripper_qpos'],    # 6
        ])

        return {
            'point_cloud': pc.astype(np.float32),
            'agent_pos': agent_pos.astype(np.float32),
        }

    def run(self, policy):
        device = policy.device
        all_rewards = []
        all_successes = []

        for ep in tqdm.tqdm(range(self.n_train), desc=f"Evaluating {self.task_name}"):
            obs = self.env.reset()
            obs_deque = collections.deque(maxlen=self.n_obs_steps)

            done = False
            step = 0
            episode_reward = 0.0
            is_success = False

            while not done and step < self.max_steps:
                # 1. Extract observations
                obs_dict_step = self._extract_obs(obs)
                obs_deque.append(obs_dict_step)

                # Pad history at episode start
                while len(obs_deque) < self.n_obs_steps:
                    obs_deque.append(obs_dict_step)

                # 2. Format as tensors: [1, n_obs_steps, ...]
                obs_dict = {
                    'point_cloud': torch.stack(
                        [torch.from_numpy(o['point_cloud']) for o in obs_deque]
                    ).unsqueeze(0).to(device),
                    'agent_pos': torch.stack(
                        [torch.from_numpy(o['agent_pos']) for o in obs_deque]
                    ).unsqueeze(0).to(device),
                }

                # 3. Predict action sequence
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                action_seq = action_dict['action'][0].cpu().numpy()

                # 4. Execute action chunk
                for action in action_seq[:self.n_action_steps]:
                    obs, reward, done, info = self.env.step(action)
                    episode_reward += reward
                    step += 1

                    if self.env._check_success():
                        is_success = True

                    if done or step >= self.max_steps:
                        break

            all_rewards.append(episode_reward)
            all_successes.append(1.0 if is_success else 0.0)

        # Log results
        log_data = {
            'test/mean_reward': np.mean(all_rewards),
            'test/success_rate': np.mean(all_successes),
            'test_mean_score': np.mean(all_successes),
        }

        return log_data

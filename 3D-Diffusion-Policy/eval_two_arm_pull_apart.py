"""
Evaluation Script for TwoArmPullApart with 3D Diffusion Policy.

Loads a trained checkpoint, runs the policy in closed-loop, and reports
success rate and reward.  Records a 3-panel MP4 (front view | depth | PC).

Usage:
    python eval_two_arm_pull_apart.py \\
        --checkpoint outputs/train/TwoArmPullApart/.../checkpoints/latest.ckpt \\
        [--num_episodes 50] [--render] [--video eval_pull_apart.mp4]

Agent pos  : [36] = joint_pos(6)×2 + gripper_qpos(6)×2 + wrist_ft(6)×2
Action dim : [7]  = arm1 joint delta(6) + arm1 gripper(1)
"""

import os
import sys
import gc
import argparse
import numpy as np
import torch
import collections
import mujoco

_ROBOSUITE_ROOT = "/home/rosi/robosuite"
if _ROBOSUITE_ROOT not in sys.path:
    sys.path.insert(0, _ROBOSUITE_ROOT)

import robosuite as suite
import robosuite.utils.camera_utils as camera_utils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dill
from train import TrainDP3Workspace

try:
    from pytorch3d.ops import sample_farthest_points
    HAS_FPS = True
except ImportError:
    sample_farthest_points = None
    HAS_FPS = False

CAMERA_NAME  = "birdview"
FRONT_CAMERA = "frontview"
CAMERA_HEIGHT = 256
CAMERA_WIDTH  = 256
NUM_POINTS    = 1024

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


def depth_to_pointcloud(depth_img, intrinsics, extrinsics):
    depth_img = np.squeeze(depth_img)
    h, w = depth_img.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x, y, depth = x.flatten(), y.flatten(), depth_img.flatten()
    valid = depth > 0
    x, y, depth = x[valid], y[valid], depth[valid]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    z_cam = depth
    x_cam = (x - cx) * z_cam / fx
    y_cam = (y - cy) * z_cam / fy
    pts_cam = np.vstack((x_cam, y_cam, z_cam, np.ones_like(z_cam)))
    return (np.linalg.inv(extrinsics) @ pts_cam)[:3, :].T


def downsample_pc(point_cloud, num_points=NUM_POINTS):
    if len(point_cloud) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    if len(point_cloud) < num_points:
        pad_idx = np.random.choice(len(point_cloud), num_points - len(point_cloud), replace=True)
        point_cloud = np.vstack([point_cloud, point_cloud[pad_idx]])
    if HAS_FPS:
        pc_tensor = torch.tensor(point_cloud, dtype=torch.float32).unsqueeze(0)
        sampled_pc, _ = sample_farthest_points(pc_tensor, K=num_points)
        return sampled_pc.squeeze(0).numpy()
    return point_cloud[np.random.choice(len(point_cloud), num_points, replace=False)]


def extract_obs(env, obs):
    """Extract point cloud + 36-D proprioceptive state."""
    depth_map = np.nan_to_num(obs[f"{CAMERA_NAME}_depth"], nan=0.0, posinf=1.0, neginf=0.0)
    depth_map = np.clip(depth_map, 0.0, 1.0)
    real_depth = camera_utils.get_real_depth_map(env.sim, depth_map)
    intrinsics = camera_utils.get_camera_intrinsic_matrix(
        env.sim, CAMERA_NAME, CAMERA_HEIGHT, CAMERA_WIDTH)
    extrinsics = camera_utils.get_camera_extrinsic_matrix(env.sim, CAMERA_NAME)

    pc = depth_to_pointcloud(real_depth, intrinsics, extrinsics)
    pc = downsample_pc(pc, NUM_POINTS)

    agent_pos = np.concatenate([
        obs['robot0_joint_pos'],    # 6
        obs['robot0_gripper_qpos'], # 6
        obs['robot1_joint_pos'],    # 6
        obs['robot1_gripper_qpos'], # 6
        obs['robot0_wrist_ft'],     # 6
        obs['robot1_wrist_ft'],     # 6
    ])

    return {
        'point_cloud': pc.astype(np.float32),
        'agent_pos':   agent_pos.astype(np.float32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    type=str, required=True)
    parser.add_argument("--num_episodes",  type=int, default=50)
    parser.add_argument("--max_steps",     type=int, default=600)
    parser.add_argument("--render",        action="store_true")
    parser.add_argument("--video",         type=str, default="eval_pull_apart.mp4")
    parser.add_argument("--seed",          type=int, default=0)
    parser.add_argument("--device",        type=str, default=None)
    parser.add_argument("--inference-steps", type=int, default=3)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Load Policy ──
    print(f"Loading checkpoint: {args.checkpoint}")
    payload   = torch.load(open(args.checkpoint, 'rb'), pickle_module=dill, map_location='cpu')
    workspace = TrainDP3Workspace(payload['cfg'])
    workspace.load_payload(payload=payload)
    cfg = workspace.cfg

    policy = workspace.ema_model if (cfg.training.use_ema and workspace.ema_model) else workspace.model
    policy.eval()

    if hasattr(policy, 'num_inference_steps'):
        policy.num_inference_steps = args.inference_steps

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    policy.to(device)

    n_obs_steps    = cfg.n_obs_steps
    n_action_steps = cfg.n_action_steps
    print(f"  n_obs_steps:    {n_obs_steps}")
    print(f"  n_action_steps: {n_action_steps}")
    print(f"  inference_steps: {args.inference_steps}")

    # ── Video setup ──
    import cv2 as _cv2
    VIDEO_W = CAMERA_WIDTH * 3
    VIDEO_H = CAMERA_HEIGHT
    video_writer = None
    _DILATE_K = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (6, 6))

    def _render_pc(pc, intr, extr):
        pts_h   = np.hstack([pc, np.ones((len(pc), 1))]).T
        pts_cam = (extr @ pts_h)[:3]
        in_front = pts_cam[2] > 0.01
        if not in_front.any():
            return np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        pts_cam = pts_cam[:, in_front]
        u = np.round(pts_cam[0] * intr[0, 0] / pts_cam[2] + intr[0, 2]).astype(int)
        v = np.round(pts_cam[1] * intr[1, 1] / pts_cam[2] + intr[1, 2]).astype(int)
        in_bounds = (u >= 0) & (u < CAMERA_WIDTH) & (v >= 0) & (v < CAMERA_HEIGHT)
        u, v = u[in_bounds], v[in_bounds]
        d = pts_cam[2, in_bounds]
        img = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        if d.size:
            d_norm = ((d - d.min()) / (d.max() - d.min() + 1e-8) * 255).astype(np.uint8)
            cols   = _cv2.applyColorMap(d_norm[:, None, None], _cv2.COLORMAP_VIRIDIS)[:, 0, :]
            img[v, u] = cols
        return _cv2.dilate(img, _DILATE_K)

    def _label(panel, text):
        _cv2.putText(panel, text, (6, 22), _cv2.FONT_HERSHEY_SIMPLEX,
                     0.65, (255, 255, 255), 2, _cv2.LINE_AA)
        return panel

    def _obs_to_frame(obs, pc, intr, extr):
        front_bgr = _cv2.flip(_cv2.cvtColor(obs[f"{FRONT_CAMERA}_image"],
                                            _cv2.COLOR_RGB2BGR), 0)
        depth   = np.squeeze(obs[f"{CAMERA_NAME}_depth"])
        valid   = depth[depth > 0]
        d_norm  = np.clip((depth - valid.min()) / (valid.max() - valid.min() + 1e-8),
                          0, 1) if valid.size else depth
        depth_col = _cv2.applyColorMap((d_norm * 255).astype(np.uint8), _cv2.COLORMAP_TURBO)
        pc_bgr    = _render_pc(pc, intr, extr)
        _label(front_bgr, "Front View")
        _label(depth_col, "Depth (birdview)")
        _label(pc_bgr,    "Point Cloud")
        return np.hstack([front_bgr, depth_col, pc_bgr])

    if args.video:
        fourcc       = _cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = _cv2.VideoWriter(args.video, fourcc, 20, (VIDEO_W, VIDEO_H))
        print(f"  Recording → {args.video}")

    # ── Create environment ──
    print("\nInitializing TwoArmPullApart environment...")
    env = suite.make(
        env_name="TwoArmPullApart",
        robots=["UR10e", "UR10e"],
        env_configuration="opposed",
        controller_configs=[JOINT_POS_CONTROLLER_CONFIG, JOINT_POS_CONTROLLER_CONFIG],
        gripper_types="Robotiq140Gripper",
        has_renderer=args.render,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=[CAMERA_NAME, FRONT_CAMERA],
        camera_depths=True,
        camera_heights=CAMERA_HEIGHT,
        camera_widths=CAMERA_WIDTH,
        control_freq=20,
        ignore_done=True,
        reward_shaping=True,
    )

    _intr = camera_utils.get_camera_intrinsic_matrix(env.sim, CAMERA_NAME, CAMERA_HEIGHT, CAMERA_WIDTH)
    _extr = camera_utils.get_camera_extrinsic_matrix(env.sim, CAMERA_NAME)

    # ── Evaluation loop ──
    print(f"\nRunning {args.num_episodes} evaluation episodes...")
    all_rewards   = []
    all_successes = []

    for ep in range(args.num_episodes):
        obs = env.reset()
        if args.render:
            env.render()

        obs_deque       = collections.deque(maxlen=n_obs_steps)
        episode_reward  = 0.0
        is_success      = False

        for step in range(args.max_steps):
            obs_dict_step = extract_obs(env, obs)
            obs_deque.append(obs_dict_step)
            while len(obs_deque) < n_obs_steps:
                obs_deque.append(obs_dict_step)

            obs_dict = {
                'point_cloud': torch.stack(
                    [torch.from_numpy(o['point_cloud']) for o in obs_deque]
                ).unsqueeze(0).to(device),
                'agent_pos': torch.stack(
                    [torch.from_numpy(o['agent_pos']) for o in obs_deque]
                ).unsqueeze(0).to(device),
            }

            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)
            action_seq = action_dict['action'][0].cpu().numpy()
            del obs_dict, action_dict

            for action in action_seq[:n_action_steps]:
                obs, reward, done, info = env.step(action)  # 7-D action; arm0 handled internally
                episode_reward += reward
                if args.render:
                    env.render()
                if env._check_success():
                    is_success = True
                if done:
                    break

            if video_writer is not None:
                video_writer.write(_obs_to_frame(obs, obs_dict_step['point_cloud'], _intr, _extr))

            if done:
                break

        all_rewards.append(episode_reward)
        all_successes.append(is_success)
        status = "SUCCESS" if is_success else "FAIL"
        print(f"  Episode {ep+1:3d}/{args.num_episodes}: reward={episode_reward:7.2f}  {status}")

        if video_writer is not None:
            card = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
            color = (80, 220, 80) if is_success else (80, 80, 220)
            _cv2.putText(card, f"Ep {ep+1}: {'SUCCESS' if is_success else 'FAIL'}",
                         (20, VIDEO_H // 2), _cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            for _ in range(20):
                video_writer.write(card)

        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved → {args.video}")

    success_rate = np.mean(all_successes) * 100
    mean_reward  = np.mean(all_rewards)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Episodes:     {args.num_episodes}")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Mean Reward:  {mean_reward:.2f}")
    print(f"  Successes:    {int(sum(all_successes))}/{args.num_episodes}")
    print("=" * 50)

    env.close()


if __name__ == "__main__":
    main()

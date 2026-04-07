"""
Deployment & Evaluation Script for TwoArmLift with 3D Diffusion Policy.

Loads a trained checkpoint, instantiates the TwoArmLift environment with
the exact sensor and control configurations used during data collection,
and runs the policy in closed-loop to test the task success rate.

Usage:
    python eval_two_arm_lift.py --checkpoint <path_to_checkpoint> [--num_episodes 50] [--render]
"""

import os
import sys
import gc
import argparse
import numpy as np
import torch
import collections
import mujoco
import yaml

# Ensure the full robosuite installation (with camera_utils) takes precedence
# over the stub package installed in the conda env.
_ROBOSUITE_ROOT = "/home/swapneel_hmgics/workspace/robosuite"
if _ROBOSUITE_ROOT not in sys.path:
    sys.path.insert(0, _ROBOSUITE_ROOT)

import robosuite as suite
import robosuite.utils.camera_utils as camera_utils

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dill
from train import TrainDP3Workspace

try:
    from pytorch3d.ops import sample_farthest_points
    HAS_FPS = True
except ImportError:
    sample_farthest_points = None
    HAS_FPS = False

# ─────────────── Config (must match data collection) ─────────────

CAMERA_NAME  = "birdview"
FRONT_CAMERA = "frontview"
CAMERA_HEIGHT = 256
CAMERA_WIDTH  = 256
NUM_POINTS    = 1024

JOINT_POS_CONTROLLER_CONFIG = {
    "type": "BASIC",
    "body_parts": {
        "right": {                  # arm name directly under body_parts (not under "arms")
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


# ─────────────────── Point Cloud Utilities ───────────────────────

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

    if HAS_FPS:
        # Always run FPS on CPU — avoids repeated CUDA allocations every obs step
        # which fragments CUDA memory and can cause segfaults over many episodes.
        pc_tensor = torch.tensor(point_cloud, dtype=torch.float32).unsqueeze(0)
        sampled_pc, _ = sample_farthest_points(pc_tensor, K=num_points)
        return sampled_pc.squeeze(0).numpy()
    else:
        idx = np.random.choice(len(point_cloud), num_points, replace=False)
        return point_cloud[idx]


# ──────────────────── Observation Extraction ─────────────────────

def extract_obs(env, obs):
    """Extract point cloud + dual-arm proprioceptive state from raw obs."""
    # Point cloud from depth
    depth_key = f"{CAMERA_NAME}_depth"
    depth_map = obs[depth_key]
    # NaN-safe clip: replace NaN with 0 before clamping so the assertion in
    # camera_utils.get_real_depth_map always passes.
    depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=1.0, neginf=0.0)
    depth_map = np.clip(depth_map, 0.0, 1.0)
    real_depth = camera_utils.get_real_depth_map(env.sim, depth_map)
    intrinsics = camera_utils.get_camera_intrinsic_matrix(
        env.sim, CAMERA_NAME, CAMERA_HEIGHT, CAMERA_WIDTH
    )
    extrinsics = camera_utils.get_camera_extrinsic_matrix(env.sim, CAMERA_NAME)

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


# ──────────────── Cable Placement (mirrors collect_config.yaml) ──

def place_cables(env, cfg):
    """
    Override cable positions after env.reset() using the same logic and
    config as collect_two_arm_lift_data.py so eval and training see
    identical object distributions.
    """
    op    = cfg["object_placement"]
    model = env.sim.model._model
    data  = env.sim.data._data

    x         = op["cable_x"]
    y         = op["cable_y"]
    z         = 0.804
    noise_xy  = op["noise_xy"]
    angle_deg = op["cable_angle_deg"]
    noise_ang = op["noise_angle_deg"]

    pos1 = np.array([
        x + np.random.uniform(-noise_xy, noise_xy),
        y + np.random.uniform(-noise_xy, noise_xy),
        z,
    ])
    angle1 = np.deg2rad(angle_deg + np.random.uniform(-noise_ang, noise_ang))
    quat1  = np.array([np.cos(angle1 / 2), 0.0, 0.0, np.sin(angle1 / 2)])

    jnt1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wire_harness_joint0")
    adr1    = model.jnt_qposadr[jnt1_id]
    data.qpos[adr1 : adr1 + 7] = np.concatenate([pos1, quat1])

    angle2 = angle1 + np.pi / 2
    quat2  = np.array([np.cos(angle2 / 2), 0.0, 0.0, np.sin(angle2 / 2)])
    pos2   = pos1 + np.array([
        op.get("cross_offset_x", 0.0),
        op.get("cross_offset_y", 0.0),
        0.01,
    ])

    jnt2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wire_harness_2_joint0")
    adr2    = model.jnt_qposadr[jnt2_id]
    data.qpos[adr2 : adr2 + 7] = np.concatenate([pos2, quat2])

    mujoco.mj_forward(model, data)


# ──────────────────────── Main Loop ──────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DP3 on TwoArmLift")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained .ckpt file")
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=400,
                        help="Max steps per episode")
    parser.add_argument("--render", action="store_true",
                        help="Open the MuJoCo viewer for visualization")
    parser.add_argument("--video", type=str, default="eval_result.mp4",
                        help="Path to save the evaluation MP4 video. "
                             "Set to '' to disable recording.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override, e.g. 'cpu' or 'cuda:0'. "
                             "Defaults to cuda if available.")
    parser.add_argument("--collect-config", type=str,
                        default="/home/swapneel_hmgics/workspace/robosuite/collect_config.yaml",
                        help="Path to collect_config.yaml for cable placement. "
                             "Pass '' to disable (use robosuite default randomisation).")
    parser.add_argument("--inference-steps", type=int, default=3,
                        help="DDIM denoising steps at eval time (fewer = faster). "
                             "Policy was trained with 10; 3 is usually sufficient.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Load cable placement config ──
    collect_cfg = None
    if args.collect_config:
        with open(args.collect_config) as f:
            collect_cfg = yaml.safe_load(f)
        op = collect_cfg["object_placement"]
        print(f"  Cable config: {args.collect_config}")
        print(f"  Cable XY: ({op['cable_x']}, {op['cable_y']})  "
              f"noise_xy={op['noise_xy']}  "
              f"angle={op['cable_angle_deg']}°±{op['noise_angle_deg']}°")

    # ── 1. Load Policy ──
    # create_from_checkpoint loads without map_location so CUDA tensors are
    # allocated on the training GPU even when we want CPU eval.
    # Replicate its logic with map_location='cpu' to avoid OOM.
    print(f"Loading checkpoint: {args.checkpoint}")
    payload = torch.load(open(args.checkpoint, 'rb'), pickle_module=dill, map_location='cpu')
    workspace = TrainDP3Workspace(payload['cfg'])
    workspace.load_payload(payload=payload)
    cfg = workspace.cfg

    # Use EMA model if available
    if cfg.training.use_ema and workspace.ema_model is not None:
        policy = workspace.ema_model
    else:
        policy = workspace.model

    policy.eval()
    # Override DDIM inference steps for faster eval
    if hasattr(policy, 'num_inference_steps'):
        policy.num_inference_steps = args.inference_steps
    if hasattr(policy, 'noise_scheduler') and hasattr(policy.noise_scheduler, 'set_timesteps'):
        pass  # scheduler timesteps are set on each predict_action call
    print(f"  inference_steps: {args.inference_steps} (trained with 10)")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)

    n_obs_steps = cfg.n_obs_steps
    n_action_steps = cfg.n_action_steps

    print(f"  n_obs_steps:    {n_obs_steps}")
    print(f"  n_action_steps: {n_action_steps}")

    # ── 2. Video writer setup ──
    import cv2 as _cv2

    video_writer = None
    video_fps    = 20
    # Video frame: [Front RGB | Birdview Depth | Point Cloud]  →  width = 3 × CAMERA_WIDTH
    VIDEO_W = CAMERA_WIDTH * 3
    VIDEO_H = CAMERA_HEIGHT

    # Dilation kernel for projecting sparse (1024-pt) point cloud to image
    _DILATE_K = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (6, 6))

    def _render_pc(pc, intr, extr):
        """
        Project 3D world-frame point cloud through the birdview camera and
        return a CAMERA_HEIGHT × CAMERA_WIDTH BGR image coloured by depth.
        Pure numpy + cv2 — deterministic, no matplotlib, no flickering.
        """
        # world → camera frame
        pts_h   = np.hstack([pc, np.ones((len(pc), 1))]).T   # 4×N
        pts_cam = (extr @ pts_h)[:3]                          # 3×N

        in_front = pts_cam[2] > 0.01
        if not in_front.any():
            return np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        pts_cam = pts_cam[:, in_front]

        # Project to pixel coordinates
        u = np.round(pts_cam[0] * intr[0, 0] / pts_cam[2] + intr[0, 2]).astype(int)
        v = np.round(pts_cam[1] * intr[1, 1] / pts_cam[2] + intr[1, 2]).astype(int)

        in_bounds = (u >= 0) & (u < CAMERA_WIDTH) & (v >= 0) & (v < CAMERA_HEIGHT)
        u, v = u[in_bounds], v[in_bounds]
        d    = pts_cam[2, in_bounds]

        img = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        if d.size:
            d_norm = ((d - d.min()) / (d.max() - d.min() + 1e-8) * 255).astype(np.uint8)
            cols   = _cv2.applyColorMap(d_norm[:, None, None],
                                        _cv2.COLORMAP_VIRIDIS)[:, 0, :]  # N×3
            img[v, u] = cols

        # Dilate so the 1024 sparse dots become visible patches
        img = _cv2.dilate(img, _DILATE_K)
        return img

    def _label(panel, text):
        _cv2.putText(panel, text, (6, 22), _cv2.FONT_HERSHEY_SIMPLEX,
                     0.65, (255, 255, 255), 2, _cv2.LINE_AA)
        return panel

    def _obs_to_frame(obs, pc, intr, extr):
        """Three-panel BGR frame: Front RGB | Birdview Depth | Point Cloud (projected)."""
        # Panel 1 – Frontview from robosuite obs (same pipeline as birdview,
        # no extra context or renderer needed).
        front     = obs[f"{FRONT_CAMERA}_image"]
        front_bgr = _cv2.flip(_cv2.cvtColor(front, _cv2.COLOR_RGB2BGR), 0)

        # Panel 2 – Birdview depth colourmap
        depth = np.squeeze(obs[f"{CAMERA_NAME}_depth"])
        valid = depth[depth > 0]
        d_norm    = np.clip((depth - valid.min()) / (valid.max() - valid.min() + 1e-8),
                            0, 1) if valid.size else depth
        depth_col = _cv2.applyColorMap((d_norm * 255).astype(np.uint8), _cv2.COLORMAP_TURBO)

        # Panel 3 – Point cloud projected through birdview camera
        pc_bgr = _render_pc(pc, intr, extr)

        _label(front_bgr, "Front View")
        _label(depth_col, "Depth (birdview)")
        _label(pc_bgr,    "Point Cloud")

        return np.hstack([front_bgr, depth_col, pc_bgr])

    def _make_title_card(ep_idx, n_episodes, is_success, ep_reward):
        card         = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
        status_color = (80, 220, 80) if is_success else (80, 80, 220)
        status_text  = "SUCCESS" if is_success else "FAIL"
        _cv2.putText(card, f"Episode {ep_idx}/{n_episodes}",
                     (20, VIDEO_H // 2 - 40), _cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)
        _cv2.putText(card, status_text,
                     (20, VIDEO_H // 2 + 10), _cv2.FONT_HERSHEY_SIMPLEX, 1.4, status_color, 3)
        _cv2.putText(card, f"Reward: {ep_reward:.2f}",
                     (20, VIDEO_H // 2 + 60), _cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        return card

    if args.video:
        fourcc       = _cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = _cv2.VideoWriter(args.video, fourcc, video_fps, (VIDEO_W, VIDEO_H))
        print(f"  Recording video → {args.video}  ({VIDEO_W}×{VIDEO_H})")

    # ── 3. Create Environment ──
    print("\nInitializing TwoArmLift environment...")
    env = suite.make(
        env_name="TwoArmLift",
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

    # Cache birdview camera matrices (static — compute once)
    _intr = camera_utils.get_camera_intrinsic_matrix(
        env.sim, CAMERA_NAME, CAMERA_HEIGHT, CAMERA_WIDTH)
    _extr = camera_utils.get_camera_extrinsic_matrix(env.sim, CAMERA_NAME)


    # ── 4. Evaluation Loop ──
    print(f"\nRunning {args.num_episodes} evaluation episodes...")
    all_rewards = []
    all_successes = []

    for ep in range(args.num_episodes):
        obs = env.reset()
        if collect_cfg is not None:
            place_cables(env, collect_cfg)
            obs = env._get_observations()
        if args.render:
            env.render()

        obs_deque = collections.deque(maxlen=n_obs_steps)
        episode_reward = 0.0
        is_success = False

        for step in range(args.max_steps):
            # A. Extract and process observations
            obs_dict_step = extract_obs(env, obs)
            obs_deque.append(obs_dict_step)

            # Pad at episode start
            while len(obs_deque) < n_obs_steps:
                obs_deque.append(obs_dict_step)

            # B. Format tensors for policy: [1, n_obs_steps, ...]
            obs_dict = {
                'point_cloud': torch.stack(
                    [torch.from_numpy(o['point_cloud']) for o in obs_deque]
                ).unsqueeze(0).to(device),
                'agent_pos': torch.stack(
                    [torch.from_numpy(o['agent_pos']) for o in obs_deque]
                ).unsqueeze(0).to(device),
            }

            # C. Predict action chunk via diffusion
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

            action_seq = action_dict['action'][0].cpu().numpy()
            # Explicitly free CUDA tensors — prevents fragmentation over long runs
            del obs_dict, action_dict

            # D. Execute action chunk in environment
            for action in action_seq[:n_action_steps]:
                obs, reward, done, info = env.step(action)
                episode_reward += reward

                if args.render:
                    env.render()

                if env._check_success():
                    is_success = True

                if done:
                    break

            # Record one frame per policy step (not per action) to reduce overhead
            if video_writer is not None:
                video_writer.write(_obs_to_frame(
                    obs, obs_dict_step['point_cloud'], _intr, _extr,
                ))

            if done:
                break

        all_rewards.append(episode_reward)
        all_successes.append(is_success)

        status = "SUCCESS" if is_success else "FAIL"
        print(f"  Episode {ep+1:3d}/{args.num_episodes}: "
              f"reward={episode_reward:7.2f}  {status}")

        # Write a 1-second title card between episodes
        if video_writer is not None:
            card = _make_title_card(ep + 1, args.num_episodes, is_success, episode_reward)
            for _ in range(video_fps):
                video_writer.write(card)

        # Release memory before the next env.reset()
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ── 5. Report Results ──
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

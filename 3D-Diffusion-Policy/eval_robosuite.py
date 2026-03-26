import os
import torch
import numpy as np
import robosuite as suite
import robosuite.utils.camera_utils as camera_utils

# We use the workspace to properly deserialize the model configuration
from train import TrainDP3Workspace

# Optional: Fast PC downsampling matching training
try:
    from pytorch3d.ops import sample_farthest_points
except ImportError:
    sample_farthest_points = None

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
    if len(point_cloud) < num_points:
        pad_size = num_points - len(point_cloud)
        pad_points = point_cloud[np.random.choice(len(point_cloud), pad_size, replace=True)]
        point_cloud = np.vstack([point_cloud, pad_points])
    
    if sample_farthest_points is not None:
        pc_tensor = torch.tensor(point_cloud, dtype=torch.float32).unsqueeze(0)
        if torch.cuda.is_available():
            pc_tensor = pc_tensor.cuda()
        sampled_pc, _ = sample_farthest_points(pc_tensor, K=num_points)
        return sampled_pc.squeeze(0).cpu().numpy()
    else:
        idx = np.random.choice(len(point_cloud), num_points, replace=False)
        return point_cloud[idx]

def main():
    # 1. UPDATE THIS PATH once your training outputs a checkpoint!
    checkpoint_path = "/home/rosi/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/2026.03.17/23.00.18_train_dp3_robosuite_lift/checkpoints/latest.ckpt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Waiting for checkpoint to be created at {checkpoint_path}...")
        return

    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # 2. Load the DP3 policy via the Workspace abstraction
    payload = torch.load(checkpoint_path)
    cfg = payload['cfg']
    workspace = TrainDP3Workspace.create_from_checkpoint(checkpoint_path)
    policy = workspace.model
    policy.eval()
    if torch.cuda.is_available():
        policy.cuda()

    # 3. Initialize the GUI Environment
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=True,            # This pops open the MuJoCo Viewer
        has_offscreen_renderer=True,  # Required for depth map rendering
        use_camera_obs=True,
        camera_names="frontview",
        camera_depths=True,
        camera_heights=256,
        camera_widths=256,
        control_freq=20,
    )

    obs = env.reset()
    env.render()
    
    # 4. Extract action/observation horizons from the training config
    n_obs_steps = cfg.n_obs_steps
    n_action_steps = cfg.n_action_steps
    max_steps = 500
    
    obs_deque = []
    
    print("Starting deployment loop...")
    for step in range(max_steps):
        # A. Process Vision exactly like the conversion pipeline
        depth_map = obs["frontview_depth"]
        depth_map = np.clip(depth_map, 0.0, 1.0) 
        
        real_depth = camera_utils.get_real_depth_map(env.sim, depth_map)        
        intrinsics = camera_utils.get_camera_intrinsic_matrix(env.sim, "frontview", 256, 256)
        extrinsics = camera_utils.get_camera_extrinsic_matrix(env.sim, "frontview")
        
        pc = depth_to_pointcloud(real_depth, intrinsics, extrinsics)
        pc = downsample_pc(pc, num_points=1024)
        
        # B. Process Proprioception
        agent_pos = np.concatenate([
            obs['robot0_eef_pos'],
            obs['robot0_eef_quat'],
            obs['robot0_gripper_qpos']
        ])
        
        # C. Manage Observation History Queue
        obs_dict_step = {
            'point_cloud': pc.astype(np.float32),
            'agent_pos': agent_pos.astype(np.float32)
        }
        obs_deque.append(obs_dict_step)
        if len(obs_deque) > n_obs_steps:
            obs_deque.pop(0)
            
        # Pad history at the very beginning of the episode
        while len(obs_deque) < n_obs_steps:
            obs_deque.append(obs_dict_step)
            
        # D. Format Tensors for DP3 Model
        obs_dict = {
            'point_cloud': torch.stack([torch.from_numpy(o['point_cloud']) for o in obs_deque]).unsqueeze(0),
            'agent_pos': torch.stack([torch.from_numpy(o['agent_pos']) for o in obs_deque]).unsqueeze(0)
        }
        if torch.cuda.is_available():
            obs_dict = {k: v.cuda() for k, v in obs_dict.items()}
        
        # E. Predict Action Chunk via Diffusion
        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict)
            
        action_seq = action_dict['action'][0].cpu().numpy()
        
        # F. Execute Action Chunk in Simulator
        for action in action_seq[:n_action_steps]:
            obs, reward, done, info = env.step(action)
            env.render()
            
            if done: break
        if done: break

    print("Deployment episode complete.")

if __name__ == "__main__":
    main()
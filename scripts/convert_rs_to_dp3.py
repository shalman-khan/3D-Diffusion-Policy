import os
import argparse
import json
import h5py
import zarr
import numpy as np
import torch
import robosuite as suite
import robosuite.utils.camera_utils as camera_utils

# Try importing FPS from pytorch3d (which is included in the DP3 repo)
try:
    from pytorch3d.ops import sample_farthest_points
except ImportError:
    print("Warning: pytorch3d not found. Please ensure you are in the DP3 environment.")
    print("Falling back to random uniform sampling (not recommended for final training).")
    sample_farthest_points = None

def depth_to_pointcloud(depth_img, camera_intrinsics, camera_extrinsics):
    """
    Converts a depth image to a 3D point cloud in world coordinates.
    """
    # Add this line to drop the trailing channel dimension (e.g., turns 256x256x1 into 256x256)
    depth_img = np.squeeze(depth_img) 
    
    h, w = depth_img.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.flatten()
    y = y.flatten()
    depth = depth_img.flatten()

    # Filter out zero depth (background)
    valid = depth > 0
    x, y, depth = x[valid], y[valid], depth[valid]

    # Unproject to camera coordinates
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    
    z_cam = depth
    x_cam = (x - cx) * z_cam / fx
    y_cam = (y - cy) * z_cam / fy
    
    points_cam = np.vstack((x_cam, y_cam, z_cam, np.ones_like(z_cam)))
    
    # Transform to world coordinates using inverse of extrinsics
    cam_to_world = np.linalg.inv(camera_extrinsics)
    points_world = (cam_to_world @ points_cam)[:3, :].T
    
    return points_world

def downsample_pc(point_cloud, num_points=1024):
    """
    Downsamples the point cloud to exactly `num_points` using Farthest Point Sampling.
    """
    if len(point_cloud) < num_points:
        # Pad with zeros or randomly repeat points if too few
        pad_size = num_points - len(point_cloud)
        pad_points = point_cloud[np.random.choice(len(point_cloud), pad_size, replace=True)]
        point_cloud = np.vstack([point_cloud, pad_points])
    
    if sample_farthest_points is not None:
        # Pytorch3D expects shape (Batch, N, 3)
        pc_tensor = torch.tensor(point_cloud, dtype=torch.float32).unsqueeze(0)
        if torch.cuda.is_available():
            pc_tensor = pc_tensor.cuda()
            
        sampled_pc, _ = sample_farthest_points(pc_tensor, K=num_points)
        return sampled_pc.squeeze(0).cpu().numpy()
    else:
        # Fallback to random uniform sampling
        idx = np.random.choice(len(point_cloud), num_points, replace=False)
        return point_cloud[idx]

def convert_dataset(hdf5_path, zarr_path, camera_name="frontview", num_points=1024):
    print(f"Loading HDF5 from {hdf5_path}...")
    f = h5py.File(hdf5_path, "r")
    env_name = f["data"].attrs["env"]
    env_info = json.loads(f["data"].attrs["env_info"])

    # Initialize Robosuite Environment
    print(f"Initializing Robosuite environment: {env_name}...")
    env = suite.make(
        **env_info,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_names=camera_name,
        camera_depths=True,
        camera_heights=256,
        camera_widths=256,
    )

    demos = list(f["data"].keys())
    
    all_point_clouds = []
    all_states = []
    all_actions = []
    episode_ends = []
    
    current_step = 0

    print(f"Converting {len(demos)} demonstrations...")
    for idx, demo in enumerate(demos):
        print(f"Processing episode {idx+1}/{len(demos)}")
        ep_group = f[f"data/{demo}"]
        
        states = ep_group["states"][()]
        actions = ep_group["actions"][()]
        
        env.reset()
        
        for i in range(len(states)):
            # 1. Set Simulator State
            env.sim.set_state_from_flattened(states[i])
            env.sim.forward()
            
            # 2. Get Observations
            obs = env._get_observations()
            
            # 3. Extract and unproject Depth
            depth_map = obs[f"{camera_name}_depth"]
            real_depth = camera_utils.get_real_depth_map(env.sim, depth_map)
            
            intrinsics = camera_utils.get_camera_intrinsic_matrix(env.sim, camera_name, 256, 256)
            extrinsics = camera_utils.get_camera_extrinsic_matrix(env.sim, camera_name)
            
            pc_world = depth_to_pointcloud(real_depth, intrinsics, extrinsics)
            
            # 4. Downsample Point Cloud
            pc_downsampled = downsample_pc(pc_world, num_points=num_points)
            
            # 5. Extract Low-Dim Robot State (Example: Proprioception)
            # You can customize this based on what your policy needs!
            robot_state = np.concatenate([
                obs['robot0_eef_pos'],
                obs['robot0_eef_quat'],
                obs['robot0_gripper_qpos']
            ])
            
            # 6. Buffer
            all_point_clouds.append(pc_downsampled)
            all_states.append(robot_state)
            all_actions.append(actions[i])
            
            current_step += 1
            
        episode_ends.append(current_step)

    f.close()

    # Save to Zarr
    print(f"Saving to Zarr archive at {zarr_path}...")
    root = zarr.open(zarr_path, mode='w')
    
    data_grp = root.create_group('data')
    data_grp.create_dataset('point_cloud', data=np.array(all_point_clouds, dtype=np.float32), chunks=(100, num_points, 3))
    data_grp.create_dataset('state', data=np.array(all_states, dtype=np.float32), chunks=(1000, all_states[0].shape[0]))
    data_grp.create_dataset('action', data=np.array(all_actions, dtype=np.float32), chunks=(1000, all_actions[0].shape[0]))
    
    meta_grp = root.create_group('meta')
    meta_grp.create_dataset('episode_ends', data=np.array(episode_ends, dtype=np.int64))

    print(f"✅ Conversion complete! Total transitions: {current_step}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_path", type=str, required=True, help="Path to Robosuite .hdf5 file")
    parser.add_argument("--zarr_path", type=str, required=True, help="Output path for the .zarr archive")
    parser.add_argument("--camera", type=str, default="frontview", help="Camera name to extract PC from")
    args = parser.parse_args()

    convert_dataset(args.hdf5_path, args.zarr_path, args.camera)
import os
import glob
import json
import numpy as np
import zarr
import open3d as o3d
import robosuite as suite
import robosuite.utils.camera_utils as cam_utils

def unproject_depth_to_pc(depth_map, intrinsics, extrinsics):
    """
    Converts a depth map to a 3D point cloud using camera matrices.
    """
    # Force the depth map to 2D by squeezing out the channel dimension
    depth_map = np.squeeze(depth_map)
    H, W = depth_map.shape
    
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = x.flatten()
    y = y.flatten()
    d = depth_map.flatten()

    # Filter out background (e.g., depth = 0 or very far)
    valid = (d > 0.01) & (d < 2.0)
    x = x[valid]
    y = y[valid]
    d = d[valid]

    # Pinhole unprojection
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Robosuite/MuJoCo cameras look along the -Z axis
    x_cam = (x - cx) * d / fx
    y_cam = (y - cy) * d / fy
    z_cam = -d 
    pts_cam = np.vstack((x_cam, y_cam, z_cam, np.ones_like(x_cam)))

    # Transform to world frame
    pts_world = (extrinsics @ pts_cam)[:3, :].T

    # Downsample to 1024 points (DP3 Standard)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world)
    if len(pcd.points) > 1024:
        pcd = pcd.farthest_point_down_sample(1024)
    else:
        # If not enough points, duplicate them to reach 1024 (edge case)
        pts = np.asarray(pcd.points)
        pad = 1024 - len(pts)
        pts = np.vstack([pts, pts[:pad]])
        pcd.points = o3d.utility.Vector3dVector(pts)

    return np.asarray(pcd.points)

def convert_to_zarr(base_dir, zarr_path):
    zroot = zarr.open(zarr_path, mode='w')
    
    all_pcs, all_states, all_actions, episode_ends = [], [], [], []
    current_end = 0

    # 1. Find all episode directories
    ep_dirs = sorted(glob.glob(os.path.join(base_dir, "ep_*")))
    
    print(f"Found {len(ep_dirs)} episodes. Starting conversion...")

    env = None

    for ep_dir in ep_dirs:
        print(f"Processing {ep_dir}...")
        
        # 2. Load all state files in order first
        npz_files = sorted(glob.glob(os.path.join(ep_dir, "state_*.npz")))
        if not npz_files:
            continue
            
        # 3. Read Episode Metadata (env name) directly from the first .npz file
        first_data = np.load(npz_files[0], allow_pickle=True)
        env_name = str(first_data['env'])
        
        # 4. Initialize Environment (Only once, or if env changes)
        if env is None:
            env = suite.make(
                env_name=env_name,
                robots="Panda", # CHANGE THIS if you used a different robot
                has_renderer=False,
                has_offscreen_renderer=True, # Crucial for depth rendering
                use_camera_obs=True,
                camera_names="frontview",    # Match the camera you want
                camera_depths=True,
                camera_widths=256,
                camera_heights=256,
            )

        # Get Camera Matrices
        K = cam_utils.get_camera_intrinsic_matrix(env.sim, "frontview", 256, 256)
        R = env.sim.data.get_camera_xmat("frontview")
        t = env.sim.data.get_camera_xpos("frontview")
        
        # Build 4x4 Extrinsic matrix
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = R
        extrinsics[:3, 3] = t

        for npz_file in npz_files:
            data = np.load(npz_file, allow_pickle=True)
            
            state_arr = np.array(data['states']).flatten()
            
            # Skip empty boundary frames
            if len(state_arr) == 0:
                continue
            
            # 5. Extract actions using the correct key based on your .npz structure
            if 'action_infos' in data.files:
                raw_info = data['action_infos']
                
                # Check if it's a 0-d scalar object
                if raw_info.shape == ():
                    action_arr = raw_info.item()['actions']
                # Check if it's a populated array
                elif len(raw_info) > 0:
                    action_arr = raw_info[0]['actions']
                # Catch empty arrays (first/last frame) and pad with zeros
                else:
                    action_arr = np.zeros(7, dtype=np.float32)
                    
            elif 'action' in data.files:
                action_arr = data['action']
            else:
                action_arr = data['action_applied']
            
            # 6. Set MuJoCo to this exact state
            env.sim.set_state_from_flattened(state_arr)
            env.sim.forward() # Move simulation forward 1 tick to update sensors
            
            # 7. Extract Rendered Observations
            obs = env._get_observations()
            
            # Robosuite depth maps are normalized [0, 1]. Get real depth in meters.
            real_depth = cam_utils.get_real_depth_map(env.sim, obs['frontview_depth'])
            
            # 8. Convert Depth to Point Cloud
            pc = unproject_depth_to_pc(real_depth, K, extrinsics)
            
            # 9. Extract Proprioception (Robot arm state + Gripper)
            eef_pos = obs['robot0_eef_pos']
            eef_quat = obs['robot0_eef_quat']
            gripper_qpos = obs['robot0_gripper_qpos']
            low_dim_state = np.concatenate([eef_pos, eef_quat, gripper_qpos], axis=-1)
            
            # Append step data
            all_pcs.append(pc)
            all_states.append(low_dim_state)
            all_actions.append(action_arr)
            
        # Update episode boundaries for Zarr
        episode_ends.append(len(all_states))

    # 10. Save everything to Zarr format
    print("Writing to Zarr...")
    data_group = zroot.create_group('data')
    data_group.create_dataset('point_cloud', data=np.array(all_pcs, dtype=np.float32))
    data_group.create_dataset('state', data=np.array(all_states, dtype=np.float32))
    data_group.create_dataset('action', data=np.array(all_actions, dtype=np.float32))
    meta_group = zroot.create_group('meta')
    meta_group.create_dataset('episode_ends', data=np.array(episode_ends, dtype=np.int64))
    print(f"Success! Saved to {zarr_path}")
    
if __name__ == "__main__":
    # Point this to your temporary folder
    DATA_DIR = "/tmp/1773645178_327953/"
    ZARR_OUT = "dp3_robosuite_dataset.zarr"
    
    convert_to_zarr(DATA_DIR, ZARR_OUT)
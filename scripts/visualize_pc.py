import zarr
import open3d as o3d
import numpy as np
import time

ZARR_PATH = "dp3_robosuite_dataset.zarr"

def play_point_clouds(zarr_path, start_frame=0, max_frames=200, fps=10):
    zroot = zarr.open(zarr_path, mode='r')
    point_clouds = zroot['data/point_cloud']
    
    total_frames = point_clouds.shape[0]
    print(f"Total frames available: {total_frames}")
    
    # Initialize Open3D Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Zarr Point Cloud Playback", width=800, height=600)
    
    pcd = o3d.geometry.PointCloud()
    
    frames_to_play = min(max_frames, total_frames - start_frame)
    
    for i in range(frames_to_play):
        frame_idx = start_frame + i
        
        # Load the 1024 points for this frame
        pts = point_clouds[frame_idx]
        pcd.points = o3d.utility.Vector3dVector(pts)
        
        if i == 0:
            vis.add_geometry(pcd)
            # You might need to rotate the view manually with your mouse 
            # the first time the window opens to see it clearly.
        else:
            vis.update_geometry(pcd)
            
        vis.poll_events()
        vis.update_renderer()
        
        time.sleep(1.0 / fps)
        
    vis.destroy_window()

if __name__ == "__main__":
    # Plays the first 200 frames at 10 frames per second
    play_point_clouds(ZARR_PATH, start_frame=0, max_frames=200, fps=10)
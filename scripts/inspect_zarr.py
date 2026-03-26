import zarr
import numpy as np

# Point this to your generated zarr file
ZARR_PATH = "dp3_robosuite_dataset.zarr"

# Open the zarr directory in read-only mode
zroot = zarr.open(ZARR_PATH, mode='r')

print("--- ZARR DATASET STRUCTURE ---")
print(f"Keys at root: {list(zroot.keys())}")

# Print shapes of the data
print("\n--- DATA SHAPES ---")
print(f"Point Clouds: {zroot['data/point_cloud'].shape}  --> (Total Steps, Num Points, 3D Coordinates)")
print(f"States:       {zroot['data/state'].shape}        --> (Total Steps, State Dimension)")
print(f"Actions:      {zroot['data/action'].shape}       --> (Total Steps, Action Dimension)")

# Print Episode Information
ep_ends = zroot['meta/episode_ends'][:]
print("\n--- EPISODE INFO ---")
print(f"Total Episodes: {len(ep_ends)}")
print(f"Episode end indices: {ep_ends}")

# Print the very first action to verify it looks correct
print("\n--- SAMPLE ACTION ---")
print(f"Action at Step 0: {zroot['data/action'][0]}")
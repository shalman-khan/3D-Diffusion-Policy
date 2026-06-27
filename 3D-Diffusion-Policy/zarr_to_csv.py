#!/usr/bin/env python3
"""
zarr_to_csv.py
==============
Exports state + action arrays from a DP3 zarr dataset to CSV,
with a descriptive point_cloud column for each timestep.

Usage:
    python3 zarr_to_csv.py [--zarr PATH] [--output PATH]
"""

import argparse
import numpy as np
import pandas as pd
import zarr
from pathlib import Path

DEFAULT_ZARR = (
    "/home/rosi/3D-Diffusion-Policy/3D-Diffusion-Policy/data/"
    "real_cable_pull_trimmed.zarr"
)
DEFAULT_OUT = "cable_pull_state_action.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr",   default=DEFAULT_ZARR, help="Path to zarr dataset")
    parser.add_argument("--output", default=DEFAULT_OUT,  help="Output CSV path")
    args = parser.parse_args()

    print(f"Opening zarr: {args.zarr}")
    z = zarr.open(args.zarr, "r")

    states  = z["data/state"][:]        # (T, 14)
    actions = z["data/action"][:]       # (T, 14)
    ends    = z["meta/episode_ends"][:]  # (E,)
    pc      = z["data/point_cloud"]     # lazy — shape only

    T           = states.shape[0]
    n_channels  = pc.shape[-1]
    n_pts       = pc.shape[1]
    n_episodes  = len(ends)

    pc_type  = "XYZRGB" if n_channels == 6 else "XYZ"
    pc_label = f"{pc_type}_pointcloud_{n_pts}pts"

    print(f"  Episodes  : {n_episodes}")
    print(f"  Timesteps : {T}")
    print(f"  PC label  : {pc_label}")

    # Build episode index column
    episode_idx = np.zeros(T, dtype=np.int32)
    prev = 0
    for ep, end in enumerate(ends):
        episode_idx[prev:end] = ep
        prev = end

    cols_s = [f"state_{i}"  for i in range(states.shape[1])]
    cols_a = [f"action_{i}" for i in range(actions.shape[1])]

    df = pd.DataFrame(
        np.hstack([states, actions]),
        columns=cols_s + cols_a,
    )
    df.insert(0, "episode", episode_idx)
    df.insert(1, "timestep", np.arange(T, dtype=np.int32))
    df["point_cloud"] = pc_label

    out = Path(args.output)
    df.to_csv(out, index=False)
    print(f"\nSaved → {out.resolve()}  ({out.stat().st_size / 1024:.1f} KB, {len(df)} rows)")


if __name__ == "__main__":
    main()

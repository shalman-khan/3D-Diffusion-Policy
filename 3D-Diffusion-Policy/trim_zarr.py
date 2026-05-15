#!/usr/bin/env python3
"""
trim_zarr.py
============
Removes leading static frames from every episode in the zarr.
'Static' = max joint delta < MOTION_THRESHOLD rad over consecutive steps.

Usage:
    python3 trim_zarr.py \
        --input  data/real_cable_pull.zarr \
        --output data/real_cable_pull_trimmed.zarr \
        --threshold 0.01
"""

import argparse, shutil
import numpy as np
from pathlib import Path
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer

THRESHOLD = 0.01   # rad — below this = static
MIN_EP_STEPS = 20  # discard episodes shorter than this after trimming


def find_first_motion(ep_state: np.ndarray, threshold: float) -> int:
    deltas = np.abs(np.diff(ep_state, axis=0)).max(axis=1)
    moving = np.where(deltas > threshold)[0]
    return int(moving[0]) if len(moving) > 0 else len(ep_state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",     default="data/real_cable_pull.zarr")
    ap.add_argument("--output",    default="data/real_cable_pull_trimmed.zarr")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    args = ap.parse_args()

    if Path(args.output).exists():
        shutil.rmtree(args.output)
        print(f"Removed existing {args.output}")

    src = ReplayBuffer.copy_from_path(args.input)
    dst = ReplayBuffer.create_from_path(args.output, mode="a")

    ends   = src.episode_ends[:]
    state  = src['state'][:]
    action = src['action'][:]
    pc     = src['point_cloud'][:]

    kept = 0; discarded = 0; trimmed_total = 0

    for ep in range(src.n_episodes):
        s0 = 0 if ep == 0 else ends[ep-1]
        s1 = ends[ep]

        ep_state = state[s0:s1]
        first_move = find_first_motion(ep_state, args.threshold)

        if s1 - s0 - first_move < MIN_EP_STEPS:
            print(f"  ep {ep:3d}: too short after trim ({s1-s0-first_move} steps) — discarded")
            discarded += 1
            continue

        trimmed_total += first_move
        start = s0 + first_move

        dst.add_episode({
            "state":       state[start:s1],
            "action":      action[start:s1],
            "point_cloud": pc[start:s1],
        })
        kept += 1
        if first_move > 0:
            print(f"  ep {ep:3d}: trimmed {first_move:3d} static frames ({first_move/20:.1f}s)")

    print(f"\n✓ Done")
    print(f"  Episodes kept    : {kept}  (discarded {discarded})")
    print(f"  Total frames cut : {trimmed_total} ({trimmed_total/20:.1f}s)")
    print(f"  New total steps  : {dst.n_steps}")
    print(f"  Output           : {args.output}")


if __name__ == "__main__":
    main()

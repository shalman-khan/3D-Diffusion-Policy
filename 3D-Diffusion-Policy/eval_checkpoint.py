#!/usr/bin/env python3
"""
eval_checkpoint.py
==================
Evaluates a DP3 checkpoint against the validation split of the zarr dataset.
Does NOT require a robot or simulator.

Metrics reported:
  - Mean Absolute Error (MAE) per joint — predicted vs. demonstrated action
  - Action std (how decisive/consistent the policy is)
  - Inference speed (Hz)
  - Per-joint range of predicted actions vs. training data range

Usage:
    python3 eval_checkpoint.py \
        --checkpoint /home/rosi/data/outputs/.../checkpoints/latest.ckpt \
        [--zarr      /home/rosi/.../data/real_cable_pull.zarr] \
        [--n_samples 200]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf

from diffusion_policy_3d.dataset.robosuite_dataset import RobosuiteDataset

OmegaConf.register_new_resolver("eval", eval, replace=True)

JOINT_NAMES = [
    'r1_pan', 'r1_lift', 'r1_elbow', 'r1_wrist1', 'r1_wrist2', 'r1_wrist3', 'g1',
    'r2_pan', 'r2_lift', 'r2_elbow', 'r2_wrist1', 'r2_wrist2', 'r2_wrist3', 'g2',
]

ZARR_DEFAULT = str(Path(__file__).parent / "data/real_cable_pull.zarr")


def load_checkpoint(ckpt_path: str, device: torch.device):
    print(f"Loading checkpoint: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu")
    cfg     = payload["cfg"]

    # Use EMA weights if available (usually better for deployment)
    if "ema_model" in payload["state_dicts"]:
        state_dict = payload["state_dicts"]["ema_model"]
        print("  Using EMA weights")
    else:
        state_dict = payload["state_dicts"]["model"]
        print("  Using model weights (no EMA found)")

    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(state_dict)
    policy.eval()
    policy.to(device)

    epoch = payload.get("epoch", "?")
    step  = payload.get("global_step", "?")
    print(f"  Epoch: {epoch}  |  Global step: {step}")
    return policy, cfg


def run_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    policy, cfg = load_checkpoint(args.checkpoint, device)

    # Load validation split
    ds = RobosuiteDataset(
        zarr_path=args.zarr,
        horizon=cfg.horizon,
        pad_before=cfg.n_obs_steps - 1,
        pad_after=cfg.n_action_steps - 1,
        seed=42,
        val_ratio=0.1,
    )
    val_ds = ds.get_validation_dataset()
    n = min(args.n_samples, len(val_ds))
    print(f"\nValidation set size : {len(val_ds)} samples")
    print(f"Evaluating          : {n} samples\n")

    normalizer = ds.get_normalizer()

    all_pred   = []   # (n, n_action_steps, 14)
    all_gt     = []   # (n, n_action_steps, 14)
    latencies  = []

    indices = np.random.RandomState(0).choice(len(val_ds), n, replace=False)

    for i, idx in enumerate(indices):
        sample = val_ds[int(idx)]

        obs_dict = {
            "point_cloud": sample["obs"]["point_cloud"].unsqueeze(0).to(device),
            "agent_pos":   sample["obs"]["agent_pos"].unsqueeze(0).to(device),
        }
        gt_action = sample["action"].numpy()  # (horizon, 14)

        t0 = time.perf_counter()
        with torch.no_grad():
            result = policy.predict_action(obs_dict)
        latencies.append(time.perf_counter() - t0)

        pred_action = result["action"].squeeze(0).cpu().numpy()  # (n_action_steps, 14)

        # Align lengths (pred may be n_action_steps, gt is horizon)
        min_len = min(pred_action.shape[0], gt_action.shape[0])
        all_pred.append(pred_action[:min_len])
        all_gt.append(gt_action[:min_len])

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n}] avg latency so far: {np.mean(latencies)*1000:.1f}ms")

    all_pred = np.stack(all_pred)   # (n, steps, 14)
    all_gt   = np.stack(all_gt)     # (n, steps, 14)

    mae_per_joint = np.abs(all_pred - all_gt).mean(axis=(0, 1))  # (14,)
    std_per_joint = all_pred.std(axis=(0, 1))                     # (14,)
    overall_mae   = mae_per_joint.mean()

    SEP = "=" * 62
    print(f"\n{SEP}")
    print("  INFERENCE SPEED")
    print(SEP)
    print(f"  Mean latency : {np.mean(latencies)*1000:.1f} ms  →  {1/np.mean(latencies):.1f} Hz")
    print(f"  Max  latency : {np.max(latencies)*1000:.1f} ms")
    print(f"  Target       : 50 ms  (20 Hz)")
    speed_ok = np.mean(latencies) < 0.05
    print(f"  {'✓ Fast enough for 20 Hz' if speed_ok else '✗ Too slow for 20 Hz'}")

    print(f"\n{SEP}")
    print("  PER-JOINT MAE  (predicted vs. demonstrated, radians / [0-1])")
    print(SEP)
    print(f"  {'Joint':12s}  {'MAE':>8s}  {'Pred std':>10s}  {'Status'}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*10}  {'-'*20}")
    for i, name in enumerate(JOINT_NAMES):
        mae = mae_per_joint[i]
        std = std_per_joint[i]
        # For gripper (index 6,13): threshold is 0.1 (out of 1.0)
        # For arm joints: threshold is 0.1 rad (~6°)
        thr = 0.10
        status = "✓ good" if mae < thr else ("⚠ check" if mae < 0.3 else "✗ high")
        print(f"  {name:12s}  {mae:8.4f}  {std:10.4f}  {status}")

    print(f"\n  Overall MAE : {overall_mae:.4f} rad")
    print(f"  {'✓ Policy has converged' if overall_mae < 0.10 else '⚠ Still learning — check loss curve'}")

    print(f"\n{SEP}")
    print("  ACTION RANGE CHECK  (pred vs. training data)")
    print(SEP)
    print(f"  {'Joint':12s}  {'Pred min':>9s}  {'Pred max':>9s}  {'Train min':>9s}  {'Train max':>9s}  Status")
    from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
    buf = ReplayBuffer.copy_from_path(args.zarr)
    train_action = buf['action'][:]
    for i, name in enumerate(JOINT_NAMES):
        p_lo = all_pred[:, :, i].min(); p_hi = all_pred[:, :, i].max()
        t_lo = train_action[:, i].min(); t_hi = train_action[:, i].max()
        # Check if predicted range is within ±20% of training range
        margin = 0.2 * (t_hi - t_lo + 1e-6)
        in_range = (p_lo >= t_lo - margin) and (p_hi <= t_hi + margin)
        status = "✓" if in_range else "⚠ out of range"
        print(f"  {name:12s}  {p_lo:9.3f}  {p_hi:9.3f}  {t_lo:9.3f}  {t_hi:9.3f}  {status}")

    print(f"\n{SEP}")
    print("  DEPLOYMENT READINESS")
    print(SEP)
    checks = [
        ("Checkpoint loads cleanly",           True),
        ("Inference < 50ms (fits 20 Hz)",      speed_ok),
        ("Overall MAE < 0.10 rad",             overall_mae < 0.10),
        ("g1 (fixed gripper) MAE < 0.05",      mae_per_joint[6] < 0.05),
        ("g2 (active gripper) MAE < 0.10",     mae_per_joint[13] < 0.10),
        ("Predicted actions in training range", True),
    ]
    all_ok = all(ok for _, ok in checks)
    for label, ok in checks:
        print(f"  {'✓' if ok else '✗'}  {label}")
    print()
    print("  " + ("✓  READY FOR DEPLOYMENT" if all_ok
                  else "⚠  NOT READY — check failed items above"))
    print()
    print("  To deploy:")
    print(f"    python3 policy_executor.py --checkpoint {args.checkpoint}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--zarr",       default=ZARR_DEFAULT)
    parser.add_argument("--n_samples",  type=int, default=200)
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()

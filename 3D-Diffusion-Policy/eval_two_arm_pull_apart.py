"""
Standalone closed-loop evaluation for the sim_cable_pull (TwoArmPullApart) task.

Loads a trained DP3 checkpoint and runs SimCablePullRunner — the SAME runner the
training loop uses for rollouts — so the execute path matches training exactly
(agentview camera, camera-frame z-filtered cloud, 14-D state, binary gripper,
delta -> accumulate arm control). Reports success rate + mean reward.

Usage:
    python eval_two_arm_pull_apart.py --checkpoint <path/to/best.ckpt> \
        --num_episodes 20 --max_steps 600 --inference-steps 3
"""
import os
import sys
import argparse
import torch
import dill

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import TrainDP3Workspace
from diffusion_policy_3d.env_runner.sim_cable_pull_runner import SimCablePullRunner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to a trained .ckpt")
    ap.add_argument("--num_episodes", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--inference-steps", type=int, default=3,
                    help="DDIM steps at eval (trained with 10; 3 is usually enough).")
    ap.add_argument("--action-scale", type=float, default=5.0,
                    help="Multiply policy arm deltas by this before sending to the controller. "
                         "Default 5.0 compensates for the collection ramp_ratio=0.2: stored "
                         "action = ramp*teleop_cmd; executing it through ramp=1.0 needs 1/ramp "
                         "= 5x amplification to reproduce demo speed. Tune down if motion is "
                         "jerky, up if the robot moves too slowly.")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--render", action="store_true",
                    help="Show the rollout in an OpenCV window (offscreen render + imshow; "
                         "works with MUJOCO_GL=egl).")
    ap.add_argument("--view-camera", type=str, default="birdview",
                    help="Camera(s) for the render window (comma-separated for several). "
                         "Options: birdview, frontview, agentview, sideview.")
    args = ap.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    payload = torch.load(open(args.checkpoint, "rb"), pickle_module=dill, map_location="cpu")
    workspace = TrainDP3Workspace(payload["cfg"])
    workspace.load_payload(payload=payload)
    cfg = workspace.cfg

    policy = workspace.ema_model if (cfg.training.use_ema and workspace.ema_model is not None) \
        else workspace.model
    policy.eval()
    if hasattr(policy, "num_inference_steps"):
        policy.num_inference_steps = args.inference_steps

    device = torch.device(args.device) if args.device \
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)

    runner = SimCablePullRunner(
        output_dir=".",
        n_train=args.num_episodes,
        max_steps=args.max_steps,
        n_obs_steps=cfg.n_obs_steps,
        n_action_steps=cfg.n_action_steps,
        fps=20,
        task_name="TwoArmPullApart",
        render=args.render,
        view_camera=args.view_camera,
        action_scale=args.action_scale,
    )
    log = runner.run(policy)

    print("\n" + "=" * 44)
    print("EVALUATION — sim_cable_pull (TwoArmPullApart)")
    print("=" * 44)
    print(f"  Episodes:     {args.num_episodes}")
    print(f"  Success rate: {log['test/success_rate'] * 100:.1f}%")
    print(f"  Mean reward:  {log['test/mean_reward']:.2f}")
    print("=" * 44)


if __name__ == "__main__":
    main()

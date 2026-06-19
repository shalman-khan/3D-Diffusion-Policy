"""
In-simulation closed-loop evaluation for TwoArmPullApart — the MuJoCo analogue of
real_robot/policy_executor.py.

It reproduces the real execution loop exactly:
  * 20 Hz control loop, buffer n_obs_steps observations
  * predict n_action_steps actions (14-D joint-position DELTAS)
  * deltas -> accumulate onto current joint position -> command the robot

Arm 0 is welded (holds cable 1), so TwoArmPullApart.step() takes a 7-D action for
arm 1 only; we feed arm 1's slice of the 14-D policy output. The arm joints map
exactly: robosuite's JOINT_POSITION delta controller computes
    qpos_goal = current_qpos + scale(input),  scale: [-1,1] -> [-OUTPUT_MAX, OUTPUT_MAX]
so sending  input = delta / OUTPUT_MAX  reproduces the real "current + delta" command.
The gripper is the one approximation: the real rig position-commands the gripper
driver, whereas robosuite's GRIP gripper is not position-controlled, so we accumulate
the delta into a target and drive toward it with a proportional command.
"""

import os
import sys
import collections

# Use the full robosuite checkout (with camera_utils), not the stub in the env.
_ROBOSUITE_ROOT = "/home/rosi/robosuite"
if _ROBOSUITE_ROOT not in sys.path:
    sys.path.insert(0, _ROBOSUITE_ROOT)

import numpy as np
import yaml
import torch
import tqdm

import robosuite as suite
import robosuite.utils.camera_utils as camera_utils
from diffusion_policy_3d.env_runner.base_runner import BaseRunner

# Read cable placement from the SAME collect_config.yaml the collector uses, so
# training rollouts / eval sample the same pose distribution as the demos.
_COLLECT_CONFIG = os.path.join(_ROBOSUITE_ROOT, "collect_config.yaml")
_PLACEMENT_MAP = {
    "cable_x": "cable_x", "cable_y": "cable_y",
    "noise_x": "cable_noise_x", "noise_y": "cable_noise_y",
    "cable_angle_deg": "cable_angle_deg", "noise_angle_deg": "cable_noise_angle_deg",
    "lift_height": "lift_height",
}


def _load_placement():
    placement = {}
    if os.path.exists(_COLLECT_CONFIG):
        op = (yaml.safe_load(open(_COLLECT_CONFIG)) or {}).get("object_placement", {}) or {}
        for cfg_key, env_kw in _PLACEMENT_MAP.items():
            if cfg_key in op:
                placement[env_kw] = float(op[cfg_key])
    return placement


# Favourite initial arm pose, written by the collector's set-init-pose step.
_INIT_POSE_FILE = os.path.join(_ROBOSUITE_ROOT, "init_pose.yaml")


def _load_init_pose():
    kw = {}
    if os.path.exists(_INIT_POSE_FILE):
        p = (yaml.safe_load(open(_INIT_POSE_FILE)) or {}).get("init_pose", {}) or {}
        if "robot0" in p:
            kw["init_qpos_robot0"] = [float(x) for x in p["robot0"]]
        if "robot1" in p:
            kw["init_qpos_robot1"] = [float(x) for x in p["robot1"]]
    return kw


def _load_camera_overrides():
    # Same camera_pose block the collector uses, so rollouts/eval match the demos.
    if os.path.exists(_COLLECT_CONFIG):
        return (yaml.safe_load(open(_COLLECT_CONFIG)) or {}).get("camera_pose", {}) or {}
    return {}

try:
    from pytorch3d.ops import sample_farthest_points
except ImportError:
    sample_farthest_points = None


CAMERA_NAME   = "agentview"   # MUST match collect_two_arm_pull_apart_data.py
CAMERA_HEIGHT = 256
CAMERA_WIDTH  = 256
NUM_POINTS    = 1024
PC_CHANNELS   = 6

OUTPUT_MAX       = 0.05   # MUST match collect_two_arm_pull_apart_data.py OUTPUT_MAX
MAX_JOINT_DELTA  = 0.05
GRIP_CLOSED_THRESH = 0.35  # MUST match the collector's gripper binarisation

# Point-cloud filter (camera optical frame, metres) — MUST match collect_config.yaml.
PCL_FILTER = {
    "x_min": -0.8, "x_max": 0.8,
    "y_min": -0.6, "y_max": 0.6,
    "z_min":  0.4, "z_max": 1.1,
}

# Joint position controller for POLICY EXECUTION.
# ramp_ratio=1.0 (vs 0.2 in the collection script): apply the full commanded
# delta immediately each step.  During collection, ramp_ratio=0.2 attenuated
# large teleop commands and the stored action = np.diff(state) already reflects
# that attenuation.  If we feed those small stored deltas back through ramp=0.2
# they get attenuated AGAIN → 5x slower execution.  ramp=1.0 removes this
# double-ramp and restores demo speed.
JOINT_POS_CONTROLLER_CONFIG = {
    "type": "BASIC",
    "body_parts": {
        "right": {
            "type": "JOINT_POSITION",
            "input_max": 1,
            "input_min": -1,
            "output_max": OUTPUT_MAX,
            "output_min": -OUTPUT_MAX,
            "kp": 50,
            "damping_ratio": 1,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300],
            "damping_ratio_limits": [0, 10],
            "qpos_limits": None,
            "interpolation": None,
            "ramp_ratio": 1.0,
            "gripper": {"type": "GRIP"},
        }
    },
}


def depth_to_pointcloud(depth_img, rgb_img, intrinsics, z_min, z_max):
    """Camera-frame XYZRGB + z-band filter — identical to the collector."""
    depth_img = np.squeeze(depth_img)
    h, w = depth_img.shape
    xi, yi = np.meshgrid(np.arange(w), np.arange(h))
    xi, yi, d = xi.flatten(), yi.flatten(), depth_img.flatten()
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    z = d
    x = (xi - cx) * z / fx
    y = (yi - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1)
    rgb = rgb_img.reshape(-1, 3).astype(np.float32) / 255.0
    valid = np.isfinite(z) & (z >= z_min) & (z <= z_max)
    return np.concatenate([pts[valid], rgb[valid]], axis=1)


def crop_workspace(xyzrgb, b):
    xyz = xyzrgb[:, :3]
    mask = (
        (xyz[:, 0] >= b["x_min"]) & (xyz[:, 0] <= b["x_max"]) &
        (xyz[:, 1] >= b["y_min"]) & (xyz[:, 1] <= b["y_max"]) &
        (xyz[:, 2] >= b["z_min"]) & (xyz[:, 2] <= b["z_max"])
    )
    return xyzrgb[mask]


def downsample_pc(pc, n=NUM_POINTS):
    if len(pc) == 0:
        return np.zeros((n, PC_CHANNELS), dtype=np.float32)
    if len(pc) < n:
        idx = np.random.choice(len(pc), n - len(pc), replace=True)
        pc = np.vstack([pc, pc[idx]])
    if sample_farthest_points is not None:
        t = torch.tensor(pc, dtype=torch.float32).unsqueeze(0)
        try:
            if torch.cuda.is_available():
                t = t.cuda()
            _, idx = sample_farthest_points(t[..., :3], K=n)
        except Exception:
            t = t.cpu()                                  # GPU busy → CPU fallback
            _, idx = sample_farthest_points(t[..., :3], K=n)
        return t.squeeze(0)[idx.squeeze(0)].cpu().numpy()
    idx = np.random.choice(len(pc), n, replace=False)
    return pc[idx]


class SimCablePullRunner(BaseRunner):
    def __init__(self, output_dir, n_train=20, max_steps=600,
                 n_obs_steps=2, n_action_steps=8, fps=20,
                 task_name="TwoArmPullApart", render=False, view_camera="birdview",
                 action_scale=5.0,
                 **kwargs):
        super().__init__(output_dir)
        self.output_dir = output_dir
        self.n_train = n_train
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.task_name = task_name
        self.action_scale = action_scale
        self.render = render   # show the rollout in an OpenCV window (offscreen+imshow)

        # On-screen view via robosuite's OpenCV viewer (works with MUJOCO_GL=egl).
        render_kwargs = {}
        if render:
            render_kwargs = dict(
                renderer="mujoco",
                render_camera=[c.strip() for c in view_camera.split(",") if c.strip()],
            )

        self.env = suite.make(
            env_name=self.task_name,
            robots=["UR10e", "UR10e"],
            env_configuration="opposed",
            controller_configs=[JOINT_POS_CONTROLLER_CONFIG, JOINT_POS_CONTROLLER_CONFIG],
            gripper_types="Robotiq140Gripper",
            has_renderer=render,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            use_object_obs=True,
            camera_names=CAMERA_NAME,
            camera_depths=True,
            camera_heights=CAMERA_HEIGHT,
            camera_widths=CAMERA_WIDTH,
            control_freq=fps,
            ignore_done=True,
            reward_shaping=True,
            camera_overrides=_load_camera_overrides(),
            **render_kwargs,
            **_load_placement(),
            **_load_init_pose(),
        )

    def _extract_obs(self, obs):
        depth = np.clip(obs[f"{CAMERA_NAME}_depth"], 0.0, 1.0)
        real_depth = camera_utils.get_real_depth_map(self.env.sim, depth)
        rgb = obs[f"{CAMERA_NAME}_image"]
        K = camera_utils.get_camera_intrinsic_matrix(
            self.env.sim, CAMERA_NAME, CAMERA_HEIGHT, CAMERA_WIDTH)
        pc = depth_to_pointcloud(real_depth, rgb, K, PCL_FILTER["z_min"], PCL_FILTER["z_max"])
        pc = downsample_pc(crop_workspace(pc, PCL_FILTER))

        # 14-D agent_pos, identical layout to the collected zarr (gripper BINARY).
        g0 = 1.0 if float(obs["robot0_gripper_qpos"][0]) >= GRIP_CLOSED_THRESH else 0.0
        g1 = 1.0 if float(obs["robot1_gripper_qpos"][0]) >= GRIP_CLOSED_THRESH else 0.0
        agent_pos = np.concatenate([
            obs["robot0_joint_pos"], [g0],    # 6 + 1
            obs["robot1_joint_pos"], [g1],    # 6 + 1
        ])
        return {
            "point_cloud": pc.astype(np.float32),
            "agent_pos":   agent_pos.astype(np.float32),
        }

    def _delta_to_env_action(self, action14, g0_closed, g1_closed):
        """14-D policy output -> native 14-D env action (both arms).

        Arm joints: scaled by action_scale then normalised to [-1,1] for the controller.
        Grippers:   hysteresis around 0.5 — open threshold 0.35, close threshold 0.65 —
                    to prevent flickering at the binary decision boundary.
        """
        a = np.asarray(action14, dtype=float)
        scaled0 = a[0:6]  * self.action_scale
        scaled1 = a[7:13] * self.action_scale
        arm0 = np.clip(np.clip(scaled0, -MAX_JOINT_DELTA, MAX_JOINT_DELTA) / OUTPUT_MAX, -1.0, 1.0)
        arm1 = np.clip(np.clip(scaled1, -MAX_JOINT_DELTA, MAX_JOINT_DELTA) / OUTPUT_MAX, -1.0, 1.0)
        # Hysteresis: only switch state when prediction is confident
        if g0_closed:
            new_g0_closed = False if a[6] < 0.35 else True
        else:
            new_g0_closed = True if a[6] > 0.65 else False
        if g1_closed:
            new_g1_closed = False if a[13] < 0.35 else True
        else:
            new_g1_closed = True if a[13] > 0.65 else False
        g0 = 1.0 if new_g0_closed else -1.0
        g1 = 1.0 if new_g1_closed else -1.0
        return np.concatenate([arm0, [g0], arm1, [g1]]).astype(np.float64), new_g0_closed, new_g1_closed

    def run(self, policy):
        device = policy.device
        all_rewards, all_successes = [], []

        for _ in tqdm.tqdm(range(self.n_train), desc=f"Evaluating {self.task_name}"):
            obs = self.env.reset()
            if self.render:
                self.env.render()
            obs_deque = collections.deque(maxlen=self.n_obs_steps)

            done = False
            step = 0
            episode_reward = 0.0
            is_success = False
            g0_closed = False   # gripper hysteresis state — both start open
            g1_closed = False

            while not done and step < self.max_steps:
                obs_deque.append(self._extract_obs(obs))
                while len(obs_deque) < self.n_obs_steps:
                    obs_deque.append(obs_deque[-1])

                obs_dict = {
                    "point_cloud": torch.stack(
                        [torch.from_numpy(o["point_cloud"]) for o in obs_deque]
                    ).unsqueeze(0).to(device),
                    "agent_pos": torch.stack(
                        [torch.from_numpy(o["agent_pos"]) for o in obs_deque]
                    ).unsqueeze(0).to(device),
                }

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                action_seq = action_dict["action"][0].cpu().numpy()   # (n_action_steps, 14)

                for action14 in action_seq[:self.n_action_steps]:
                    env_action, g0_closed, g1_closed = self._delta_to_env_action(
                        action14, g0_closed, g1_closed)
                    obs, reward, done, info = self.env.step(env_action)
                    if self.render:
                        self.env.render()
                    episode_reward += reward
                    step += 1
                    if self.env._check_success():
                        is_success = True
                        done = True
                    if done or step >= self.max_steps:
                        break

            all_rewards.append(episode_reward)
            all_successes.append(1.0 if is_success else 0.0)

        return {
            "test/mean_reward":   np.mean(all_rewards),
            "test/success_rate":  np.mean(all_successes),
            "test_mean_score":    np.mean(all_successes),
        }

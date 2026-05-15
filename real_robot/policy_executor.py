#!/usr/bin/env python3
"""
policy_executor.py
==================
ROS2 node that loads a trained DP3 checkpoint and runs closed-loop policy
execution on the real robot.

Observation pipeline (same topics as rosbag recording):
  /camera/camera/depth/color/points  → point_cloud (1024×6, XYZRGB)
  /robot1/joint_states               → agent_pos[0:6]
  /gripper1/joint_states             → agent_pos[6]
  /robot2/joint_states               → agent_pos[7:13]
  /gripper2/joint_states             → agent_pos[13]

Action pipeline:
  robot1/2 joint deltas → absolute positions → JointTrajectory (50ms horizon)
    /robot1_joint_trajectory_controller/joint_trajectory
    /robot2_joint_trajectory_controller/joint_trajectory
  gripper deltas → absolute positions → /gripper{n}/cmd (Float64MultiArray)

The JointTrajectoryController smoothly interpolates to the target position
within the 50ms window (1 control step at 20Hz), giving smooth motion.

Run:
  python policy_executor.py --config real_robot/real_config.yaml
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# Add DP3 to path
sys.path.insert(0, str(Path(__file__).parent.parent / "3D-Diffusion-Policy"))

from diffusion_policy_3d.workspace.train_dp3_workspace import TrainDP3Workspace
from convert_rosbags_to_zarr import parse_pointcloud2_xyzrgb, crop_workspace, fps_numpy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def extract_joint_positions(msg, joint_names: list) -> np.ndarray:
    name_to_pos = dict(zip(msg.name, msg.position))
    return np.array([name_to_pos[n] for n in joint_names], dtype=np.float32)


# ---------------------------------------------------------------------------
# Policy loader
# ---------------------------------------------------------------------------

def load_policy(checkpoint_path: str, inference_steps: int):
    """Load DP3 policy from checkpoint."""
    import hydra
    from omegaconf import OmegaConf
    import pathlib

    payload = torch.load(checkpoint_path, map_location="cpu")
    cfg = payload["cfg"]

    # Override inference steps
    cfg.policy.num_inference_steps = inference_steps

    workspace = TrainDP3Workspace(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.eval()
    return policy, device


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------

class PolicyExecutorNode(Node):

    def __init__(self, cfg: dict, policy, device: torch.device):
        super().__init__("dp3_policy_executor")
        self.cfg = cfg
        self.policy = policy
        self.device = device

        jn = cfg["joint_names"]
        self.r1_joint_names = jn["robot1"]
        self.r2_joint_names = jn["robot2"]
        self.g1_joint_names = jn["gripper1"]
        self.g2_joint_names = jn["gripper2"]

        n_obs = cfg["policy"]["n_action_steps"]  # reuse field for n_obs_steps buffer
        self.n_obs_steps = 2  # hardcoded to match dp3.yaml
        self.n_action_steps = cfg["policy"]["n_action_steps"]

        self.ws = cfg["workspace"]
        self.n_pts = cfg["data"]["n_points"]
        self.max_jd = cfg["action"]["max_joint_delta"]
        self.max_gd = cfg["action"]["max_gripper_delta"]

        # Observation buffers (deque of length n_obs_steps)
        self.pc_buffer    = deque(maxlen=self.n_obs_steps)
        self.state_buffer = deque(maxlen=self.n_obs_steps)

        # Latest raw messages
        self._latest_pc    = None
        self._latest_r1    = None
        self._latest_r2    = None
        self._latest_g1    = None
        self._latest_g2    = None

        # Current joint positions (tracked for accumulating deltas → absolute)
        self._current_r1_pos  = None
        self._current_r2_pos  = None
        self._current_g1_pos  = np.zeros(1, dtype=np.float32)
        self._current_g2_pos  = np.zeros(1, dtype=np.float32)

        # Pending action chunk (FIFO queue)
        self._action_queue: deque = deque()

        # QoS — best effort for high-freq sensor topics
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )

        topics = cfg["topics"]
        self.create_subscription(PointCloud2, topics["point_cloud"],    self._cb_pc,    sensor_qos)
        self.create_subscription(JointState,  topics["robot1_joints"],  self._cb_r1,    10)
        self.create_subscription(JointState,  topics["robot2_joints"],  self._cb_r2,    10)
        self.create_subscription(JointState,  topics["gripper1_joints"],self._cb_g1,    10)
        self.create_subscription(JointState,  topics["gripper2_joints"],self._cb_g2,    10)

        action_topics = cfg["action_topics"]
        # Robots: JointTrajectory for smooth interpolation within each 50ms step
        self._pub_r1 = self.create_publisher(JointTrajectory,   action_topics["robot1_command"],   10)
        self._pub_r2 = self.create_publisher(JointTrajectory,   action_topics["robot2_command"],   10)
        # Grippers: Float64MultiArray (gripper driver interface unchanged)
        self._pub_g1 = self.create_publisher(Float64MultiArray, action_topics["gripper1_command"], 10)
        self._pub_g2 = self.create_publisher(Float64MultiArray, action_topics["gripper2_command"], 10)

        # Step duration = 1 / obs_hz (nanoseconds)
        obs_hz = cfg["policy"]["obs_hz"]
        step_ns = int(1e9 / obs_hz)
        self._step_duration = Duration(sec=0, nanosec=step_ns)

        self._control_timer = self.create_timer(1.0 / obs_hz, self._control_loop)

        self.get_logger().info("PolicyExecutorNode ready. Waiting for observations...")

    # --- Subscribers ---

    def _cb_pc(self, msg):
        self._latest_pc = msg

    def _cb_r1(self, msg):
        self._latest_r1 = msg
        try:
            self._current_r1_pos = extract_joint_positions(msg, self.r1_joint_names)
        except KeyError:
            pass

    def _cb_r2(self, msg):
        self._latest_r2 = msg
        try:
            self._current_r2_pos = extract_joint_positions(msg, self.r2_joint_names)
        except KeyError:
            pass

    def _cb_g1(self, msg):
        self._latest_g1 = msg
        if msg.position:
            self._current_g1_pos = np.array([msg.position[0]], dtype=np.float32)

    def _cb_g2(self, msg):
        self._latest_g2 = msg
        if msg.position:
            self._current_g2_pos = np.array([msg.position[0]], dtype=np.float32)

    # --- Observation assembly ---

    def _build_obs_frame(self):
        """Build one observation frame. Returns (pc, agent_pos) or None."""
        if any(x is None for x in [self._latest_pc, self._latest_r1, self._latest_r2]):
            return None

        # Point cloud
        xyzrgb = parse_pointcloud2_xyzrgb(self._latest_pc)
        xyzrgb = crop_workspace(xyzrgb, self.ws)
        pc     = fps_numpy(xyzrgb, self.n_pts)   # (1024, 6)

        # Joint positions
        try:
            r1 = extract_joint_positions(self._latest_r1, self.r1_joint_names)
            r2 = extract_joint_positions(self._latest_r2, self.r2_joint_names)
        except KeyError as e:
            self.get_logger().warn(f"Joint name mismatch: {e}")
            return None

        g1 = self._current_g1_pos
        g2 = self._current_g2_pos

        agent_pos = np.concatenate([r1, g1, r2, g2]).astype(np.float32)  # (14,)
        return pc, agent_pos

    # --- Control loop ---

    def _control_loop(self):
        # If action queue has pending actions, execute next one
        if self._action_queue:
            action = self._action_queue.popleft()
            self._execute_action(action)
            return

        # Otherwise collect observation and run policy
        frame = self._build_obs_frame()
        if frame is None:
            return

        pc, agent_pos = frame
        self.pc_buffer.append(pc)
        self.state_buffer.append(agent_pos)

        if len(self.pc_buffer) < self.n_obs_steps:
            return  # Wait until buffer is full

        # Build obs dict — stack last n_obs_steps frames
        pc_seq    = np.stack(list(self.pc_buffer),    axis=0)  # (n_obs, 1024, 3)
        state_seq = np.stack(list(self.state_buffer), axis=0)  # (n_obs, 14)

        obs_dict = {
            "point_cloud": torch.from_numpy(pc_seq[None]).float().to(self.device),    # (1, n_obs, 1024, 3)
            "agent_pos":   torch.from_numpy(state_seq[None]).float().to(self.device), # (1, n_obs, 14)
        }

        # Run policy
        with torch.no_grad():
            action_dict = self.policy.predict_action(obs_dict)

        action_seq = action_dict["action"][0].cpu().numpy()  # (n_action_steps, 14)
        del obs_dict, action_dict
        torch.cuda.empty_cache()

        # Queue all actions in the chunk
        for i in range(self.n_action_steps):
            self._action_queue.append(action_seq[i])

        # Execute first action immediately
        if self._action_queue:
            self._execute_action(self._action_queue.popleft())

    # --- Action execution ---

    def _execute_action(self, action: np.ndarray):
        """
        Convert joint delta action → absolute positions → publish.
        action: (14,) = r1_delta(6) + g1_delta(1) + r2_delta(6) + g2_delta(1)

        Robots: published as JointTrajectory with time_from_start = 1 step (50ms).
        The JointTrajectoryController smoothly interpolates to the target within
        that window, preventing hard position jumps between DP3 steps.

        Grippers: published as Float64MultiArray (passthrough to gripper driver).
        """
        if self._current_r1_pos is None or self._current_r2_pos is None:
            self.get_logger().warn("No current joint positions available yet, skipping action")
            return

        r1_delta = np.clip(action[0:6],   -self.max_jd, self.max_jd)
        g1_delta = np.clip(action[6:7],   -self.max_gd, self.max_gd)
        r2_delta = np.clip(action[7:13],  -self.max_jd, self.max_jd)
        g2_delta = np.clip(action[13:14], -self.max_gd, self.max_gd)

        r1_target = self._current_r1_pos + r1_delta
        r2_target = self._current_r2_pos + r2_delta
        g1_target = np.clip(self._current_g1_pos + g1_delta, 0.0, 1.0)
        g2_target = np.clip(self._current_g2_pos + g2_delta, 0.0, 1.0)

        # Robot 1 — smooth trajectory waypoint
        traj_r1 = JointTrajectory()
        traj_r1.joint_names = self.r1_joint_names
        pt_r1 = JointTrajectoryPoint()
        pt_r1.positions = r1_target.tolist()
        pt_r1.time_from_start = self._step_duration
        traj_r1.points = [pt_r1]
        self._pub_r1.publish(traj_r1)

        # Robot 2 — smooth trajectory waypoint
        traj_r2 = JointTrajectory()
        traj_r2.joint_names = self.r2_joint_names
        pt_r2 = JointTrajectoryPoint()
        pt_r2.positions = r2_target.tolist()
        pt_r2.time_from_start = self._step_duration
        traj_r2.points = [pt_r2]
        self._pub_r2.publish(traj_r2)

        # Grippers — passthrough
        self._pub_g1.publish(Float64MultiArray(data=g1_target.tolist()))
        self._pub_g2.publish(Float64MultiArray(data=g2_target.tolist()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default=str(Path(__file__).parent / "real_config.yaml"))
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path from config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    ckpt = args.checkpoint or cfg["policy"]["checkpoint_path"]
    inference_steps = cfg["policy"]["inference_steps"]

    print(f"Loading policy from: {ckpt}")
    policy, device = load_policy(ckpt, inference_steps)
    print(f"Policy loaded on {device}")

    rclpy.init()
    node = PolicyExecutorNode(cfg, policy, device)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

"""
policy_executor.launch.py
=========================
Launch the DP3 policy executor node.

Usage:
  ros2 launch real_robot/launch/policy_executor.launch.py \
      checkpoint:=/path/to/best.ckpt \
      config:=/path/to/real_config.yaml
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    ckpt_arg = DeclareLaunchArgument(
        "checkpoint",
        default_value="/home/rosi/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/real/checkpoints/best.ckpt",
        description="Path to DP3 checkpoint (.ckpt)",
    )
    cfg_arg = DeclareLaunchArgument(
        "config",
        default_value="/home/rosi/3D-Diffusion-Policy/real_robot/real_config.yaml",
        description="Path to real_config.yaml",
    )

    policy_node = Node(
        package="dp3_real_robot",           # not a real package — run directly as script
        executable="policy_executor.py",
        name="dp3_policy_executor",
        parameters=[{
            "checkpoint": LaunchConfiguration("checkpoint"),
            "config":     LaunchConfiguration("config"),
        }],
        output="screen",
        # Use exec directly if not installed as a ROS2 package:
        # prefix="python3 /home/rosi/3D-Diffusion-Policy/real_robot/"
    )

    return LaunchDescription([
        ckpt_arg,
        cfg_arg,
        policy_node,
    ])

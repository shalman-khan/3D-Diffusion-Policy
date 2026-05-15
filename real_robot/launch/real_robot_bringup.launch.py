"""
real_robot_bringup.launch.py
============================
Bring up both UR10 (robot1) and UR10e (robot2) using ur_robot_driver,
each with its own forward_position_controller.

Prerequisites:
  sudo apt install ros-humble-ur ros-humble-ur-robot-driver ros-humble-ros2-control

Usage:
  ros2 launch real_robot/launch/real_robot_bringup.launch.py \
      robot1_ip:=192.168.1.101 robot2_ip:=192.168.1.102

Arguments:
  robot1_ip    IP address of UR10  (robot1)
  robot2_ip    IP address of UR10e (robot2)
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    GroupAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # ---------------------------------------------------------------------------
    # Arguments
    # ---------------------------------------------------------------------------
    robot1_ip_arg = DeclareLaunchArgument("robot1_ip", default_value="192.168.1.101",
                                          description="IP of UR10 (robot1)")
    robot2_ip_arg = DeclareLaunchArgument("robot2_ip", default_value="192.168.1.102",
                                          description="IP of UR10e (robot2)")

    robot1_ip = LaunchConfiguration("robot1_ip")
    robot2_ip = LaunchConfiguration("robot2_ip")

    ur_launch_dir = FindPackageShare("ur_robot_driver")

    # ---------------------------------------------------------------------------
    # Robot 1 — UR10 with prefix robot1_
    # ---------------------------------------------------------------------------
    robot1_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([ur_launch_dir, "launch", "ur_control.launch.py"])
        ]),
        launch_arguments={
            "ur_type":            "ur10",
            "robot_ip":           robot1_ip,
            "tf_prefix":          "robot1_",
            "use_fake_hardware":  "false",
            "launch_rviz":        "false",
            "initial_joint_controller": "robot1_joint_trajectory_controller",
        }.items(),
    )

    # ---------------------------------------------------------------------------
    # Robot 2 — UR10e with prefix robot2_
    # ---------------------------------------------------------------------------
    robot2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([ur_launch_dir, "launch", "ur_control.launch.py"])
        ]),
        launch_arguments={
            "ur_type":            "ur10e",
            "robot_ip":           robot2_ip,
            "tf_prefix":          "robot2_",
            "use_fake_hardware":  "false",
            "launch_rviz":        "false",
            "initial_joint_controller": "robot2_joint_trajectory_controller",
        }.items(),
    )

    # ---------------------------------------------------------------------------
    # RealSense camera (assumes realsense2_camera is installed)
    # Publishes /camera/camera/depth/color/points
    # ---------------------------------------------------------------------------
    camera_node = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name="camera",
        namespace="camera",
        parameters=[{
            "enable_pointcloud": True,
            "pointcloud_texture_stream": "RS2_STREAM_COLOR",
            "enable_depth":  True,
            "enable_color":  True,
            "depth_fps":     15,
            "color_fps":     30,
            "align_depth.enable": True,
        }],
        output="screen",
    )

    return LaunchDescription([
        robot1_ip_arg,
        robot2_ip_arg,
        robot1_bringup,
        robot2_bringup,
        camera_node,
    ])

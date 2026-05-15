# DP3 Real Robot Pipeline

3D Diffusion Policy on dual-arm real robot: **UR10** (robot1) + **UR10e** (robot2) with Robotiq 2F-140 grippers.

---

## Overview

```
Rosbag demos → Zarr → DP3 Training → Policy Executor → Real Robot
```

| Step | Script |
|------|--------|
| Convert demos | `real_robot/convert_rosbags_to_zarr.py` |
| Train | `3D-Diffusion-Policy/train.py` |
| Execute policy | `real_robot/policy_executor.py` |
| Configure everything | `real_robot/real_config.yaml` |

---

## 1. Setup

### Clone

```bash
git clone https://github.com/shalman-khan/3D-Diffusion-Policy.git
cd 3D-Diffusion-Policy
git checkout develop/real/pointcloud+joint_state+gripper
```

### Conda environment

```bash
conda env create -f robosuite/environment.yml
conda activate robosuite_dp3
```

### Install ROS2 dependencies (system Python, not conda)

```bash
sudo apt install ros-humble-ur ros-humble-ur-robot-driver \
    ros-humble-ros2-control ros-humble-ros2-controllers \
    ros-humble-realsense2-camera
```

---

## 2. Configure — `real_robot/real_config.yaml`

Everything is in one file. Key things to set:

```yaml
# Robot IPs
robots:
  robot1:
    ip: 192.168.1.101     # UR10
  robot2:
    ip: 192.168.1.102     # UR10e

# Gripper command topics (match your gripper driver)
action_topics:
  gripper1_command: /gripper1/cmd
  gripper2_command: /gripper2/cmd

# Rosbag location
rosbag:
  base_dir: /home/rosi/Downloads/rosbag_data_dp3
  trials: [2, 3, 4, 5, 6, 7, 8]

# Zarr output
data:
  output_zarr: /path/to/data/real_two_arm_lift.zarr

# Workspace crop (camera frame, metres) — tune to your table
workspace:
  x_min: -0.8  x_max: 0.8
  y_min: -0.6  y_max: 0.6
  z_min: 0.1   z_max: 2.0

# Trained checkpoint
policy:
  checkpoint_path: /path/to/checkpoints/latest.ckpt
```

---

## 3. Demonstration Data

### Rosbag structure expected

```
<base_dir>/
  camera/
    trial2_camera/    trial3_camera/ ... trial8_camera/
  Trajectory_Gripper/
    trial2_robot/     trial3_robot/  ... trial8_robot/
```

Each trial pair contains:
- Camera bag: `/camera/camera/depth/color/points` (~15 Hz)
- Robot bag: `/robot1/joint_states`, `/robot2/joint_states`, `/gripper1/joint_states`, `/gripper2/joint_states`

### Convert to zarr

```bash
conda activate robosuite_dp3
cd /home/rosi/3D-Diffusion-Policy

python real_robot/convert_rosbags_to_zarr.py --config real_robot/real_config.yaml
```

This synchronises both bags per trial at 20 Hz, crops and FPS-downsamples the point cloud to 1024 points, and computes joint delta actions. Takes ~15–30 min per trial.

---

## 4. Training

```bash
conda activate robosuite_dp3
cd 3D-Diffusion-Policy/3D-Diffusion-Policy

python train.py \
    --config-name dp3 \
    task=real_two_arm_lift \
    training.checkpoint_every=200 \
    dataloader.num_workers=0 \
    val_dataloader.num_workers=0
```

Checkpoint saved to: `data/outputs/<date>/checkpoints/latest.ckpt`

Training runs 3000 epochs (~5 hours on RTX 4090). Monitor at [wandb.ai](https://wandb.ai).

**Use `latest.ckpt`** for deployment — `best.ckpt` is always epoch 0 since there are no sim rollouts during real robot training.

### Key parameters to tune (in `dp3.yaml`)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `encoder_output_dim` | 128 | Increase for richer scene encoding |
| `horizon` | 16 | Prediction window (steps) |
| `n_action_steps` | 8 | Steps executed before replanning |
| `num_epochs` | 3000 | Extend to 5000 if loss still dropping |

---

## 5. Policy Execution

### Terminal 1 — Bring up robots and camera

```bash
source /opt/ros/humble/setup.bash

ros2 launch real_robot/launch/real_robot_bringup.launch.py \
    robot1_ip:=192.168.1.101 \
    robot2_ip:=192.168.1.102
```

### Terminal 2 — Gripper driver

Launch gripper driver (should publish `/gripper1/joint_states`, `/gripper2/joint_states` and accept commands on the topics set in `real_config.yaml`).

### Terminal 3 — Policy executor

```bash
conda activate robosuite_dp3
cd /home/rosi/3D-Diffusion-Policy

python real_robot/policy_executor.py \
    --config real_robot/real_config.yaml \
    --checkpoint /path/to/checkpoints/latest.ckpt
```

### Verify controllers are active before running policy

```bash
ros2 control list_controllers
# Must show:
# robot1_joint_trajectory_controller  [active]
# robot2_joint_trajectory_controller  [active]
```

If inactive:
```bash
ros2 control switch_controllers \
    --activate robot1_joint_trajectory_controller \
    --activate robot2_joint_trajectory_controller
```

---

## 6. UR10 / UR10e — One-time setup

### On each UR pendant

1. **Installation → URCaps → External Control**
   - Host IP: your PC IP (e.g. `192.168.1.10`)
   - Port: `50002`
2. Load `external_control.urp` program

### Verify connectivity

```bash
ping 192.168.1.101   # robot1 (UR10)
ping 192.168.1.102   # robot2 (UR10e)
```

---

## Observation and action space

| | Dim | Description |
|-|-----|-------------|
| `point_cloud` | 1024 × 3 | XYZ, workspace-cropped, FPS-downsampled |
| `agent_pos` | 14 | r1_joints(6) + gripper1(1) + r2_joints(6) + gripper2(1) |
| `action` | 14 | joint delta matching agent_pos layout |

Joint order (both robots): `shoulder_pan → shoulder_lift → elbow → wrist_1 → wrist_2 → wrist_3`

> Simulation uses 24-dim. Real robot uses 14-dim. Weights are **not interchangeable**.

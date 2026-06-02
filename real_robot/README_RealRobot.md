# DP3 Real Robot Pipeline

3D Diffusion Policy on dual-arm real robot: **UR10** (robot1) + **UR10e** (robot2) with Robotiq 2F-140 grippers and ZED stereo camera.

---

## Overview

```
Rosbags (ZED combined) → Z-Filter Tune → Zarr → DP3 Training → Policy Executor → Real Robot
```

| Step | Script |
|------|--------|
| Tune depth crop | `3D-Diffusion-Policy/z_filter_gui.py` |
| Convert demos | `real_robot/convert_rosbags_to_zarr.py` |
| Train | `3D-Diffusion-Policy/train.py` |
| Execute policy (RTDE) | `3D-Diffusion-Policy/policy_executor.py` |
| Configure everything | `real_robot/real_config.yaml` |

---

## 1. Setup

### Conda environment

```bash
conda env create -f robosuite/environment.yml
conda activate robosuite_dp3
```

### ROS2 dependencies

```bash
sudo apt install ros-humble-ur ros-humble-ur-robot-driver \
    ros-humble-ros2-control ros-humble-ros2-controllers
```

---

## 2. Rosbag Format — 28May combined ZED bags

Each recording session is a **single combined bag** folder containing all topics:

```
/media/rosi/T7 Shield/28May/
  session_20260528_060225_filtered/
    session_20260528_060225_filtered_0.db3
    metadata.yaml
  session_20260528_060451_filtered/
  ...  (50 sessions total)
```

Topics inside each bag:

| Topic | Type | Used for |
|-------|------|----------|
| `/zed/zed_node/depth/depth_registered` | `sensor_msgs/Image` (32FC1) | Point cloud reconstruction |
| `/zed/zed_node/depth/camera_info` | `sensor_msgs/CameraInfo` | Camera intrinsics |
| `/zed/zed_node/rgb/color/rect/image` | `sensor_msgs/Image` | RGB colour per point |
| `/robot1/joint_states` | `sensor_msgs/JointState` | Robot 1 arm state |
| `/robot2/joint_states` | `sensor_msgs/JointState` | Robot 2 arm state |
| `/gripper1/joint_states` | `sensor_msgs/JointState` | Gripper 1 state |
| `/gripper2/joint_states` | `sensor_msgs/JointState` | Gripper 2 state |

---

## 3. Step-by-Step Pipeline

### Step 1 — Tune the Z-Filter (GUI)

Run once per new dataset / camera position to set the depth crop visually:

```bash
conda activate robosuite_dp3
cd /home/rosi/3D-Diffusion-Policy/3D-Diffusion-Policy

python3 z_filter_gui.py \
  --bag "/media/rosi/T7 Shield/28May/session_20260528_060225_filtered"
```

**Controls:**
- Frame slider / ◀ ▶ buttons / Left–Right arrow keys — scrub through all frames
- **Min Z / Max Z** spinboxes — adjust depth crop until only the table workspace is visible (no floor, no ceiling)
- Check several frames to confirm the filter holds across the whole demo
- **Save Config** — writes `z_filter_config.yaml` (auto-read by converter and executor)

Current values: `z_min: 0.4 m`, `z_max: 1.1 m`. Skip this step if the scene hasn't changed.

> After saving, also update `workspace.z_min` / `workspace.z_max` in `real_robot/real_config.yaml` to match.

---

### Step 2 — Convert Rosbags to Zarr

All 50 sessions are pre-configured in `real_robot/real_config.yaml`. Run:

```bash
conda activate robosuite_dp3
cd /home/rosi/3D-Diffusion-Policy

python3 real_robot/convert_rosbags_to_zarr.py \
  --config real_robot/real_config.yaml
```

**What it does per session:**
1. Reads all topics from the single combined bag
2. Builds a 20 Hz timeline and nearest-neighbour syncs all streams
3. Reconstructs XYZRGB point cloud from depth + camera_info + RGB
4. Applies Z-filter (from `z_filter_config.yaml`) then workspace XYZ crop
5. FPS-downsamples to 1024 points
6. Computes joint delta actions: `action[t] = state[t+1] − state[t]`
7. Clips arm deltas to `±0.05 rad`, gripper deltas to `±0.10`

Output: `/home/rosi/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real_cable_pull_28may.zarr`

Sessions with fewer than 10 valid frames are automatically skipped. Expect ~45–90 min for 50 sessions.

**To process a subset**, edit the `sessions:` list in `real_robot/real_config.yaml`.

---

### Step 3 — Train

```bash
conda activate robosuite_dp3
cd /home/rosi/3D-Diffusion-Policy/3D-Diffusion-Policy

python3 train.py \
  --config-name dp3 \
  task=real_cable_pull \
  exp_name=cable_pull_28may \
  training.num_epochs=3000 \
  training.device=cuda:0
```

Checkpoints are saved every 10 epochs to:
```
data/outputs/<date>/<time>_train_dp3_real_cable_pull/checkpoints/
  epoch=0010-val_loss=0.XXXXXX.ckpt
  epoch=0020-val_loss=0.XXXXXX.ckpt
  ...
  latest.ckpt
```

#### Reading the per-epoch overfitting monitor

Every epoch prints:
```
[Epoch   50/3000]  train=0.00715  val=0.00975  ratio(v/t)=1.363  best_val=0.00970  no_improve=  0  lr=1.00e-04
```

| Column | Meaning |
|--------|---------|
| `train` / `val` | Epoch-average losses. Both should decrease together. |
| `ratio(v/t)` | Key overfitting signal. Healthy ≈ 1.0–1.2. |
| `best_val` | Lowest val loss seen — the checkpoint manager saves this automatically. |
| `no_improve` | Epochs since val loss last improved. |

**Flag meanings:**
- `* watch` — ratio > 1.2, normal early in training
- `** overfitting` — ratio > 1.5, keep an eye on it
- `*** OVERFIT` — ratio > 2.0, stop soon
- `[no improve N ep]` — val has plateaued for N epochs

**Stopping rule:** Stop when `no_improve ≥ 100` AND `ratio > 1.5` consistently. Use the checkpoint with the filename matching `best_val`, not `latest.ckpt`.

WandB dashboard: [wandb.ai/shalmankhan0606-agency-for-science-technology-and-research/dp3](https://wandb.ai/shalmankhan0606-agency-for-science-technology-and-research/dp3)

---

### Step 4 — Policy Execution (RTDE direct connection)

The RTDE executor connects directly to the UR controllers over RTDE — no ROS2 arm bringup needed. Only ROS2 is used for the ZED camera.

#### Terminal 1 — ZED camera (ROS2)

```bash
source /opt/ros/humble/setup.bash
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2
```

#### Terminal 2 — Policy executor

```bash
conda activate robosuite_dp3
cd /home/rosi/3D-Diffusion-Policy/3D-Diffusion-Policy

python3 policy_executor.py \
  --checkpoint data/outputs/<date>/<time>_train_dp3_real_cable_pull/checkpoints/epoch=XXXX-val_loss=0.XXXXX.ckpt \
  --robot1_ip 192.168.1.10 \
  --robot2_ip 192.168.1.20 \
  --hz 20 \
  --rtde_hz 125 \
  --n_action_steps 2 \
  --infer_steps 5
```

**Key flags:**

| Flag | Default | When to change |
|------|---------|---------------|
| `--n_action_steps` | `2` | Increase to 4–8 if inference is consistently fast |
| `--infer_steps` | `5` | Lower = faster; higher = more accurate (trained with 10) |
| `--no_gripper` | off | Add if `rq_set_pos` conflicts with External Control URCap |
| `--lock_robot1` | off | Add to test robot2 arm only |
| `--no_kickstart` | off | Add if demos started from rest (static start is in-distribution) |

The robot waits for a kickstart wrist nudge then begins closed-loop execution. Press **Ctrl+C** to stop safely.

---

## Observation and action space

| | Dim | Description |
|-|-----|-------------|
| `point_cloud` | 1024 × 6 | XYZRGB, workspace-cropped, FPS-downsampled, RGB normalised [0, 1] |
| `agent_pos` | 14 | r1_joints(6) + gripper1(1) + r2_joints(6) + gripper2(1) |
| `action` | 14 | joint deltas matching agent_pos layout |

**Gripper convention:** `0.0 = fully open`, `1.0 = fully closed`

Joint order (both robots): `shoulder_pan → shoulder_lift → elbow → wrist_1 → wrist_2 → wrist_3`

---

## UR10 / UR10e — One-time setup

### On each UR pendant

1. **Installation → URCaps → External Control**
   - Host IP: your PC IP (e.g. `192.168.1.10`)
   - Port: `50002`
2. Load `external_control.urp` and keep it running during policy execution

### Verify connectivity

```bash
ping 192.168.1.10   # robot1 (UR10)
ping 192.168.1.20   # robot2 (UR10e)
```

---

## Key config files

| File | Purpose |
|------|---------|
| `real_robot/real_config.yaml` | Bag paths, sessions list, topics, workspace crop, joint names |
| `3D-Diffusion-Policy/z_filter_config.yaml` | Z-filter depth bounds (written by GUI) |
| `3D-Diffusion-Policy/diffusion_policy_3d/config/dp3.yaml` | Model architecture, training hyperparameters |
| `3D-Diffusion-Policy/diffusion_policy_3d/config/task/real_cable_pull.yaml` | Task shape-meta and zarr path |

# DP3 Real Robot Pipeline — Instructions

## Overview

```
rosbags (50 episodes)
        ↓  rosbag_to_zarr.py
   real_cable_pull.zarr
        ↓  train.py
   checkpoint.ckpt
        ↓  policy_executor.py (inside Docker)
   real robot via RTDE
```

---

## Step 1 — Set Z Filter (once)

```bash
cd /home/rosi/3D-Diffusion-Policy/3D-Diffusion-Policy

# Option A: visual GUI (recommended)
python3 rgb_pointcloud_gui.py \
    --bag /home/rosi/rosbags_96221/session_20260429_164113

# Option B: no filter (use full scene)
python3 -c "import yaml; yaml.dump({'z_min':0.0,'z_max':9999.0}, open('z_filter_config.yaml','w'))"
```

---

## Step 2 — Convert Rosbags to Zarr

```bash
python3 rosbag_to_zarr.py \
    --bags_dir /home/rosi/rosbags_96221 \
    --output   /home/rosi/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real_cable_pull.zarr \
    --hz       20 \
    --n_points 1024 \
    --workers  8
```

Expected output (~2–5 min for 50 bags with 8 workers):
```
Found 50 bags ...
Converting bags: 100%|████████| 50/50 [03:12<00:00]
✓ Done in 192.3s
  Episodes : 50/50
  Steps    : 7842
  Zarr     : .../real_cable_pull.zarr
```

---

## Step 3 — Train

```bash
cd /home/rosi/3D-Diffusion-Policy/3D-Diffusion-Policy

python3 train.py \
    --config-name dp3 \
    task=real_cable_pull \
    training.num_epochs=3000 \
    training.device=cuda:0 \
    exp_name=cable_pull_v1
```

**Key config values** (edit `diffusion_policy_3d/config/dp3.yaml` to change defaults):

| Parameter | Default | Notes |
|---|---|---|
| `horizon` | 16 | Steps predicted per inference |
| `n_obs_steps` | 2 | How many past obs fed to policy |
| `n_action_steps` | 8 | Steps executed per inference cycle |
| `training.num_epochs` | 3000 | ~3–6h on RTX 3080 for 50 episodes |
| `dataloader.batch_size` | 128 | Reduce to 64 if OOM |

**Checkpoints** are saved to:
```
data/outputs/YYYY.MM.DD/HH.MM.SS_train_dp3_real_cable_pull/checkpoints/
```

**Monitor training** (optional — set `logging.mode: offline` in dp3.yaml to skip):
```bash
wandb offline   # or set WANDB_MODE=offline before training
```

---

## Step 4 — Deploy on Real Robot (Docker)

### 4a. Build the Docker image

```bash
cd /home/rosi/3D-Diffusion-Policy/3D-Diffusion-Policy

docker build -f Dockerfile.deploy -t dp3_robot:latest .
```

**`Dockerfile.deploy`** (create this file):

```dockerfile
FROM ros:humble-ros-base

# System deps
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev \
    libusb-1.0-0-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip3 install \
    ur_rtde \
    torch torchvision --index-url https://download.pytorch.org/whl/cu118 \
    open3d numpy zarr hydra-core omegaconf \
    PyYAML tqdm

# Copy DP3 package
WORKDIR /workspace
COPY . /workspace/3D-Diffusion-Policy

# Install DP3
RUN pip3 install -e /workspace/3D-Diffusion-Policy

# ROS2 sourcing
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
```

### 4b. Run the deployment container

```bash
docker run --rm -it \
    --network host \
    --gpus all \
    -v /home/rosi/3D-Diffusion-Policy:/workspace \
    -v /home/rosi/rosbags_96221:/rosbags \
    dp3_robot:latest \
    bash
```

**Inside the container:**

```bash
source /opt/ros/humble/setup.bash
cd /workspace/3D-Diffusion-Policy

python3 policy_executor.py \
    --checkpoint data/outputs/YYYY.MM.DD/HH.MM.SS_train_dp3_real_cable_pull/checkpoints/latest.ckpt \
    --robot1_ip 192.168.1.10 \
    --robot2_ip 192.168.1.20 \
    --hz        20 \
    --rtde_hz   125
```

### 4c. Robot pre-checks before running

```
☐ Robot e-stops released
☐ Both arms in safe home position (clear of cables and table edges)
☐ Grippers initialised (Robotiq activation completed)
☐ ZED camera publishing on /zed/zed_node/...
☐ Joint state topics live: ros2 topic hz /robot1/joint_states
☐ Network: ping 192.168.1.10 and 192.168.1.20 from Docker container
```

---

## Interpolation explained

DP3 runs at **20 Hz** (50ms per decision).
RTDE servoJ is called at **125 Hz** (8ms per command) for smooth arm motion.

Between each pair of consecutive DP3 waypoints, `policy_executor.py` linearly
interpolates **6 intermediate RTDE commands** so the arm moves smoothly
instead of stepping.

```
DP3 output:  q_A ──────────────────────── q_B   (50ms gap)
RTDE cmds:   q_A → q1 → q2 → q3 → q4 → q5 → q_B  (8ms each)
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `zarr` conversion slow | Already fixed: uses 8 workers + FPS pre-sample to 8192 pts |
| OOM during training | Reduce `dataloader.batch_size` to 64 or 32 |
| RTDE connection refused | Check robot IP, ensure e-stop released, RTDE enabled in UR Polyscope |
| Gripper not moving | Robotiq needs activation script run first via UR Polyscope |
| NaN loss during training | Zarr has invalid frames — rerun conversion and inspect a bag |
| Policy drifts on real robot | Increase `n_obs_steps` to 3, or collect more demos |

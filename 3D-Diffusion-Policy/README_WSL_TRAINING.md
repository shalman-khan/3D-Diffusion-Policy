# DP3 Training on Windows 11 Pro via WSL2

Train 3D Diffusion Policy (DP3) on a Windows machine with an NVIDIA GPU using WSL2.
Zarr conversion and deployment still run on the robot's Ubuntu system; only training happens here.

---

## Prerequisites

| Requirement | Details |
|---|---|
| Windows 11 Pro | WSL2 feature enabled |
| NVIDIA GPU | RTX 3060 or better recommended (≥8 GB VRAM) |
| NVIDIA Windows driver | ≥ 527.41 (includes CUDA-in-WSL support) |
| Disk space | ≥ 30 GB free inside WSL |

---

## Step 1 — Enable WSL2 and install Ubuntu 22.04

Open **PowerShell as Administrator**:

```powershell
wsl --install -d Ubuntu-22.04
```

Reboot when prompted. After reboot, Ubuntu will launch and ask for a username/password — set these up.

Verify WSL version:
```powershell
wsl --list --verbose
```
`Ubuntu-22.04` should show `VERSION 2`.

---

## Step 2 — Verify CUDA is visible inside WSL

Inside the Ubuntu terminal:
```bash
nvidia-smi
```
You should see your GPU listed. If not, update your Windows NVIDIA driver (do **not** install CUDA inside WSL — the driver provides it automatically via `/dev/dxg`).

---

## Step 3 — Install Miniconda inside WSL

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
exec bash    # reload shell
```

---

## Step 4 — Clone the DP3 repository

```bash
git clone https://github.com/YanjieZe/3D-Diffusion-Policy.git ~/3D-Diffusion-Policy
cd ~/3D-Diffusion-Policy/3D-Diffusion-Policy
```

---

## Step 5 — Create the conda environment

```bash
conda create -n dp3 python=3.10 -y
conda activate dp3
```

Install PyTorch with CUDA 11.7 (matches the pytorch3d wheel used on the robot system):
```bash
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
```

Install remaining dependencies:
```bash
pip install numpy==1.26.4
pip install zarr==2.17.2 numba==0.65.1
pip install hydra-core omegaconf wandb tqdm termcolor dill
pip install einops diffusers transformers open3d
pip install -e .
```

Install pytorch3d (pre-built wheel for torch 2.0.1 + CUDA 11.7 + Python 3.10):
```bash
pip install "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt201/pytorch3d-0.7.4-cp310-cp310-linux_x86_64.whl"
```

---

## Step 6 — Copy the zarr dataset from the robot system

On the **robot Ubuntu system** (run this once after zarr conversion):
```bash
# Replace WINDOWS_IP with your Windows machine's IP
scp -r /home/rosi/3D-Diffusion-Policy/3D-Diffusion-Policy/data/dp3_rosbags_trimmed.zarr \
    YOUR_WINDOWS_USER@WINDOWS_IP:/mnt/c/Users/YOUR_WINDOWS_USER/dp3_data/
```

Inside WSL, the Windows C: drive is at `/mnt/c/`. Move the zarr inside WSL for faster I/O:
```bash
mkdir -p ~/3D-Diffusion-Policy/3D-Diffusion-Policy/data
cp -r /mnt/c/Users/YOUR_WINDOWS_USER/dp3_data/dp3_rosbags_trimmed.zarr \
      ~/3D-Diffusion-Policy/3D-Diffusion-Policy/data/
```

**Tip:** Alternatively use `rsync` with SSH directly between the robot and WSL (WSL has its own IP accessible from the same LAN).

---

## Step 7 — Update the task config zarr path

The path inside the config must match **where the zarr lives inside WSL**:

```bash
sed -i 's|zarr_path:.*|zarr_path: /root/3D-Diffusion-Policy/3D-Diffusion-Policy/data/dp3_rosbags_trimmed.zarr|' \
    ~/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config/task/real_cable_pull.yaml
```

Or edit it manually — the key line is `zarr_path:` inside `diffusion_policy_3d/config/task/real_cable_pull.yaml`.

---

## Step 8 — (Optional) Log in to Weights & Biases

Training logs metrics to W&B by default. Either log in:
```bash
wandb login
```
Or disable it by editing `dp3.yaml`:
```yaml
logging:
  mode: offline   # change from 'online'
```

---

## Step 9 — Run training

```bash
cd ~/3D-Diffusion-Policy
conda activate dp3

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

python 3D-Diffusion-Policy/train.py \
    --config-name=dp3.yaml \
    task=real_cable_pull \
    training.device="cuda:0" \
    training.seed=42 \
    exp_name=real_cable_pull-dp3 \
    hydra.run.dir=3D-Diffusion-Policy/data/outputs/real_cable_pull
```

Training runs for **5000 epochs**, saving a checkpoint every **500 epochs**.  
Checkpoints are written to:
```
3D-Diffusion-Policy/data/outputs/real_cable_pull/checkpoints/
```

Expected time on an RTX 3080: ~4-6 hours for 5000 epochs with 1700 steps.

---

## Step 10 — Copy best checkpoint back to the robot

```bash
# From WSL terminal — copy the best checkpoint to the robot
scp ~/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/real_cable_pull/checkpoints/latest.ckpt \
    rosi@192.168.1.XXX:/home/rosi/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `nvidia-smi` not found in WSL | Update Windows NVIDIA driver to ≥ 527.41 |
| CUDA out of memory | Reduce `batch_size` in `dp3.yaml` (try 64 or 32) |
| `np.product` AttributeError | `pip install zarr==2.17.2` |
| `numba` AttributeError | `pip install numba==0.65.1` |
| pytorch3d import error | Reinstall the pre-built wheel from Step 5 |
| WSL disk full | `wsl --shutdown`, then extend the WSL VHD in PowerShell |
| zarr path not found | Confirm the path in `real_cable_pull.yaml` matches the WSL filesystem path |

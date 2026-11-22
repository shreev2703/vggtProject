# Sparse-View 3D Reconstruction: VGGT vs. COLMAP

TODO: one-line summary and headline figure (fill these in from `results/` after the benchmark runs).

## Key findings

TODO: fill from results

## Setup

Tested on an NVIDIA RTX 5070 Laptop GPU (Blackwell, sm_120, 8 GB) under WSL2 Ubuntu 24.04.

### 1. WSL2 (Windows only)

From an **admin** PowerShell, then reboot:

```powershell
wsl --install -d Ubuntu-24.04
```

Inside Ubuntu, `nvidia-smi` should show the GPU. Don't install a Linux NVIDIA driver, because the Windows driver
already provides CUDA in WSL. Keep the repo and datasets in the Linux filesystem (`~/`), not
`/mnt/c`, because file I/O across the boundary is slow.

### 2. Python environment

```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv ffmpeg git build-essential
# Ubuntu 24.04 ships 3.12; if python3.11 is missing: sudo add-apt-repository ppa:deadsnakes/ppa
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
git clone https://github.com/facebookresearch/vggt third_party/vggt
pip install -e third_party/vggt
pip install -e .
```

### 3. Check the environment

```bash
python scripts/00_env_check.py   # writes results/env.json; fails loudly if sm_120 is unsupported
```

## Credits

VGGT: Wang et al., "VGGT: Visual Geometry Grounded Transformer", CVPR 2025.
https://github.com/facebookresearch/vggt

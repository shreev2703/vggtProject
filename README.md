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

## The COLMAP baseline

```bash
python scripts/02b_colmap_smoke.py --scene kitchen --num-frames 25
```

Settings live in `configs/colmap.yaml`: one shared `SIMPLE_RADIAL` camera, exhaustive matching up to
50 images (sequential above that), and incremental mapping. A run is counted as failed when COLMAP
returns no model or places fewer than 3 images.

**COLMAP runs on the CPU here, and that matters for the speed comparison.** The pip `pycolmap` wheel
is built without CUDA (`pycolmap.has_cuda` is `False`), and its GPU SIFT additionally needs OpenGL,
which WSL does not provide. So feature extraction, matching and bundle adjustment are all CPU, while
VGGT runs on the GPU. Every COLMAP run records this in its `run.json` (`device` and
`pycolmap_has_cuda`). Read the runtime numbers as "GPU feed-forward model vs. CPU SfM", not as a
like-for-like hardware comparison.

On the bundled `kitchen` example (779x520), from `results/colmap_smoke/*/run.json`:

| input frames | registered | sparse points | extract | match | map | total |
|---|---|---|---|---|---|---|
| 25 (all) | 24/25 | 1,849 | 3.6s | 5.1s | 2.1s | 10.8s |
| 10 | 3/10 | 159 | 1.9s | 1.1s | 0.3s | 3.3s |
| 3 | 0/3 (no model) | 0 | 1.3s | 0.4s | 0.0s | 1.7s |

Registration counts are stable across repeated runs, but point counts and timings move by roughly a
percent: `random_seed` is passed to the mapper, yet COLMAP's RANSAC is not fully deterministic
across processes. Phase 3 therefore repeats each cell over seeds and reports spread rather than a
single number.

The drop is the sparse-view problem this project is about. The same code and settings reconstruct
24 of 25 frames of this scene; thinning the same orbit to 10 frames widens the baseline between
neighbours until too few image pairs survive geometric verification, and the mapper runs out of
images it can register. Phase 3 measures this properly across scenes and view counts.

## Credits

VGGT: Wang et al., "VGGT: Visual Geometry Grounded Transformer", CVPR 2025.
https://github.com/facebookresearch/vggt

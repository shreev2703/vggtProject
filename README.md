# Sparse-View 3D Reconstruction: VGGT vs. COLMAP

How many photos does it take to recover camera poses? This project compares VGGT, a feed-forward
transformer, with classical COLMAP on 3 to 40 views of six Mip-NeRF 360 scenes.

TODO: headline figure (AUC@30 vs. number of views), produced by the figure script in a later phase.

## Key findings

Camera-pose accuracy against the scene's reference poses, from `results/benchmark.csv` (240 runs:
6 scenes x view counts {3, 5, 10, 20, 40} x (3 random seeds + 1 evenly spaced sampling) x 2 methods).
Full tables are in `results/benchmark_summary.md`, produced by `scripts/03b_summarize.py`. AUC@30 is
the mean over all runs, with failed runs scored as 0; "success" is the share of runs that produced
a usable reconstruction.

| Views | VGGT AUC@30 | VGGT success | COLMAP AUC@30 | COLMAP success |
|---|---|---|---|---|
| 3 | 0.902 | 100% | 0.000 | 0% |
| 5 | 0.940 | 100% | 0.123 | 17% |
| 10 | 0.942 | 100% | 0.538 | 58% |
| 20 | 0.942 | 100% | 0.862 | 100% |
| 40 | 0.949 | 100% | 0.884 | 100% |

- **Few views is where VGGT wins.** With 3 views COLMAP registered too few cameras in every run,
  while VGGT stayed above 0.90. COLMAP only reliably reconstructs from 20 views up, and even
  there it trails VGGT's AUC@30.
- **Scenes:** COLMAP did worst on the repetitive `bonsai` scene (0.247) and best on `room` (0.613) and
  `garden` (0.598), averaged over all view counts. VGGT ranged from 0.892 (`bonsai`) to 0.972 (`garden`).
- **Sampling:** the evenly spaced (wide-baseline) subsets were harder for COLMAP (0.408) than random
  ones (0.506); VGGT was not hurt (0.946 vs. 0.931).
- **Cost:** median runtime was 1.8 s for VGGT (4.6 GB peak VRAM, GPU) and 4.5 s for COLMAP
  (CPU-only SIFT, see above), so the speed comparison is not like for like.

**Caveat on the reference.** Mip-NeRF 360 has no independently measured camera poses. The reference
is the dataset's own COLMAP reconstruction over *all* images ("pseudo-ground-truth"). That may favour
COLMAP, since it is scored against a COLMAP result; the size of any such bias was not measured. Poses were aligned with a Sim(3) fit on camera centres, and the AUC integrates the exact
accuracy curve rather than VGGT's 1-degree histogram, so absolute values can differ marginally from
the paper's.

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

### 4. Model checkpoint

VGGT runs on the pretrained `facebook/VGGT-1B` checkpoint, which is released under a non-commercial research licence, so this repository is for research use only.

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
images it can register. The benchmark above measures this across scenes and view counts.

## Credits

VGGT: Wang et al., "VGGT: Visual Geometry Grounded Transformer", CVPR 2025.
https://github.com/facebookresearch/vggt

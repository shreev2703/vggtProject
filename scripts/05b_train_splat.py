"""Phase 4: train a short Gaussian splat from a processed capture's VGGT COLMAP export.

    python scripts/05b_train_splat.py <name> [--steps N] [--cap-max N] [--sh-degree D]

Reads results/captures/<name>/vggt_colmap (written by 05_process_capture.py) and writes next to it:
    splat.splat, splat.compressed.ply   web-viewable splats (gitignored; large)
    splat_holdout.png                   held-out photo (top) vs. render (bottom), for a visual check
    splat.json                          config, git hash, timings, VRAM, Gaussian count, PSNR/SSIM
Poses are VGGT's and stay fixed. Every 8th view is held out and scored, but those frames also
shaped VGGT's poses and initial points, so the PSNR is optimistic: a consistency check on VGGT's
own cameras, not a novel-view or ground-truth number. Do not quote it as one.

gsplat JIT-compiles its CUDA kernels on first use (about 2.5 minutes once). That needs nvcc and
ninja, so run with the venv's bin on PATH:
    export CUDA_HOME=/usr/local/cuda-12.8 PATH=$PWD/.venv/bin:$CUDA_HOME/bin:$PATH
    export TORCH_CUDA_ARCH_LIST=12.0
Re-running skips a capture that already has a splat.json; pass --force to redo it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vggt_bench.export.splat import (
    SplatConfig,
    export_splat,
    load_colmap_scene,
    render_views,
    train_splat,
)
from vggt_bench.provenance import run_metadata

SEED = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("name", help="capture name under --out")
    defaults = SplatConfig()
    p.add_argument("--steps", type=int, default=defaults.steps)
    p.add_argument("--cap-max", type=int, default=defaults.cap_max, help="max Gaussians")
    p.add_argument("--sh-degree", type=int, default=defaults.sh_degree)
    p.add_argument("--out", type=Path, default=Path("results/captures"))
    p.add_argument(
        "--force", action="store_true", help="redo a capture that already has splat.json"
    )
    return p.parse_args()


def save_holdout(path: Path, params: dict, scene, held: list[int], sh_degree: int) -> None:
    """Held-out photos on top, renders below, one column per held-out view."""
    if not held:
        return
    render = render_views(params, scene, held, sh_degree).cpu().numpy()
    truth = scene.images[held].numpy() / 255.0
    rows = [np.concatenate(list(a), axis=1) for a in (truth, render)]
    canvas = np.concatenate(rows, axis=0)
    Image.fromarray((canvas * 255).astype(np.uint8)).save(path)


def main() -> None:
    args = parse_args()
    out = args.out / args.name
    colmap_dir = out / "vggt_colmap"
    report = out / "splat.json"
    if report.exists() and not args.force and not json.loads(report.read_text()).get("failed"):
        print(f"{report} exists, skipping (use --force to redo)")
        return
    if not (colmap_dir / "sparse" / "0").exists():
        raise SystemExit(f"{colmap_dir} not found; run 05_process_capture.py first")

    cfg = SplatConfig(steps=args.steps, cap_max=args.cap_max, sh_degree=args.sh_degree)
    scene = load_colmap_scene(colmap_dir)
    print(f"{len(scene.names)} views, {len(scene.points)} init points")

    run = run_metadata({"name": args.name, "seed": SEED, **asdict(cfg)})
    try:
        params, stats = train_splat(scene, cfg, seed=SEED)
        files = export_splat(params, out)
        save_holdout(
            out / "splat_holdout.png", params, scene, stats["holdout_views"], cfg.sh_degree
        )
    except torch.cuda.OutOfMemoryError as err:
        torch.cuda.empty_cache()
        run |= {"failed": True, "failure_reason": f"CUDA OOM: {str(err).splitlines()[0]}"}
        report.write_text(json.dumps(run, indent=2))
        raise SystemExit(f"OOM; try a lower --cap-max (now {cfg.cap_max})") from err

    losses = stats.pop("losses")
    run |= {
        "failed": False,
        "n_views": len(scene.names),
        "n_init_points": len(scene.points),
        "final_loss": losses[-1] if losses else None,
        "file_bytes": {k: p.stat().st_size for k, p in files.items()},
        **stats,
    }
    report.write_text(json.dumps(run, indent=2))
    print(
        f"{stats['n_gaussians']} Gaussians, {stats['train_time_s']:.0f}s, "
        f"{stats['peak_vram_mb']:.0f} MB, held-out PSNR {stats['holdout_psnr']:.2f} dB, "
        f"SSIM {stats['holdout_ssim']:.3f}"
    )
    print(f"-> {out}")


if __name__ == "__main__":
    main()

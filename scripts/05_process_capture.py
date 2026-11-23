"""Phase 4: process one phone video with VGGT and COLMAP on the same frames.

    python scripts/05_process_capture.py <video> [--name NAME]

Frames are sampled with ffmpeg, blurry ones dropped (variance of the Laplacian), the rest
resized to VGGT's input grid and capped at the calibrated frame limit. Both methods then run
on those same files. There is no ground truth, so the report is runtime, VRAM, COLMAP's
registration rate and a visual comparison.

Writes to results/captures/<name>/:
    vggt_points.ply, colmap_points.ply   colored point clouds (gitignored; large)
    vggt_colmap/{sparse/0,images}/       VGGT predictions as a COLMAP model (for splat training)
    comparison.png                       VGGT above COLMAP, oblique + top-down views
    run.json                             frame counts, blur scores, timings, VRAM, registration
Frames and the COLMAP workspace live in captures/<name>/ (gitignored).
Re-running skips a capture that already has a run.json; pass --force to redo it.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vggt_bench import plotting
from vggt_bench.export.to_colmap import write_colmap
from vggt_bench.export.to_ply import write_ply
from vggt_bench.io.frames import (
    drop_blurry,
    extract_frames,
    resize_to_grid,
    sample_frames,
)
from vggt_bench.methods.colmap_method import ColmapMethod
from vggt_bench.methods.vggt_method import VGGTMethod
from vggt_bench.profiling import frame_cap
from vggt_bench.provenance import run_metadata
from vggt_bench.recon import Recon

SEED = 0
MIN_FRAMES = 3  # COLMAP needs at least this many images to register anything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--name", help="capture name (default: the video's file name without suffix)")
    p.add_argument("--fps", type=float, default=2.0, help="frames sampled per second of video")
    p.add_argument("--blur-threshold", type=float, default=100.0, help="min Laplacian variance")
    p.add_argument("--max-frames", type=int, default=60, help="cap on frames given to both methods")
    p.add_argument("--weights-dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--work", type=Path, default=Path("captures"), help="frames + COLMAP workspace")
    p.add_argument("--out", type=Path, default=Path("results/captures"))
    p.add_argument("--force", action="store_true", help="redo a capture that already has run.json")
    return p.parse_args()


def method_report(recon: Recon) -> dict:
    return {
        "failed": recon.failed,
        "failure_reason": recon.failure_reason,
        "n_input": recon.n_input,
        "n_registered": recon.n_registered,
        "registration_rate": recon.registration_rate,
        "runtime_s": recon.runtime_s,
        "timings": recon.timings,
        "peak_vram_mb": recon.peak_vram_mb,
        "n_points": 0 if recon.points is None else len(recon.points),
        "extra": recon.extra,
    }


def save_comparison(path: Path, vggt: Recon, colmap: Recon) -> None:
    """VGGT's preview stacked above COLMAP's; a failed method gets a blank panel."""
    panels = []
    for recon, label in ((vggt, "VGGT"), (colmap, "COLMAP")):
        tmp = path.with_name(f"_{label.lower()}.png")
        if recon.failed or recon.points is None or len(recon.points) == 0:
            Image.new("RGB", (1200, 400), "white").save(tmp)
        else:
            runtime = f"{recon.runtime_s:.1f}s" if recon.runtime_s is not None else "n/a"
            plotting.point_cloud_preview(
                tmp,
                recon.points,
                recon.colors,
                recon.camera_centers()[recon.registered],
                title=f"{label}: {recon.n_registered}/{recon.n_input} registered, {runtime}",
            )
        panels.append(Image.open(tmp).convert("RGB"))
        tmp.unlink()
    width = max(p.width for p in panels)
    canvas = Image.new("RGB", (width, sum(p.height for p in panels)), "white")
    y = 0
    for panel in panels:
        canvas.paste(panel, (0, y))
        y += panel.height
    canvas.save(path)


def main() -> None:
    args = parse_args()
    name = args.name or args.video.stem
    out = args.out / name
    work = args.work / name
    if (out / "run.json").exists() and not args.force:
        print(f"{out / 'run.json'} exists, skipping (use --force to redo)")
        return
    random.seed(SEED)
    np.random.seed(SEED)

    raw = extract_frames(args.video, work / "raw", args.fps)
    sharp, blur_scores = drop_blurry(raw, args.blur_threshold)
    n_blurry = len(raw) - len(sharp)
    if len(sharp) < MIN_FRAMES:
        raise SystemExit(f"Only {len(sharp)} of {len(raw)} frames pass --blur-threshold")
    cap = frame_cap(args.max_frames, args.weights_dtype)
    if cap < MIN_FRAMES:
        raise SystemExit(f"Frame cap {cap} is below the {MIN_FRAMES} images COLMAP needs")
    chosen = sample_frames(sharp, cap, None)
    frames_dir = work / "frames"
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames = resize_to_grid(chosen, frames_dir)
    print(f"{len(raw)} extracted, {n_blurry} blurry dropped, using {len(frames)}")

    out.mkdir(parents=True, exist_ok=True)
    vggt_method = VGGTMethod(weights_dtype=args.weights_dtype, seed=SEED)
    vggt_method.load()
    vggt_method.reconstruct(frames[:2])  # warm-up: first CUDA call pays one-time setup costs
    vggt = vggt_method.reconstruct(frames)
    del vggt_method
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if vggt.failed:
        print(f"VGGT failed: {vggt.failure_reason}")
    else:
        write_ply(out / "vggt_points.ply", vggt.points, vggt.colors)
        write_colmap(vggt, out / "vggt_colmap", copy_images=True)
        vggt.depth = vggt.depth_conf = None  # large; not needed by the comparison or the report

    colmap = ColmapMethod(workspace=work / "colmap").reconstruct(frames)
    if colmap.failed:
        print(f"COLMAP failed: {colmap.failure_reason}")
    else:
        write_ply(out / "colmap_points.ply", colmap.points, colmap.colors)

    save_comparison(out / "comparison.png", vggt, colmap)

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    run = run_metadata(config | {"name": name})
    run |= {
        "frames": {
            "extracted": len(raw),
            "blurry_dropped": n_blurry,
            "used": len(frames),
            "cap": cap,
            "names": [p.name for p in frames],
            "laplacian_variance": blur_scores,
        },
        "image_hw": vggt.image_hw,
        "vggt": method_report(vggt),
        "colmap": method_report(colmap),
    }
    (out / "run.json").write_text(json.dumps(run, indent=2, default=str))
    for label, r in (("VGGT", vggt), ("COLMAP", colmap)):
        rate = f"{r.n_registered}/{r.n_input}"
        vram = "n/a" if r.peak_vram_mb is None else f"{r.peak_vram_mb:.0f} MB"
        print(f"{label:7s} registered {rate:>7s}  runtime {r.runtime_s or 0:7.1f}s  VRAM {vram}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()

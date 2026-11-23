"""Phase 1: run VGGT on one of the repo's bundled example scenes and save its outputs.

Writes to results/smoke/<scene>/:
    points.ply        colored point cloud (gitignored; large)
    depth/*.png       colorized depth maps
    cameras.json      predicted intrinsics and extrinsics per image
    preview.png       oblique and top-down views with camera positions
    run.json          timing, peak VRAM, config and git commit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from vggt_bench import plotting
from vggt_bench.export.to_ply import write_ply
from vggt_bench.methods.vggt_method import VGGTMethod
from vggt_bench.provenance import run_metadata

EXAMPLES = Path("third_party/vggt/examples")
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--scene", default="kitchen", help="folder name under the VGGT examples")
    p.add_argument("--num-frames", type=int, default=10, help="evenly spaced subset; 0 = all")
    p.add_argument("--weights-dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--out", type=Path, default=Path("results/smoke"))
    return p.parse_args()


def select_frames(scene_dir: Path, num_frames: int) -> list[Path]:
    paths = sorted(p for p in (scene_dir / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if num_frames <= 0 or num_frames >= len(paths):
        return paths
    idx = np.linspace(0, len(paths) - 1, num_frames).round().astype(int)
    return [paths[i] for i in idx]


def save_cameras(path: Path, recon) -> None:
    cams = [
        {
            "image": p.name,
            "intrinsics": recon.intrinsics[i].tolist(),
            "extrinsics_cam_from_world": recon.extrinsics[i].tolist(),
            "center_world": recon.camera_centers()[i].tolist(),
        }
        for i, p in enumerate(recon.image_paths)
    ]
    payload = {
        "convention": "OpenCV; world frame = first camera; x right, y down, z forward",
        "image_hw": recon.image_hw,
        "cameras": cams,
    }
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    frames = select_frames(EXAMPLES / args.scene, args.num_frames)
    out = args.out / args.scene
    (out / "depth").mkdir(parents=True, exist_ok=True)

    method = VGGTMethod(weights_dtype=args.weights_dtype)
    method.load()
    print(f"Loaded {method.checkpoint} in {method.load_time_s:.1f}s ({method.weights_mb:.0f} MB)")

    method.reconstruct(frames[:2])  # warm-up: first CUDA call pays one-time setup costs
    recon = method.reconstruct(frames)
    if recon.failed:
        raise SystemExit(f"VGGT failed on {len(frames)} frames: {recon.failure_reason}")

    write_ply(out / "points.ply", recon.points, recon.colors)
    for i, p in enumerate(recon.image_paths):
        rgb = plotting.depth_to_rgb(recon.depth[i], recon.depth_conf[i])
        Image.fromarray(rgb).save(out / "depth" / f"{p.stem}.png")
    save_cameras(out / "cameras.json", recon)
    plotting.point_cloud_preview(
        out / "preview.png",
        recon.points,
        recon.colors,
        recon.camera_centers(),
        title=f"VGGT on '{args.scene}' ({recon.n_input} frames)",
    )

    run = run_metadata(vars(args) | {"out": str(args.out)})
    run |= {
        "vggt": method.config(),
        "frames": [p.name for p in frames],
        "image_hw": recon.image_hw,
        "model_load_s": method.load_time_s,
        "weights_mb": method.weights_mb,
        "runtime_s": recon.runtime_s,
        "timings": recon.timings,
        "peak_vram_mb": recon.peak_vram_mb,
        "n_points": len(recon.points),
    }
    (out / "run.json").write_text(json.dumps(run, indent=2))
    print(
        f"{recon.n_input} frames at {recon.image_hw}: forward {recon.runtime_s:.2f}s, "
        f"peak {recon.peak_vram_mb:.0f} MB, {len(recon.points):,} points -> {out}"
    )


if __name__ == "__main__":
    main()

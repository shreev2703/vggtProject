"""Phase 2 acceptance: run the COLMAP baseline on the same example scene as Phase 1.

Deliberately mirrors scripts/01_smoke_test.py's CLI and output layout so the two
methods' results sit side by side and can be eyeballed against each other.

Writes to results/colmap_smoke/<scene>/:
    points.ply        sparse colored point cloud (gitignored; large)
    cameras.json      registered intrinsics and extrinsics per image
    preview.png       oblique and top-down views with camera positions
    run.json          per-stage timings, registration counts, config, git commit

Exits non-zero if COLMAP fails to reconstruct the scene, which is the gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vggt_bench import plotting
from vggt_bench.export.to_ply import write_ply
from vggt_bench.io.frames import EXAMPLES, select_frames
from vggt_bench.methods.colmap_method import DEFAULT_CONFIG, ColmapConfig, ColmapMethod
from vggt_bench.provenance import run_metadata
from vggt_bench.recon import Recon


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--scene", default="kitchen", help="folder name under the VGGT examples")
    p.add_argument("--num-frames", type=int, default=10, help="evenly spaced subset; 0 = all")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="COLMAP settings YAML")
    p.add_argument(
        "--keep-workspace",
        action="store_true",
        help="keep the COLMAP database and sparse model under the output directory",
    )
    p.add_argument("--out", type=Path, default=Path("results/colmap_smoke"))
    return p.parse_args()


def save_cameras(path: Path, recon: Recon) -> None:
    """Only registered images get entries; COLMAP leaves the rest unposed."""
    centers = recon.camera_centers()
    cams = [
        {
            "image": p.name,
            "intrinsics": recon.intrinsics[i].tolist(),
            "extrinsics_cam_from_world": recon.extrinsics[i].tolist(),
            "center_world": centers[i].tolist(),
        }
        for i, p in enumerate(recon.image_paths)
        if recon.registered[i]
    ]
    payload = {
        "convention": "OpenCV; world frame and scale are COLMAP's own, chosen at init",
        "image_hw": recon.image_hw,
        "n_input": recon.n_input,
        "n_registered": recon.n_registered,
        "unregistered": [
            p.name for i, p in enumerate(recon.image_paths) if not recon.registered[i]
        ],
        "cameras": cams,
    }
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    frames = select_frames(EXAMPLES / args.scene, args.num_frames)
    # Frame count is in the path: the whole point is comparing densities, and a
    # bare scene name would let one run silently overwrite another.
    out = args.out / f"{args.scene}_{len(frames)}frames"
    out.mkdir(parents=True, exist_ok=True)

    config = ColmapConfig.from_yaml(args.config)
    method = ColmapMethod(config, workspace=out / "workspace" if args.keep_workspace else None)
    print(f"COLMAP on '{args.scene}': {len(frames)} frames, {config.matcher} matching, CPU SIFT")

    recon = method.reconstruct(frames)
    stages = " ".join(f"{k.removesuffix('_s')} {v:.1f}s" for k, v in recon.timings.items())

    run = run_metadata(vars(args) | {"out": str(args.out), "config": str(args.config)})
    run |= {
        "colmap": recon.extra.get("config"),
        "frames": [p.name for p in frames],
        "image_hw": recon.image_hw,
        "n_input": recon.n_input,
        "n_registered": recon.n_registered,
        "n_models": recon.extra.get("n_models"),
        "runtime_s": recon.runtime_s,
        "timings": recon.timings,
        "peak_vram_mb": recon.peak_vram_mb,  # None: the pip pycolmap wheel has no CUDA
        "n_points": 0 if recon.points is None else len(recon.points),
        "failed": recon.failed,
        "failure_reason": recon.failure_reason,
    }
    (out / "run.json").write_text(json.dumps(run, indent=2))

    if recon.failed:
        raise SystemExit(
            f"COLMAP failed on {len(frames)} frames: {recon.failure_reason} "
            f"({recon.n_registered}/{recon.n_input} registered, {stages}) -> {out / 'run.json'}"
        )

    write_ply(out / "points.ply", recon.points, recon.colors)
    save_cameras(out / "cameras.json", recon)
    plotting.point_cloud_preview(
        out / "preview.png",
        recon.points,
        recon.colors,
        recon.camera_centers()[recon.registered],  # unregistered rows are NaN
        title=f"COLMAP on '{args.scene}' ({recon.n_registered}/{recon.n_input} registered)",
    )
    print(
        f"{recon.n_registered}/{recon.n_input} registered at {recon.image_hw}: "
        f"{stages}, total {recon.runtime_s:.1f}s, {len(recon.points):,} points -> {out}"
    )


if __name__ == "__main__":
    main()

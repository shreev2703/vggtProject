"""Make the benchmark figures from results/benchmark.csv and results/memory_profile.json.

Deterministic, CPU only. Every plotted number comes from those two files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from vggt_bench import plotting
from vggt_bench.eval import aggregate
from vggt_bench.provenance import run_metadata

FIGURES = {
    "auc30": "auc30_vs_views.png",
    "success": "success_vs_views.png",
    "runtime": "runtime_vs_views.png",
    "vram": "vggt_vram_vs_views.png",
    "scene_type": "auc30_by_scene_type.png",
}
AUC_YLIM = (0, 1.02)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=Path("results/benchmark.csv"))
    p.add_argument("--profile", type=Path, default=Path("results/memory_profile.json"))
    p.add_argument("--outdir", type=Path, default=Path("results/figures"))
    p.add_argument(
        "--scene-type-n", type=int, default=10, help="view count for the scene-type bars"
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    df = pd.read_csv(args.csv)
    if df.auc_30.isna().any():
        raise SystemExit("auc_30 has NaNs; failed runs must be recorded as 0, not dropped")
    profile = json.loads(args.profile.read_text())
    out = args.outdir

    plotting.band_lines(
        out / FIGURES["auc30"],
        aggregate.per_views(df, "auc_30"),
        "AUC@30 (failed runs = 0)",
        "Pose accuracy vs. number of views",
        ylim=AUC_YLIM,
        band_label="mean over scenes and samplings, band = 1 std",
    )

    plotting.band_lines(
        out / FIGURES["success"],
        aggregate.success_per_views(df),
        "success rate",
        "Usable reconstructions vs. number of views",
        ylim=AUC_YLIM,
        legend_loc="center right",
    )

    plotting.band_lines(
        out / FIGURES["runtime"],
        aggregate.per_views(df[df.success], "runtime_s", stat="median"),
        "runtime (s, log scale)",
        "Runtime vs. number of views",
        log_y=True,
        band_label="successful runs only, median, band = IQR; COLMAP is CPU-only SIFT, VGGT is GPU",
    )

    frames, peak, total = aggregate.vram_profile(profile)
    plotting.vram_vs_frames(
        out / FIGURES["vram"],
        frames,
        peak,
        total,
        aggregate.vggt_vram_median(df),
        "VGGT peak VRAM vs. number of frames",
    )

    plotting.scene_type_bars(
        out / FIGURES["scene_type"],
        aggregate.by_scene_type(df, args.scene_type_n),
        args.scene_type_n,
    )

    meta = run_metadata(
        {
            "csv": str(args.csv),
            "profile": str(args.profile),
            "rows": len(df),
            "scene_type_n": args.scene_type_n,
            "figures": list(FIGURES.values()),
        }
    )
    (out / "figures_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {len(FIGURES)} figures to {out}")


if __name__ == "__main__":
    main()

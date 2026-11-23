"""Aggregations of results/benchmark.csv behind the figures. Pure pandas, no matplotlib.

Failed runs carry AUC 0 in the CSV, so plain means already count them as zero.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd


def per_views(df: pd.DataFrame, col: str, stat: Literal["mean", "median"] = "mean") -> pd.DataFrame:
    """Centre and band of `col` for every (method, n_views).

    stat="mean": centre = mean, band = +-1 sample std, clipped to [0, 1] (for AUC-like columns).
    stat="median": centre = median, band = 25th to 75th percentile (unclipped).
    NaNs are dropped, so `n_runs` counts the runs that contributed; cells with none are omitted.
    """
    rows = []
    for (method, n_views), g in df.groupby(["method", "n_views"]):
        x = g[col].dropna().to_numpy(dtype=float)
        if len(x) == 0:  # nothing measured for this cell: leave a gap instead of crashing
            continue
        if stat == "mean":
            centre = float(x.mean())
            std = float(x.std(ddof=1)) if len(x) > 1 else 0.0
            lo, hi = max(0.0, centre - std), min(1.0, centre + std)
        else:
            centre = float(np.median(x))
            lo, hi = (float(v) for v in np.percentile(x, [25, 75]))
        rows.append(
            {
                "method": method,
                "n_views": int(n_views),
                "center": centre,
                "lo": lo,
                "hi": hi,
                "n_runs": len(x),
            }
        )
    return pd.DataFrame(rows)


def success_per_views(df: pd.DataFrame) -> pd.DataFrame:
    """Share of runs with a usable reconstruction, per (method, n_views), as `center`."""
    out = df.groupby(["method", "n_views"]).success.mean()
    return out.rename("center").reset_index()


def by_scene_type(df: pd.DataFrame, n_views: int) -> pd.DataFrame:
    """Mean AUC@30 and success rate per (scene_type, method) at one view count."""
    g = df[df.n_views == n_views].groupby(["scene_type", "method"])
    return g[["auc_30", "success"]].mean().reset_index()


def vggt_vram_median(df: pd.DataFrame) -> dict[int, float]:
    """Median peak VRAM (MB) logged by the benchmark's VGGT runs, per view count."""
    vggt = df[(df.method == "vggt") & df.peak_vram_mb.notna()]
    return {int(n): float(v) for n, v in vggt.groupby("n_views").peak_vram_mb.median().items()}


def vram_profile(profile: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    """(frames, peak allocated MB, total VRAM MB) of the bf16 profile, ok rows only."""
    prof = profile["profiles"]["bf16"]
    ok = sorted((r for r in prof["rows"] if r["ok"]), key=lambda r: r["n_frames"])
    frames = np.array([r["n_frames"] for r in ok])
    peak = np.array([r["peak_allocated_mb"] for r in ok], dtype=float)
    return frames, peak, float(prof["total_vram_mb"])

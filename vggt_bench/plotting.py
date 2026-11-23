"""Shared figure style so every plot in the README looks consistent."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open windows
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

METHOD_COLORS = {"vggt": "#1D9E75", "colmap": "#534AB7"}
NEUTRAL = "#5F5E5A"
LIMIT = "#A32D2D"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "legend.frameon": False,
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def depth_to_rgb(depth: np.ndarray, conf: np.ndarray | None = None) -> np.ndarray:
    """Colorize a depth map (near = bright). Low-confidence pixels are greyed out."""
    valid = np.isfinite(depth)
    lo, hi = np.percentile(depth[valid], [2, 98])
    norm = np.clip((depth - lo) / max(hi - lo, 1e-8), 0, 1)
    rgb = plt.get_cmap("turbo_r")(norm)[..., :3]
    if conf is not None:
        rgb[conf < np.percentile(conf, 25)] = 0.5
    return (rgb * 255).astype(np.uint8)


def point_cloud_preview(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    cam_centers: np.ndarray,
    title: str,
    max_points: int = 40_000,
    seed: int = 0,
) -> None:
    """Two views of a point cloud: oblique 3D and top-down (x vs z), with cameras.

    Coordinates are OpenCV world = first camera's frame: x right, y down, z forward.
    """
    apply_style()
    rng = np.random.default_rng(seed)
    if len(points) > max_points:
        idx = rng.choice(len(points), max_points, replace=False)
        points, colors = points[idx], colors[idx]
    c = colors / 255.0

    # Robust limits so a few far outliers don't shrink the scene to a dot.
    lo = np.percentile(points, 1, axis=0)
    hi = np.percentile(points, 99, axis=0)

    fig = plt.figure(figsize=(11, 5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    # Plot (x, z, -y) so "up" in the scene is up in the figure.
    ax3d.scatter(points[:, 0], points[:, 2], -points[:, 1], c=c, s=0.3, linewidths=0)
    ax3d.scatter(cam_centers[:, 0], cam_centers[:, 2], -cam_centers[:, 1], c=LIMIT, s=12)
    ax3d.set_xlim(lo[0], hi[0])
    ax3d.set_ylim(lo[2], hi[2])
    ax3d.set_zlim(-hi[1], -lo[1])
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("z (forward)")
    ax3d.set_zlabel("-y (up)")
    ax3d.view_init(elev=25, azim=-60)
    ax3d.set_title("Oblique view")

    ax2d = fig.add_subplot(1, 2, 2)
    ax2d.scatter(points[:, 0], points[:, 2], c=c, s=0.3, linewidths=0)
    ax2d.scatter(cam_centers[:, 0], cam_centers[:, 2], c=LIMIT, s=14, label="cameras")
    ax2d.set_xlim(min(lo[0], cam_centers[:, 0].min()), max(hi[0], cam_centers[:, 0].max()))
    ax2d.set_ylim(min(lo[2], cam_centers[:, 2].min()), max(hi[2], cam_centers[:, 2].max()))
    ax2d.set_aspect("equal")
    ax2d.set_xlabel("x")
    ax2d.set_ylabel("z (forward)")
    ax2d.set_title("Top-down view")
    ax2d.legend(loc="upper right")

    fig.suptitle(title)
    save(fig, path)


VIEW_TICKS = [3, 5, 10, 20, 40]
METHOD_LABELS = {"vggt": "VGGT (GPU)", "colmap": "COLMAP (CPU SIFT)"}


def _views_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(VIEW_TICKS, [str(n) for n in VIEW_TICKS])
    ax.minorticks_off()
    ax.set_xlabel("number of input views")


def band_lines(
    path: Path,
    stats,
    ylabel: str,
    title: str,
    ylim: tuple[float, float] | None = None,
    log_y: bool = False,
    band_label: str | None = None,
    legend_loc: str = "best",
) -> None:
    """One line per method over view count.

    `stats` has method, n_views, center and, for a band, lo and hi (aggregate.per_views).
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, g in stats.groupby("method"):
        g = g.sort_values("n_views")
        color = METHOD_COLORS.get(method, NEUTRAL)
        ax.plot(
            g.n_views, g.center, marker="o", color=color, label=METHOD_LABELS.get(method, method)
        )
        if "lo" in g:
            ax.fill_between(g.n_views, g.lo, g.hi, color=color, alpha=0.15, linewidth=0)
    _views_axis(ax)
    if log_y:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title if band_label is None else f"{title}\n({band_label})", fontsize=10)
    ax.legend(loc=legend_loc)
    save(fig, path)


def vram_vs_frames(
    path: Path, frames, peak_mb, total_mb: float, bench_median: dict[int, float], title: str
) -> None:
    """Calibrated VGGT peak memory curve with the benchmark's per-N medians overlaid."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    color = METHOD_COLORS["vggt"]
    ax.plot(frames, peak_mb / 1024, marker="o", ms=3, color=color, label="calibrated (Phase 1)")
    ns = sorted(bench_median)
    ax.scatter(
        ns,
        [bench_median[n] / 1024 for n in ns],
        color=NEUTRAL,
        zorder=3,
        s=22,
        label="benchmark median",
    )
    ax.axhline(total_mb / 1024, color=LIMIT, lw=1, label="total VRAM")
    ax.set_xlabel("input frames")
    ax.set_ylabel("peak VRAM (GB)")
    ax.set_title(title, fontsize=10)
    ax.legend()
    save(fig, path)


def scene_type_bars(path: Path, stats, n_views: int) -> None:
    """Grouped bars: mean AUC@30 per scene type and method, success rate printed above."""
    apply_style()
    types = sorted(stats.scene_type.unique())
    present = set(stats.method)
    methods = [m for m in METHOD_COLORS if m in present]
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, method in enumerate(methods):
        g = stats[stats.method == method].set_index("scene_type").reindex(types)
        x = np.arange(len(types)) + (i - (len(methods) - 1) / 2) * width
        bars = ax.bar(
            x,
            g.auc_30.fillna(0),
            width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS.get(method, method),
        )
        for rect, succ in zip(bars, g.success, strict=True):
            ax.annotate(
                "n/a" if pd.isna(succ) else f"{succ:.0%}",
                (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(np.arange(len(types)), types)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("mean AUC@30 (failures = 0)")
    ax.set_title(f"AUC@30 by scene type at {n_views} views (label = success rate)", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    save(fig, path)

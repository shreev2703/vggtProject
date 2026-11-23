"""Shared figure style so every plot in the README looks consistent."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open windows
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

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

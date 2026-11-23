"""Method-agnostic reconstruction result shared by VGGT and COLMAP."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Recon:
    """One reconstruction of one image set.

    Camera convention (OpenCV, same as VGGT and COLMAP): extrinsics map world
    points into camera coordinates, x_cam = R @ x_world + t, with x right,
    y down and z forward.
    """

    method: str
    image_paths: list[Path]
    image_hw: tuple[int, int] | None = None  # resolution the intrinsics refer to

    # Per input image; rows for unregistered images are NaN.
    extrinsics: np.ndarray | None = None  # (N, 3, 4) camera-from-world [R | t]
    intrinsics: np.ndarray | None = None  # (N, 3, 3)
    registered: np.ndarray | None = None  # (N,) bool

    points: np.ndarray | None = None  # (M, 3) world coordinates
    colors: np.ndarray | None = None  # (M, 3) uint8 RGB
    depth: np.ndarray | None = None  # (N, H, W), VGGT only
    depth_conf: np.ndarray | None = None  # (N, H, W), VGGT only

    runtime_s: float | None = None  # reconstruction only, model load excluded
    peak_vram_mb: float | None = None
    timings: dict[str, float] = field(default_factory=dict)  # per-stage breakdown
    extra: dict[str, Any] = field(default_factory=dict)

    failed: bool = False
    failure_reason: str | None = None

    @property
    def n_input(self) -> int:
        return len(self.image_paths)

    @property
    def n_registered(self) -> int:
        return 0 if self.registered is None else int(self.registered.sum())

    @property
    def registration_rate(self) -> float:
        """Fraction of input images that were registered (0.0 for an empty input)."""
        return self.n_registered / self.n_input if self.n_input else 0.0

    def camera_centers(self) -> np.ndarray:
        """World-space camera positions C = -R^T t, shape (N, 3)."""
        assert self.extrinsics is not None
        rot = self.extrinsics[:, :, :3]
        trans = self.extrinsics[:, :, 3]
        return -np.einsum("nji,nj->ni", rot, trans)

    @classmethod
    def failure(cls, method: str, image_paths: list[Path], reason: str, **kwargs: Any) -> Recon:
        n = len(image_paths)
        return cls(
            method=method,
            image_paths=image_paths,
            registered=np.zeros(n, dtype=bool),
            failed=True,
            failure_reason=reason,
            **kwargs,
        )

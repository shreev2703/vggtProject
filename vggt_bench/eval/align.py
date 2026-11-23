"""Similarity (Sim(3)) alignment of predicted cameras onto reference cameras.

VGGT and COLMAP both reconstruct in an arbitrary world frame with arbitrary
scale, so poses are only comparable after fitting one similarity transform.
The fit uses camera *centres* only: rotations are scale-invariant, so they
carry no information about the scale and only get rotated by the fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class AlignmentError(ValueError):
    """Too few usable cameras to fit a similarity transform."""


@dataclass
class Sim3:
    scale: float
    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """s * R @ p + t for row-major points of shape (..., 3)."""
        return self.scale * np.asarray(points) @ self.rotation.T + self.translation


def umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> Sim3:
    """Least-squares similarity mapping src onto dst, both (N, 3), N >= 3."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"src and dst must both be (N, 3); got {src.shape} and {dst.shape}")
    if len(src) < 3:
        raise AlignmentError(f"need at least 3 points to fit a Sim(3), got {len(src)}")

    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    var_s = (xs**2).sum() / len(src)
    if var_s <= 0:
        raise AlignmentError("source points are all identical")

    cov = xd.T @ xs / len(src)
    u, d, vt = np.linalg.svd(cov)
    # Flip the last axis when the fit would be a reflection. Collinear points
    # give a rank-1 covariance, where this is what keeps the result a rotation.
    s_fix = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_fix[2, 2] = -1.0
    rot = u @ s_fix @ vt
    scale = float(np.trace(np.diag(d) @ s_fix) / var_s)
    trans = mu_d - scale * rot @ mu_s
    return Sim3(scale=scale, rotation=rot, translation=trans)


def _centers(ext: np.ndarray) -> np.ndarray:
    return -np.einsum("nji,nj->ni", ext[:, :, :3], ext[:, :, 3])


def align_extrinsics(
    pred: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None
) -> tuple[Sim3, np.ndarray]:
    """Fit pred's camera centres onto ref's and return pred in the reference frame.

    pred, ref: (N, 3, 4) camera-from-world. mask selects the cameras to fit on
    (default: all finite rows). Rows outside the mask stay NaN in the output.
    """
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    use = np.isfinite(pred).all(axis=(1, 2))
    if mask is not None:
        use &= np.asarray(mask, dtype=bool)
    if int(use.sum()) < 3:
        raise AlignmentError(f"need at least 3 registered cameras to align, got {int(use.sum())}")

    fit = umeyama_sim3(_centers(pred[use]), _centers(ref[use]))

    out = np.full_like(pred, np.nan)
    new_rot = pred[use][:, :, :3] @ fit.rotation.T
    new_c = fit.apply(_centers(pred[use]))
    new_t = -np.einsum("nij,nj->ni", new_rot, new_c)
    out[use] = np.concatenate([new_rot, new_t[:, :, None]], axis=2)
    return fit, out

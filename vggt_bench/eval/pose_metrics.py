"""Relative pose error and AUC, the metrics the VGGT paper reports.

Errors are over camera *pairs*, so they do not depend on which camera is the
anchor. Rotation error is the geodesic angle between the predicted and
reference relative rotations. Translation error is the angle between baseline
directions, which makes it scale free. Pair error is max(RRE, RTE); AUC@tau is
the normalised area under the cumulative accuracy curve up to tau degrees.

Deviation from the VGGT code: it averages a 1-degree histogram's cumulative
sum; this integrates the exact step curve instead. The two agree to within the
histogram's bin width, and the exact form is easy to verify by hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np

from vggt_bench.eval.align import AlignmentError, align_extrinsics
from vggt_bench.io.datasets import Scene
from vggt_bench.recon import Recon

THRESHOLDS = (5.0, 10.0, 30.0)


def _centers(ext: np.ndarray) -> np.ndarray:
    return -np.einsum("nji,nj->ni", ext[:, :, :3], ext[:, :, 3])


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def relative_pose_errors(
    pred: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pair RRE and RTE in degrees over pairs where both cameras are usable.

    pred must already be expressed in the reference frame (see align_extrinsics),
    because the translation error compares baseline directions in world space.
    """
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    ok = np.isfinite(pred).all(axis=(1, 2)) & np.isfinite(ref).all(axis=(1, 2))
    if mask is not None:
        ok &= np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(ok)
    if len(idx) < 2:
        return np.empty(0), np.empty(0)

    i, j = (np.array(p) for p in zip(*combinations(idx, 2), strict=True))
    rp, rr = pred[:, :, :3], ref[:, :, :3]
    rel_p = rp[j] @ rp[i].transpose(0, 2, 1)
    rel_r = rr[j] @ rr[i].transpose(0, 2, 1)
    trace = np.trace(rel_p.transpose(0, 2, 1) @ rel_r, axis1=1, axis2=2)
    rre = np.degrees(np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0)))

    cp, cr = _centers(pred), _centers(ref)
    dp, dr = _unit(cp[j] - cp[i]), _unit(cr[j] - cr[i])
    rte = np.degrees(np.arccos(np.clip((dp * dr).sum(axis=1), -1.0, 1.0)))
    return rre, rte


def pose_auc(
    errors: np.ndarray, thresholds: Sequence[float] = THRESHOLDS
) -> dict[float, float]:
    """Normalised area under the accuracy-vs-threshold step curve, in [0, 1]."""
    errors = np.asarray(errors, dtype=np.float64)
    if len(errors) == 0:
        return {float(t): 0.0 for t in thresholds}
    srt = np.sort(errors)
    out = {}
    for t in thresholds:
        below = srt[srt < t]
        # recall is k/n on [below[k-1], below[k]) and 1.0-ish steps up at each error
        edges = np.r_[below, t]
        recall = np.arange(1, len(below) + 1) / len(srt)
        area = float((recall * np.diff(edges)).sum())
        out[float(t)] = float(np.clip(area / t, 0.0, 1.0))
    return out


def _zero(thresholds: Sequence[float], frac: float = 0.0) -> dict[str, float]:
    out = {f"auc_{int(t)}": 0.0 for t in thresholds}
    out.update(rre_median=float("nan"), rte_median=float("nan"), n_pairs=0, frac_registered=frac)
    return out


def evaluate_poses(
    recon: Recon, scene: Scene, thresholds: Sequence[float] = THRESHOLDS
) -> dict[str, float]:
    """Score a reconstruction's cameras against the scene's reference poses.

    Cameras are matched to the reference by image file name, so any sampled
    subset in any order is scored against the right poses. A failed recon, or
    one with fewer than 3 aligned cameras, scores AUC 0 rather than NaN.
    """
    frac = recon.n_registered / recon.n_input if recon.n_input else 0.0
    if recon.failed or recon.extrinsics is None or recon.registered is None:
        return _zero(thresholds, frac)

    row = {p.name: k for k, p in enumerate(scene.images)}
    missing = [p.name for p in recon.image_paths if p.name not in row]
    if missing:
        raise KeyError(f"images not in scene {scene.name!r}: {missing[:3]}")
    ref = scene.ref_extrinsics[[row[p.name] for p in recon.image_paths]]

    try:
        _, aligned = align_extrinsics(recon.extrinsics, ref, recon.registered)
    except AlignmentError:
        return _zero(thresholds, frac)

    rre, rte = relative_pose_errors(aligned, ref, recon.registered)
    if len(rre) == 0:
        return _zero(thresholds, frac)
    aucs = pose_auc(np.maximum(rre, rte), thresholds)
    out = {f"auc_{int(t)}": aucs[float(t)] for t in thresholds}
    out.update(
        rre_median=float(np.median(rre)),
        rte_median=float(np.median(rte)),
        n_pairs=len(rre),
        frac_registered=frac,
    )
    return out

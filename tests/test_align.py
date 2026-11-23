"""Sim(3) alignment of predicted cameras onto reference cameras."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vggt_bench.eval.align import AlignmentError, Sim3, align_extrinsics, umeyama_sim3

R_KNOWN = Rotation.from_euler("xyz", [20, -35, 50], degrees=True).as_matrix()
T_KNOWN = np.array([1.0, -2.0, 3.0])


def _cameras(n: int = 6, seed: int = 0) -> np.ndarray:
    """n random camera-from-world [R|t] poses."""
    rng = np.random.default_rng(seed)
    rots = Rotation.random(n, random_state=seed).as_matrix()
    trans = rng.normal(size=(n, 3))
    return np.concatenate([rots, trans[:, :, None]], axis=2)


def _centers(ext: np.ndarray) -> np.ndarray:
    return -np.einsum("nji,nj->ni", ext[:, :, :3], ext[:, :, 3])


def _push(ext: np.ndarray, sim: Sim3) -> np.ndarray:
    """Express cameras in a world that is `sim` applied to the original one."""
    rot = ext[:, :, :3]
    new_rot = rot @ sim.rotation.T
    new_c = sim.apply(_centers(ext))
    new_t = -np.einsum("nij,nj->ni", new_rot, new_c)
    return np.concatenate([new_rot, new_t[:, :, None]], axis=2)


def test_umeyama_recovers_a_known_similarity() -> None:
    src = np.random.default_rng(1).normal(size=(8, 3))
    dst = 2.5 * src @ R_KNOWN.T + T_KNOWN

    fit = umeyama_sim3(src, dst)

    assert fit.scale == pytest.approx(2.5, abs=1e-9)
    np.testing.assert_allclose(fit.rotation, R_KNOWN, atol=1e-9)
    np.testing.assert_allclose(fit.translation, T_KNOWN, atol=1e-9)


def test_umeyama_is_exact_for_the_identity_transform() -> None:
    pts = np.random.default_rng(2).normal(size=(5, 3))

    fit = umeyama_sim3(pts, pts)

    assert fit.scale == pytest.approx(1.0)
    np.testing.assert_allclose(fit.rotation, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(fit.translation, 0.0, atol=1e-9)


def test_umeyama_handles_collinear_points_without_reflecting() -> None:
    line = np.array([[0.0, 0, 0], [1, 0, 0], [3, 0, 0]])
    fit = umeyama_sim3(line, line @ R_KNOWN.T)

    assert np.linalg.det(fit.rotation) == pytest.approx(1.0)
    np.testing.assert_allclose(fit.apply(line), line @ R_KNOWN.T, atol=1e-9)


def test_align_extrinsics_maps_transformed_cameras_back_onto_the_reference() -> None:
    ref = _cameras()
    pred = _push(ref, Sim3(scale=3.0, rotation=R_KNOWN, translation=T_KNOWN))
    # pred lives in a different world with a different scale; fold it into the
    # camera centres so only the similarity is unknown
    _, aligned = align_extrinsics(pred, ref)

    np.testing.assert_allclose(_centers(aligned), _centers(ref), atol=1e-6)
    np.testing.assert_allclose(aligned[:, :, :3], ref[:, :, :3], atol=1e-6)


def test_align_extrinsics_uses_only_masked_cameras() -> None:
    ref = _cameras()
    pred = _push(ref, Sim3(scale=2.0, rotation=R_KNOWN, translation=T_KNOWN))
    pred[2] = np.nan
    mask = np.array([True, True, False, True, True, True])

    _, aligned = align_extrinsics(pred, ref, mask)

    assert np.isnan(aligned[2]).all()
    keep = mask
    np.testing.assert_allclose(_centers(aligned[keep]), _centers(ref[keep]), atol=1e-6)


def test_align_extrinsics_raises_below_three_cameras() -> None:
    ref = _cameras()
    with pytest.raises(AlignmentError):
        align_extrinsics(ref[:2], ref[:2])


def test_align_extrinsics_raises_when_the_mask_leaves_under_three() -> None:
    ref = _cameras()
    mask = np.array([True, True, False, False, False, False])
    with pytest.raises(AlignmentError):
        align_extrinsics(ref, ref, mask)

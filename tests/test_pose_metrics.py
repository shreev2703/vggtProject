"""Relative pose error and AUC, including the partial-registration edge cases."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vggt_bench.eval.align import align_extrinsics
from vggt_bench.eval.pose_metrics import evaluate_poses, pose_auc, relative_pose_errors
from vggt_bench.io.datasets import Scene
from vggt_bench.recon import Recon


def _ext(rots: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Camera-from-world [R|t] with the given rotations and world-space centres."""
    t = -np.einsum("nij,nj->ni", rots, centers)
    return np.concatenate([rots, t[:, :, None]], axis=2)


def _ring(n: int = 6, seed: int = 0) -> np.ndarray:
    rots = Rotation.random(n, random_state=seed).as_matrix()
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    centers = np.stack([np.cos(ang), np.sin(ang), 0.2 * np.sin(3 * ang)], axis=1)
    return _ext(rots, centers)


def _scene(ref: np.ndarray) -> Scene:
    paths = [Path(f"{i:03d}.png") for i in range(len(ref))]
    k = np.zeros((len(ref), 3, 3))
    return Scene("mip360", "s", "easy", Path("."), paths, ref, k, (1, 1), 0)


def _recon(pred: np.ndarray, paths: list[Path], registered: np.ndarray | None = None) -> Recon:
    reg = np.ones(len(pred), bool) if registered is None else registered
    return Recon(method="x", image_paths=paths, extrinsics=pred, registered=reg)


def test_errors_are_zero_for_identical_poses() -> None:
    ref = _ring()
    rre, rte = relative_pose_errors(ref, ref)
    assert len(rre) == len(rte) == 15
    np.testing.assert_allclose(rre, 0.0, atol=1e-4)
    np.testing.assert_allclose(rte, 0.0, atol=1e-4)


def test_relative_rotation_error_on_a_known_yaw() -> None:
    centers = np.array([[0.0, 0, 0], [1, 0, 0]])
    ref = _ext(np.stack([np.eye(3)] * 2), centers)
    yaw = Rotation.from_euler("y", 10, degrees=True).as_matrix()
    pred = _ext(np.stack([np.eye(3), yaw]), centers)

    rre, rte = relative_pose_errors(pred, ref)

    assert rre == pytest.approx([10.0])
    assert rte == pytest.approx([0.0], abs=1e-6)


def test_relative_translation_error_is_an_angle_between_baselines() -> None:
    eye = np.stack([np.eye(3)] * 2)
    ref = _ext(eye, np.array([[0.0, 0, 0], [1, 0, 0]]))
    rotated = Rotation.from_euler("z", 30, degrees=True).apply([1.0, 0, 0])
    pred = _ext(eye, np.array([[0.0, 0, 0], rotated]))

    _, rte = relative_pose_errors(pred, ref)

    assert rte == pytest.approx([30.0])


def test_errors_are_scale_invariant() -> None:
    ref = _ring()
    pred = ref.copy()
    pred[:, :, :3] = pred[:, :, :3] @ Rotation.from_euler("z", 4, degrees=True).as_matrix().T
    scaled_ref = ref.copy()
    scaled_ref[:, :, 3] *= 7.0  # same rotations, centres scaled 7x

    out = []
    for r in (ref, scaled_ref):
        _, aligned = align_extrinsics(pred, r)
        out.append(relative_pose_errors(aligned, r))

    np.testing.assert_allclose(out[0][0], out[1][0], atol=1e-4)
    np.testing.assert_allclose(out[0][1], out[1][1], atol=1e-4)


def test_pairs_with_an_unregistered_camera_are_skipped() -> None:
    ref = _ring(4)
    mask = np.array([True, True, False, True])
    rre, rte = relative_pose_errors(ref, ref, mask)
    assert len(rre) == len(rte) == 3


def test_pose_auc_is_one_when_every_error_is_zero() -> None:
    assert pose_auc(np.zeros(5)) == {5.0: 1.0, 10.0: 1.0, 30.0: 1.0}


def test_pose_auc_is_zero_when_every_error_exceeds_the_threshold() -> None:
    assert pose_auc(np.full(4, 90.0)) == {5.0: 0.0, 10.0: 0.0, 30.0: 0.0}


def test_pose_auc_matches_a_hand_computed_case() -> None:
    # recall is 0.5 on [0, 15) and 1.0 on [15, 30]: (0.5*15 + 1.0*15) / 30
    assert pose_auc(np.array([0.0, 15.0]))[30.0] == pytest.approx(0.75)


def test_pose_auc_of_no_errors_is_zero() -> None:
    assert pose_auc(np.array([])) == {5.0: 0.0, 10.0: 0.0, 30.0: 0.0}


def test_evaluate_poses_scores_a_perfect_recon_as_one() -> None:
    ref = _ring()
    scene = _scene(ref)
    m = evaluate_poses(_recon(ref, scene.images), scene)
    assert m["auc_30"] == pytest.approx(1.0)
    assert m["n_pairs"] == 15
    assert m["frac_registered"] == 1.0


def test_evaluate_poses_matches_a_subset_by_image_name() -> None:
    ref = _ring()
    scene = _scene(ref)
    order = [4, 1, 3]  # a sampled subset, in a different order from the scene
    recon = _recon(ref[order], [scene.images[i] for i in order])
    assert evaluate_poses(recon, scene)["auc_30"] == pytest.approx(1.0)


def test_evaluate_poses_scores_a_failed_recon_as_zero() -> None:
    scene = _scene(_ring())
    m = evaluate_poses(Recon.failure("x", scene.images, "boom"), scene)
    assert m["auc_30"] == 0.0
    assert m["n_pairs"] == 0
    assert m["frac_registered"] == 0.0


def test_evaluate_poses_scores_a_two_camera_recon_as_zero() -> None:
    ref = _ring()
    scene = _scene(ref)
    reg = np.array([True, True, False, False, False, False])
    pred = ref.copy()
    pred[~reg] = np.nan
    m = evaluate_poses(_recon(pred, scene.images, reg), scene)
    assert m["auc_30"] == 0.0
    assert m["n_pairs"] == 0
    assert m["frac_registered"] == pytest.approx(2 / 6)


def test_pairs_with_coincident_cameras_are_skipped() -> None:
    eye = np.stack([np.eye(3)] * 3)
    centers = np.array([[0.0, 0, 0], [0.0, 0, 0], [1, 0, 0]])
    ref = _ext(eye, centers)
    rre, rte = relative_pose_errors(ref, ref)
    assert len(rre) == len(rte) == 2

"""Scene loading: reference poses must follow the images, never their position."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap
import pytest
from PIL import Image

from vggt_bench.io.datasets import load_scene

FULL_W, FULL_H = 400, 200
SMALL_W, SMALL_H = 100, 50  # images_4 copy: 4x smaller


def _rot_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def _build(tmp_path: Path, names: list[str], extra_images: list[str] | None = None):
    """A scene dir with a sparse/0 model for `names` and images_4 files for names+extra."""
    root = tmp_path / "mip360"
    scene = root / "garden"
    sparse = scene / "sparse" / "0"
    sparse.mkdir(parents=True)
    rec = pycolmap.Reconstruction()
    cam = pycolmap.Camera.create_from_model_name(1, "PINHOLE", 300.0, FULL_W, FULL_H)
    rec.add_camera_with_trivial_rig(cam)
    for i, name in enumerate(names):
        quat_xyzw = pycolmap.Rotation3d(_rot_z(10.0 * (i + 1))).quat
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(quat_xyzw), np.array([float(i), 0.0, 1.0]))
        img = pycolmap.Image(name=name, camera_id=1, image_id=i + 1)
        rec.add_image_with_trivial_frame(img, pose)
    rec.write(str(sparse))
    small = scene / "images_4"
    small.mkdir()
    for name in [*names, *(extra_images or [])]:
        Image.new("RGB", (SMALL_W, SMALL_H)).save(small / name)
    cfg = {
        "root": str(root),
        "images_subdir": "images_4",
        "sparse_subdir": "sparse/0",
        "scenes": {"garden": {"scene_type": "easy"}},
    }
    return cfg


def test_load_scene_reads_poses_in_image_order(tmp_path: Path) -> None:
    names = ["b.png", "a.png", "c.png"]  # registered out of filename order
    scene = load_scene("garden", _build(tmp_path, names))

    assert [p.name for p in scene.images] == ["a.png", "b.png", "c.png"]
    # a.png was the 2nd registered image: rotation 20 deg, translation (1, 0, 1)
    np.testing.assert_allclose(scene.ref_extrinsics[0][:, :3], _rot_z(20.0), atol=1e-6)
    np.testing.assert_allclose(scene.ref_extrinsics[0][:, 3], [1.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(scene.ref_extrinsics[1][:, 3], [0.0, 0.0, 1.0], atol=1e-6)


def test_load_scene_scales_intrinsics_to_the_images_4_copy(tmp_path: Path) -> None:
    scene = load_scene("garden", _build(tmp_path, ["a.png", "b.png"]))
    ratio = SMALL_W / FULL_W

    K = scene.ref_intrinsics[0]
    assert K[0, 0] == pytest.approx(300.0 * ratio)
    assert K[1, 1] == pytest.approx(300.0 * ratio)
    assert K[0, 2] == pytest.approx(FULL_W / 2 * ratio)
    assert K[1, 2] == pytest.approx(FULL_H / 2 * ratio)
    assert scene.image_hw == (SMALL_H, SMALL_W)
    assert scene.scene_type == "easy"


def test_load_scene_drops_images_missing_from_the_reference(tmp_path: Path) -> None:
    cfg = _build(tmp_path, ["a.png", "b.png"], extra_images=["z.png"])
    scene = load_scene("garden", cfg)

    assert [p.name for p in scene.images] == ["a.png", "b.png"]
    assert scene.n_dropped == 1
    assert scene.ref_extrinsics.shape == (2, 3, 4)


def test_load_scene_rejects_an_unknown_scene_name(tmp_path: Path) -> None:
    cfg = _build(tmp_path, ["a.png"])
    with pytest.raises(KeyError, match="nope"):
        load_scene("nope", cfg)

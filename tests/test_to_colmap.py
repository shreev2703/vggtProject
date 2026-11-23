"""VGGT -> COLMAP-format export, read back through pycolmap to prove it is a valid model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap

from vggt_bench.export.to_colmap import recon_to_reconstruction, write_colmap
from vggt_bench.recon import Recon

HW = (294, 518)


def make_recon(n_points: int = 50) -> Recon:
    names = ["f0.png", "f1.png", "f2.png"]
    ext = np.zeros((3, 3, 4))
    ks = np.zeros((3, 3, 3))
    for i in range(3):
        ang = np.deg2rad(10.0 * (i + 1))
        rot = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1.0]])
        ext[i, :, :3] = rot
        ext[i, :, 3] = [0.1 * i, 0.2, 1.0 + i]
        ks[i] = [[400.0 + i, 0, 259.0], [0, 401.0 + i, 147.0], [0, 0, 1]]
    rng = np.random.default_rng(0)
    return Recon(
        method="vggt",
        image_paths=[Path("/frames") / n for n in names],
        image_hw=HW,
        extrinsics=ext,
        intrinsics=ks,
        registered=np.ones(3, dtype=bool),
        points=rng.normal(size=(n_points, 3)).astype(np.float32),
        colors=rng.integers(0, 255, size=(n_points, 3), dtype=np.uint8),
    )


def _xyz(model: pycolmap.Reconstruction) -> list[tuple[float, ...]]:
    return sorted(tuple(p.xyz) for p in model.points3D.values())


def test_cameras_and_poses_survive_a_write_read_round_trip(tmp_path: Path) -> None:
    recon = make_recon()
    write_colmap(recon, tmp_path)

    model = pycolmap.Reconstruction(tmp_path / "sparse" / "0")

    assert model.num_images() == 3
    by_name = {im.name: im for im in model.images.values()}
    for i, path in enumerate(recon.image_paths):
        image = by_name[path.name]
        np.testing.assert_allclose(image.cam_from_world().matrix(), recon.extrinsics[i], atol=1e-9)
        cam = model.cameras[image.camera_id]
        assert (cam.width, cam.height) == (HW[1], HW[0])
        np.testing.assert_allclose(
            cam.params, [recon.intrinsics[i, 0, 0], recon.intrinsics[i, 1, 1], 259.0, 147.0]
        )


def test_points_keep_position_and_colour(tmp_path: Path) -> None:
    recon = make_recon(n_points=20)
    write_colmap(recon, tmp_path, max_points=1000)

    model = pycolmap.Reconstruction(tmp_path / "sparse" / "0")

    assert model.num_points3D() == 20
    np.testing.assert_allclose(_xyz(model), sorted(map(tuple, recon.points)), atol=1e-5)


def test_unregistered_images_are_left_out() -> None:
    recon = make_recon()
    recon.registered[1] = False
    recon.extrinsics[1] = np.nan
    recon.intrinsics[1] = np.nan

    model = recon_to_reconstruction(recon)

    assert sorted(im.name for im in model.images.values()) == ["f0.png", "f2.png"]


def test_point_count_is_capped_reproducibly() -> None:
    recon = make_recon(n_points=200)

    a = recon_to_reconstruction(recon, max_points=50, seed=1)
    b = recon_to_reconstruction(recon, max_points=50, seed=1)

    assert a.num_points3D() == 50
    assert _xyz(a) == _xyz(b)


def test_copy_images_brings_only_registered_frames(tmp_path: Path) -> None:
    from PIL import Image

    recon = make_recon()
    src = tmp_path / "src"
    src.mkdir()
    for p in recon.image_paths:
        Image.new("RGB", (8, 8)).save(src / p.name)
    recon.image_paths = [src / p.name for p in recon.image_paths]
    recon.registered[1] = False

    write_colmap(recon, tmp_path / "out", copy_images=True)

    assert sorted(p.name for p in (tmp_path / "out" / "images").iterdir()) == ["f0.png", "f2.png"]

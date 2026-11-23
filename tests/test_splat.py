"""Splat training helpers: scene loading, view split, initialisation, SSIM, short training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from test_to_colmap import HW, make_recon

from vggt_bench.export.splat import (
    SplatConfig,
    export_splat,
    init_gaussians,
    load_colmap_scene,
    split_views,
    ssim,
    train_splat,
)
from vggt_bench.export.to_colmap import write_colmap

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat needs CUDA")


def _write_scene(tmp_path: Path, n_points: int = 50) -> tuple[Path, object]:
    recon = make_recon(n_points=n_points)
    src = tmp_path / "src"
    src.mkdir()
    ramp = np.linspace(60, 200, HW[1], dtype=np.uint8)[None, :, None]  # smooth, so learnable
    for p in recon.image_paths:
        Image.fromarray(np.broadcast_to(ramp, (HW[0], HW[1], 3)).copy()).save(src / p.name)
    recon.image_paths = [src / p.name for p in recon.image_paths]
    write_colmap(recon, tmp_path / "scene", copy_images=True)
    return tmp_path / "scene", recon


def test_scene_round_trips_poses_intrinsics_and_images(tmp_path: Path) -> None:
    scene_dir, recon = _write_scene(tmp_path)

    scene = load_colmap_scene(scene_dir)

    assert scene.images.shape == (3, HW[0], HW[1], 3)
    assert scene.viewmats.shape == (3, 4, 4)
    for i in range(3):  # make_recon names are f0, f1, f2 -> sorted order matches
        np.testing.assert_allclose(scene.viewmats[i, :3].numpy(), recon.extrinsics[i], atol=1e-6)
        np.testing.assert_allclose(scene.viewmats[i, 3].numpy(), [0, 0, 0, 1])
        np.testing.assert_allclose(scene.ks[i].numpy(), recon.intrinsics[i], atol=1e-4)
    assert scene.points.shape == (50, 3)
    assert scene.colors.shape == (50, 3)
    assert scene.colors.max() <= 1.0


def test_split_holds_out_every_nth_view_without_overlap() -> None:
    train, held = split_views(20, every=8)

    assert held == [0, 8, 16]
    assert sorted(train + held) == list(range(20))


def test_split_always_keeps_a_training_view() -> None:
    train, held = split_views(3, every=8)

    assert len(train) >= 1
    assert not set(train) & set(held)


def test_init_scale_tracks_point_spacing() -> None:
    grid = torch.stack(torch.meshgrid(*[torch.arange(4.0)] * 3, indexing="ij"), -1).reshape(-1, 3)
    colors = torch.full((len(grid), 3), 0.5)

    a = init_gaussians(grid, colors, sh_degree=3)
    b = init_gaussians(grid * 2, colors, sh_degree=3)

    assert a["means"].shape == (64, 3)
    assert a["sh0"].shape == (64, 1, 3) and a["shN"].shape == (64, 15, 3)
    assert a["quats"].shape == (64, 4) and a["opacities"].shape == (64,)
    assert all(torch.isfinite(v).all() for v in a.values())
    torch.testing.assert_close(b["scales"] - a["scales"], torch.full((64, 3), np.log(2.0)))


def test_ssim_is_one_for_identical_images_and_lower_for_noise() -> None:
    torch.manual_seed(0)
    x = torch.rand(1, 32, 32, 3)

    assert ssim(x, x).item() == pytest.approx(1.0, abs=1e-5)
    assert ssim(x, torch.rand(1, 32, 32, 3)).item() < 0.5


@needs_cuda
def test_short_training_lowers_loss_and_exports_files(tmp_path: Path) -> None:
    scene_dir, _ = _write_scene(tmp_path, n_points=200)
    scene = load_colmap_scene(scene_dir)
    cfg = SplatConfig(steps=60, cap_max=2000, sh_degree=1, holdout_every=3)

    params, stats = train_splat(scene, cfg, seed=0)

    losses = stats["losses"]
    assert np.mean(losses[-10:]) < np.mean(losses[:10])
    assert stats["n_gaussians"] <= cfg.cap_max
    assert np.isfinite(stats["holdout_psnr"])
    files = export_splat(params, tmp_path / "out")
    assert files["splat"].stat().st_size > 0
    assert files["ply_compressed"].stat().st_size > 0

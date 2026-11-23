"""Write a VGGT Recon as a COLMAP sparse model (cameras.bin / images.bin / points3D.bin).

VGGT's own helper (third_party/vggt/vggt/dependency/np_to_pycolmap.py) targets an older pycolmap
and fails on 4.x, so this uses the 4.x rig/frame API. The output is what gsplat and NeRF
trainers read: one PINHOLE camera per registered image, plus the point cloud as Points3D
with empty tracks (VGGT predicts no 2D feature tracks).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap

from vggt_bench.recon import Recon


def recon_to_reconstruction(
    recon: Recon, max_points: int = 100_000, seed: int = 0
) -> pycolmap.Reconstruction:
    """Build an in-memory pycolmap model from the registered images and points of `recon`."""
    assert recon.extrinsics is not None and recon.intrinsics is not None
    assert recon.registered is not None and recon.image_hw is not None
    height, width = recon.image_hw
    model = pycolmap.Reconstruction()

    next_id = 1
    for i, path in enumerate(recon.image_paths):
        if not recon.registered[i]:
            continue
        k = recon.intrinsics[i]
        camera = pycolmap.Camera(
            model="PINHOLE",
            width=width,
            height=height,
            params=[k[0, 0], k[1, 1], k[0, 2], k[1, 2]],
            camera_id=next_id,
        )
        model.add_camera_with_trivial_rig(camera)
        pose = pycolmap.Rigid3d(
            pycolmap.Rotation3d(np.asarray(recon.extrinsics[i, :, :3], dtype=np.float64)),
            np.asarray(recon.extrinsics[i, :, 3], dtype=np.float64),
        )
        image = pycolmap.Image(name=path.name, camera_id=next_id, image_id=next_id)
        model.add_image_with_trivial_frame(image, pose)
        next_id += 1

    if recon.points is not None and len(recon.points):
        points, colors = recon.points, recon.colors
        if len(points) > max_points:
            keep = np.sort(np.random.default_rng(seed).choice(len(points), max_points, False))
            points = points[keep]
            colors = None if colors is None else colors[keep]
        if colors is None:
            colors = np.full((len(points), 3), 128, dtype=np.uint8)
        for xyz, rgb in zip(points, colors, strict=True):
            model.add_point3D(xyz.astype(np.float64), pycolmap.Track(), rgb.astype(np.uint8))
    return model


def write_colmap(recon: Recon, out_dir: Path, max_points: int = 100_000, seed: int = 0) -> Path:
    """Write out_dir/sparse/0/ in binary and text form; return the sparse/0 path."""
    sparse = out_dir / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    model = recon_to_reconstruction(recon, max_points=max_points, seed=seed)
    model.write_binary(str(sparse))
    model.write_text(str(sparse))
    return sparse

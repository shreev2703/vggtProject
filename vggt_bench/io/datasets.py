"""Benchmark scenes: images plus reference camera poses.

Mip-NeRF 360 ships a COLMAP model per scene (``sparse/0``) computed over *all*
images. It is pseudo-ground-truth, not a measured one, and every result built on
it must say so. Poses are matched to images by file name, never by position, so
a file missing from the model cannot shift every later pose by one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
import yaml
from PIL import Image

from vggt_bench.io.frames import list_images

log = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/datasets.yaml")


@dataclass
class Scene:
    dataset: str
    name: str
    scene_type: str
    image_dir: Path
    images: list[Path]
    ref_extrinsics: np.ndarray  # (N, 3, 4) OpenCV camera-from-world
    ref_intrinsics: np.ndarray  # (N, 3, 3) at the resolution of `images`
    image_hw: tuple[int, int]
    n_dropped: int


def load_dataset_config(path: Path = DEFAULT_CONFIG, dataset: str = "mip360") -> dict:
    """One dataset's block of configs/datasets.yaml, tagged with its name."""
    with open(path) as f:
        cfg = yaml.safe_load(f)[dataset]
    cfg["dataset"] = dataset
    return cfg


def load_scene(name: str, cfg: dict) -> Scene:
    scenes = cfg["scenes"]
    if name not in scenes:
        raise KeyError(f"unknown scene {name!r}; configured: {sorted(scenes)}")

    scene_dir = Path(cfg["root"]) / name
    image_dir = scene_dir / cfg["images_subdir"]
    rec = pycolmap.Reconstruction(str(scene_dir / cfg["sparse_subdir"]))
    by_name = {img.name: img for img in rec.images.values()}

    on_disk = list_images(scene_dir, cfg["images_subdir"])
    kept = [p for p in on_disk if p.name in by_name]
    n_dropped = len(on_disk) - len(kept)
    if n_dropped:
        log.warning("%s: dropped %d image(s) absent from the reference model", name, n_dropped)
    if not kept:
        raise ValueError(f"{name}: no image in {image_dir} matches the reference model")

    with Image.open(kept[0]) as im:
        width, height = im.size

    extrinsics = np.empty((len(kept), 3, 4))
    intrinsics = np.empty((len(kept), 3, 3))
    for i, path in enumerate(kept):
        image = by_name[path.name]
        camera = image.camera
        # Scale by the measured width ratio rather than assuming exactly 4.
        scale = width / camera.width
        K = np.asarray(camera.calibration_matrix(), dtype=float).copy()
        K[:2] *= scale
        extrinsics[i] = np.asarray(image.cam_from_world().matrix())
        intrinsics[i] = K

    return Scene(
        dataset=cfg.get("dataset", "mip360"),
        name=name,
        scene_type=scenes[name]["scene_type"],
        image_dir=image_dir,
        images=kept,
        ref_extrinsics=extrinsics,
        ref_intrinsics=intrinsics,
        image_hw=(height, width),
        n_dropped=n_dropped,
    )

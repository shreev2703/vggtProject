"""COLMAP wrapped behind the ReconMethod interface.

Uses pycolmap 4.2, whose options API differs from older releases: SIFT settings
live under ``FeatureExtractionOptions.sift``, pair selection is its own
``*PairingOptions`` object, the camera model is set through
``ImageReaderOptions``, and image poses are owned by frames rather than being
settable on ``Image``.

The pip wheel is built without CUDA (``pycolmap.has_cuda`` is False) and GPU
SIFT would additionally need OpenGL, which WSL does not provide, so feature
extraction, matching and bundle adjustment all run on the CPU. That is recorded
in ``config()`` because it changes how the runtime comparison against VGGT
should be read.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap
import yaml

from vggt_bench.recon import Recon

DEFAULT_CONFIG = Path("configs/colmap.yaml")
MIN_REGISTERED = 3  # spec section 6.2: fewer than this counts as a failure

# The pip pycolmap wheel has no CUDA and GPU SIFT would need OpenGL, which WSL
# does not provide. Declared once so the options and the recorded provenance
# cannot disagree about what actually ran.
DEVICE = pycolmap.Device.cpu
USE_GPU = DEVICE != pycolmap.Device.cpu


@dataclass
class ColmapConfig:
    """COLMAP settings, mirroring the keys in ``configs/colmap.yaml``."""

    camera_model: str = "SIMPLE_RADIAL"
    camera_mode: str = "SINGLE"
    matcher: str = "exhaustive"
    exhaustive_max_images: int = 50
    sequential_overlap: int = 10
    sequential_quadratic_overlap: bool = True
    max_image_size: int = 1024
    max_num_features: int = 8192
    min_model_size: int = 3
    min_num_matches: int = 15
    multiple_models: bool = True
    max_num_models: int = 50
    extract_colors: bool = True
    random_seed: int = 0
    num_threads: int = -1

    def __post_init__(self) -> None:
        if self.matcher not in ("exhaustive", "sequential"):
            raise ValueError(f"matcher must be 'exhaustive' or 'sequential', got {self.matcher!r}")
        if self.camera_mode not in pycolmap.CameraMode.__members__:
            valid = sorted(pycolmap.CameraMode.__members__)
            raise ValueError(f"camera_mode must be one of {valid}, got {self.camera_mode!r}")
        if self.min_model_size < MIN_REGISTERED:
            # Otherwise COLMAP would happily keep a model this code then marks
            # failed, and the two thresholds for one spec rule drift apart.
            raise ValueError(
                f"min_model_size must be >= {MIN_REGISTERED} (the registration floor "
                f"from spec section 6.2), got {self.min_model_size}"
            )

    @property
    def camera_mode_enum(self) -> pycolmap.CameraMode:
        return pycolmap.CameraMode.__members__[self.camera_mode]

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> ColmapConfig:
        """Load settings from YAML.

        A path given explicitly must exist: silently falling back to defaults
        would let a typo'd --config report success while running other settings.
        Only the shipped default is allowed to be absent.
        """
        explicit = path is not None
        path = Path(path) if explicit else DEFAULT_CONFIG
        if not path.exists():
            if explicit:
                raise FileNotFoundError(f"COLMAP config not found: {path}")
            return cls()
        values = yaml.safe_load(path.read_text()) or {}
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")
        return cls(**values)


class ColmapMethod:
    name = "colmap"

    def __init__(
        self,
        config: ColmapConfig | None = None,
        workspace: Path | None = None,
    ) -> None:
        """
        Args:
            config: COLMAP settings; defaults to ``configs/colmap.yaml`` if that
                file exists, otherwise to ``ColmapConfig()``.
            workspace: keep the COLMAP database and sparse model here instead of
                in a temporary directory that is deleted afterwards. Useful for
                inspecting a run in the COLMAP GUI and as gsplat input.
        """
        self.cfg = config if config is not None else ColmapConfig.from_yaml()
        self.workspace = Path(workspace) if workspace is not None else None

    def config(self, matcher_used: str | None = None) -> dict[str, Any]:
        """Settings plus the device provenance the README has to state.

        DEVICE is a single recorded fact rather than one key per stage: it feeds
        the option objects too, so the provenance cannot disagree with what ran.
        """
        return {
            **asdict(self.cfg),
            "matcher_used": matcher_used,
            "pycolmap_version": pycolmap.__version__,
            "pycolmap_has_cuda": bool(pycolmap.has_cuda),
            "device": DEVICE.name,
        }

    # -- pycolmap option objects ------------------------------------------------

    def _reader_options(self) -> pycolmap.ImageReaderOptions:
        return pycolmap.ImageReaderOptions(camera_model=self.cfg.camera_model)

    def _extraction_options(self) -> pycolmap.FeatureExtractionOptions:
        opts = pycolmap.FeatureExtractionOptions(
            max_image_size=self.cfg.max_image_size,
            num_threads=self.cfg.num_threads,
            use_gpu=USE_GPU,
        )
        opts.sift.max_num_features = self.cfg.max_num_features
        return opts

    def _matching_options(self) -> pycolmap.FeatureMatchingOptions:
        return pycolmap.FeatureMatchingOptions(
            num_threads=self.cfg.num_threads,
            use_gpu=USE_GPU,
        )

    def _mapper_options(self) -> pycolmap.IncrementalPipelineOptions:
        cfg = self.cfg
        return pycolmap.IncrementalPipelineOptions(
            min_model_size=cfg.min_model_size,
            min_num_matches=cfg.min_num_matches,
            multiple_models=cfg.multiple_models,
            max_num_models=cfg.max_num_models,
            extract_colors=cfg.extract_colors,
            random_seed=cfg.random_seed,
            num_threads=cfg.num_threads,
        )

    def _choose_matcher(self, n_input: int) -> str:
        """Exhaustive below the image-count ceiling, sequential above it."""
        cfg = self.cfg
        if cfg.matcher == "exhaustive" and n_input > cfg.exhaustive_max_images:
            return "sequential"
        return cfg.matcher

    # -- pipeline ---------------------------------------------------------------

    def reconstruct(self, image_paths: list[Path]) -> Recon:
        image_paths = [Path(p) for p in image_paths]
        image_dir, image_names = _split_image_paths(image_paths)
        # A local, not instance state: one method instance is reused across a
        # whole Phase 3 sweep, so nothing about a run may outlive it.
        matcher = self._choose_matcher(len(image_paths))
        timings: dict[str, float] = {}
        detail: str | None = None

        with self._open_workspace() as work:
            database = work / "database.db"
            sparse = work / "sparse"
            _clear_stale(database, sparse)
            sparse.mkdir(parents=True, exist_ok=True)

            try:
                with _timer(timings, "extract_s"):
                    pycolmap.extract_features(
                        database_path=database,
                        image_path=image_dir,
                        image_names=image_names,
                        camera_mode=self.cfg.camera_mode_enum,
                        reader_options=self._reader_options(),
                        extraction_options=self._extraction_options(),
                        device=DEVICE,
                    )

                with _timer(timings, "match_s"):
                    self._match(database, matcher)

                with _timer(timings, "map_s"):
                    models = pycolmap.incremental_mapping(
                        database_path=database,
                        image_path=image_dir,
                        output_path=sparse,
                        options=self._mapper_options(),
                    )
            except Exception as exc:
                # A bad image or a COLMAP internal error must not abort a whole
                # benchmark sweep: record it as a failed run and let the caller
                # carry on, the way VGGTMethod turns an OOM into a failed Recon.
                # The reason stays one of a few fixed values so Phase 3 can group
                # by it; the unbounded text goes in extra.
                recon = Recon.failure(self.name, image_paths, "colmap_error")
                detail = f"{type(exc).__name__}: {exc}"
            else:
                recon = self._finish(models, image_paths)

        # One exit, so no future branch can ship a run without its provenance.
        recon.timings = dict(timings)
        recon.runtime_s = sum(timings.values())
        recon.extra["config"] = self.config(matcher_used=matcher)
        if detail is not None:
            recon.extra["failure_detail"] = detail
        # peak_vram_mb stays None: this pipeline never touches the GPU.
        return recon

    def _match(self, database: Path, matcher: str) -> None:
        cfg = self.cfg
        if matcher == "sequential":
            pycolmap.match_sequential(
                database_path=database,
                matching_options=self._matching_options(),
                pairing_options=pycolmap.SequentialPairingOptions(
                    overlap=cfg.sequential_overlap,
                    quadratic_overlap=cfg.sequential_quadratic_overlap,
                ),
                device=DEVICE,
            )
        else:
            pycolmap.match_exhaustive(
                database_path=database,
                matching_options=self._matching_options(),
                device=DEVICE,
            )

    @contextmanager
    def _open_workspace(self) -> Iterator[Path]:
        """The persistent workspace if one was given, else a temporary directory."""
        if self.workspace is not None:
            self.workspace.mkdir(parents=True, exist_ok=True)
            yield self.workspace
        else:
            with tempfile.TemporaryDirectory(prefix="colmap_") as tmp:
                yield Path(tmp)

    def _finish(
        self,
        models: dict[int, pycolmap.Reconstruction],
        image_paths: list[Path],
    ) -> Recon:
        """Turn pycolmap output into a Recon, applying the spec failure rules."""
        if not models:
            return Recon.failure(self.name, image_paths, "no_model")

        recon = self._to_recon(pick_largest_model(models), image_paths)
        recon.extra["n_models"] = len(models)
        if recon.n_registered < MIN_REGISTERED:
            recon.failed = True
            recon.failure_reason = "too_few_registered"
        return recon

    def _to_recon(self, model: pycolmap.Reconstruction, image_paths: list[Path]) -> Recon:
        n = len(image_paths)
        index_of = {p.name: i for i, p in enumerate(image_paths)}
        # float32 to match the extrinsics VGGTMethod produces, so Phase 3 sees
        # one dtype from both methods.
        extrinsics = np.full((n, 3, 4), np.nan, dtype=np.float32)
        intrinsics = np.full((n, 3, 3), np.nan, dtype=np.float32)
        registered = np.zeros(n, dtype=bool)
        image_hw: tuple[int, int] | None = None

        for image in model.images.values():
            i = index_of.get(image.name)
            if i is None or not image.has_pose:
                continue
            camera = image.camera
            # cam_from_world, num_reg_images and num_points3D are methods in
            # pycolmap 4.2, not properties.
            extrinsics[i] = np.asarray(image.cam_from_world().matrix())
            intrinsics[i] = np.asarray(camera.calibration_matrix())
            registered[i] = True
            image_hw = (int(camera.height), int(camera.width))

        # One pass, preallocated: pycolmap hands back a fresh small array per
        # point, so a list comprehension would build two throwaway copies of
        # every point in the cloud.
        points = np.empty((model.num_points3D(), 3), dtype=np.float32)
        colors = np.empty((model.num_points3D(), 3), dtype=np.uint8)
        for i, point in enumerate(model.points3D.values()):
            points[i] = point.xyz
            colors[i] = point.color

        return Recon(
            method=self.name,
            image_paths=image_paths,
            image_hw=image_hw,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            registered=registered,
            points=points,
            colors=colors,
            # depth and depth_conf stay None: COLMAP sparse produces neither.
        )


def pick_largest_model(models: dict[int, pycolmap.Reconstruction]) -> pycolmap.Reconstruction:
    """The model with the most registered images (spec section 6.2).

    Ties break on more 3D points, then on the lower model id, so repeated runs
    over the same inputs pick the same model.
    """
    best_id = max(models, key=lambda i: (models[i].num_reg_images(), models[i].num_points3D(), -i))
    return models[best_id]


def _clear_stale(database: Path, sparse: Path) -> None:
    """Start a reused workspace from scratch.

    COLMAP skips extraction and matching for images already in the database. A
    second run over a persistent workspace would then record near-zero
    extract_s and match_s, which is a fabricated benchmark number, and would
    reconstruct the previous frame set if the inputs had changed.
    """
    database.unlink(missing_ok=True)
    if sparse.exists():
        shutil.rmtree(sparse)


def _split_image_paths(image_paths: list[Path]) -> tuple[Path, list[str]]:
    """COLMAP wants one image root plus names, so all inputs must share a parent.

    Passing names rather than copying files is what lets a 10-frame subset of a
    25-image scene be reconstructed in place.
    """
    if not image_paths:
        raise ValueError("no input images")
    parents = {p.parent.resolve() for p in image_paths}
    if len(parents) > 1:
        raise ValueError(
            "all input images must live in one directory, got: "
            + ", ".join(str(p) for p in sorted(parents))
        )
    return parents.pop(), [p.name for p in image_paths]


@contextmanager
def _timer(into: dict[str, float], key: str) -> Iterator[None]:
    """Wall-clock timing. Not profiling.gpu_timer, which synchronises CUDA."""
    start = time.perf_counter()
    try:
        yield
    finally:
        into[key] = time.perf_counter() - start

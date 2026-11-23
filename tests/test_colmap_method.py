"""Phase 2: COLMAP conversion, model selection and failure rules.

These build pycolmap Reconstruction objects in memory instead of running SfM, so
the suite stays fast and needs no image data. In pycolmap 4.2 a pose belongs to
a frame, not an Image, so a registered image is added with
``add_image_with_trivial_frame(image, cam_from_world)`` and an unregistered one
with the single-argument overload.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap
import pytest

from vggt_bench.methods.colmap_method import (
    DEFAULT_CONFIG,
    ColmapConfig,
    ColmapMethod,
    _clear_stale,
    _split_image_paths,
    pick_largest_model,
)

IMAGE_DIR = Path("/scene/images")
WIDTH, HEIGHT = 640, 480
FOCAL = 500.0


def make_camera(camera_id: int = 1) -> pycolmap.Camera:
    return pycolmap.Camera.create_from_model_name(camera_id, "SIMPLE_RADIAL", FOCAL, WIDTH, HEIGHT)


def make_pose(tx: float) -> pycolmap.Rigid3d:
    """A pose with a non-identity rotation, so a transposed matrix would show."""
    rotation = pycolmap.Rotation3d(np.array([0.0, 0.0, np.deg2rad(30.0)]))
    return pycolmap.Rigid3d(rotation, np.array([tx, 0.0, 2.0]))


def make_model(
    names: list[str],
    poses: dict[str, pycolmap.Rigid3d] | None = None,
    n_points: int = 4,
) -> pycolmap.Reconstruction:
    """A reconstruction holding `names`; only names in `poses` are registered."""
    poses = poses or {}
    model = pycolmap.Reconstruction()
    model.add_camera_with_trivial_rig(make_camera())
    for image_id, name in enumerate(names, start=1):
        image = pycolmap.Image(name=name, camera_id=1, image_id=image_id)
        if name in poses:
            model.add_image_with_trivial_frame(image, poses[name])
        else:
            model.add_image_with_trivial_frame(image)
    for i in range(n_points):
        model.add_point3D(
            np.array([float(i), 1.0, 2.0]),
            pycolmap.Track(),
            np.array([i, 10, 20], dtype=np.uint8),
        )
    return model


def paths(*names: str) -> list[Path]:
    return [IMAGE_DIR / n for n in names]


# -- conversion ---------------------------------------------------------------


def test_to_recon_marks_only_posed_images_as_registered() -> None:
    poses = {"a.png": make_pose(0.0), "c.png": make_pose(1.0)}
    model = make_model(["a.png", "b.png", "c.png"], poses)

    recon = ColmapMethod()._to_recon(model, paths("a.png", "b.png", "c.png"))

    assert recon.method == "colmap"
    assert recon.registered.tolist() == [True, False, True]
    assert recon.n_registered == 2
    assert recon.n_input == 3


def test_to_recon_leaves_unregistered_rows_as_nan() -> None:
    model = make_model(["a.png", "b.png"], {"a.png": make_pose(0.0)})

    recon = ColmapMethod()._to_recon(model, paths("a.png", "b.png"))

    assert np.isfinite(recon.extrinsics[0]).all()
    assert np.isnan(recon.extrinsics[1]).all()
    assert np.isnan(recon.intrinsics[1]).all()


def test_to_recon_round_trips_the_pose_matrix() -> None:
    pose = make_pose(0.5)
    model = make_model(["a.png"], {"a.png": pose})

    recon = ColmapMethod()._to_recon(model, paths("a.png"))

    np.testing.assert_allclose(recon.extrinsics[0], np.asarray(pose.matrix()), atol=1e-12)
    # Recon documents x_cam = R @ x_world + t, so the centre is -R^T t.
    rotation = np.asarray(pose.matrix())[:, :3]
    expected_centre = -rotation.T @ np.asarray(pose.matrix())[:, 3]
    np.testing.assert_allclose(recon.camera_centers()[0], expected_centre, atol=1e-12)


def test_to_recon_copies_intrinsics_and_resolution_from_the_shared_camera() -> None:
    model = make_model(["a.png"], {"a.png": make_pose(0.0)})

    recon = ColmapMethod()._to_recon(model, paths("a.png"))

    np.testing.assert_allclose(recon.intrinsics[0], np.asarray(make_camera().calibration_matrix()))
    assert recon.image_hw == (HEIGHT, WIDTH)


def test_to_recon_returns_points_and_colors_in_recon_dtypes() -> None:
    model = make_model(["a.png"], {"a.png": make_pose(0.0)}, n_points=7)

    recon = ColmapMethod()._to_recon(model, paths("a.png"))

    assert recon.points.shape == (7, 3)
    assert recon.colors.shape == (7, 3)
    assert recon.points.dtype == np.float32
    assert recon.colors.dtype == np.uint8


def test_to_recon_leaves_depth_unset_because_colmap_sparse_has_none() -> None:
    model = make_model(["a.png"], {"a.png": make_pose(0.0)})

    recon = ColmapMethod()._to_recon(model, paths("a.png"))

    assert recon.depth is None
    assert recon.depth_conf is None


def test_to_recon_ignores_images_that_are_not_inputs() -> None:
    """A persistent workspace can hold a model built from a different frame set."""
    model = make_model(["a.png", "stale.png"], {"a.png": make_pose(0.0)})

    recon = ColmapMethod()._to_recon(model, paths("a.png"))

    assert recon.n_input == 1
    assert recon.registered.tolist() == [True]


def test_to_recon_handles_an_empty_point_cloud() -> None:
    model = make_model(["a.png"], {"a.png": make_pose(0.0)}, n_points=0)

    recon = ColmapMethod()._to_recon(model, paths("a.png"))

    assert recon.points.shape == (0, 3)
    assert recon.colors.shape == (0, 3)


# -- model selection ----------------------------------------------------------


def test_pick_largest_model_prefers_the_most_registered_images() -> None:
    small = make_model(["a.png", "b.png"], {"a.png": make_pose(0.0)})
    large = make_model(
        ["a.png", "b.png", "c.png"],
        {"a.png": make_pose(0.0), "b.png": make_pose(1.0), "c.png": make_pose(2.0)},
    )

    assert pick_largest_model({0: small, 1: large}) is large
    assert pick_largest_model({0: large, 1: small}) is large


def test_pick_largest_model_breaks_registration_ties_on_point_count() -> None:
    poses = {"a.png": make_pose(0.0), "b.png": make_pose(1.0)}
    few = make_model(["a.png", "b.png"], poses, n_points=3)
    many = make_model(["a.png", "b.png"], poses, n_points=30)

    assert pick_largest_model({0: few, 1: many}) is many


def test_pick_largest_model_is_deterministic_on_a_full_tie() -> None:
    poses = {"a.png": make_pose(0.0)}
    first = make_model(["a.png"], poses, n_points=5)
    second = make_model(["a.png"], poses, n_points=5)

    assert pick_largest_model({0: first, 1: second}) is first
    assert pick_largest_model({1: second, 0: first}) is first


# -- failure rules ------------------------------------------------------------


def test_no_model_is_a_failure() -> None:
    recon = ColmapMethod()._finish({}, paths("a.png", "b.png"))

    assert recon.failed
    assert recon.failure_reason == "no_model"
    assert recon.n_registered == 0


def test_fewer_than_three_registered_is_a_failure_that_keeps_its_poses() -> None:
    model = make_model(
        ["a.png", "b.png", "c.png"],
        {"a.png": make_pose(0.0), "b.png": make_pose(1.0)},
    )

    recon = ColmapMethod()._finish({0: model}, paths("a.png", "b.png", "c.png"))

    assert recon.failed
    assert recon.failure_reason == "too_few_registered"
    # Phase 3 plots partial failures, so the mask and poses must survive.
    assert recon.registered.tolist() == [True, True, False]
    assert np.isfinite(recon.extrinsics[0]).all()


def test_three_registered_is_a_success() -> None:
    names = ["a.png", "b.png", "c.png"]
    model = make_model(names, {n: make_pose(float(i)) for i, n in enumerate(names)})

    recon = ColmapMethod()._finish({0: model}, paths(*names))

    assert not recon.failed
    assert recon.failure_reason is None
    assert recon.n_registered == 3


def test_finish_records_the_model_count() -> None:
    names = ["a.png", "b.png", "c.png"]
    model = make_model(names, {n: make_pose(float(i)) for i, n in enumerate(names)})

    recon = ColmapMethod()._finish({0: model, 1: model}, paths(*names))

    assert recon.extra["n_models"] == 2


def test_config_records_the_device_provenance() -> None:
    config = ColmapMethod(ColmapConfig()).config(matcher_used="exhaustive")

    assert config["device"] == "cpu"
    assert config["matcher_used"] == "exhaustive"
    assert config["pycolmap_has_cuda"] is False


# -- config -------------------------------------------------------------------


def test_shipped_config_matches_the_spec() -> None:
    """min_model_size is the one that bites: pycolmap defaults it to 10, which
    would report "no model" for every run with fewer than 10 views."""
    config = ColmapConfig.from_yaml(DEFAULT_CONFIG)

    assert config.min_model_size == 3
    assert pycolmap.IncrementalPipelineOptions().min_model_size == 10
    assert config.camera_model == "SIMPLE_RADIAL"
    assert config.camera_mode == "SINGLE"
    assert config.matcher == "exhaustive"
    assert config.exhaustive_max_images == 50


def test_min_model_size_cannot_drop_below_the_registration_floor() -> None:
    """Otherwise COLMAP keeps a model this code then marks failed."""
    with pytest.raises(ValueError, match="min_model_size"):
        ColmapConfig(min_model_size=2)


def test_config_values_reach_the_pycolmap_option_objects() -> None:
    config = ColmapConfig(max_image_size=512, max_num_features=1234, random_seed=7)
    method = ColmapMethod(config)

    assert method._extraction_options().max_image_size == 512
    assert method._extraction_options().sift.max_num_features == 1234
    assert method._mapper_options().random_seed == 7
    assert method._reader_options().camera_model == "SIMPLE_RADIAL"
    # The wheel has no CUDA, so GPU must stay off on both stages.
    assert method._extraction_options().use_gpu is False
    assert method._matching_options().use_gpu is False


def test_an_explicit_config_path_that_is_missing_is_an_error() -> None:
    """Silently using defaults would let a typo'd --config report false success."""
    with pytest.raises(FileNotFoundError):
        ColmapConfig.from_yaml(Path("configs/does-not-exist.yaml"))


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "colmap.yaml"
    path.write_text("matcher: exhaustive\nmatchr: typo\n")

    with pytest.raises(ValueError, match="unknown keys"):
        ColmapConfig.from_yaml(path)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"matcher": "vocabtree"}, "exhaustive"),
        ({"camera_mode": "single"}, "camera_mode"),  # the enum member is upper case
    ],
)
def test_bad_config_values_are_rejected_at_load_not_mid_run(
    kwargs: dict[str, str], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        ColmapConfig(**kwargs)


# -- pipeline errors ----------------------------------------------------------


def test_a_colmap_error_becomes_a_failed_recon_not_an_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep must be able to log a bad scene and continue to the next one."""
    image = tmp_path / "a.png"
    image.write_bytes(b"not an image")
    monkeypatch.setattr(
        "pycolmap.extract_features",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("cannot read image")),
    )

    recon = ColmapMethod().reconstruct([image])

    assert recon.failed
    assert recon.n_registered == 0
    # The reason stays one of a few fixed values so Phase 3's CSV can group by
    # it; the unbounded message goes in extra.
    assert recon.failure_reason == "colmap_error"
    assert "cannot read image" in recon.extra["failure_detail"]
    assert "extract_s" in recon.timings  # the stage that raised is still timed
    assert recon.extra["config"]["device"] == "cpu"  # provenance even on failure


# -- matcher choice -----------------------------------------------------------


@pytest.mark.parametrize(
    ("matcher", "n_input", "expected"),
    [
        ("exhaustive", 3, "exhaustive"),
        ("exhaustive", 50, "exhaustive"),  # the ceiling itself still exhaustive
        ("exhaustive", 51, "sequential"),  # above it, pairs get quadratic
        ("sequential", 3, "sequential"),  # an explicit choice is never overridden
        ("sequential", 500, "sequential"),
    ],
)
def test_matcher_choice(matcher: str, n_input: int, expected: str) -> None:
    method = ColmapMethod(ColmapConfig(matcher=matcher, exhaustive_max_images=50))

    assert method._choose_matcher(n_input) == expected


# -- image path handling ------------------------------------------------------


def test_split_image_paths_returns_one_root_and_the_names() -> None:
    directory, names = _split_image_paths(paths("00.png", "03.png"))

    assert directory == IMAGE_DIR.resolve()
    assert names == ["00.png", "03.png"]


def test_split_image_paths_rejects_inputs_spanning_directories() -> None:
    with pytest.raises(ValueError, match="one directory"):
        _split_image_paths([Path("/a/00.png"), Path("/b/01.png")])


def test_split_image_paths_rejects_an_empty_list() -> None:
    with pytest.raises(ValueError, match="no input images"):
        _split_image_paths([])


# -- workspace reuse ----------------------------------------------------------


def test_clear_stale_removes_a_previous_runs_database_and_models(tmp_path: Path) -> None:
    """Reusing a database makes COLMAP skip work and report fake near-zero timings."""
    database = tmp_path / "database.db"
    sparse = tmp_path / "sparse"
    database.write_bytes(b"stale")
    (sparse / "0").mkdir(parents=True)
    (sparse / "0" / "cameras.bin").write_bytes(b"stale")

    _clear_stale(database, sparse)

    assert not database.exists()
    assert not sparse.exists()


def test_clear_stale_is_a_no_op_on_a_fresh_workspace(tmp_path: Path) -> None:
    _clear_stale(tmp_path / "database.db", tmp_path / "sparse")  # must not raise

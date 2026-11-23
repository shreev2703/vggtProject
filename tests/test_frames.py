"""Frame selection shared by the reconstruction scripts.

Both methods must see the same subset for any side-by-side number to mean
anything, so this picks evenly across the trajectory rather than the first N.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vggt_bench.io.frames import list_images, sample_frames, select_frames


@pytest.fixture
def scene(tmp_path: Path) -> Path:
    images = tmp_path / "images"
    images.mkdir()
    for i in range(10):
        (images / f"{i:02d}.png").write_bytes(b"")
    (images / "notes.txt").write_text("not an image")
    return tmp_path


def test_lists_only_images_in_filename_order(scene: Path) -> None:
    names = [p.name for p in list_images(scene)]

    assert names == [f"{i:02d}.png" for i in range(10)]


def test_selects_an_evenly_spaced_subset_spanning_the_trajectory(scene: Path) -> None:
    names = [p.name for p in select_frames(scene, 4)]

    assert names == ["00.png", "03.png", "06.png", "09.png"]
    # First and last matter: a subset must cover the whole camera path.
    assert names[0] == "00.png"
    assert names[-1] == "09.png"


@pytest.mark.parametrize("num_frames", [0, -1, 10, 99])
def test_non_positive_or_oversized_counts_return_every_frame(scene: Path, num_frames: int) -> None:
    assert len(select_frames(scene, num_frames)) == 10


def test_selection_is_deterministic(scene: Path) -> None:
    assert select_frames(scene, 7) == select_frames(scene, 7)


def test_requesting_one_frame_is_allowed(scene: Path) -> None:
    assert [p.name for p in select_frames(scene, 1)] == ["00.png"]


def _touch_images(root: Path, n: int) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = [root / f"{i:03d}.png" for i in range(n)]
    for p in paths:
        p.write_bytes(b"")
    return paths


def test_sample_frames_is_reproducible_for_a_seed(tmp_path: Path) -> None:
    paths = _touch_images(tmp_path, 20)
    assert sample_frames(paths, 5, seed=0) == sample_frames(paths, 5, seed=0)


def test_sample_frames_differs_between_seeds(tmp_path: Path) -> None:
    paths = _touch_images(tmp_path, 20)
    assert sample_frames(paths, 5, seed=0) != sample_frames(paths, 5, seed=1)


def test_sample_frames_keeps_original_order_and_is_unique(tmp_path: Path) -> None:
    paths = _touch_images(tmp_path, 20)
    got = sample_frames(paths, 6, seed=3)
    assert got == sorted(got, key=paths.index)
    assert len(set(got)) == 6


def test_sample_frames_without_a_seed_is_evenly_spaced(tmp_path: Path) -> None:
    paths = _touch_images(tmp_path, 21)
    assert sample_frames(paths, 3, seed=None) == [paths[0], paths[10], paths[20]]


def test_sample_frames_clamps_to_the_available_frames(tmp_path: Path) -> None:
    paths = _touch_images(tmp_path, 4)
    assert sample_frames(paths, 10, seed=0) == paths

"""Frame selection shared by the reconstruction scripts.

Both methods must see the same subset for any side-by-side number to mean
anything, so this picks evenly across the trajectory rather than the first N.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vggt_bench.io.frames import list_images, select_frames


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

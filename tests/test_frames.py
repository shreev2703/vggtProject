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


# --- Phase 4 capture helpers ---------------------------------------------------------------


def _checkerboard(size: int = 64, cell: int = 4):
    import numpy as np

    yy, xx = np.indices((size, size))
    return (((yy // cell + xx // cell) % 2) * 255).astype("uint8")


def _save_gray(path: Path, arr) -> Path:
    from PIL import Image

    Image.fromarray(arr).convert("RGB").save(path)
    return path


def test_grid_size_matches_vggt_crop_mode() -> None:
    from vggt_bench.io.frames import vggt_grid_size

    # 1920x1080 landscape: width 518, height round(1080*518/1920/14)*14 = 294.
    assert vggt_grid_size(1920, 1080) == (518, 294)
    # Portrait is centre-cropped to a square, as VGGT does.
    assert vggt_grid_size(1080, 1920) == (518, 518)


def test_grid_size_is_a_fixed_point() -> None:
    from vggt_bench.io.frames import vggt_grid_size

    w, h = vggt_grid_size(1280, 720)
    assert vggt_grid_size(w, h) == (w, h)
    assert w % 14 == 0 and h % 14 == 0


def test_laplacian_variance_ranks_sharp_above_blurred(tmp_path: Path) -> None:
    from scipy.ndimage import uniform_filter

    from vggt_bench.io.frames import laplacian_variance

    sharp = _checkerboard()
    blurred = uniform_filter(sharp.astype(float), size=7).astype("uint8")
    s = laplacian_variance(_save_gray(tmp_path / "s.png", sharp))
    b = laplacian_variance(_save_gray(tmp_path / "b.png", blurred))

    assert s > b


def test_drop_blurry_keeps_sharp_frames_and_reports_every_score(tmp_path: Path) -> None:
    from scipy.ndimage import uniform_filter

    from vggt_bench.io.frames import drop_blurry, laplacian_variance

    sharp = _save_gray(tmp_path / "a_sharp.png", _checkerboard())
    soft = _save_gray(
        tmp_path / "b_soft.png",
        uniform_filter(_checkerboard().astype(float), size=9).astype("uint8"),
    )
    threshold = (laplacian_variance(sharp) + laplacian_variance(soft)) / 2

    kept, scores = drop_blurry([sharp, soft], threshold)

    assert kept == [sharp]
    assert set(scores) == {"a_sharp.png", "b_soft.png"}


def test_resize_to_grid_writes_vggt_sized_pngs_in_order(tmp_path: Path) -> None:
    from PIL import Image

    from vggt_bench.io.frames import resize_to_grid, vggt_grid_size

    src = tmp_path / "src"
    src.mkdir()
    paths = []
    for i in range(3):
        p = src / f"f{i}.jpg"
        Image.new("RGB", (640, 360), (i * 40, 0, 0)).save(p)
        paths.append(p)

    out = resize_to_grid(paths, tmp_path / "out")

    assert [p.stem for p in out] == ["f0", "f1", "f2"]
    for p in out:
        assert p.suffix == ".png"
        with Image.open(p) as im:
            assert im.size == vggt_grid_size(640, 360)


def test_extract_frames_samples_at_requested_fps(tmp_path: Path) -> None:
    import shutil
    import subprocess

    from vggt_bench.io.frames import extract_frames

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    video = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x90:rate=10:duration=3",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )

    frames = extract_frames(video, tmp_path / "raw", fps=2.0)

    assert 5 <= len(frames) <= 7  # 3 s at 2 fps
    assert all(p.exists() for p in frames)


def test_extract_frames_raises_on_a_bad_video(tmp_path: Path) -> None:
    from vggt_bench.io.frames import extract_frames

    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")

    with pytest.raises(RuntimeError, match="ffmpeg"):
        extract_frames(bad, tmp_path / "raw")

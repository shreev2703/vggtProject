"""Choosing input frames for a reconstruction.

Phase 1's scripts carry their own copy of this logic and are left alone
deliberately. New scripts import it from here so the Phase 3 sweep and the
Phase 4 captures do not grow a third and fourth variant: both methods have to
see the *same* frame subset for any side-by-side number to mean anything.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import laplace

EXAMPLES = Path("third_party/vggt/examples")
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def list_images(scene_dir: Path, subdir: str = "images") -> list[Path]:
    """Every image in a scene's images folder (default images/), in filename order."""
    return sorted(p for p in (scene_dir / subdir).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def select_frames(scene_dir: Path, num_frames: int) -> list[Path]:
    """An evenly spaced subset of a scene's frames; all of them when num_frames <= 0.

    Evenly spaced rather than the first N, so a subset still spans the whole
    camera trajectory. That is what makes the view count, and not the amount of
    the scene covered, the variable under test.
    """
    return sample_frames(list_images(scene_dir), num_frames, None)


def sample_frames(paths: list[Path], n: int, seed: int | None) -> list[Path]:
    """Pick n frames from an ordered list, identically for every method.

    seed=None gives the evenly spaced, wide-baseline subset. An int gives a
    reproducible random draw without replacement, returned in original order.
    n >= len(paths) returns everything: a scene with fewer images than n is
    clamped rather than padded with duplicates.
    """
    if n <= 0 or n >= len(paths):
        return list(paths)
    if seed is None:
        idx = np.linspace(0, len(paths) - 1, n).round().astype(int)
    else:
        idx = np.sort(np.random.default_rng(seed).choice(len(paths), n, replace=False))
    return [paths[i] for i in idx]


# --- Phase 4: phone-video captures ----------------------------------------------------------

VGGT_SIZE = 518
VGGT_PATCH = 14


def vggt_grid_size(width: int, height: int, target: int = VGGT_SIZE) -> tuple[int, int]:
    """(W, H) that VGGT's load_and_preprocess_images(mode="crop") produces for an image.

    Width becomes `target`; height keeps the aspect ratio rounded to the 14 px patch grid, then
    is centre-cropped to at most `target`. Saving frames at this size makes VGGT's own
    preprocessing an identity, so its intrinsics apply to the saved files directly.
    """
    new_h = round(height * (target / width) / VGGT_PATCH) * VGGT_PATCH
    return target, min(new_h, target)


def extract_frames(video: Path, out_dir: Path, fps: float = 2.0) -> list[Path]:
    """Sample a video at `fps` into out_dir/frame_000001.jpg, ... and return them in order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-i", str(video),
        "-vf", f"fps={fps}", "-q:v", "2", str(out_dir / "frame_%06d.jpg"),
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True)
    frames = sorted(out_dir.glob("frame_*.jpg"))
    if proc.returncode != 0 or not frames:
        raise RuntimeError(f"ffmpeg failed on {video}: {proc.stderr.strip() or 'no frames'}")
    return frames


def laplacian_variance(path: Path) -> float:
    """Sharpness score: variance of the Laplacian of the grayscale image (low = blurry)."""
    with Image.open(path) as im:
        gray = np.asarray(im.convert("L"), dtype=np.float64)
    return float(laplace(gray).var())


def drop_blurry(paths: list[Path], threshold: float) -> tuple[list[Path], dict[str, float]]:
    """Keep frames whose sharpness is at least `threshold`; also return every frame's score."""
    scores = {p.name: laplacian_variance(p) for p in paths}
    return [p for p in paths if scores[p.name] >= threshold], scores


def resize_to_grid(paths: list[Path], out_dir: Path) -> list[Path]:
    """Write each frame as a PNG at its vggt_grid_size, centre-cropping tall frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for p in paths:
        with Image.open(p) as im:
            rgb = im.convert("RGB")
            w, h = vggt_grid_size(*rgb.size)
            new_h = round(rgb.height * (w / rgb.width) / VGGT_PATCH) * VGGT_PATCH
            rgb = rgb.resize((w, new_h), Image.Resampling.BICUBIC)
            if new_h > h:
                top = (new_h - h) // 2
                rgb = rgb.crop((0, top, w, top + h))
            dest = out_dir / f"{p.stem}.png"
            rgb.save(dest)
        out.append(dest)
    return out

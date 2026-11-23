"""Choosing input frames for a reconstruction.

Phase 1's scripts carry their own copy of this logic and are left alone
deliberately. New scripts import it from here so the Phase 3 sweep and the
Phase 4 captures do not grow a third and fourth variant: both methods have to
see the *same* frame subset for any side-by-side number to mean anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

EXAMPLES = Path("third_party/vggt/examples")
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def list_images(scene_dir: Path) -> list[Path]:
    """Every image in a scene's images/ folder, in filename order."""
    return sorted(p for p in (scene_dir / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS)


def select_frames(scene_dir: Path, num_frames: int) -> list[Path]:
    """An evenly spaced subset of a scene's frames; all of them when num_frames <= 0.

    Evenly spaced rather than the first N, so a subset still spans the whole
    camera trajectory. That is what makes the view count, and not the amount of
    the scene covered, the variable under test.
    """
    paths = list_images(scene_dir)
    if num_frames <= 0 or num_frames >= len(paths):
        return paths
    idx = np.linspace(0, len(paths) - 1, num_frames).round().astype(int)
    return [paths[i] for i in idx]

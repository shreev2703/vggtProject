"""GPU timing and memory helpers, plus the VGGT frame-count guard."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch

MB = 2**20
DEFAULT_PROFILE = Path("results/memory_profile.json")


def total_vram_mb() -> float:
    return torch.cuda.get_device_properties(0).total_memory / MB


def allocated_mb() -> float:
    return torch.cuda.memory_allocated() / MB


def reset_peak() -> None:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def peak_allocated_mb() -> float:
    return torch.cuda.max_memory_allocated() / MB


def peak_reserved_mb() -> float:
    return torch.cuda.max_memory_reserved() / MB


@contextmanager
def gpu_timer() -> Iterator[dict[str, float]]:
    """Wall-clock timer that waits for queued GPU work before and after.

    CUDA calls are asynchronous, so without synchronize() the timer would stop
    before the GPU has actually finished.
    """
    result: dict[str, float] = {}
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield result
    finally:
        torch.cuda.synchronize()
        result["elapsed_s"] = time.perf_counter() - start


def free_gpu_memory() -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def max_frames_from_profile(weights_dtype: str, path: Path = DEFAULT_PROFILE) -> int | None:
    """Largest frame count that ran without OOM during calibration, if calibrated."""
    if not path.exists():
        return None
    profile = json.loads(path.read_text())["profiles"].get(weights_dtype)
    return None if profile is None else profile["max_ok_frames"]


def frame_cap(max_frames: int, weights_dtype: str, path: Path = DEFAULT_PROFILE) -> int:
    """`max_frames`, lowered to the calibrated no-OOM frame count when a profile exists."""
    limit = max_frames_from_profile(weights_dtype, path)
    return max_frames if limit is None else min(max_frames, limit)

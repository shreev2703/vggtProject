"""Sped-up, silent preview clip of a capture video for the demo site."""

from __future__ import annotations

import subprocess
from pathlib import Path

TARGET_SECONDS = 8.0
WIDTH = 640


def probe_duration(src: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return float(out.strip())


def speedup_for(duration: float, target: float = TARGET_SECONDS) -> float:
    """Speed factor that fits `duration` seconds into about `target`; never slows a short clip."""
    return max(1.0, duration / target)


def ffmpeg_args(src: Path, dst: Path, speed: float, width: int = WIDTH) -> list[str]:
    vf = f"setpts=PTS/{speed:.4f},fps=30,scale={width}:-2"
    return [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-an", "-vf", vf,
        "-c:v", "libx264", "-crf", "28", "-preset", "slow", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(dst),
    ]  # fmt: skip


def make_clip(src: Path, dst: Path) -> float:
    """Write `dst` and return the speed factor used."""
    speed = speedup_for(probe_duration(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(ffmpeg_args(src, dst, speed), check=True)
    return speed

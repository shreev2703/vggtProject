"""Run metadata (git commit, time, config) attached to every result file."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from typing import Any


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_is_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def run_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Everything needed to trace a result back to the code and settings that made it."""
    meta: dict[str, Any] = {
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": config,
    }
    try:
        import torch

        meta["torch"] = torch.__version__
        if torch.cuda.is_available():
            meta["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return meta

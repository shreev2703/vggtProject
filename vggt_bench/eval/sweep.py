"""Planning and bookkeeping for the benchmark sweep.

Kept apart from scripts/03_run_benchmark.py so the parts that decide *what* runs
and *what is already done* are testable without a GPU.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

COLUMNS: tuple[str, ...] = (
    "dataset", "scene", "scene_type", "method", "n_views", "seed", "sampling",
    "auc_5", "auc_10", "auc_30", "rre_median", "rte_median", "n_pairs",
    "success", "n_input", "n_registered", "frac_registered",
    "runtime_s", "model_load_s", "peak_vram_mb", "failure_reason",
    "git_commit", "timestamp_utc",
)  # fmt: skip

# (dataset, scene, method, n_views, "seed<k>" | "wide")
RunKey = tuple[str, str, str, int, str]


def sampling_label(sampling: str, seed: int | None) -> str:
    return "wide" if sampling == "wide" else f"seed{seed}"


def run_key(row: dict[str, Any]) -> RunKey:
    """Identity of a run; works on a planned run and on a CSV row alike."""
    seed = row.get("seed")
    seed = None if seed in (None, "") else int(seed)
    return (
        str(row["dataset"]),
        str(row["scene"]),
        str(row["method"]),
        int(row["n_views"]),
        sampling_label(str(row["sampling"]), seed),
    )


def plan_runs(cfg: dict, methods: Sequence[str], max_frames: int | None) -> list[dict]:
    """Every (scene, N, sampling, method) run in a deterministic order.

    max_frames is VGGT's calibrated memory cap; it removes larger view counts for
    VGGT only, never for COLMAP. Sampling is the outer loop over method so both
    methods are handed the identical frame subset back to back.
    """
    runs: list[dict] = []
    samplings: list[tuple[str, int | None]] = [("random", s) for s in cfg["seeds"]]
    samplings.append(("wide", None))
    for scene, info in cfg["scenes"].items():
        for n in cfg["view_counts"]:
            for sampling, seed in samplings:
                for method in methods:
                    if method == "vggt" and max_frames is not None and n > max_frames:
                        continue
                    runs.append(
                        {
                            "dataset": cfg["dataset"],
                            "scene": scene,
                            "scene_type": info["scene_type"],
                            "method": method,
                            "n_views": n,
                            "seed": seed,
                            "sampling": sampling,
                        }
                    )
    return runs


def done_keys(csv_path: Path) -> set[RunKey]:
    """Keys of rows already in the CSV; a missing file or torn last line is fine."""
    if not csv_path.exists():
        return set()
    # An unterminated last line is a torn write even if its key columns look
    # intact; append_row will cut it, so the run must not count as done.
    text = _complete_lines(csv_path.read_bytes()).decode()
    keys: set[RunKey] = set()
    for row in csv.DictReader(text.splitlines(keepends=True)):
        try:
            keys.add(run_key(row))
        except (KeyError, TypeError, ValueError):
            continue
    return keys


def append_row(csv_path: Path, row: dict[str, Any]) -> None:
    """Append one row, writing the header on first use and flushing immediately."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _drop_torn_tail(csv_path)
    new = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore", restval="")
        if new:
            writer.writeheader()
        writer.writerow({k: ("" if row.get(k) is None else row[k]) for k in COLUMNS})
        f.flush()


def _complete_lines(data: bytes) -> bytes:
    """data without a trailing line that never got its newline."""
    return data if not data or data.endswith(b"\n") else data[: data.rfind(b"\n") + 1]


def _drop_torn_tail(csv_path: Path) -> None:
    """Cut a last line left unterminated by an interrupted write, so the next
    row starts on a fresh line instead of being glued onto it."""
    if csv_path.exists():
        data = csv_path.read_bytes()
        if (kept := _complete_lines(data)) != data:
            csv_path.write_bytes(kept)

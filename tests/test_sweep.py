"""Sweep planning and the resumable results CSV."""

from __future__ import annotations

import csv
from pathlib import Path

from vggt_bench.eval.sweep import COLUMNS, append_row, done_keys, plan_runs, run_key

CFG = {
    "dataset": "mip360",
    "view_counts": [3, 5, 10, 20, 40],
    "seeds": [0, 1, 2],
    "scenes": {"garden": {"scene_type": "easy"}, "room": {"scene_type": "textureless"}},
}


def test_plan_runs_covers_every_combination() -> None:
    runs = plan_runs(CFG, ["vggt", "colmap"], max_frames=None)
    assert len(runs) == 2 * 5 * (3 + 1) * 2
    assert len({run_key(r) for r in runs}) == len(runs)


def test_plan_runs_labels_the_wide_baseline_sampling() -> None:
    runs = plan_runs(CFG, ["vggt"], max_frames=None)
    assert {r["sampling"] for r in runs} == {"random", "wide"}
    assert all(r["seed"] is None for r in runs if r["sampling"] == "wide")
    assert all(isinstance(r["seed"], int) for r in runs if r["sampling"] == "random")


def test_plan_runs_caps_view_counts_by_the_memory_profile() -> None:
    runs = plan_runs(CFG, ["vggt", "colmap"], max_frames=10)
    vggt_n = {r["n_views"] for r in runs if r["method"] == "vggt"}
    colmap_n = {r["n_views"] for r in runs if r["method"] == "colmap"}
    assert vggt_n == {3, 5, 10}
    assert colmap_n == {3, 5, 10, 20, 40}


def test_plan_runs_is_deterministic() -> None:
    assert plan_runs(CFG, ["vggt", "colmap"], None) == plan_runs(CFG, ["vggt", "colmap"], None)


def _row(**kw) -> dict:
    base = {
        "dataset": "mip360", "scene": "garden", "method": "vggt",
        "n_views": 3, "seed": 0, "sampling": "random",
    }  # fmt: skip
    return base | kw


def test_done_keys_reads_back_appended_rows(tmp_path: Path) -> None:
    path = tmp_path / "b.csv"
    append_row(path, _row())
    append_row(path, _row(seed=None, sampling="wide", method="colmap"))

    keys = done_keys(path)

    assert run_key(_row()) in keys
    assert run_key(_row(seed=None, sampling="wide", method="colmap")) in keys
    assert len(keys) == 2


def test_done_keys_is_empty_for_a_missing_csv(tmp_path: Path) -> None:
    assert done_keys(tmp_path / "nope.csv") == set()


def test_done_keys_tolerates_a_truncated_final_line(tmp_path: Path) -> None:
    path = tmp_path / "b.csv"
    append_row(path, _row())
    with open(path, "a") as f:
        f.write("mip360,garden,easy,vg")  # killed mid-write

    assert done_keys(path) == {run_key(_row())}


def test_append_row_writes_the_header_once(tmp_path: Path) -> None:
    path = tmp_path / "b.csv"
    append_row(path, _row())
    append_row(path, _row(n_views=5))

    lines = path.read_text().splitlines()
    assert lines[0].split(",") == list(COLUMNS)
    assert sum(line.startswith("dataset,") for line in lines) == 1
    with open(path) as f:
        assert len(list(csv.DictReader(f))) == 2


def test_append_row_after_a_torn_line_keeps_the_csv_clean(tmp_path: Path) -> None:
    path = tmp_path / "b.csv"
    append_row(path, _row())
    with open(path, "a") as f:
        f.write("mip360,garden,easy,vg")
    append_row(path, _row(n_views=5))

    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert [r["n_views"] for r in rows] == ["3", "5"]

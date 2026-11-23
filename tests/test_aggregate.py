"""Aggregation helpers behind the benchmark figures."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vggt_bench.eval.aggregate import (
    by_scene_type,
    per_views,
    success_per_views,
    vggt_vram_median,
    vram_profile,
)

COLS = ["method", "n_views", "auc_30", "success", "runtime_s", "peak_vram_mb", "scene_type"]
NAN = np.nan


def toy() -> pd.DataFrame:
    # colmap at N=3: one success (0.6) and one failure (0.0, no runtime, no VRAM)
    return pd.DataFrame(
        [
            ("vggt", 3, 1.0, True, 1.0, 3000.0, "easy"),
            ("vggt", 3, 0.8, True, 3.0, 3200.0, "reflective"),
            ("colmap", 3, 0.6, True, 10.0, NAN, "easy"),
            ("colmap", 3, 0.0, False, NAN, NAN, "reflective"),
            ("vggt", 10, 0.9, True, 2.0, 4000.0, "easy"),
            ("colmap", 10, 0.5, True, 20.0, NAN, "easy"),
        ],
        columns=COLS,
    )


def profile(*extra_rows: dict) -> dict:
    rows = [
        {"n_frames": 1, "ok": True, "peak_allocated_mb": 3000.0},
        {"n_frames": 8, "ok": True, "peak_allocated_mb": 4000.0},
        *extra_rows,
    ]
    return {"profiles": {"bf16": {"total_vram_mb": 8000.0, "rows": rows}}}


def auc_stats() -> pd.DataFrame:
    return per_views(toy(), "auc_30").set_index(["method", "n_views"])


def test_mean_includes_failed_runs_as_zero() -> None:
    out = auc_stats()
    assert out.loc[("colmap", 3), "center"] == pytest.approx(0.3)  # (0.6 + 0.0) / 2
    assert out.loc[("vggt", 3), "center"] == pytest.approx(0.9)
    assert out.loc[("colmap", 3), "n_runs"] == 2


def test_std_band_is_clipped_to_unit_interval() -> None:
    out = auc_stats()
    std = np.std([0.6, 0.0], ddof=1)
    assert out.loc[("colmap", 3), "lo"] == pytest.approx(max(0.0, 0.3 - std))
    assert out.loc[("vggt", 3), "hi"] == pytest.approx(1.0)  # 0.9 + std > 1, clipped
    assert (out.lo >= 0).all() and (out.hi <= 1).all()


def test_single_run_has_zero_width_band() -> None:
    out = auc_stats()
    row = out.loc[("vggt", 10)]
    assert row.lo == row.center == row.hi == pytest.approx(0.9)


def test_median_and_iqr_ignore_nan_runtimes() -> None:
    out = per_views(toy(), "runtime_s", stat="median").set_index(["method", "n_views"])
    assert out.loc[("colmap", 3), "center"] == pytest.approx(10.0)  # nan dropped
    assert out.loc[("colmap", 3), "n_runs"] == 1
    assert out.loc[("vggt", 3), "center"] == pytest.approx(2.0)
    assert out.loc[("vggt", 3), "lo"] == pytest.approx(1.5)  # 25th pct of [1, 3]
    assert out.loc[("vggt", 3), "hi"] == pytest.approx(2.5)


def test_success_rate_per_views() -> None:
    out = success_per_views(toy()).set_index(["method", "n_views"])["center"]
    assert out[("colmap", 3)] == pytest.approx(0.5)
    assert out[("vggt", 3)] == pytest.approx(1.0)
    assert out[("colmap", 10)] == pytest.approx(1.0)


def test_by_scene_type_filters_to_n_and_covers_every_pair() -> None:
    out = by_scene_type(toy(), n_views=3).set_index(["scene_type", "method"])
    assert set(out.index) == {
        ("easy", "vggt"),
        ("easy", "colmap"),
        ("reflective", "vggt"),
        ("reflective", "colmap"),
    }
    assert out.loc[("reflective", "colmap"), "auc_30"] == pytest.approx(0.0)
    assert out.loc[("reflective", "colmap"), "success"] == pytest.approx(0.0)
    assert out.loc[("easy", "vggt"), "auc_30"] == pytest.approx(1.0)  # N=10 row excluded


def test_vram_profile_drops_failed_rows() -> None:
    failed = {"n_frames": 72, "ok": False, "peak_allocated_mb": 7900.0}
    frames, peak, total = vram_profile(profile(failed))
    assert list(frames) == [1, 8]
    assert list(peak) == [3000.0, 4000.0]
    assert total == 8000.0


def test_vggt_vram_median_ignores_colmap_and_nan() -> None:
    assert vggt_vram_median(toy()) == {3: 3100.0, 10: 4000.0}


def _load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "04_make_figures.py"
    spec = importlib.util.spec_from_file_location("make_figures", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_make_figures_writes_every_png_and_meta(tmp_path: Path) -> None:
    toy().to_csv(tmp_path / "b.csv", index=False)
    (tmp_path / "p.json").write_text(json.dumps(profile()))
    out = tmp_path / "figs"
    mod = _load_script()
    mod.main(
        ["--csv", str(tmp_path / "b.csv"), "--profile", str(tmp_path / "p.json")]
        + ["--outdir", str(out), "--scene-type-n", "3"]
    )
    for name in [*mod.FIGURES.values(), "figures_meta.json"]:
        assert (out / name).stat().st_size > 0, name


def test_group_with_no_measurements_is_omitted_not_a_crash() -> None:
    out = per_views(toy(), "peak_vram_mb", stat="median")
    assert set(out.method) == {"vggt"}  # colmap has no VRAM values at all

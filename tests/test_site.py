"""Demo-site builder: scene discovery, asset shrinking, results table and page rendering."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vggt_bench.export.to_ply import read_ply, write_ply
from vggt_bench.site import (
    SPLAT_ROW,
    TEMPLATES,
    build_site,
    check_budget,
    discover_scenes,
    downsample_points,
    prune_splat,
    results_table,
)

KW = {
    "csv": Path("results/benchmark.csv"),
    "figure": Path("results/figures/auc30_vs_views.png"),
}
ROW = SPLAT_ROW.itemsize  # antimatter15 .splat row: 32 bytes


def write_scene(root: Path, name: str, *, vggt_failed: bool = False, splat: str = "ok") -> Path:
    d = root / name
    d.mkdir(parents=True)
    run = {
        "frames": {"used": 20},
        "vggt": {
            "failed": vggt_failed,
            "n_input": 20,
            "n_registered": 20,
            "runtime_s": 1.5,
            "peak_vram_mb": 4000.0,
        },
        "colmap": {
            "failed": False,
            "n_input": 20,
            "n_registered": 12,
            "runtime_s": 6.0,
            "peak_vram_mb": None,
        },
    }
    (d / "run.json").write_text(json.dumps(run))
    (d / "comparison.png").write_bytes(b"png")
    if not vggt_failed:
        pts = np.random.default_rng(0).normal(size=(500, 3))
        cols = np.random.default_rng(1).integers(0, 255, size=(500, 3))
        write_ply(d / "vggt_points.ply", pts, cols)
        write_ply(d / "colmap_points.ply", pts[:100], cols[:100])
    if splat == "ok":
        (d / "splat.json").write_text(json.dumps({"failed": False, "n_gaussians": 10}))
        (d / "splat.splat").write_bytes(bytes(ROW * 10))
    elif splat == "failed":
        (d / "splat.json").write_text(json.dumps({"failed": True, "failure_reason": "CUDA OOM"}))
    return d


def test_discover_skips_failed_vggt_and_tolerates_missing_splat(tmp_path):
    write_scene(tmp_path, "b_ok")
    write_scene(tmp_path, "a_nosplat", splat="none")
    write_scene(tmp_path, "c_splatfail", splat="failed")
    write_scene(tmp_path, "d_bad", vggt_failed=True)
    scenes = discover_scenes(tmp_path)
    assert [s.name for s in scenes] == ["a_nosplat", "b_ok", "c_splatfail"]
    by_name = {s.name: s for s in scenes}
    assert by_name["b_ok"].has_splat
    assert not by_name["a_nosplat"].has_splat
    assert not by_name["c_splatfail"].has_splat


def test_discover_ignores_underscore_prefixed_staging_dirs_only_when_asked(tmp_path):
    write_scene(tmp_path, "_staging")
    assert [s.name for s in discover_scenes(tmp_path)] == ["_staging"]
    assert discover_scenes(tmp_path, exclude={"_staging"}) == []


def test_downsample_is_deterministic_capped_and_keeps_colour_alignment():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(1000, 3)).astype(np.float32)
    cols = np.stack([np.arange(1000) % 256] * 3, axis=1).astype(np.uint8)
    p1, c1 = downsample_points(pts, cols, 100, seed=0)
    p2, c2 = downsample_points(pts, cols, 100, seed=0)
    assert len(p1) == 100
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(c1, c2)
    # each kept point still carries its own colour
    idx = np.array([np.flatnonzero((pts == p).all(axis=1))[0] for p in p1])
    np.testing.assert_array_equal(c1, cols[idx])


def test_downsample_returns_everything_when_under_cap():
    pts = np.zeros((10, 3), np.float32)
    p, c = downsample_points(pts, None, 100, seed=0)
    assert len(p) == 10 and c is None


def make_splat(opacities: list[int], scale: float = 1.0) -> bytes:
    rows = np.zeros(
        len(opacities),
        dtype=[("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)],
    )
    rows["scale"] = scale
    rows["rgba"][:, 3] = opacities
    rows["pos"][:, 0] = np.arange(len(opacities))
    return rows.tobytes()


def test_prune_splat_keeps_most_opaque_rows_and_valid_size(tmp_path):
    src, dst = tmp_path / "in.splat", tmp_path / "out.splat"
    src.write_bytes(make_splat([10, 250, 30, 200, 5]))
    kept = prune_splat(src, dst, 2)
    assert kept == 2
    data = dst.read_bytes()
    assert len(data) == 2 * ROW
    rows = np.frombuffer(
        data, dtype=[("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)]
    )
    assert sorted(rows["pos"][:, 0]) == [1.0, 3.0]  # the 250 and 200 opacity rows


def test_prune_splat_copies_when_under_cap(tmp_path):
    src, dst = tmp_path / "in.splat", tmp_path / "out.splat"
    src.write_bytes(make_splat([1, 2, 3]))
    assert prune_splat(src, dst, 100) == 3
    assert dst.read_bytes() == src.read_bytes()


def test_prune_splat_rejects_truncated_file(tmp_path):
    src = tmp_path / "bad.splat"
    src.write_bytes(bytes(ROW + 5))
    with pytest.raises(ValueError):
        prune_splat(src, tmp_path / "out.splat", 1)


def test_results_table_matches_the_csv_and_the_summary():
    df = pd.read_csv("results/benchmark.csv")
    table = results_table(df)
    assert set(table.columns) >= {
        "n_views",
        "method",
        "auc_30",
        "success",
        "runtime_s",
        "peak_vram_mb",
    }
    row = table[(table.method == "vggt") & (table.n_views == 3)].iloc[0]
    assert row.auc_30 == pytest.approx(df[(df.method == "vggt") & (df.n_views == 3)].auc_30.mean())
    assert row.success == pytest.approx(0.0 + 1.0)  # VGGT: 100% success at 3 views
    colmap3 = table[(table.method == "colmap") & (table.n_views == 3)].iloc[0]
    assert colmap3.auc_30 == 0.0 and colmap3.success == 0.0


def test_check_budget(tmp_path):
    (tmp_path / "a.bin").write_bytes(bytes(1000))
    check_budget(tmp_path, 2000)
    with pytest.raises(ValueError, match="over budget"):
        check_budget(tmp_path, 500)


def test_zero_scene_build_writes_landing_only(tmp_path):
    out = tmp_path / "site"
    manifest = build_site(
        tmp_path / "no_captures",
        out,
        csv=Path("results/benchmark.csv"),
        figure=Path("results/figures/auc30_vs_views.png"),
    )
    assert (out / "index.html").exists()
    assert not (out / "scenes").exists() or not any((out / "scenes").iterdir())
    assert manifest["scenes"] == {}
    assert "coming soon" in (out / "index.html").read_text().lower()


def test_build_site_full_scene_and_no_unfilled_placeholders(tmp_path):
    caps = tmp_path / "caps"
    write_scene(caps, "plaza")
    write_scene(caps, "statue", splat="none")
    out = tmp_path / "site"
    manifest = build_site(
        caps,
        out,
        csv=Path("results/benchmark.csv"),
        figure=Path("results/figures/auc30_vs_views.png"),
        max_points=50,
        max_gaussians=5,
    )
    plaza = out / "scenes" / "plaza"
    for f in ("index.html", "vggt.ply", "colmap.ply", "scene.splat", "comparison.png"):
        assert (plaza / f).exists(), f
    assert len(read_ply(plaza / "vggt.ply")[0]) == 50
    assert (plaza / "scene.splat").stat().st_size == 5 * ROW
    assert not (out / "scenes" / "statue" / "scene.splat").exists()
    assert set(manifest["scenes"]) == {"plaza", "statue"}
    for page in [
        out / "index.html",
        plaza / "index.html",
        out / "scenes" / "statue" / "index.html",
    ]:
        assert not re.search(r"\$\{?[a-z_]+\}?", page.read_text()), page


def test_rebuild_clears_stale_scene_dirs(tmp_path):
    caps = tmp_path / "caps"
    write_scene(caps, "old")
    out = tmp_path / "site"
    kw = {
        "csv": Path("results/benchmark.csv"),
        "figure": Path("results/figures/auc30_vs_views.png"),
    }
    build_site(caps, out, **kw)
    assert (out / "scenes" / "old").exists()
    build_site(caps, out, exclude={"old"}, **kw)
    assert not (out / "scenes" / "old").exists()


def test_scene_without_comparison_png_still_builds(tmp_path):
    caps = tmp_path / "caps"
    (write_scene(caps, "plaza") / "comparison.png").unlink()
    out = tmp_path / "site"
    build_site(caps, out, **KW)
    assert not (out / "scenes" / "plaza" / "comparison.png").exists()
    assert "No comparison image" in (out / "scenes" / "plaza" / "index.html").read_text()


def test_failed_build_leaves_previous_site_intact(tmp_path):
    caps = tmp_path / "caps"
    write_scene(caps, "plaza")
    out = tmp_path / "site"
    build_site(caps, out, **KW)
    before = (out / "index.html").read_text()
    with pytest.raises(ValueError, match="over budget"):
        build_site(caps, out, budget_mb=0.0001, **KW)
    assert (out / "scenes" / "plaza" / "index.html").exists()
    assert (out / "index.html").read_text() == before
    assert not (out / ".scenes_new").exists()


def test_scene_names_are_url_encoded_in_links(tmp_path):
    caps = tmp_path / "caps"
    write_scene(caps, "my room")
    out = tmp_path / "site"
    build_site(caps, out, **KW)
    assert "scenes/my%20room/index.html" in (out / "index.html").read_text()
    assert (
        'data-src="scene.splat"' in (out / "scenes" / "my room" / "index.html").read_text()
    )


def test_landing_quotes_experiment_size_from_the_csv(tmp_path):
    out = tmp_path / "site"
    build_site(tmp_path / "none", out, **KW)
    text = (out / "index.html").read_text()
    df = pd.read_csv("results/benchmark.csv")
    assert f"mean over {df.scene.nunique()} scenes" in text
    assert f"{df.n_views.min()} to {df.n_views.max()} views" in text


def test_committed_static_assets_match_their_templates():
    for name in ("site.css", "pointcloud.js", "splat.js"):
        committed = Path("site") / name
        assert committed.read_bytes() == (TEMPLATES / name).read_bytes(), (
            f"site/{name} is stale: rerun scripts/06_build_site.py"
        )

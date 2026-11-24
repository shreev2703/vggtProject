"""Static demo-site builder: discover captured scenes, shrink their assets, render the pages.

Pure logic lives here so it can be tested; `scripts/06_build_site.py` is a thin CLI over
`build_site`. Pages are filled from `string.Template` files in `site_templates/`, and a missing
placeholder raises instead of shipping a literal `$name` to the browser.
"""

from __future__ import annotations

import html
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from string import Template
from urllib.parse import quote

import numpy as np
import pandas as pd

from vggt_bench.eval.aggregate import per_views, success_per_views
from vggt_bench.export.to_ply import read_ply, write_ply
from vggt_bench.provenance import run_metadata

TEMPLATES = Path(__file__).parent / "site_templates"
STATIC_FILES = ("site.css", "pointcloud.js", "splat.js")
METHODS = (("VGGT", "vggt"), ("COLMAP", "colmap"))  # (label, key in the CSV)
DEFAULT_MAX_POINTS = 150_000
DEFAULT_MAX_GAUSSIANS = 200_000
DEFAULT_BUDGET_MB = 15.0
SPLAT_ROW = np.dtype([("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])


@dataclass
class Scene:
    name: str
    path: Path
    run: dict
    splat: dict | None = None

    @property
    def has_splat(self) -> bool:
        return (
            self.splat is not None
            and not self.splat.get("failed")
            and (self.path / "splat.splat").exists()
        )


def discover_scenes(
    captures_dir: Path, only: set[str] | None = None, exclude: set[str] | None = None
) -> list[Scene]:
    """Scenes under `captures_dir` whose VGGT run succeeded, sorted by name.

    A missing or failed splat is recorded (`has_splat` False) rather than dropping the scene.
    """
    if not captures_dir.is_dir():
        return []
    scenes = []
    for d in sorted(p for p in captures_dir.iterdir() if (p / "run.json").exists()):
        if (only and d.name not in only) or (exclude and d.name in exclude):
            continue
        run = json.loads((d / "run.json").read_text())
        if run["vggt"].get("failed") or not (d / "vggt_points.ply").exists():
            continue
        splat_json = d / "splat.json"
        splat = json.loads(splat_json.read_text()) if splat_json.exists() else None
        scenes.append(Scene(d.name, d, run, splat))
    return scenes


def downsample_points(
    points: np.ndarray, colors: np.ndarray | None, max_points: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray | None]:
    """Seeded random subset of at most `max_points`; colours stay attached to their points."""
    if len(points) <= max_points:
        return points, colors
    idx = np.sort(np.random.default_rng(seed).choice(len(points), max_points, replace=False))
    return points[idx], None if colors is None else colors[idx]


def prune_splat(src: Path, dst: Path, max_gaussians: int) -> int:
    """Copy an antimatter15 `.splat`, keeping the `max_gaussians` with the largest opacity x volume.

    Returns the number of Gaussians written.
    """
    size = src.stat().st_size
    if size % SPLAT_ROW.itemsize:
        raise ValueError(f"{src} is not a multiple of {SPLAT_ROW.itemsize} bytes")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if size // SPLAT_ROW.itemsize <= max_gaussians:
        shutil.copy(src, dst)
        return size // SPLAT_ROW.itemsize
    rows = np.frombuffer(src.read_bytes(), dtype=SPLAT_ROW)
    weight = rows["rgba"][:, 3].astype(np.float64) * rows["scale"].astype(np.float64).prod(axis=1)
    keep = np.sort(np.argpartition(weight, -max_gaussians)[-max_gaussians:])
    dst.write_bytes(rows[keep].tobytes())
    return max_gaussians


def check_budget(scene_dir: Path, max_bytes: int) -> int:
    """Total size of a built scene directory; raises if it exceeds `max_bytes`."""
    total = sum(f.stat().st_size for f in scene_dir.rglob("*") if f.is_file())
    if total > max_bytes:
        raise ValueError(
            f"{scene_dir.name} is {total / 1e6:.1f} MB, over budget {max_bytes / 1e6:.1f} MB"
        )
    return total


def results_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per (method, n_views): mean AUC@30, success rate, median runtime and median peak VRAM.

    Built from `eval.aggregate` so the site quotes the same statistics as the figures. Failed runs
    carry AUC 0 in the CSV, so the mean already scores them as zero.
    """
    keys = ["method", "n_views"]

    def centre(col: str, stat: str = "mean") -> pd.DataFrame:
        out = per_views(df, col, stat).reindex(columns=[*keys, "center"])
        return out.rename(columns={"center": col})

    table = centre("auc_30").merge(
        success_per_views(df).rename(columns={"center": "success"}), on=keys
    )
    for col in ("runtime_s", "peak_vram_mb"):
        table = table.merge(centre(col, "median"), on=keys, how="left")
    return table


def _gb(mb: float | None) -> str:
    return "n/a" if mb is None or pd.isna(mb) else f"{mb / 1024:.1f} GB"


def _seconds(s: float | None) -> str:
    return "n/a" if s is None or pd.isna(s) else f"{s:.1f} s"


def _registered(m: dict) -> str:
    return f"{m.get('n_registered', 0)}/{m.get('n_input', 0)}"


def _results_html(table: pd.DataFrame) -> str:
    head = "<tr><th>Views</th>" + "".join(
        f"<th>{label} AUC@30</th><th>{label} success</th>"
        f"<th>{label} median runtime</th><th>{label} median VRAM</th>"
        for label, _ in METHODS
    )
    rows = [head + "</tr>"]
    for n in sorted(table.n_views.unique()):
        cells = [f"<td>{n}</td>"]
        for _, key in METHODS:
            r = table[(table.method == key) & (table.n_views == n)]
            if r.empty:
                cells += ["<td>n/a</td>"] * 4
                continue
            r = r.iloc[0]
            cells += [
                f"<td>{r.auc_30:.3f}</td>",
                f"<td>{r.success:.0%}</td>",
                f"<td>{_seconds(r.runtime_s)}</td>",
                f"<td>{_gb(r.peak_vram_mb)}</td>",
            ]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table>" + "".join(rows) + "</table>"


def _method_row(label: str, m: dict) -> str:
    if m.get("failed"):
        reason = html.escape(str(m.get("failure_reason")))
        return f"<tr><td>{label}</td><td colspan=4>failed: {reason}</td></tr>"
    return (
        f"<tr><td>{label}</td><td>{_registered(m)}</td>"
        f"<td>{_seconds(m.get('runtime_s'))}</td><td>{_gb(m.get('peak_vram_mb'))}</td>"
        f"<td>{m.get('n_points', 0):,}</td></tr>"
    )


def _load(name: str) -> Template:
    return Template((TEMPLATES / name).read_text())


def _facts(df: pd.DataFrame) -> dict[str, int]:
    """Experiment size quoted on the landing page, taken from the CSV rather than typed in."""
    per_cell = df.groupby(["scene", "method", "n_views"]).size().max()
    return {
        "n_scenes": int(df.scene.nunique()),
        "min_views": int(df.n_views.min()),
        "max_views": int(df.n_views.max()),
        "n_samplings": int(per_cell),
    }


def render_landing(df: pd.DataFrame, scenes: list[Scene]) -> str:
    if scenes:
        cards = "".join(
            f'<li><a href="scenes/{quote(s.name)}/index.html">{html.escape(s.name)}</a>: '
            f"{s.run['frames']['used']} frames, "
            f"VGGT {_registered(s.run['vggt'])} registered, "
            f"COLMAP {_registered(s.run['colmap'])}"
            "</li>"
            for s in scenes
        )
        scene_list = f"<ul>{cards}</ul>"
    else:
        scene_list = "<p>Scenes: coming soon.</p>"
    return _load("landing.html").substitute(
        results=_results_html(results_table(df)), scenes=scene_list, **_facts(df)
    )


def render_scene(scene: Scene, splat_gaussians: int | None) -> str:
    name = html.escape(scene.name)
    if scene.has_splat:
        splat = (
            '<div id="splat" aria-label="Interactive Gaussian splat viewer" '
            'data-src="scene.splat"></div>'
            f"<p class=note>Drag to orbit, right-drag to pan, scroll to zoom. "
            f"{splat_gaussians:,} Gaussians. Trained from the VGGT poses and "
            "points, so it shows how well the splat fits the capture; it is not a novel-view "
            "quality score.</p>"
        )
    else:
        reason = (scene.splat or {}).get("failure_reason", "no splat was trained for this scene")
        splat = f"<p class=note>No splat for this scene: {html.escape(str(reason))}</p>"
    comparison = (
        '<img class="wide" src="comparison.png" alt="VGGT point cloud above COLMAP point cloud">'
        if (scene.path / "comparison.png").exists()
        else "<p class=note>No comparison image for this scene.</p>"
    )
    colmap_pc = "colmap.ply" if (scene.path / "colmap_points.ply").exists() else ""
    capture = (
        '<video class="wide" src="capture.mp4" autoplay loop muted playsinline controls '
        'aria-label="Sped-up phone video of the captured scene"></video>'
        "<p class=note>The phone video these frames come from, sped up.</p>"
        if (scene.path / "capture.mp4").exists()
        else ""
    )
    methods = _method_row("VGGT", scene.run["vggt"]) + _method_row("COLMAP", scene.run["colmap"])
    return _load("scene.html").substitute(
        name=name,
        frames=scene.run["frames"]["used"],
        colmap_ply=colmap_pc,
        capture=capture,
        splat=splat,
        comparison=comparison,
        methods=methods,
    )


def _build_scene(s: Scene, d: Path, max_points: int, max_gaussians: int, budget_bytes: int) -> dict:
    d.mkdir(parents=True)
    for src, dst in (("vggt_points.ply", "vggt.ply"), ("colmap_points.ply", "colmap.ply")):
        if (s.path / src).exists():
            pts, cols = read_ply(s.path / src)
            pts, cols = downsample_points(pts, cols, max_points)
            write_ply(d / dst, pts, cols)
    n_gauss = None
    if s.has_splat:
        n_gauss = prune_splat(s.path / "splat.splat", d / "scene.splat", max_gaussians)
    if (s.path / "comparison.png").exists():
        shutil.copy(s.path / "comparison.png", d / "comparison.png")
    if (s.path / "capture.mp4").exists():
        shutil.copy(s.path / "capture.mp4", d / "capture.mp4")
    (d / "index.html").write_text(render_scene(s, n_gauss))
    return {"bytes": check_budget(d, budget_bytes), "n_gaussians": n_gauss}


def build_site(
    captures_dir: Path,
    out: Path,
    csv: Path,
    figure: Path,
    max_points: int = DEFAULT_MAX_POINTS,
    max_gaussians: int = DEFAULT_MAX_GAUSSIANS,
    budget_mb: float = DEFAULT_BUDGET_MB,
    only: set[str] | None = None,
    exclude: set[str] | None = None,
) -> dict:
    """Write `out/index.html`, `out/scenes/<name>/...` and a manifest. `out/vendor` is left alone.

    Everything is built and checked first (scenes go into `out/.scenes_new`); `out` is only
    changed once the whole build has succeeded, so a failure leaves the previous site intact.
    """
    df = pd.read_csv(csv)
    figure_bytes = figure.read_bytes()
    scenes = discover_scenes(captures_dir, only, exclude)
    out.mkdir(parents=True, exist_ok=True)
    staging = out / ".scenes_new"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()

    sizes: dict[str, dict] = {}
    try:
        for s in scenes:
            sizes[s.name] = _build_scene(
                s, staging / s.name, max_points, max_gaussians, int(budget_mb * 1e6)
            )
        landing = render_landing(df, scenes)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    shutil.rmtree(out / "scenes", ignore_errors=True)
    staging.rename(out / "scenes")
    for f in STATIC_FILES:
        shutil.copy(TEMPLATES / f, out / f)
    (out / "figures").mkdir(exist_ok=True)
    (out / "figures" / "auc30_vs_views.png").write_bytes(figure_bytes)
    (out / ".nojekyll").touch()
    (out / "index.html").write_text(landing)
    manifest = run_metadata(
        {"max_points": max_points, "max_gaussians": max_gaussians, "budget_mb": budget_mb}
    ) | {"scenes": sizes}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest

"""Build the static demo site from results/ and results/captures/.

    python scripts/06_build_site.py                       # writes site/
    python scripts/06_build_site.py --out /tmp/preview    # build somewhere else
    python scripts/06_build_site.py --exclude _staging    # leave a scene out

Writes site/index.html (headline figure, results table, scene list), one page per scene under
site/scenes/<name>/ (downsampled point clouds, pruned splat, comparison.png) and manifest.json
(config, git hash, per-scene sizes). site/vendor/ (three.js, antimatter15/splat) is left untouched.
A scene over --budget-mb fails the build. Serve locally with `python -m http.server -d site`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_bench.site import (
    DEFAULT_BUDGET_MB,
    DEFAULT_MAX_GAUSSIANS,
    DEFAULT_MAX_POINTS,
    build_site,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--captures", type=Path, default=Path("results/captures"))
    p.add_argument("--out", type=Path, default=Path("site"))
    p.add_argument("--csv", type=Path, default=Path("results/benchmark.csv"))
    p.add_argument("--figure", type=Path, default=Path("results/figures/auc30_vs_views.png"))
    p.add_argument("--max-points", type=int, default=DEFAULT_MAX_POINTS)
    p.add_argument("--max-gaussians", type=int, default=DEFAULT_MAX_GAUSSIANS)
    p.add_argument(
        "--budget-mb", type=float, default=DEFAULT_BUDGET_MB, help="max size of one scene directory"
    )
    p.add_argument("--only", nargs="+", help="build just these scenes")
    p.add_argument("--exclude", nargs="+", help="skip these scenes")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_site(
        args.captures,
        args.out,
        csv=args.csv,
        figure=args.figure,
        max_points=args.max_points,
        max_gaussians=args.max_gaussians,
        budget_mb=args.budget_mb,
        only=set(args.only) if args.only else None,
        exclude=set(args.exclude) if args.exclude else None,
    )
    for name, info in manifest["scenes"].items():
        print(f"{name}: {info['bytes'] / 1e6:.1f} MB, splat {info['n_gaussians'] or 'none'}")
    print(f"-> {args.out} ({len(manifest['scenes'])} scenes)")


if __name__ == "__main__":
    main()

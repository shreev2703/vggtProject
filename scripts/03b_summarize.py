"""Summarise results/benchmark.csv into results/benchmark_summary.md.

Every number the README quotes about the benchmark comes from this file. Failed
runs count as AUC 0 in the means, and the success rate is reported beside them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def to_md(df: pd.DataFrame) -> str:
    """GitHub markdown table without the optional `tabulate` dependency."""
    head = [df.index.name or "", *map(str, df.columns)]
    rows = [[str(i), *("" if pd.isna(v) else f"{v:g}" for v in r)] for i, r in df.iterrows()]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def table(df: pd.DataFrame, index: str) -> str:
    g = df.groupby([index, "method"])
    out = pd.DataFrame(
        {
            "auc_30": g.auc_30.mean(),
            "auc_30_std": g.auc_30.std(),
            "auc_10": g.auc_10.mean(),
            "success": g.success.mean(),
            "runs": g.size(),
        }
    ).unstack("method")
    out = out.round(3)
    out.columns = [f"{m} {c}" for c, m in out.columns]
    return to_md(out.sort_index(axis=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=Path("results/benchmark.csv"))
    p.add_argument("--out", type=Path, default=Path("results/benchmark_summary.md"))
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    runtime = df.groupby("method")[["runtime_s", "peak_vram_mb"]].median().round(1)
    parts = [
        f"# Benchmark summary ({len(df)} runs from `{args.csv}`)",
        "AUC@30 is the mean over all runs; failed runs count as 0.",
        "## By number of views",
        table(df, "n_views"),
        "## By scene",
        table(df.assign(scene=df.scene + " (" + df.scene_type + ")"), "scene"),
        "## By sampling (random seeds vs wide-baseline)",
        table(df, "sampling"),
        "## Median runtime (s) and peak VRAM (MB)",
        to_md(runtime),
    ]
    args.out.write_text("\n\n".join(parts) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

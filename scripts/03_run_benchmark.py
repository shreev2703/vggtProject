"""Phase 3: sweep VGGT and COLMAP over scenes, view counts and samplings.

One row per run is appended to results/benchmark.csv the moment it finishes, and
runs already in the CSV are skipped, so an interrupted sweep resumes where it
stopped. Both methods are handed the identical frame subset. A failed run
(including CUDA OOM) is recorded as a row with success=False and AUC 0, not a
crash. Reference poses are dense-COLMAP pseudo-ground-truth (see the README).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vggt_bench.eval.pose_metrics import evaluate_poses  # noqa: E402
from vggt_bench.eval.sweep import append_row, done_keys, plan_runs, run_key  # noqa: E402
from vggt_bench.io.datasets import Scene, load_dataset_config, load_scene  # noqa: E402
from vggt_bench.io.frames import sample_frames  # noqa: E402
from vggt_bench.provenance import git_commit, run_metadata  # noqa: E402

SEED = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--scenes", nargs="+", help="subset of the configured scenes")
    p.add_argument("--methods", nargs="+", default=["vggt", "colmap"], choices=["vggt", "colmap"])
    p.add_argument("--view-counts", nargs="+", type=int, help="override the configured N values")
    p.add_argument("--seeds", nargs="+", type=int, help="override the configured sampling seeds")
    p.add_argument("--csv", type=Path, default=Path("results/benchmark.csv"))
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    p.add_argument("--limit", type=int, help="stop after this many new runs")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    cfg = load_dataset_config()
    if args.scenes:
        unknown = sorted(set(args.scenes) - set(cfg["scenes"]))
        if unknown:
            raise SystemExit(f"unknown scenes {unknown}; configured: {sorted(cfg['scenes'])}")
        cfg["scenes"] = {s: cfg["scenes"][s] for s in args.scenes}
    if args.view_counts:
        cfg["view_counts"] = args.view_counts
    if args.seeds:
        cfg["seeds"] = args.seeds
    return cfg


def make_method(name: str):
    if name == "vggt":
        from vggt_bench.methods.vggt_method import VGGTMethod

        return VGGTMethod(seed=SEED)
    from vggt_bench.methods.colmap_method import ColmapMethod

    return ColmapMethod()


def free_gpu() -> None:
    try:
        from vggt_bench import profiling

        profiling.free_gpu_memory()
    except Exception:  # noqa: BLE001 - cleanup must never mask the original failure
        pass


def run_one(method, frames: list[Path], scene: Scene) -> dict:
    """Reconstruct and score; every failure mode comes back as a row, never raises."""
    import torch

    t0 = time.perf_counter()
    try:
        recon = method.reconstruct(frames)
    except torch.cuda.OutOfMemoryError:
        free_gpu()
        return _failed("cuda_oom", len(frames), time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001 - a bad run must not kill a multi-hour sweep
        free_gpu()
        return _failed(f"{type(exc).__name__}: {exc}"[:300], len(frames), time.perf_counter() - t0)

    m = evaluate_poses(recon, scene)
    reason = recon.failure_reason
    if reason is None and m["n_pairs"] == 0:
        reason = "fewer_than_3_cameras_registered"
    free_gpu()
    return m | {
        "success": reason is None,
        "n_input": recon.n_input,
        "n_registered": recon.n_registered,
        "runtime_s": recon.runtime_s,
        "peak_vram_mb": recon.peak_vram_mb,
        "failure_reason": reason,
    }


def _failed(reason: str, n_input: int, elapsed: float) -> dict:
    return {
        "auc_5": 0.0, "auc_10": 0.0, "auc_30": 0.0, "n_pairs": 0, "frac_registered": 0.0,
        "success": False, "n_input": n_input, "n_registered": 0,
        "runtime_s": elapsed, "failure_reason": reason,
    }  # fmt: skip


def write_meta(csv_path: Path, cfg: dict, args: argparse.Namespace, methods: dict) -> None:
    meta_path = csv_path.with_name(f"{csv_path.stem}_meta.json")
    meta = {"invocations": []}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    config = {
        "dataset": {k: v for k, v in cfg.items() if k != "scenes"}
        | {"scenes": list(cfg["scenes"])},
        "methods": {n: m.config() for n, m in methods.items()},
        "seed": SEED,
        "reference": "pseudo-ground-truth: dense COLMAP over all images (Mip-NeRF 360 sparse/0)",
        "cli": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    meta["invocations"].append(run_metadata(config))
    meta_path.write_text(json.dumps(meta, indent=2))


def main() -> None:
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)

    cfg = build_config(args)
    max_frames = None
    if "vggt" in args.methods:
        from vggt_bench.profiling import max_frames_from_profile

        max_frames = max_frames_from_profile("bf16")
    plan = plan_runs(cfg, args.methods, max_frames)
    done = done_keys(args.csv)
    pending = [r for r in plan if run_key(r) not in done]
    print(f"{len(plan)} planned, {len(plan) - len(pending)} already done, {len(pending)} to run")
    if args.dry_run or not pending:
        return
    if args.limit is not None:
        pending = pending[: args.limit]

    methods = {n: make_method(n) for n in args.methods}
    write_meta(args.csv, cfg, args, methods)
    commit = git_commit()
    scenes: dict[str, Scene] = {}

    for i, run in enumerate(pending, 1):
        name = run["scene"]
        if name not in scenes:
            scenes[name] = load_scene(name, cfg)
        scene = scenes[name]
        # Same subset for every method: it depends only on (scene, N, seed).
        frames = sample_frames(scene.images, run["n_views"], run["seed"])
        method = methods[run["method"]]
        load_s = None
        if run["method"] == "vggt" and method.model is None:
            try:
                method.load()
            except Exception as exc:  # noqa: BLE001
                raise SystemExit(f"VGGT failed to load: {exc}") from exc
        if run["method"] == "vggt":
            load_s = method.load_time_s

        result = run_one(method, frames, scene)
        row = run | result | {
            "model_load_s": load_s,
            "git_commit": commit,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        append_row(args.csv, row)
        tag = "ok " if row["success"] else "FAIL"
        print(
            f"[{i}/{len(pending)}] {tag} {name} {run['method']:6s} N={run['n_views']:<2d} "
            f"{run['sampling']}{'' if run['seed'] is None else run['seed']} "
            f"auc30={row['auc_30']:.3f} reg={row['n_registered']}/{row['n_input']} "
            f"{row.get('failure_reason') or ''}",
            flush=True,
        )


if __name__ == "__main__":
    main()

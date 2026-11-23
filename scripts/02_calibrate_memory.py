"""Phase 1: measure VGGT peak VRAM and runtime as the number of input frames grows.

Doubles N until the GPU runs out of memory, then bisects to find the largest N
that fits. Frames are cycled from one example scene: memory and runtime depend
on the frame count and resolution, not on image content.

Each N runs in a fresh subprocess. Under WSL, running out of GPU memory can
surface as a CUDA driver error instead of a clean OutOfMemoryError, and after
that the CUDA context in that process cannot be trusted for further runs.

Writes results/memory_profile.json (one profile per setting; re-running with
other flags adds to the file) and results/figures/vggt_memory_vs_frames.png.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

EXAMPLES = Path("third_party/vggt/examples")
FIGURE_PATH = Path("results/figures/vggt_memory_vs_frames.png")
DOUBLING = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
RESULT_PREFIX = "RESULT "


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--scene", default="kitchen")
    p.add_argument("--weights-dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument(
        "--allocator",
        choices=["expandable", "default"],
        default="expandable",
        help="PyTorch CUDA allocator mode (expandable_segments on or off)",
    )
    p.add_argument("--repeats", type=int, default=3, help="timed runs per N (median reported)")
    p.add_argument("--bisect-steps", type=int, default=4)
    p.add_argument("--child", type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def profile_label(args: argparse.Namespace) -> str:
    return args.weights_dtype + ("" if args.allocator == "expandable" else "_default_alloc")


def frames_for(scene: str, n: int) -> list[Path]:
    pool = sorted((EXAMPLES / scene / "images").iterdir())
    return [pool[i % len(pool)] for i in range(n)]


# --- child process: one N, one fresh CUDA context ---------------------------------


def measure_child(args: argparse.Namespace) -> dict[str, Any]:
    from vggt_bench import profiling
    from vggt_bench.methods.vggt_method import VGGTMethod

    n = args.child
    method = VGGTMethod(weights_dtype=args.weights_dtype)
    method.load()
    method.reconstruct(frames_for(args.scene, 2))  # warm-up

    row: dict[str, Any] = {
        "n_frames": n,
        "weights_mb": method.weights_mb,
        "total_vram_mb": profiling.total_vram_mb(),
        "vggt": method.config(),
    }
    runtimes: list[float] = []
    peaks: list[float] = []
    for _ in range(args.repeats):
        recon = method.reconstruct(frames_for(args.scene, n))
        if recon.failed:
            return row | {"ok": False, "failure_reason": recon.failure_reason}
        runtimes.append(recon.runtime_s)
        peaks.append(recon.peak_vram_mb)

    row |= {
        "runtime_s_median": statistics.median(runtimes),
        "runtime_s_all": runtimes,
        "peak_allocated_mb": max(peaks),
        "peak_reserved_mb": profiling.peak_reserved_mb(),
        "image_hw": recon.image_hw,
    }
    if row["peak_reserved_mb"] > row["total_vram_mb"]:
        # The Windows driver can spill into system RAM instead of failing. That "works"
        # but is much slower, so it counts as not fitting.
        return row | {"ok": False, "failure_reason": "exceeds_vram_spilled"}
    return row | {"ok": True}


# --- parent process: the sweep ------------------------------------------------------


def run_one(args: argparse.Namespace, n: int) -> dict[str, Any]:
    env = os.environ | {
        "PYTORCH_CUDA_ALLOC_CONF": f"expandable_segments:{args.allocator == 'expandable'}"
    }
    cmd = [
        sys.executable,
        __file__,
        "--child",
        str(n),
        "--scene",
        args.scene,
        "--weights-dtype",
        args.weights_dtype,
        "--repeats",
        str(args.repeats),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line.removeprefix(RESULT_PREFIX))
    err = [ln for ln in proc.stderr.splitlines() if ln.strip()]
    return {
        "n_frames": n,
        "ok": False,
        "failure_reason": "process_crashed",
        "error": err[-1] if err else f"exit code {proc.returncode}",
    }


def sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def run(n: int) -> bool:
        row = run_one(args, n)
        rows.append(row)
        if row["ok"]:
            status = f"{row['runtime_s_median']:.2f}s, peak {row['peak_allocated_mb']:.0f} MB"
        else:
            status = f"{row['failure_reason']} {row.get('error', '')}".strip()
        print(f"  N={n:4d}: {status}", flush=True)
        return row["ok"]

    last_ok, first_fail = 0, None
    for n in DOUBLING:
        if run(n):
            last_ok = n
        else:
            first_fail = n
            break

    if first_fail is not None:
        lo, hi = last_ok, first_fail
        for _ in range(args.bisect_steps):
            if hi - lo <= 1:
                break
            mid = (lo + hi) // 2
            if run(mid):
                lo = mid
            else:
                hi = mid
    return sorted(rows, key=lambda r: r["n_frames"])


def plot_profiles(profiles: dict[str, dict]) -> None:
    from vggt_bench import plotting

    plotting.apply_style()
    fig, (ax_mem, ax_time) = plotting.plt.subplots(1, 2, figsize=(10, 4))
    styles = {"bf16": "-", "fp32": "--", "bf16_default_alloc": ":"}
    color = plotting.METHOD_COLORS["vggt"]
    for label, prof in profiles.items():
        ok = [r for r in prof["rows"] if r["ok"]]
        n = [r["n_frames"] for r in ok]
        name = label.replace("_default_alloc", " weights, default allocator")
        name = name if "allocator" in name else f"{name} weights"
        ls = styles.get(label, "-.")
        ax_mem.plot(
            n,
            [r["peak_allocated_mb"] / 1024 for r in ok],
            ls,
            marker="o",
            ms=3,
            color=color,
            label=f"{name} (max {prof['max_ok_frames']})",
        )
        ax_time.plot(
            n, [r["runtime_s_median"] for r in ok], ls, marker="o", ms=3, color=color, label=name
        )
    any_prof = next(iter(profiles.values()))
    ax_mem.axhline(any_prof["total_vram_mb"] / 1024, color=plotting.LIMIT, lw=1, label="total VRAM")
    ax_mem.set_xlabel("input frames")
    ax_mem.set_ylabel("peak allocated VRAM (GB)")
    ax_mem.set_title("Peak GPU memory")
    ax_mem.legend(fontsize=8)
    ax_time.set_xlabel("input frames")
    ax_time.set_ylabel("forward pass (s)")
    ax_time.set_title("Runtime (warm GPU, model load excluded)")
    ax_time.legend(fontsize=8)
    hw = any_prof["image_hw"]
    fig.suptitle(f"VGGT-1B on {any_prof['gpu']}, {hw[1]}x{hw[0]} input")
    plotting.save(fig, FIGURE_PATH)


def main() -> None:
    args = parse_args()
    if args.child is not None:
        print(RESULT_PREFIX + json.dumps(measure_child(args)), flush=True)
        return

    from vggt_bench import profiling
    from vggt_bench.provenance import run_metadata

    label = profile_label(args)
    print(f"profile '{label}'")
    rows = sweep(args)
    ok = [r for r in rows if r["ok"]]
    ref = ok[0] if ok else rows[0]

    profile = run_metadata({k: v for k, v in vars(args).items() if k != "child"})
    profile |= {
        "total_vram_mb": ref.get("total_vram_mb"),
        "weights_mb": ref.get("weights_mb"),
        "image_hw": ref.get("image_hw"),
        "vggt": ref.get("vggt"),
        "max_ok_frames": max((r["n_frames"] for r in ok), default=0),
        "min_failed_frames": min((r["n_frames"] for r in rows if not r["ok"]), default=None),
        "rows": rows,
    }

    path = profiling.DEFAULT_PROFILE
    data = json.loads(path.read_text()) if path.exists() else {"profiles": {}}
    data["profiles"][label] = profile
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    plot_profiles(data["profiles"])
    print(f"max frames that fit: {profile['max_ok_frames']} -> {path}, {FIGURE_PATH}")


if __name__ == "__main__":
    main()

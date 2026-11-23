"""Phase 1: measure VGGT peak VRAM and runtime as the number of input frames grows.

Doubles N until the GPU runs out of memory, then bisects to find the largest N
that fits. Frames are cycled from one example scene: memory and runtime depend
on the frame count and resolution, not on image content.

Writes results/memory_profile.json (one profile per weights dtype; re-running
with another dtype adds to the file) and results/figures/vggt_memory_vs_frames.png.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from vggt_bench import plotting, profiling
from vggt_bench.methods.vggt_method import VGGTMethod
from vggt_bench.provenance import run_metadata

EXAMPLES = Path("third_party/vggt/examples")
PROFILE_PATH = profiling.DEFAULT_PROFILE
FIGURE_PATH = Path("results/figures/vggt_memory_vs_frames.png")
DOUBLING = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--scene", default="kitchen")
    p.add_argument("--weights-dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--repeats", type=int, default=3, help="timed runs per N (median reported)")
    p.add_argument("--bisect-steps", type=int, default=4)
    return p.parse_args()


def frame_pool(scene: str) -> list[Path]:
    return sorted((EXAMPLES / scene / "images").iterdir())


def frames_for(pool: list[Path], n: int) -> list[Path]:
    return [pool[i % len(pool)] for i in range(n)]


def measure(method: VGGTMethod, pool: list[Path], n: int, repeats: int) -> dict[str, Any]:
    frames = frames_for(pool, n)
    runtimes: list[float] = []
    peaks: list[float] = []
    for _ in range(repeats):
        recon = method.reconstruct(frames)
        if recon.failed:
            return {"n_frames": n, "ok": False, "failure_reason": recon.failure_reason}
        runtimes.append(recon.runtime_s)
        peaks.append(recon.peak_vram_mb)
    reserved = profiling.peak_reserved_mb()
    if reserved > profiling.total_vram_mb():
        # Under WSL/Windows the driver can spill to system RAM instead of raising OOM.
        # That "works" but is many times slower, so it counts as not fitting.
        return {
            "n_frames": n,
            "ok": False,
            "failure_reason": "exceeds_vram_spilled",
            "peak_reserved_mb": reserved,
            "runtime_s_median": statistics.median(runtimes),
        }
    return {
        "n_frames": n,
        "ok": True,
        "runtime_s_median": statistics.median(runtimes),
        "runtime_s_all": runtimes,
        "peak_allocated_mb": max(peaks),
        "peak_reserved_mb": reserved,
        "image_hw": recon.image_hw,
    }


def calibrate(method: VGGTMethod, pool: list[Path], args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []

    def run(n: int) -> bool:
        row = measure(method, pool, n, args.repeats)
        rows.append(row)
        status = (
            f"{row['runtime_s_median']:.2f}s, peak {row['peak_allocated_mb']:.0f} MB"
            if row["ok"]
            else row["failure_reason"]
        )
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


def plot_profiles(profiles: dict[str, dict], total_mb: float) -> None:
    plotting.apply_style()
    fig, (ax_mem, ax_time) = plotting.plt.subplots(1, 2, figsize=(10, 4))
    styles = {"fp32": "-", "bf16": "--"}
    for dtype, prof in profiles.items():
        ok = [r for r in prof["rows"] if r["ok"]]
        n = [r["n_frames"] for r in ok]
        label = f"{dtype} weights (max {prof['max_ok_frames']} frames)"
        color = plotting.METHOD_COLORS["vggt"]
        ax_mem.plot(n, [r["peak_allocated_mb"] / 1024 for r in ok], styles[dtype], marker="o",
                    color=color, label=label)
        ax_time.plot(n, [r["runtime_s_median"] for r in ok], styles[dtype], marker="o",
                     color=color, label=f"{dtype} weights")
    ax_mem.axhline(total_mb / 1024, color=plotting.LIMIT, lw=1, label="total VRAM")
    ax_mem.set_xlabel("input frames")
    ax_mem.set_ylabel("peak VRAM (GB)")
    ax_mem.set_title("Peak GPU memory")
    ax_mem.legend()
    ax_time.set_xlabel("input frames")
    ax_time.set_ylabel("forward pass (s)")
    ax_time.set_title("Runtime (warm GPU, model load excluded)")
    ax_time.legend()
    gpu = next(iter(profiles.values()))["gpu"]
    hw = next(iter(profiles.values()))["image_hw"]
    fig.suptitle(f"VGGT-1B on {gpu}, {hw[1]}x{hw[0]} input")
    plotting.save(fig, FIGURE_PATH)


def main() -> None:
    args = parse_args()
    pool = frame_pool(args.scene)
    method = VGGTMethod(weights_dtype=args.weights_dtype)
    method.load()
    total_mb = profiling.total_vram_mb()
    print(f"weights {args.weights_dtype}: {method.weights_mb:.0f} MB of {total_mb:.0f} MB VRAM")

    method.reconstruct(frames_for(pool, 2))  # warm-up
    rows = calibrate(method, pool, args)
    ok = [r for r in rows if r["ok"]]

    profile = run_metadata(vars(args))
    profile |= {
        "gpu": profile.get("gpu"),
        "total_vram_mb": total_mb,
        "weights_mb": method.weights_mb,
        "image_hw": ok[0]["image_hw"] if ok else None,
        "vggt": method.config(),
        "max_ok_frames": max((r["n_frames"] for r in ok), default=0),
        "min_oom_frames": min((r["n_frames"] for r in rows if not r["ok"]), default=None),
        "rows": rows,
    }

    data = json.loads(PROFILE_PATH.read_text()) if PROFILE_PATH.exists() else {"profiles": {}}
    data["profiles"][args.weights_dtype] = profile
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps(data, indent=2))
    plot_profiles(data["profiles"], total_mb)
    print(f"max frames without OOM: {profile['max_ok_frames']} -> {PROFILE_PATH}, {FIGURE_PATH}")


if __name__ == "__main__":
    main()

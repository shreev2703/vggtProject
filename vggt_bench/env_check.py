"""Phase 0 environment check: GPU, VRAM, CUDA arch support, package versions."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

REQUIRED_ARCH = "sm_120"  # Blackwell (RTX 50-series)
PACKAGES = ["torch", "torchvision", "vggt", "pycolmap", "gsplat", "open3d", "numpy"]


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def is_wsl() -> bool:
    return "microsoft" in platform.uname().release.lower()


def gpu_info() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
    }
    if not torch.cuda.is_available():
        return info

    props = torch.cuda.get_device_properties(0)
    info.update(
        gpu_name=props.name,
        total_vram_mb=round(props.total_memory / 2**20),
        compute_capability=f"{props.major}.{props.minor}",
        bf16_supported=torch.cuda.is_bf16_supported(),
    )
    # A real kernel launch catches "no kernel image is available" errors
    # that arch_list alone can miss.
    try:
        x = torch.randn(256, 256, device="cuda")
        (x @ x).sum().item()
        info["kernel_launch_ok"] = True
    except RuntimeError as exc:
        info["kernel_launch_ok"] = False
        info["kernel_launch_error"] = str(exc)
    return info


def collect() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "is_wsl": is_wsl(),
        "git_commit": git_commit(),
        "packages": {p: package_version(p) for p in PACKAGES},
    }
    try:
        report["gpu"] = gpu_info()
    except ImportError:
        report["gpu"] = {"error": "torch is not installed"}
    return report


def problems(report: dict[str, Any]) -> list[str]:
    """Return human-readable reasons the environment is not ready."""
    issues: list[str] = []
    gpu = report["gpu"]
    if "error" in gpu:
        issues.append(gpu["error"])
        return issues
    if not gpu["cuda_available"]:
        issues.append("CUDA is not available to torch (CPU-only build or driver issue).")
        return issues
    if REQUIRED_ARCH not in gpu["arch_list"]:
        issues.append(
            f"{REQUIRED_ARCH} is missing from torch.cuda.get_arch_list() = {gpu['arch_list']}. "
            "Blackwell needs a CUDA 12.8+ build: "
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
        )
    if not gpu.get("kernel_launch_ok", False):
        issues.append(f"Test CUDA kernel failed: {gpu.get('kernel_launch_error')}")
    for pkg in ("vggt", "pycolmap"):
        if report["packages"][pkg] is None:
            issues.append(f"Package '{pkg}' is not installed.")
    if not report["python"].startswith("3.11"):
        issues.append(f"Python {report['python']} found; this project pins 3.11.")
    return issues


def main(out_path: Path = Path("results/env.json")) -> int:
    report = collect()
    issues = problems(report)
    report["ok"] = not issues
    report["issues"] = issues

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    if issues:
        print("\nENV CHECK FAILED:", file=sys.stderr)
        for i in issues:
            print(f"  - {i}", file=sys.stderr)
        return 1
    print(f"\nENV CHECK PASSED. Saved to {out_path}")
    return 0

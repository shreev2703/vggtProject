"""Frame-count guard derived from results/memory_profile.json."""

from __future__ import annotations

import json
from pathlib import Path

from vggt_bench.profiling import frame_cap


def _profile(tmp_path: Path, max_ok: int) -> Path:
    path = tmp_path / "memory_profile.json"
    path.write_text(json.dumps({"profiles": {"bf16": {"max_ok_frames": max_ok}}}))
    return path


def test_frame_cap_lowers_the_request_to_the_calibrated_limit(tmp_path: Path) -> None:
    assert frame_cap(100, "bf16", _profile(tmp_path, 70)) == 70


def test_frame_cap_keeps_a_smaller_request(tmp_path: Path) -> None:
    assert frame_cap(40, "bf16", _profile(tmp_path, 70)) == 40


def test_frame_cap_passes_the_request_through_without_a_profile(tmp_path: Path) -> None:
    assert frame_cap(60, "bf16", tmp_path / "missing.json") == 60
    assert frame_cap(60, "fp32", _profile(tmp_path, 70)) == 60  # dtype not calibrated

"""Common interface so the benchmark can swap reconstruction methods."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from vggt_bench.recon import Recon


class ReconMethod(Protocol):
    name: str

    def reconstruct(self, image_paths: list[Path]) -> Recon: ...

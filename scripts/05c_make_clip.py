"""Make the sped-up preview clip results/captures/<name>/capture.mp4 from captures/<name>.<ext>.

Usage: python scripts/05c_make_clip.py NAME [NAME ...]
"""

import sys
from pathlib import Path

from vggt_bench.clip import make_clip


def main(names: list[str]) -> None:
    for name in names:
        src = next(Path("captures").glob(f"{name}.*"), None)
        if src is None:
            raise SystemExit(f"no video captures/{name}.* found")
        dst = Path("results/captures") / name / "capture.mp4"
        speed = make_clip(src, dst)
        print(f"{name}: {speed:.1f}x -> {dst} ({dst.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])

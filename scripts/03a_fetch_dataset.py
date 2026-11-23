"""Download Mip-NeRF 360 and extract only the configured scenes' images_4/ and sparse/.

A no-op for scenes already present. The reference poses are COLMAP's own output
over all images (pseudo-ground-truth), which every downstream number must say.
"""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vggt_bench.io.datasets import load_dataset_config  # noqa: E402

CHUNK = 1 << 20


def scene_present(root: Path, scene: str, cfg: dict) -> bool:
    d = root / scene
    return (
        (d / cfg["images_subdir"]).is_dir()
        and any((d / cfg["images_subdir"]).iterdir())
        and (d / cfg["sparse_subdir"] / "images.bin").is_file()
    )


def download(url: str, dest: Path) -> None:
    """Resumable download: continues a partial file with an HTTP Range request."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req) as resp:
        if have and resp.status != 206:  # server ignored the Range header
            have = 0
        total = int(resp.headers.get("Content-Length", 0)) + have
        done = have
        with open(dest, "ab" if have else "wb") as out:
            while chunk := resp.read(CHUNK):
                out.write(chunk)
                done += len(chunk)
                print(f"\r{done / 1e9:6.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    print()


def main() -> None:
    cfg = load_dataset_config()
    root = Path(cfg["root"])
    wanted = [s for s in cfg["scenes"] if not scene_present(root, s, cfg)]
    if not wanted:
        print("all scenes already present")
    else:
        zip_path = root.parent / "360_v2.zip"
        if not zip_path.exists() or not zipfile.is_zipfile(zip_path):
            download(cfg["url"], zip_path)
        prefixes = {
            s: (f"{s}/{cfg['images_subdir']}/", f"{s}/{cfg['sparse_subdir']}/") for s in wanted
        }
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                scene = member.split("/", 1)[0]
                if scene in prefixes and member.startswith(prefixes[scene]):
                    zf.extract(member, root)
        for s in wanted:
            if not scene_present(root, s, cfg):
                raise SystemExit(f"scene {s} missing images_4/ or sparse/0 after extraction")
    print(
        "NOTE: reference poses are pseudo-ground-truth (dense COLMAP over all "
        "images), not independently measured."
    )


if __name__ == "__main__":
    main()

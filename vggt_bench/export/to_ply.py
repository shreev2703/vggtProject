"""Write colored point clouds as binary PLY (readable by MeshLab, Open3D, three.js)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    points = np.asarray(points, dtype=np.float32)
    n = len(points)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if colors is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]

    data = np.empty(n, dtype=fields)
    data["x"], data["y"], data["z"] = points.T
    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8)
        data["red"], data["green"], data["blue"] = colors.T

    ply_types = {"<f4": "float", "u1": "uchar"}
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property {ply_types[t]} {name}" for name, t in fields]
    header.append("end_header")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(data.tobytes())


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read files written by write_ply (binary little-endian, float xyz, optional uchar rgb)."""
    with open(path, "rb") as f:
        names: list[str] = []
        n = 0
        while (line := f.readline().decode("ascii").strip()) != "end_header":
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                names.append(line.split()[-1])
        fields = [(nm, "<f4" if nm in ("x", "y", "z") else "u1") for nm in names]
        data = np.frombuffer(f.read(), dtype=fields, count=n)
    points = np.stack([data["x"], data["y"], data["z"]], axis=1)
    if "red" not in names:
        return points, None
    return points, np.stack([data["red"], data["green"], data["blue"]], axis=1)

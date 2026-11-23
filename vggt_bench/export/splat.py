"""Train a short Gaussian splat with gsplat from a COLMAP-format scene, and export it for the web.

gsplat 1.5.3 ships the rasterizer and densification strategies but no trainer, so this is a small
one: fixed poses (VGGT's), MCMC densification with a hard Gaussian cap (predictable VRAM on a
laptop GPU) and an L1 + SSIM loss, as in gsplat's reference `simple_trainer`. Parameter
conventions follow gsplat's exporter: log-scales and logit-opacities.

gsplat JIT-compiles its CUDA kernels on first use: it needs `nvcc` (CUDA_HOME) and `ninja` on PATH,
and takes a couple of minutes once.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
from PIL import Image

from vggt_bench import profiling
from vggt_bench.eval.align import camera_centers

SH_C0 = 0.28209479177387814  # degree-0 spherical-harmonic constant: rgb = sh0 * C0 + 0.5


@dataclass
class Scene:
    """Images, OpenCV-convention cameras and the initial point cloud of one capture."""

    names: list[str]
    images: torch.Tensor  # (N, H, W, 3) uint8; converted to float on the GPU per use
    viewmats: torch.Tensor  # (N, 4, 4) camera-from-world
    ks: torch.Tensor  # (N, 3, 3)
    points: torch.Tensor  # (M, 3)
    colors: torch.Tensor  # (M, 3) float in [0, 1]


@dataclass
class SplatConfig:
    steps: int = 3000
    cap_max: int = 300_000
    sh_degree: int = 3
    holdout_every: int = 8
    sh_degree_interval: int = 1000
    ssim_weight: float = 0.2
    opacity_reg: float = 0.01
    scale_reg: float = 0.01


def load_colmap_scene(colmap_dir: Path) -> Scene:
    """Read colmap_dir/sparse/0 and colmap_dir/images (the layout `write_colmap` produces)."""
    model = pycolmap.Reconstruction(str(colmap_dir / "sparse" / "0"))
    images = sorted(model.images.values(), key=lambda im: im.name)
    if not images:
        raise ValueError(f"{colmap_dir} has no registered images")
    frames, viewmats, ks = [], [], []
    for im in images:
        cam = model.cameras[im.camera_id]
        if cam.model.name != "PINHOLE":
            raise ValueError(f"expected PINHOLE cameras, got {cam.model.name}")
        ks.append(np.asarray(cam.calibration_matrix(), dtype=np.float64))
        viewmat = np.eye(4)
        viewmat[:3] = im.cam_from_world().matrix()
        viewmats.append(viewmat)
        with Image.open(colmap_dir / "images" / im.name) as pil:
            frames.append(np.asarray(pil.convert("RGB")))
    points = [model.points3D[i] for i in sorted(model.points3D)]
    return Scene(
        names=[im.name for im in images],
        images=torch.from_numpy(np.stack(frames)),
        viewmats=torch.tensor(np.stack(viewmats), dtype=torch.float32),
        ks=torch.tensor(ks, dtype=torch.float32),
        points=torch.tensor(np.array([p.xyz for p in points]), dtype=torch.float32).reshape(-1, 3),
        colors=torch.tensor(np.array([p.color for p in points]), dtype=torch.float32).reshape(-1, 3)
        / 255.0,
    )


def split_views(n: int, every: int) -> tuple[list[int], list[int]]:
    """(train, held-out) view indices: every `every`-th view is held out for evaluation.

    Always keeps at least one training view.
    """
    held = list(range(0, n, every))
    train = [i for i in range(n) if i % every]
    return (train, held) if train else (held, [])


def init_gaussians(
    points: torch.Tensor, colors: torch.Tensor, sh_degree: int
) -> dict[str, torch.Tensor]:
    """One Gaussian per point; log-scale from the mean distance to its 3 nearest neighbours."""
    n = len(points)
    k = min(3, n - 1)
    if k < 1:
        dist = torch.ones(n, device=points.device)
    else:
        nearest = []
        rows = max(
            1, 2**26 // n
        )  # rows per chunk: bounds the (rows x M) distance matrix to ~256 MB
        for chunk in points.split(rows):
            d = torch.cdist(chunk, points)
            nearest.append(d.topk(k + 1, largest=False).values[:, 1:].mean(dim=1))  # skip self
        dist = torch.cat(nearest).clamp_min(1e-7)
    n_rest = (sh_degree + 1) ** 2 - 1
    return {
        "means": points.clone(),
        "scales": dist.log()[:, None].repeat(1, 3),
        "quats": F.pad(torch.zeros(n, 3, device=points.device), (1, 0), value=1.0),
        "opacities": torch.full((n,), math.log(0.1 / 0.9), device=points.device),
        "sh0": ((colors - 0.5) / SH_C0)[:, None, :],
        "shN": torch.zeros(n, n_rest, 3, device=points.device),
    }


def ssim(a: torch.Tensor, b: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Mean SSIM of two (B, H, W, 3) images in [0, 1], Gaussian window, valid convolution."""
    coords = torch.arange(window, dtype=a.dtype, device=a.device) - window // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :]).expand(3, 1, window, window).contiguous()

    x, y = a.permute(0, 3, 1, 2), b.permute(0, 3, 1, 2)
    # one convolution for all five moments instead of five launches
    mu_x, mu_y, xx, yy, xy = F.conv2d(
        torch.cat([x, y, x * x, y * y, x * y]), kernel, groups=3
    ).chunk(5)
    var_x, var_y, cov = xx - mu_x**2, yy - mu_y**2, xy - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    )
    return score.mean()


def _rasterize(
    params: dict[str, torch.Tensor],
    scales: torch.Tensor,
    opacities: torch.Tensor,
    viewmats: torch.Tensor,
    ks: torch.Tensor,
    hw: tuple[int, int],
    sh_degree: int,
) -> tuple[torch.Tensor, dict]:
    """The one place the parametrisation (log-scales, logit-opacities, SH) meets gsplat."""
    from gsplat import rasterization

    render, _, info = rasterization(
        params["means"],
        params["quats"],
        scales,
        opacities,
        torch.cat([params["sh0"], params["shN"]], dim=1),
        viewmats,
        ks,
        hw[1],
        hw[0],
        sh_degree=sh_degree,
        packed=False,
    )
    return render, info


def render_views(
    params: dict[str, torch.Tensor], scene: Scene, indices: list[int], sh_degree: int
) -> torch.Tensor:
    """Render the scene's cameras `indices` -> (len(indices), H, W, 3) clamped to [0, 1]."""
    device = params["means"].device
    with torch.no_grad():
        image, _ = _rasterize(
            params,
            params["scales"].exp(),
            params["opacities"].sigmoid(),
            scene.viewmats[indices].to(device),
            scene.ks[indices].to(device),
            tuple(scene.images.shape[1:3]),
            sh_degree,
        )
    return image.clamp(0, 1)


def _camera_extent(viewmats: torch.Tensor) -> float:
    centers = camera_centers(viewmats[:, :3].numpy())
    extent = np.linalg.norm(centers - centers.mean(axis=0), axis=1).max() * 1.1
    return float(extent) if extent > 1e-6 else 1.0


def train_splat(
    scene: Scene, cfg: SplatConfig, seed: int = 0
) -> tuple[dict[str, torch.Tensor], dict]:
    """Optimise Gaussians against the training views; return (params, stats).

    stats: losses (per step), n_gaussians, train_time_s, peak_vram_mb, holdout_psnr/ssim (NaN
    when no view is held out). Raises torch.cuda.OutOfMemoryError if the cap is too high.
    """
    from gsplat.strategy import MCMCStrategy

    if len(scene.points) == 0:
        raise ValueError("scene has no 3D points to initialise Gaussians from")
    device = torch.device("cuda")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    train_idx, held_idx = split_views(len(scene.images), cfg.holdout_every)
    images = scene.images.to(device)  # uint8; converted per use
    viewmats, ks = scene.viewmats.to(device), scene.ks.to(device)
    hw = tuple(images.shape[1:3])
    extent = _camera_extent(scene.viewmats)

    points, colors = scene.points.to(device), scene.colors.to(device)
    if len(points) > cfg.cap_max:
        keep = torch.randperm(len(points), device=device)[: cfg.cap_max].sort().values
        points, colors = points[keep], colors[keep]
    init = init_gaussians(points, colors, cfg.sh_degree)
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in init.items()})

    base_lr = {
        "means": 1.6e-4 * extent,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20,
    }
    optimizers = {
        k: torch.optim.Adam([{"params": params[k], "lr": base_lr[k], "name": k}], eps=1e-15)
        for k in params
    }
    means_decay = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / max(cfg.steps, 1))
    )
    strategy = MCMCStrategy(
        cap_max=cfg.cap_max,
        refine_start_iter=min(500, cfg.steps // 6),
        refine_stop_iter=cfg.steps,
    )
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()

    profiling.reset_peak()
    losses: list[torch.Tensor] = []
    with profiling.gpu_timer() as timer:
        for step in range(cfg.steps):
            i = rng.choice(train_idx)
            degree = min(step // cfg.sh_degree_interval, cfg.sh_degree)
            scales, opacities = params["scales"].exp(), params["opacities"].sigmoid()
            render, info = _rasterize(
                params, scales, opacities, viewmats[i : i + 1], ks[i : i + 1], hw, degree
            )
            target = images[i : i + 1].float() / 255.0
            loss = (
                (1 - cfg.ssim_weight) * F.l1_loss(render, target)
                + cfg.ssim_weight * (1 - ssim(render, target))
                + cfg.opacity_reg * opacities.mean()
                + cfg.scale_reg * scales.mean()
            )
            loss.backward()
            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
            means_decay.step()
            strategy.step_post_backward(
                params, optimizers, state, step, info, lr=means_decay.get_last_lr()[0]
            )
            losses.append(loss.detach())

    final = {k: v.detach() for k, v in params.items()}
    stats: dict = {
        "losses": torch.stack(losses).tolist() if losses else [],
        "n_gaussians": len(final["means"]),
        "train_time_s": timer["elapsed_s"],
        "peak_vram_mb": profiling.peak_allocated_mb(),
        "holdout_psnr": float("nan"),
        "holdout_ssim": float("nan"),
        "holdout_views": held_idx,
    }
    if held_idx:
        rendered = render_views(final, scene, held_idx, cfg.sh_degree)
        truth = images[held_idx].float() / 255.0
        mse = ((rendered - truth) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
        stats["holdout_psnr"] = (-10 * torch.log10(mse)).mean().item()
        stats["holdout_ssim"] = ssim(rendered, truth).item()
    return final, stats


def export_splat(params: dict[str, torch.Tensor], out_dir: Path) -> dict[str, Path]:
    """Write web-viewable splats: antimatter15 `.splat` and SuperSplat `.compressed.ply`."""
    from gsplat.exporter import export_splats

    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"splat": out_dir / "splat.splat", "ply_compressed": out_dir / "splat.compressed.ply"}
    for fmt, path in files.items():
        export_splats(
            params["means"],
            params["scales"],
            params["quats"],
            params["opacities"],
            params["sh0"],
            params["shN"],
            format=fmt,
            save_to=str(path),
        )
    return files

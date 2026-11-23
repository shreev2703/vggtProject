"""VGGT wrapped behind the ReconMethod interface.

Follows the usage in the VGGT README: `VGGT.from_pretrained`, images from
`load_and_preprocess_images` (width 518, crop mode), bf16 autocast, and
`pose_encoding_to_extri_intri` for cameras. The point cloud is built by
unprojecting the depth maps with the predicted cameras, which the README notes
is usually more accurate than the point-map head.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vggt_bench import profiling
from vggt_bench.recon import Recon

DEFAULT_CHECKPOINT = "facebook/VGGT-1B"  # non-commercial license; see README


def pick_autocast_dtype() -> torch.dtype:
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


class VGGTMethod:
    name = "vggt"

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        weights_dtype: str = "bf16",
        use_point_head: bool = False,
        conf_percentile: float = 50.0,
        max_points: int = 500_000,
        seed: int = 0,
    ) -> None:
        """
        Args:
            weights_dtype: "bf16" (default) stores the aggregator weights in bf16.
                "fp32" keeps the published weights as-is; autocast then also caches a
                bf16 copy of each weight during the forward pass, so fp32 costs about
                1.7 GB more and nearly fills an 8 GB GPU at 10 frames. On the kitchen
                example the two settings agree to <0.06 deg rotation and <0.25% focal.
                The heads always run in fp32, as in the official forward().
            use_point_head: run the point-map head too. Off by default because we
                build points from depth + cameras, and the head costs memory.
            conf_percentile: drop this percentage of lowest-confidence depth pixels
                before building the point cloud.
        """
        if weights_dtype not in ("fp32", "bf16"):
            raise ValueError(f"weights_dtype must be 'fp32' or 'bf16', got {weights_dtype!r}")
        self.checkpoint = checkpoint
        self.weights_dtype = weights_dtype
        self.use_point_head = use_point_head
        self.conf_percentile = conf_percentile
        self.max_points = max_points
        self.seed = seed
        self.autocast_dtype = pick_autocast_dtype()
        self.model: torch.nn.Module | None = None
        self.load_time_s: float | None = None
        self.weights_mb: float | None = None

    def config(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "weights_dtype": self.weights_dtype,
            "autocast_dtype": str(self.autocast_dtype).removeprefix("torch."),
            "use_point_head": self.use_point_head,
            "conf_percentile": self.conf_percentile,
            "max_points": self.max_points,
            "preprocess": "load_and_preprocess_images(mode='crop')",
        }

    def load(self) -> None:
        """Load weights onto the GPU once; reused across reconstructions."""
        if self.model is not None:
            return
        from vggt.models.vggt import VGGT

        start = time.perf_counter()
        model = VGGT.from_pretrained(self.checkpoint)
        model.track_head = None  # tracking is not used in this project
        if not self.use_point_head:
            model.point_head = None
        if self.weights_dtype == "bf16":
            model.aggregator.to(torch.bfloat16)
        model = model.to("cuda").eval()
        torch.cuda.synchronize()
        self.load_time_s = time.perf_counter() - start
        self.weights_mb = profiling.allocated_mb()
        self.model = model

    def load_images(self, image_paths: list[Path]) -> torch.Tensor:
        from vggt.utils.load_fn import load_and_preprocess_images

        return load_and_preprocess_images([str(p) for p in image_paths], mode="crop")

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Raw model forward on a (S, 3, H, W) tensor already on the GPU."""
        assert self.model is not None, "call load() first"
        with torch.autocast("cuda", dtype=self.autocast_dtype):
            return self.model(images)

    def reconstruct(self, image_paths: list[Path]) -> Recon:
        self.load()
        image_paths = [Path(p) for p in image_paths]

        t0 = time.perf_counter()
        images = self.load_images(image_paths).to("cuda")
        preprocess_s = time.perf_counter() - t0
        image_hw = (int(images.shape[-2]), int(images.shape[-1]))

        profiling.reset_peak()
        try:
            with profiling.gpu_timer() as timer:
                preds = self.forward(images)
        except torch.cuda.OutOfMemoryError:
            del images
            profiling.free_gpu_memory()
            return Recon.failure(
                self.name,
                image_paths,
                "cuda_oom",
                image_hw=image_hw,
                peak_vram_mb=profiling.peak_allocated_mb(),
            )
        except RuntimeError as exc:
            # Under WSL, exhausting VRAM can surface as a CUDA driver error instead of
            # OutOfMemoryError. The CUDA context may be unusable afterwards, so don't
            # touch the GPU again here; callers should stay under the calibrated limit.
            if "CUDA" not in str(exc):
                raise
            return Recon.failure(self.name, image_paths, f"cuda_error: {exc}", image_hw=image_hw)
        peak_mb = profiling.peak_allocated_mb()

        t1 = time.perf_counter()
        recon = self._to_recon(preds, images, image_paths, image_hw)
        postprocess_s = time.perf_counter() - t1

        recon.runtime_s = timer["elapsed_s"]
        recon.peak_vram_mb = peak_mb
        recon.timings = {
            "preprocess_s": preprocess_s,
            "forward_s": timer["elapsed_s"],
            "postprocess_s": postprocess_s,
        }
        recon.extra["config"] = self.config()
        del preds, images
        profiling.free_gpu_memory()
        return recon

    def _to_recon(
        self,
        preds: dict[str, torch.Tensor],
        images: torch.Tensor,
        image_paths: list[Path],
        image_hw: tuple[int, int],
    ) -> Recon:
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], image_hw)
        extrinsics = extrinsic[0].float().cpu().numpy()  # (S, 3, 4)
        intrinsics = intrinsic[0].float().cpu().numpy()  # (S, 3, 3)
        depth = preds["depth"][0, ..., 0].float().cpu().numpy()  # (S, H, W)
        depth_conf = preds["depth_conf"][0].float().cpu().numpy()  # (S, H, W)

        world = unproject_depth_map_to_point_map(depth[..., None], extrinsics, intrinsics)
        rgb = (images.permute(0, 2, 3, 1).float().cpu().numpy() * 255).astype(np.uint8)

        points, colors = self._filter_points(world, rgb, depth_conf)
        n = len(image_paths)
        return Recon(
            method=self.name,
            image_paths=image_paths,
            image_hw=image_hw,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            registered=np.ones(n, dtype=bool),  # feed-forward: every image gets a camera
            points=points,
            colors=colors,
            depth=depth,
            depth_conf=depth_conf,
        )

    def _filter_points(
        self, world: np.ndarray, rgb: np.ndarray, conf: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        pts = world.reshape(-1, 3)
        cols = rgb.reshape(-1, 3)
        c = conf.reshape(-1)
        keep = np.isfinite(pts).all(axis=1) & (c >= np.percentile(c, self.conf_percentile))
        pts, cols = pts[keep], cols[keep]
        if len(pts) > self.max_points:
            idx = np.random.default_rng(self.seed).choice(len(pts), self.max_points, replace=False)
            pts, cols = pts[idx], cols[idx]
        return pts.astype(np.float32), cols

"""Benchmark harness comparing VGGT and COLMAP for sparse-view reconstruction."""

import os

# Let PyTorch's CUDA allocator grow memory segments instead of carving fixed blocks.
# On an 8 GB GPU this cuts fragmentation enough to raise VGGT's frame limit well
# beyond what the default allocator allows (see results/memory_profile.json). It must
# be set before the first CUDA allocation; an explicit value in the environment wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

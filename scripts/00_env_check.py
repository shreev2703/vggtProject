"""Phase 0: verify GPU, CUDA arch (sm_120) and package versions. Writes results/env.json."""

import sys

from vggt_bench.env_check import main

if __name__ == "__main__":
    sys.exit(main())

# Benchmark summary (240 runs from `results/benchmark.csv`)

AUC@30 is the mean over all runs; failed runs count as 0.

## By number of views

| n_views | colmap auc_10 | colmap auc_30 | colmap auc_30_std | colmap runs | colmap success | vggt auc_10 | vggt auc_30 | vggt auc_30_std | vggt runs | vggt success |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0 | 0 | 0 | 24 | 0 | 0.763 | 0.902 | 0.14 | 24 | 1 |
| 5 | 0.12 | 0.123 | 0.333 | 24 | 0.167 | 0.82 | 0.94 | 0.028 | 24 | 1 |
| 10 | 0.517 | 0.538 | 0.497 | 24 | 0.583 | 0.831 | 0.942 | 0.034 | 24 | 1 |
| 20 | 0.798 | 0.862 | 0.291 | 24 | 1 | 0.833 | 0.942 | 0.034 | 24 | 1 |
| 40 | 0.816 | 0.884 | 0.24 | 24 | 1 | 0.85 | 0.949 | 0.028 | 24 | 1 |

## By scene

| scene | colmap auc_10 | colmap auc_30 | colmap auc_30_std | colmap runs | colmap success | vggt auc_10 | vggt auc_30 | vggt auc_30_std | vggt runs | vggt success |
|---|---|---|---|---|---|---|---|---|---|---|
| bicycle (easy) | 0.388 | 0.396 | 0.498 | 20 | 0.4 | 0.806 | 0.911 | 0.154 | 20 | 1 |
| bonsai (repetitive) | 0.14 | 0.247 | 0.339 | 20 | 0.55 | 0.679 | 0.892 | 0.019 | 20 | 1 |
| counter (reflective) | 0.475 | 0.492 | 0.505 | 20 | 0.5 | 0.873 | 0.957 | 0.007 | 20 | 1 |
| garden (easy) | 0.593 | 0.598 | 0.501 | 20 | 0.6 | 0.916 | 0.972 | 0.011 | 20 | 1 |
| kitchen (reflective) | 0.528 | 0.543 | 0.504 | 20 | 0.55 | 0.873 | 0.958 | 0.01 | 20 | 1 |
| room (textureless) | 0.577 | 0.613 | 0.477 | 20 | 0.7 | 0.769 | 0.92 | 0.026 | 20 | 1 |

## By sampling (random seeds vs wide-baseline)

| sampling | colmap auc_10 | colmap auc_30 | colmap auc_30_std | colmap runs | colmap success | vggt auc_10 | vggt auc_30 | vggt auc_30_std | vggt runs | vggt success |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 0.478 | 0.506 | 0.485 | 90 | 0.578 | 0.812 | 0.931 | 0.078 | 90 | 1 |
| wide | 0.366 | 0.408 | 0.47 | 30 | 0.467 | 0.841 | 0.946 | 0.03 | 30 | 1 |

## Median runtime (s) and peak VRAM (MB)

| method | runtime_s | peak_vram_mb |
|---|---|---|
| colmap | 4.5 |  |
| vggt | 1.8 | 4605.3 |

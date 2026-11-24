# Vendored viewer code

Third-party code served by the demo site. Each directory keeps the upstream licence file.
`scripts/06_build_site.py` never touches this directory.

| Path | Source | Version | Licence |
|---|---|---|---|
| `three/` (`three.module.min.js`, `loaders/PLYLoader.js`, `controls/OrbitControls.js`) | https://github.com/mrdoob/three.js via `cdn.jsdelivr.net/npm/three@0.160.0` | 0.160.0 | MIT |
| `gsplat3d/` (`gaussian-splats-3d.module.min.js`) | https://github.com/mkkellogg/GaussianSplats3D via `cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@0.4.7` | 0.4.7 | MIT |

Unmodified. It imports the bare specifier `three`, which the scene pages' import map maps to `three/`.
`site/splat.js` (from `vggt_bench/site_templates/splat.js`) is our own code that drives it.
The earlier antimatter15/splat viewer was dropped because its first-person controls were hard to use.

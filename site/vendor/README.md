# Vendored viewer code

Third-party code served by the demo site. Each directory keeps the upstream licence file.
`scripts/06_build_site.py` never touches this directory.

| Path | Source | Version | Licence |
|---|---|---|---|
| `three/` (`three.module.min.js`, `loaders/PLYLoader.js`, `controls/OrbitControls.js`) | https://github.com/mrdoob/three.js via `cdn.jsdelivr.net/npm/three@0.160.0` | 0.160.0 | MIT |
| `splat/` (`index.html`, `main.js`) | https://github.com/antimatter15/splat | commit `ba182b51b7c2ad5738cdd6741cd63336d27470fb` | MIT |

## Local change

`splat/main.js`: the `?url=` parameter is resolved against `location.href` instead of upstream's
hard-coded `https://huggingface.co/cakewalk/splat-data/resolve/main/`, so the viewer loads the
splat that ships with this site and makes no third-party request. Nothing else is modified.

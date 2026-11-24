// Gaussian-splat viewer: mkkellogg/GaussianSplats3D on the page's three.js, with OrbitControls.
import { Vector3 } from "three";
import * as GaussianSplats3D from "./vendor/gsplat3d/gaussian-splats-3d.module.min.js";

const host = document.getElementById("splat");

async function main() {
  // VGGT/COLMAP use the OpenCV convention (y down), so the orbit "up" is -y.
  const viewer = new GaussianSplats3D.Viewer({
    rootElement: host,
    cameraUp: [0, -1, 0],
    initialCameraPosition: [0, 0, 0],
    initialCameraLookAt: [0, 0, 1],
    sharedMemoryForWorkers: false, // GitHub Pages cannot send COOP/COEP headers
    gpuAcceleratedSort: false,
  });
  await viewer.addSplatScene(host.dataset.src, { splatAlphaRemovalThreshold: 5 });

  // Orbit around the middle of the scene, starting from the first camera at the origin.
  const mesh = viewer.splatMesh;
  const n = mesh.getSplatCount();
  const step = Math.max(1, Math.floor(n / 4000));
  const p = new Vector3();
  const xs = [], ys = [], zs = [];
  for (let i = 0; i < n; i += step) {
    mesh.getSplatCenter(i, p);
    xs.push(p.x); ys.push(p.y); zs.push(p.z);
  }
  const median = (a) => a.sort((u, v) => u - v)[a.length >> 1];
  viewer.controls.target.set(median(xs), median(ys), median(zs));
  viewer.camera.position.set(0, 0, 0);
  viewer.controls.update();
  viewer.start();
}

main().catch((err) => {
  console.error(err);
  host.textContent = "The splat viewer failed to load: " + err.message;
});

// Interactive point-cloud viewer: three.js PLYLoader + OrbitControls, VGGT/COLMAP toggle.
import * as THREE from "three";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const host = document.getElementById("pc");
const buttons = [...document.querySelectorAll(".toggle button")];

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101010);
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// VGGT and COLMAP use the OpenCV camera convention (y down, z forward); three.js is y up.
const root = new THREE.Group();
root.rotation.x = Math.PI;
scene.add(root);

const loader = new PLYLoader();
const clouds = new Map();
let current = null;
let dirty = true;

function resize() {
  const { clientWidth: w, clientHeight: h } = host;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  dirty = true;
}

function frame(points) {
  const geo = points.geometry;
  geo.computeBoundingSphere();
  const { center, radius } = geo.boundingSphere;
  const c = center.clone().applyEuler(root.rotation);
  controls.target.copy(c);
  camera.position.copy(c).add(new THREE.Vector3(0, 0, -radius * 2.2));
  camera.near = radius / 100;
  camera.far = radius * 100;
  camera.updateProjectionMatrix();
  // Sparse clouds (COLMAP) get larger dots than dense ones (VGGT) so both stay visible.
  points.material.size = (radius * 1.0) / Math.sqrt(geo.getAttribute("position").count);
}

let wanted = null;

async function show(src) {
  wanted = src;
  if (!clouds.has(src)) {
    const geo = await loader.loadAsync(src);
    const material = new THREE.PointsMaterial({ size: 0.01, vertexColors: geo.hasAttribute("color") });
    clouds.set(src, new THREE.Points(geo, material));
  }
  if (wanted !== src) return; // a later click superseded this load; keep the newer cloud
  if (current) root.remove(current);
  current = clouds.get(src);
  root.add(current);
  frame(current);
  dirty = true;
}

buttons.forEach((b) =>
  b.addEventListener("click", () => {
    buttons.forEach((o) => o.classList.toggle("on", o === b));
    show(b.dataset.src).catch((e) => console.error("failed to load", b.dataset.src, e));
  })
);

window.addEventListener("resize", resize);
resize();
show(buttons[0].dataset.src).catch((e) => console.error("failed to load point cloud", e));
// Draw only when something changed: the camera moving or settling, a new cloud, or a resize.
renderer.setAnimationLoop(() => {
  if (controls.update() || dirty) {
    renderer.render(scene, camera);
    dirty = false;
  }
});

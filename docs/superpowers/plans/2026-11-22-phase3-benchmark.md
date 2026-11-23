# Phase 3 Benchmark (M3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure VGGT against the COLMAP baseline on a public dataset with reference poses, across view counts, and write one tidy resumable row per run to `results/benchmark.csv`.

**Architecture:** A dataset loader turns Mip-NeRF 360 scenes into image lists plus reference poses read from the bundled COLMAP `sparse/0` models. Frame sampling draws identical subsets for both methods. Evaluation aligns predicted cameras to reference with a Sim(3) Umeyama fit, then scores relative pose error and AUC. `scripts/03_run_benchmark.py` is the only orchestration layer: it loops the sweep, skips rows already in the CSV, and records failures (including OOM) as rows rather than crashing.

**Tech Stack:** Python 3.11, numpy, scipy, pandas, pyyaml, pycolmap (reading reference models), torch (cu128), pytest, ruff.

**Spec:** `VGGT_PROJECT_SPEC.md` §7 (plus §3 repo structure, §11 engineering standards).

## Context

Phases 0–2 are done: `VGGTMethod` and `ColmapMethod` both implement `ReconMethod` and return a `Recon` (`vggt_bench/recon.py`) with extrinsics, intrinsics, a `registered` mask, timings, `peak_vram_mb` and `failed`/`failure_reason`. What is missing is everything that turns one reconstruction into a *number*: there is no dataset with reference poses, no alignment, no pose metrics, and no sweep. Phase 3 closes that gap so the README can state measured results instead of TODOs, and so the headline figure (AUC@30 vs. N views) has data behind it.

Decisions already made with the user:
- **Dataset:** Mip-NeRF 360, `http://storage.googleapis.com/gresearch/refraw360/360_v2.zip` (11.7 GB, no auth; verified reachable). Its per-scene `sparse/0` COLMAP model is **pseudo-ground-truth** (dense COLMAP over all images) and the README must say so.
- **Scenes (6):** `garden` (easy, textured), `bicycle` (easy, outdoor), `kitchen` (reflective), `counter` (reflective, indoor), `room` (textureless walls), `bonsai` (repetitive floor/foliage).
- **Resolution:** the `images_4` copy inside the zip (~1557×1038). Reference intrinsics must be scaled by the actual `images_4`-to-`images` ratio; poses are resolution independent.
- **Sweep:** N ∈ {3, 5, 10, 20, 40} × 3 random seeds + 1 wide-baseline (evenly spaced) sampling × 2 methods × 6 scenes = 240 runs. VGGT's calibrated cap is `max_ok_frames: 70` in `results/memory_profile.json`, so no N is dropped.

## Global Constraints

- Repo is canonical in **WSL at `~/vggtProject`**; run everything there, heavy commands prefixed with `rtk`.
- Never fabricate a metric: every number in `README.md`/`EXPLAINER.md` comes from an executed script writing into `results/`.
- Seed all randomness; log config + git commit into every output via the existing `vggt_bench/provenance.run_metadata()`.
- Benchmark runs are resumable: an existing matching row in `results/benchmark.csv` is skipped, never recomputed.
- Catch `torch.cuda.OutOfMemoryError`, clear the cache (`vggt_bench/profiling.free_gpu_memory()`), and record the failure as a row.
- Failed runs count as AUC = 0 in averages, and success rate is reported separately.
- Camera convention is OpenCV camera-from-world `[R|t]`, as documented in `vggt_bench/recon.py`.
- `pytest` and `ruff` must pass with zero errors before the milestone is called done.
- No AI attribution anywhere in commits, code comments or docs.

## Review Focus

Inputs the spec implies but no happy-path test covers. Each has a test assigned to the task that owns the code.

1. **Fewer than 3 cameras to align** — Umeyama is degenerate below 3 non-collinear points; `align_cameras` must raise a clear error and the runner must record `failure_reason`, not emit a nonsense AUC. (Task 2)
2. **Collinear camera centers** — a 3-view subset on a near-straight trajectory gives a rank-deficient covariance; the fit must still return a valid rotation (reflection-corrected) instead of a mirrored one. (Task 2)
3. **Partially registered reconstructions** — COLMAP places only some images, so metric functions must evaluate pairs where *both* cameras are registered and return 0-pair cases as AUC 0, not NaN. (Task 3)
4. **N larger than the scene's image count** — a scene with fewer images than N must clamp or skip explicitly rather than sample with replacement and silently duplicate frames. (Task 1)
5. **A reference model missing an image** — `images_4` may contain a file absent from `sparse/0`; the loader must drop it from the scene with a counted warning rather than mismatching poses to images by position. (Task 1)

---

### Task 1: Dataset loader and frame sampling

**Files:**
- Create: `configs/datasets.yaml`
- Create: `vggt_bench/io/datasets.py`
- Create: `scripts/03a_fetch_dataset.py`
- Modify: `vggt_bench/io/frames.py` (add seeded sampling beside the existing `list_images`/`select_frames`)
- Test: `tests/test_datasets.py`, `tests/test_frames.py` (extend)

**Interfaces:**
- Consumes: `vggt_bench.io.frames.list_images`.
- Produces:
  - `Scene` dataclass: `dataset: str`, `name: str`, `scene_type: str`, `image_dir: Path`, `images: list[Path]`, `ref_extrinsics: np.ndarray (N,3,4)`, `ref_intrinsics: np.ndarray (N,3,3)`, `image_hw: tuple[int, int]`, `n_dropped: int`.
  - `load_dataset_config(path: Path = Path("configs/datasets.yaml")) -> dict`
  - `load_scene(name: str, cfg: dict) -> Scene`
  - `frames.sample_frames(paths: list[Path], n: int, seed: int | None) -> list[Path]` — `seed=None` means evenly spaced (wide baseline); an int means a reproducible random draw without replacement, returned in original order. `n >= len(paths)` returns all paths.

- [ ] **Step 1: Write `configs/datasets.yaml`**

```yaml
mip360:
  url: http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
  root: data/mip360
  images_subdir: images_4
  sparse_subdir: sparse/0
  reference: pseudo_gt_colmap   # dense COLMAP over all images; say so in the README
  view_counts: [3, 5, 10, 20, 40]
  seeds: [0, 1, 2]
  scenes:
    garden:  {scene_type: easy}
    bicycle: {scene_type: easy}
    kitchen: {scene_type: reflective}
    counter: {scene_type: reflective}
    room:    {scene_type: textureless}
    bonsai:  {scene_type: repetitive}
```

- [ ] **Step 2: Write the failing tests in `tests/test_frames.py`**

```python
def test_sample_frames_is_reproducible_for_a_seed(tmp_path):
    paths = _touch_images(tmp_path, 20)
    assert sample_frames(paths, 5, seed=0) == sample_frames(paths, 5, seed=0)

def test_sample_frames_differs_between_seeds(tmp_path):
    paths = _touch_images(tmp_path, 20)
    assert sample_frames(paths, 5, seed=0) != sample_frames(paths, 5, seed=1)

def test_sample_frames_keeps_original_order_and_is_unique(tmp_path):
    paths = _touch_images(tmp_path, 20)
    got = sample_frames(paths, 6, seed=3)
    assert got == sorted(got, key=paths.index)
    assert len(set(got)) == 6

def test_sample_frames_without_a_seed_is_evenly_spaced(tmp_path):
    paths = _touch_images(tmp_path, 21)
    assert sample_frames(paths, 3, seed=None) == [paths[0], paths[10], paths[20]]

def test_sample_frames_clamps_to_the_available_frames(tmp_path):   # Review Focus 4
    paths = _touch_images(tmp_path, 4)
    assert sample_frames(paths, 10, seed=0) == paths
```

- [ ] **Step 3: Write the failing tests in `tests/test_datasets.py`**

Build a throwaway reference model with `pycolmap.Reconstruction()` in `tmp_path` (two `PINHOLE` cameras' worth of images with known `cam_from_world` quaternion/translation), write it with `.write(...)`, and create matching `images_4` files.

```python
def test_load_scene_reads_poses_in_image_order(tmp_path): ...        # ref_extrinsics[i] belongs to scene.images[i]
def test_load_scene_scales_intrinsics_to_the_images_4_copy(tmp_path):
    # fx/cx of ref_intrinsics equal the sparse model's fx/cx times (images_4 width / images width)
def test_load_scene_drops_images_missing_from_the_reference(tmp_path):  # Review Focus 5
    # an extra images_4 file is excluded and scene.n_dropped == 1
def test_load_scene_rejects_an_unknown_scene_name(tmp_path): ...     # raises KeyError/ValueError naming the scene
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `wsl -e bash -lc "cd ~/vggtProject && rtk .venv/bin/python -m pytest tests/test_datasets.py tests/test_frames.py -q"`
Expected: FAIL — `ImportError` / `cannot import name 'sample_frames'`.

- [ ] **Step 5: Implement `sample_frames` in `vggt_bench/io/frames.py`**

Keep `select_frames` as the even-spacing path and have `sample_frames(..., seed=None)` delegate to it; the seeded branch uses `np.random.default_rng(seed).choice(len(paths), n, replace=False)`, then sorts the indices.

- [ ] **Step 6: Implement `Scene`, `load_dataset_config` and `load_scene` in `vggt_bench/io/datasets.py`**

Read `sparse/0` with `pycolmap.Reconstruction(path)`; build `{image.name: image}`, intersect with `list_images(images_4 dir)` by basename, keep filename order, and stack `cam_from_world` as `(3,4)` matrices. Scale `fx, fy, cx, cy` by the ratio of the `images_4` width to the reference camera width (read one image's size with PIL, so the ratio is measured rather than assumed to be exactly 4).

- [ ] **Step 7: Write `scripts/03a_fetch_dataset.py`**

Downloads the zip to `data/`, extracts only the six configured scenes' `images_4/` and `sparse/` entries, verifies each scene directory afterwards, and is a no-op for scenes already present. Print the pseudo-GT caveat on completion.

- [ ] **Step 8: Run the tests and ruff**

Run: `wsl -e bash -lc "cd ~/vggtProject && rtk .venv/bin/python -m pytest -q && rtk .venv/bin/ruff check ."`
Expected: PASS, zero ruff errors.

- [ ] **Step 9: Fetch the data (long-running, once)**

Run: `wsl -e bash -lc "cd ~/vggtProject && rtk .venv/bin/python scripts/03a_fetch_dataset.py"`
Expected: six scene folders under `data/mip360/`, each with `images_4/` and `sparse/0/`.

- [ ] **Step 10: Commit**

```bash
git add configs/datasets.yaml vggt_bench/io/datasets.py vggt_bench/io/frames.py scripts/03a_fetch_dataset.py tests/test_datasets.py tests/test_frames.py
git commit -m "feat(data): load Mip-NeRF 360 scenes with reference poses and seeded frame sampling"
```

---

### Task 2: Sim(3) alignment (`eval/align.py`)

**Files:**
- Create: `vggt_bench/eval/align.py`
- Test: `tests/test_align.py`

**Interfaces:**
- Consumes: `Recon.camera_centers()` (`vggt_bench/recon.py`).
- Produces:
  - `Sim3` dataclass: `scale: float`, `rotation: np.ndarray (3,3)`, `translation: np.ndarray (3,)`, with `apply(points: np.ndarray) -> np.ndarray` (`s * R @ p + t`, row-major input).
  - `umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> Sim3` — least-squares similarity mapping `src` onto `dst`, both `(N,3)`.
  - `align_extrinsics(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None) -> tuple[Sim3, np.ndarray]` — fits on the masked cameras' centers and returns the transform plus `pred` expressed in the reference frame, shape `(N,3,4)`, unregistered rows left NaN.
  - Raises `AlignmentError` (module-level `ValueError` subclass) when fewer than 3 cameras are usable.

- [ ] **Step 1: Write the failing tests in `tests/test_align.py`**

```python
def test_umeyama_recovers_a_known_similarity():
    # 8 random points, known s=2.5, R from scipy Rotation.from_euler("xyz",[20,-35,50],degrees=True), t=[1,-2,3]
    # assert scale, rotation and translation match to 1e-9
def test_umeyama_is_exact_for_the_identity_transform(): ...
def test_umeyama_handles_collinear_points_without_reflecting():   # Review Focus 2
    # 3 points on a line, rotated by a known R; det(fit.rotation) == pytest.approx(1.0)
def test_align_extrinsics_maps_transformed_cameras_back_onto_the_reference():
    # ref extrinsics for 6 cameras; pred = same cameras pushed through a known Sim3
    # assert aligned camera centers match ref centers to 1e-6 and rotations match to 1e-6
def test_align_extrinsics_uses_only_masked_cameras():
    # one pred row is NaN with mask False; the fit still recovers the transform and that row stays NaN
def test_align_extrinsics_raises_below_three_cameras():   # Review Focus 1
    with pytest.raises(AlignmentError):
        align_extrinsics(pred[:2], ref[:2])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `wsl -e bash -lc "cd ~/vggtProject && rtk .venv/bin/python -m pytest tests/test_align.py -q"`
Expected: FAIL — module `vggt_bench.eval.align` does not exist.

- [ ] **Step 3: Implement `Sim3`, `umeyama_sim3`, `align_extrinsics` in `vggt_bench/eval/align.py`**

Classic Umeyama: centre both clouds, SVD of the cross-covariance, flip the last singular vector's sign when `det(U @ Vt) < 0`, scale as `trace(D @ S) / src_variance`. Transform extrinsics by converting each `[R|t]` to world-from-camera, applying the `Sim3` to the centre and `R_fit` to the orientation, then converting back — translation becomes `-R_new @ C_new`. Document in the docstring that camera *rotations* are scale-invariant, which is why only centres drive the fit.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `wsl -e bash -lc "cd ~/vggtProject && rtk .venv/bin/python -m pytest tests/test_align.py -q"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vggt_bench/eval/align.py tests/test_align.py
git commit -m "feat(eval): add Umeyama Sim(3) alignment of predicted cameras to reference"
```

---

### Task 3: Pose metrics (`eval/pose_metrics.py`)

**Files:**
- Create: `vggt_bench/eval/pose_metrics.py`
- Test: `tests/test_pose_metrics.py`

**Interfaces:**
- Consumes: `align_extrinsics` (Task 2).
- Produces:
  - `relative_pose_errors(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]` — per-pair RRE and RTE in **degrees** over all pairs where both cameras are registered; empty arrays when no pair qualifies.
  - `pose_auc(errors: np.ndarray, thresholds: Sequence[float] = (5.0, 10.0, 30.0)) -> dict[float, float]` — normalised area under the accuracy-vs-threshold curve, in `[0, 1]`; `{t: 0.0 for t in thresholds}` for an empty input.
  - `evaluate_poses(recon: Recon, scene: Scene, thresholds=(5.0, 10.0, 30.0)) -> dict[str, float]` — returns `auc_5`, `auc_10`, `auc_30`, `rre_median`, `rte_median`, `n_pairs`, `frac_registered`; all-zero AUCs with `n_pairs=0` when the recon failed or fewer than 3 cameras aligned.

- [ ] **Step 1: Write the failing tests in `tests/test_pose_metrics.py`**

```python
def test_errors_are_zero_for_identical_poses(): ...
def test_relative_rotation_error_on_a_known_yaw():
    # two cameras, pred's second rotated 10 deg about y relative to ref -> RRE == approx(10.0)
def test_relative_translation_error_is_an_angle_between_baselines():
    # pred baseline rotated 30 deg from ref baseline -> RTE == approx(30.0)
def test_errors_are_scale_invariant():
    # ref poses scaled by 7x produce the same RRE/RTE after alignment
def test_pairs_with_an_unregistered_camera_are_skipped():   # Review Focus 3
    # 4 cameras, mask = [True, True, False, True] -> n_pairs == 3
def test_pose_auc_is_one_when_every_error_is_zero(): ...
def test_pose_auc_is_zero_when_every_error_exceeds_the_threshold(): ...
def test_pose_auc_matches_a_hand_computed_case():
    # errors = [0, 15], threshold 30 -> auc_30 == approx(0.75)
def test_evaluate_poses_scores_a_failed_recon_as_zero():
    # Recon.failure(...) -> {"auc_30": 0.0, "n_pairs": 0, "frac_registered": 0.0}
def test_evaluate_poses_scores_a_two_camera_recon_as_zero(): ...    # Review Focus 1/3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `wsl -e bash -lc "cd ~/vggtProject && rtk .venv/bin/python -m pytest tests/test_pose_metrics.py -q"`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `relative_pose_errors` in `vggt_bench/eval/pose_metrics.py`**

For each registered pair `(i, j)`: relative rotation `R_rel = R_j @ R_i.T` for both pred and ref, RRE is the geodesic angle of `R_rel_pred.T @ R_rel_ref` (`arccos((trace - 1) / 2)`, clipped); RTE is the angle between the two baseline directions `C_j - C_i` (normalised, so it is scale free). Degrees, `float64`.

- [ ] **Step 4: Implement `pose_auc` using the per-pair worst error**

The VGGT paper's AUC@τ: per-pair error is `max(RRE, RTE)`; sort errors, build recall `arange(1, n+1) / n`, and for each threshold integrate the step curve up to τ with `np.trapezoid`, dividing by τ so a perfect method scores 1. Clamp to `[0, 1]`.

```python
def pose_auc(errors, thresholds=(5.0, 10.0, 30.0)):
    if len(errors) == 0:
        return {float(t): 0.0 for t in thresholds}
    sort = np.sort(np.asarray(errors, dtype=float))
    recall = np.arange(1, len(sort) + 1) / len(sort)
    out = {}
    for t in thresholds:
        last = int(np.searchsorted(sort, t, side="right"))
        e = np.r_[sort[:last], t]
        r = np.r_[recall[:last], recall[last - 1] if last else 0.0]
        out[float(t)] = float(np.clip(np.trapezoid(r, x=e) / t, 0.0, 1.0))
    return out
```

- [ ] **Step 5: Implement `evaluate_poses`**

Align with `align_extrinsics` on `recon.registered`, catching `AlignmentError` and the failed/`extrinsics is None` cases into the all-zero result. `frac_registered = recon.n_registered / recon.n_input`.

- [ ] **Step 6: Run the tests and ruff**

Run: `wsl -e bash -lc "cd ~/vggtProject && rtk .venv/bin/python -m pytest -q && rtk .venv/bin/ruff check ."`
Expected: PASS, zero ruff errors.

- [ ] **Step 7: Commit**

```bash
git add vggt_bench/eval/pose_metrics.py tests/test_pose_metrics.py
git commit -m "feat(eval): add relative pose error and AUC@5/10/30 metrics"
```

---

### Task 4: The sweep (`scripts/03_run_benchmark.py`)

**Files:**
- Create: `scripts/03_run_benchmark.py`
- Create: `vggt_bench/eval/sweep.py` (row key, CSV schema, resume logic — the testable part)
- Test: `tests/test_sweep.py`

**Interfaces:**
- Consumes: `load_scene`, `sample_frames`, `evaluate_poses`, `VGGTMethod`, `ColmapMethod`, `provenance.run_metadata`, `profiling.max_frames_from_profile`, `profiling.free_gpu_memory`.
- Produces:
  - `COLUMNS: tuple[str, ...]` = `dataset, scene, scene_type, method, n_views, seed, sampling, auc_5, auc_10, auc_30, rre_median, rte_median, n_pairs, success, n_input, n_registered, frac_registered, runtime_s, model_load_s, peak_vram_mb, failure_reason, git_commit, timestamp_utc`
  - `RunKey = tuple[str, str, str, int, str]` (dataset, scene, method, n_views, sampling-or-seed label)
  - `plan_runs(cfg: dict, methods: Sequence[str], max_frames: int | None) -> list[dict]`
  - `done_keys(csv_path: Path) -> set[RunKey]`
  - `append_row(csv_path: Path, row: dict) -> None` — writes the header on first use, flushes per row.

- [ ] **Step 1: Write the failing tests in `tests/test_sweep.py`**

```python
def test_plan_runs_covers_every_combination():
    # 2 scenes x 5 N x (3 seeds + wide) x 2 methods == 80 runs, all keys unique
def test_plan_runs_labels_the_wide_baseline_sampling():
    # sampling values are {"random", "wide"}; wide rows have seed is None
def test_plan_runs_caps_view_counts_by_the_memory_profile():
    # max_frames=10 drops N=20 and N=40 for vggt but keeps them for colmap
def test_done_keys_reads_back_appended_rows(tmp_path): ...
def test_done_keys_is_empty_for_a_missing_csv(tmp_path): ...
def test_append_row_writes_the_header_once(tmp_path):
    # two appends -> one header line, column order == COLUMNS
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `wsl -e bash -lc "cd ~/vggtProject && rtk .venv/bin/python -m pytest tests/test_sweep.py -q"`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `vggt_bench/eval/sweep.py`**

`plan_runs` iterates scenes × view counts × samplings × methods in a deterministic order; the VGGT cap comes from `max_frames_from_profile` and applies only to the `vggt` method. `done_keys` reads with `csv.DictReader` and tolerates a truncated final line.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `wsl -e bash -lc "cd ~/vggtProject && rtk .venv/bin/python -m pytest tests/test_sweep.py -q"`
Expected: PASS.

- [ ] **Step 5: Write `scripts/03_run_benchmark.py`**

CLI: `--scenes`, `--methods`, `--view-counts`, `--seeds`, `--csv` (default `results/benchmark.csv`), `--dry-run`, `--limit`. Load each method once (record `model_load_s` separately from `runtime_s`, per spec §7), loop `plan_runs` skipping `done_keys`, call `sample_frames` once per (scene, N, sampling) so **both methods see the identical subset**, run `reconstruct`, score with `evaluate_poses`, and `append_row` immediately so an interrupted sweep resumes. Wrap each run in `try/except (torch.cuda.OutOfMemoryError, Exception)` → row with `success=False` and `failure_reason`, then `free_gpu_memory()`. Write the sweep's `run_metadata` to `results/benchmark_meta.json`.

- [ ] **Step 6: Smoke the runner end to end**

Run: `wsl -e bash -lc "cd ~/vggtProject && rtk .venv/bin/python scripts/03_run_benchmark.py --scenes garden --view-counts 3 5 --seeds 0 --csv results/smoke_benchmark.csv"`
Expected: 4 rows (2 methods × 2 N), non-zero `auc_30` for at least VGGT, no traceback.

- [ ] **Step 7: Verify resumability**

Re-run the same command.
Expected: it reports every run skipped and the CSV is byte-identical.

- [ ] **Step 8: Commit**

```bash
git add vggt_bench/eval/sweep.py scripts/03_run_benchmark.py tests/test_sweep.py
git commit -m "feat(bench): add the resumable VGGT-vs-COLMAP sweep over scenes and view counts"
```

---

### Task 5: Full sweep and acceptance

**Files:**
- Modify: `results/benchmark.csv` (generated), `results/benchmark_meta.json` (generated)
- Modify: `README.md` (replace the Phase 3 TODOs with measured numbers only)

- [ ] **Step 1: Run the full sweep (hours; resumable, run in chunks if needed)**

Run: `wsl -e bash -lc "cd ~/vggtProject && rtk .venv/bin/python scripts/03_run_benchmark.py"`
Expected: 240 rows in `results/benchmark.csv`.

- [ ] **Step 2: Check the acceptance criteria from spec §7**

One row per run; all columns populated; failed runs present with `success=False`, `auc_30=0` and a `failure_reason`; both methods share frame subsets (spot-check two rows' `n_input`).

- [ ] **Step 3: Run the full gate**

Run: `wsl -e bash -lc "cd ~/vggtProject && rtk .venv/bin/python -m pytest -q && rtk .venv/bin/ruff check ."`
Expected: all tests pass, zero ruff errors.

- [ ] **Step 4: Update README with measured results and the pseudo-GT caveat**

State the dataset, the six scenes, that Mip-NeRF 360 reference poses are dense-COLMAP pseudo-ground-truth, and that COLMAP ran CPU-only SIFT here. Numbers come from `results/benchmark.csv` and nowhere else.

- [ ] **Step 5: Commit**

```bash
git add results/benchmark.csv results/benchmark_meta.json README.md
git commit -m "feat(results): record the Phase 3 benchmark sweep across six Mip-NeRF 360 scenes"
```

---

## Verification

1. `wsl -e bash -lc "cd ~/vggtProject && rtk .venv/bin/python -m pytest -q"` — all tests, including the new `test_align.py`, `test_pose_metrics.py`, `test_datasets.py`, `test_sweep.py`.
2. `wsl -e bash -lc "cd ~/vggtProject && rtk .venv/bin/ruff check ."` — zero errors.
3. The smoke sweep (Task 4 steps 6–7) proves the runner works end to end and resumes.
4. `results/benchmark.csv` has 240 rows with the schema in `COLUMNS`, and `results/benchmark_meta.json` records the git commit and config.
5. Sanity check the headline trend directly from the CSV: VGGT's mean `auc_30` at N=3 should be well above COLMAP's, and COLMAP's `success` rate should rise with N. (Figures are Phase 3's `04_make_figures.py`, planned separately after this plan lands.)

## Out of scope

- `04_make_figures.py` and the five spec figures — a separate, smaller plan once the CSV exists.
- `geometry_metrics.py` / Chamfer distance (spec marks it optional; Mip-NeRF 360 has no reference geometry).
- The `image_hw` per-image-resolution fix noted in the handoff — only matters if `camera_mode` moves off `SINGLE`, which this sweep does not do.

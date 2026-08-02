# RustSFM GPU PnP-f Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run single-camera unknown-focal PnP RANSAC, scoring, selection, and refinement through RustSFM's generic wgpu backend.

**Architecture:** Add a focal-aware GPU solver beside the existing fixed-intrinsic scorer. The new solver owns `R/t/log(f)` model buffers and wgpu compute passes; the mapper selects it for the unknown-focal route and falls back to the existing CPU PnP-f solver only for unavailable or invalid GPU execution. RustViewer surfaces the persisted terminal state rather than a stale progress operation.

**Tech Stack:** Rust, wgpu/WGSL, RustSFM, rustslam CPU reference solver, RustViewer project pipeline, cargo tests.

---

### Task 1: Define GPU focal-model ABI and failing parity tests

**Files:**
- Create: `RustSFM/src/gpu/pnp_focal.rs`
- Create: `RustSFM/src/gpu/shaders/pnp_focal.wgsl`
- Modify: `RustSFM/src/gpu/mod.rs`
- Test: `RustSFM/src/gpu/mod.rs`

- [ ] **Step 1: Add ABI layout tests before implementing the solver.**

  Add tests asserting 16-byte WGSL alignment for a pose-plus-log-focal model and a result record:

  ```rust
  #[test]
  fn wgpu_pnp_focal_abi_records_are_wgsl_aligned() {
      assert_eq!(std::mem::size_of::<GpuPnpFocalModel>(), 64);
      assert_eq!(std::mem::size_of::<GpuPnpFocalResult>(), 32);
  }
  ```

- [ ] **Step 2: Run the new test and verify it fails because focal GPU types do not exist.**

  Run: `cargo test -p rustsfm wgpu_pnp_focal_abi_records_are_wgsl_aligned`

  Expected: compile failure naming `GpuPnpFocalModel`.

- [ ] **Step 3: Add the ABI and public internal entry point.**

  Define `GpuPnpFocalModel` as three `vec4` pose rows plus `[f32; 4]` containing `log_focal` and padding. Define `GpuPnpFocalResult` with selected model index, inlier count, residual sum, focal, and a validity flag. Export `WgpuPnPFocalSolver` under `gpu-wgpu`; keep all WGSL-facing records `Pod + Zeroable`.

- [ ] **Step 4: Add WGSL declarations that exactly mirror Rust fields.**

  ```wgsl
  struct PnpFocalModel {
      row0: vec4<f32>, row1: vec4<f32>, row2: vec4<f32>,
      log_focal: f32, pad0: f32, pad1: f32, pad2: f32,
  }
  ```

- [ ] **Step 5: Run the ABI test and commit.**

  Run: `cargo test -p rustsfm wgpu_pnp_focal_abi_records_are_wgsl_aligned`

  Expected: PASS.

  Commit: `git commit -m "feat(rustsfm): define gpu focal pnp abi"`

### Task 2: Implement deterministic GPU hypothesis generation and pixel-space scoring

**Files:**
- Modify: `RustSFM/src/gpu/pnp_focal.rs`
- Modify: `RustSFM/src/gpu/shaders/pnp_focal.wgsl`
- Test: `RustSFM/src/gpu/mod.rs`

- [ ] **Step 1: Write a failing synthetic focal-scoring test.**

  Construct 32 points from a known `SE3` and `f=700`, inject six pixel outliers, run the new solver with seed 7 and 512 trials, and assert a valid result with at least 24 inliers and focal relative error below 5%.

- [ ] **Step 2: Verify the test fails because no dispatch implements focal candidates.**

  Run: `cargo test -p rustsfm wgpu_pnp_focal_recovers_synthetic_pose_and_focal`

  Expected: FAIL with an unavailable focal-GPU solver result.

- [ ] **Step 3: Implement deterministic four-point sampling and candidate buffers.**

  Use a counter-based integer hash of `(seed, trial, lane)` to generate four unique indices per trial. Dispatch one workgroup per trial. Invalid or duplicate samples write `valid = 0`; they never participate in best-model selection.

- [ ] **Step 4: Implement P4Pf and fallback candidates in WGSL.**

  Port the CPU algebraic P4Pf equations from `rustslam/src/tracker/solver.rs`, including finite-root filtering. In the same workgroup, generate the existing P3P-plus-focal-update fallback candidates when P4Pf returns no valid finite positive-focal root. Store every candidate as `R/t/log(f)`.

- [ ] **Step 5: Implement pixel-space scoring and deterministic best selection.**

  Project as `f * Xc.xy / Xc.z`, reject `z <= 0`, and compare squared pixel residual against `pnp_threshold_px^2`. Reduce inlier count and residual sum per candidate. Select by `(valid, inliers, -residual_sum, lower_candidate_index)` to make ties deterministic.

- [ ] **Step 6: Run the synthetic test and commit.**

  Run: `cargo test -p rustsfm wgpu_pnp_focal_recovers_synthetic_pose_and_focal`

  Expected: PASS on a compatible adapter; test prints a skip and returns PASS when no adapter exists.

  Commit: `git commit -m "feat(rustsfm): generate and score gpu focal pnp models"`

### Task 3: Add GPU inlier-mask generation, local refinement, and numerical rejection

**Files:**
- Modify: `RustSFM/src/gpu/pnp_focal.rs`
- Modify: `RustSFM/src/gpu/shaders/pnp_focal.wgsl`
- Test: `RustSFM/src/gpu/mod.rs`

- [ ] **Step 1: Write failing tests for refinement and invalid focal rejection.**

  Add one noisy synthetic scene requiring refinement and one scene whose returned focal is outside the configured ratio bounds. Assert refinement does not reduce support and invalid focal returns `None` rather than a non-finite model.

- [ ] **Step 2: Run the focused tests to verify failure.**

  Run: `cargo test -p rustsfm wgpu_pnp_focal_`

  Expected: refinement and bounds tests fail before refinement exists.

- [ ] **Step 3: Implement final mask and fixed-count Gauss-Newton passes.**

  Build the normal equations for six pose increments plus `log_focal`, reduce them on GPU, solve the damped 7x7 system in a single workgroup, and accept an iteration only when rescoring improves the same deterministic support ordering. Run at most ten iterations.

- [ ] **Step 4: Enforce all invalid-model conditions.**

  Reject non-finite values, non-positive focal, focal ratios outside `MapperConfig`, non-positive depth, singular normal equations, and fewer than four inliers. Return a typed GPU focal-solver error for device and shader failures; return `None` for valid-but-unsolved geometry.

- [ ] **Step 5: Run focused GPU tests and commit.**

  Run: `cargo test -p rustsfm wgpu_pnp_focal_`

  Expected: PASS or compatible-adapter skips.

  Commit: `git commit -m "feat(rustsfm): refine gpu focal pnp models"`

### Task 4: Route unknown-focal mapper calls through GPU PnP-f with CPU fallback

**Files:**
- Modify: `RustSFM/src/sfm/mapper.rs`
- Modify: `RustSFM/src/gpu/mod.rs`
- Test: `RustSFM/src/sfm/mapper.rs`

- [ ] **Step 1: Write a failing mapper test for unknown focal with GPU enabled.**

  Reuse the existing mapper synthetic observations, set `camera_has_prior_focal_length = false`, `ba_refine_focal_length = true`, `use_gpu_pnp = true`, and assert the focal route returns a valid absolute pose instead of the focal-estimation rejection.

- [ ] **Step 2: Verify it fails with the current rejection.**

  Run: `cargo test -p rustsfm mapper_gpu_pnp_focal_matches_cpu_reference`

  Expected: FAIL containing `gpu pnp does not support focal length estimation`.

- [ ] **Step 3: Replace the focal-route rejection with a typed focal GPU dispatch.**

  In `solve_absolute_pose_with_camera_hypotheses_and_pnp_scorer`, route `estimate_focal && config.use_gpu_pnp` to `WgpuPnPFocalSolver`; update the returned `CameraModel` focal parameters from the result. Retain the existing fixed-intrinsics scorer path unchanged.

- [ ] **Step 4: Add CPU fallback at the mapper boundary.**

  For GPU initialization, shader, dispatch, readback, or validation errors, call `solve_absolute_pose_with_focal_estimation` and record a `gpu_pnp_focal_fallback` diagnostic. Do not fall back for explicit cancellation; preserve the terminal error only when the CPU reference also fails.

- [ ] **Step 5: Run mapper differential tests and commit.**

  Run: `cargo test -p rustsfm mapper_gpu_pnp_focal_matches_cpu_reference` followed by `cargo test -p rustsfm mapper_gpu_pnp_matches_known_intrinsics_cpu_mask`

  Expected: PASS or compatible-adapter skip; known-intrinsics behavior remains unchanged.

  Commit: `git commit -m "feat(rustsfm): route unknown focal pnp through gpu"`

### Task 5: Synchronize RustViewer failure and backend presentation

**Files:**
- Modify: `RustViewer/src/pipeline/rustsfm_worker.rs`
- Modify: `RustViewer/src/app.rs`
- Modify: `RustViewer/src/project/session.rs`
- Test: `RustViewer/tests/pipeline_coordinator.rs`
- Test: `RustViewer/src/project/session.rs`

- [ ] **Step 1: Add failing presentation tests.**

  Create a failed `keyframe_sfm` stage with an error detail and a stale worker operation. Assert `ProjectSessionSummary` presents `Failed` and the UI activity text includes the persisted error, not `MatchPairBatch`.

- [ ] **Step 2: Run the test to prove the stale operation wins today.**

  Run: `cargo test -p rust-viewer project_session_failed_pose_overrides_worker_operation`

  Expected: FAIL before state precedence is corrected.

- [ ] **Step 3: Propagate PnP backend telemetry through the RustSFM worker.**

  Emit `GpuPnPFocal`, `CpuPnPFocalFallback`, and fallback reason in worker progress diagnostics. On stage completion or failure, clear the transient operation so persisted manifest state owns presentation.

- [ ] **Step 4: Give failed manifest state precedence in the UI.**

  When `keyframe_sfm` is failed, render its persisted error summary/detail and disable downstream PnP and training actions. Do not display a running operation for the failed stage.

- [ ] **Step 5: Run RustViewer tests and commit.**

  Run: `cargo test -p rust-viewer project_session_failed_pose_overrides_worker_operation` followed by `cargo test -p rust-viewer pipeline_coordinator`

  Expected: PASS.

  Commit: `git commit -m "fix(rustviewer): present focal pnp backend and failures"`

### Task 6: Verify the integrated release path

**Files:**
- Modify: `docs/superpowers/specs/2026-08-02-rustsfm-gpu-pnpf-design.md` only if verification changes an approved behavior

- [ ] **Step 1: Run the full focused test suites.**

  Run: `cargo test -p rustsfm` and `cargo test -p rust-viewer`

  Expected: PASS, with GPU tests skipped only where no wgpu adapter is available.

- [ ] **Step 2: Build the release application.**

  Run: `cargo build --release -p rust-viewer`

  Expected: release binary at `target/release/rust-viewer`.

- [ ] **Step 3: Execute the current unknown-focal fixture.**

  Run the existing RustViewer project pipeline against `test_data/flowers2/out8/Untitled.rustscanproject` and assert it either registers keyframes with GPU PnP-f or logs a CPU PnP-f fallback. It must not emit `gpu pnp does not support focal length estimation`.

- [ ] **Step 4: Review git diff and commit verification-only adjustments.**

  Run: `git diff --check` and `git status --short`

  Expected: no whitespace errors and no unrelated files staged.

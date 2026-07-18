# RustSFM wgpu PnP Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Batch known-intrinsics central-camera P3P/PnP RANSAC model scoring on wgpu while preserving RustSLAM's CPU sampling, minimal solving, local optimization, final refinement, and mapper acceptance behavior.

**Architecture:** RustSLAM receives a backend-neutral `PnPModelScorer` trait and a fallible `solve_with_model_scorer` entry point. RustSFM implements the trait with an owned wgpu PnP scoring session that uploads observations once, dispatches fixed 64-trial model batches, and reads a mask only for candidates that can improve the current support. Mapper and CLI create one scorer only for explicit `use_gpu_pnp`; unsupported routes and GPU failures propagate without CPU fallback.

**Tech Stack:** Rust 2021, wgpu 29, WGSL, nalgebra/glam, existing RustSLAM P3P/EPnP and COLMAP RANSAC rules, anyhow at the RustSFM boundary.

---

## File Map

- `RustSLAM/src/tracker/solver.rs`: backend-neutral support type, scorer trait, 64-trial PnP batching, and fallible solver entry point.
- `RustSLAM/src/tracker/mod.rs`: re-export scorer contract types for RustSFM.
- `RustSFM/src/gpu/pnp_scorer.rs`: owned wgpu PnP pipelines, observation/model ABI, summary and mask readback.
- `RustSFM/src/gpu/shaders/pnp_scoring.wgsl`: 64-lane pose projection and residual reduction.
- `RustSFM/src/gpu/mod.rs`: module wiring, public scorer export, and GPU ABI/residual tests.
- `RustSFM/src/sfm/mapper/config.rs`: `use_gpu_pnp` configuration with a disabled default.
- `RustSFM/src/sfm/mapper.rs`: scorer creation, incremental pipeline plumbing, known-intrinsics absolute-pose backend, and unsupported-route errors.
- `RustSFM/src/cli/mod.rs`: native `reconstruct --use-gpu-pnp` and COLMAP-compatible `--Mapper.use_gpu_pnp` parsing.
- `RustSFM/src/cli/project.rs`: resolved mapper option for the COLMAP-compatible command.
- `RustSFM/src/cli/commands.rs`: map parsed flags into `MapperConfig` and reject unsupported global mapper use.
- `docs/superpowers/specs/2026-07-18-rustsfm-wgpu-pnp-design.md`: approved design contract.

### Task 1: RustSLAM PnP Scorer Contract And CPU-Compatible Batching

**Files:**
- Modify: `RustSLAM/src/tracker/solver.rs`
- Modify: `RustSLAM/src/tracker/mod.rs`
- Test: `RustSLAM/src/tracker/solver.rs` test module

- [x] **Step 1: Write the failing contract and ordering tests**

Add the following public backend-neutral types in `solver.rs` before `PnPSolver`:

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PnPModelSupport {
    pub inliers: usize,
    pub residual_sum: f64,
}

pub trait PnPModelScorer {
    type Error;

    fn prepare(
        &mut self,
        normalized_points: &[[f32; 2]],
        object_points: &[[f32; 3]],
        threshold: f32,
    ) -> Result<(), Self::Error>;

    fn score_models(
        &mut self,
        models: &[SE3],
    ) -> Result<Vec<PnPModelSupport>, Self::Error>;

    fn inlier_mask(&mut self, model: &SE3) -> Result<Vec<bool>, Self::Error>;
}
```

In the solver tests, add a `RecordingPnPScorer` that records every model in `score_models`, computes CPU projection support using the existing `project_point` semantics, and increments a mask counter. Add tests with a deterministic 32-point scene:

```rust
#[test]
fn pnp_model_scorer_receives_stable_trial_order_and_selected_masks() {
    let problem = synthetic_pnp_problem_with_fixed_outliers();
    let solver = PnPSolver { ransac_max_iterations: 128, ransac_random_seed: Some(7), ..PnPSolver::new(1.0, 1.0, 0.0, 0.0) };
    let mut first_scorer = RecordingPnPScorer::default();
    let gpu_like = solver.solve_with_model_scorer(&problem, &mut first_scorer).expect("scored PnP").expect("pose");
    let mut second_scorer = RecordingPnPScorer::default();
    solver.solve_with_model_scorer(&problem, &mut second_scorer).expect("second scored PnP").expect("second pose");
    assert!(gpu_like.1.iter().filter(|&&value| value).count() >= 24);
    assert_eq!(
        first_scorer.models.iter().map(SE3::to_matrix).collect::<Vec<_>>(),
        second_scorer.models.iter().map(SE3::to_matrix).collect::<Vec<_>>(),
    );
    assert!(first_scorer.mask_calls > 0);
    assert!(first_scorer.mask_calls < first_scorer.models.len());
}

#[test]
fn pnp_model_scorer_errors_propagate() {
    let problem = synthetic_pnp_problem_with_fixed_outliers();
    let solver = PnPSolver::new(1.0, 1.0, 0.0, 0.0);
    let mut scorer = FailingPnPScorer;
    assert!(solver.solve_with_model_scorer(&problem, &mut scorer).is_err());
}
```

The test must fail to compile because `PnPModelScorer` and `solve_with_model_scorer` do not exist yet.

- [x] **Step 2: Run the contract tests and verify RED**

Run: `cargo test -p rustslam pnp_model_scorer -- --nocapture`

Expected: FAIL with unresolved `PnPModelScorer`/`solve_with_model_scorer` symbols, not with a test typo or missing fixture.

- [x] **Step 3: Add the backend contract and public re-exports**

Add `PnPModelSupport` and `PnPModelScorer` exactly as specified above and re-export them from `RustSLAM/src/tracker/mod.rs`:

```rust
pub use solver::{PnPModelScorer, PnPModelSupport, PnPFocalResult, PnPProblem, PnPSolver};
```

Keep the trait free of wgpu, anyhow, and RustSFM types. Its associated error type lets RustSFM use `anyhow::Error` without adding anyhow to RustSLAM.

- [x] **Step 4: Add the fallible 64-trial solver path**

Refactor the current `PnPSolver::solve` loop into an internal generic function with the following shape:

```rust
fn solve_with_backend<S: PnPModelScorer + ?Sized>(
    &self,
    problem: &PnPProblem,
    scorer: &mut S,
    chunk_trials: usize,
) -> Result<Option<(SE3, Vec<bool>)>, S::Error>;
```

The function normalizes points once, computes the current CPU threshold, calls `scorer.prepare`, and then processes `chunk_trials` trial indices at a time. GPU uses 64 and CPU uses 1. Each trial still calls the existing `random_indices` and `solve_p3p` in order. Candidate `SE3` models are appended in trial/solution order and passed to `score_models` once per chunk. For a support summary that can beat `best_support`, call `inlier_mask`, construct `PoseEvaluation`, and run the existing `local_optimize_pose` on CPU. Update `dynamic_max_trials` only after CPU local optimization.

Use this exact boundary helper and add its unit tests:

```rust
const PNP_GPU_CHUNK_TRIALS: usize = 64;

fn pnp_ransac_chunk_end(
    iteration: usize,
    max_trials: usize,
    dynamic_trials: usize,
    min_trials: usize,
    chunk_trials: usize,
) -> usize {
    dynamic_trials
        .max(min_trials)
        .saturating_add(1)
        .min(max_trials)
        .min(iteration.saturating_add(chunk_trials))
}
```

The existing `solve` method remains an `Option` wrapper using a CPU scorer adapter; its public behavior and deterministic seed handling remain unchanged. Add the fallible public wrapper:

```rust
pub fn solve_with_model_scorer<S: PnPModelScorer + ?Sized>(
    &self,
    problem: &PnPProblem,
    scorer: &mut S,
) -> Result<Option<(SE3, Vec<bool>)>, S::Error> {
    self.solve_with_backend(problem, scorer, PNP_GPU_CHUNK_TRIALS)
}
```

The existing CPU `solve` wrapper calls the same backend with `chunk_trials = 1`, so its per-trial dynamic abort and deterministic behavior remain unchanged. The CPU adapter must use the current `evaluate_pose` implementation, including its existing `z <= 0` projection-to-zero behavior. Do not alter focal estimation or `solve_with_estimated_focal`.

- [x] **Step 5: Run RustSLAM solver tests and commit the contract**

Run: `cargo test -p rustslam pnp_model_scorer -- --nocapture`

Run: `cargo test -p rustslam tracker::solver -- --nocapture`

Expected: the new recording/error tests and all existing solver tests pass with zero failures.

```bash
git add RustSLAM/src/tracker/solver.rs RustSLAM/src/tracker/mod.rs
git commit -m "feat(rustslam): add batchable PnP scoring contract"
```

### Task 2: wgpu PnP Scoring Session

**Files:**
- Create: `RustSFM/src/gpu/pnp_scorer.rs`
- Create: `RustSFM/src/gpu/shaders/pnp_scoring.wgsl`
- Modify: `RustSFM/src/gpu/mod.rs`
- Test: `RustSFM/src/gpu/mod.rs` GPU tests

- [x] **Step 1: Write failing ABI and residual tests**

Add tests guarded by `#[cfg(feature = "gpu-wgpu")]` that skip when `WgpuContext::try_new_optional()` returns `None`:

```rust
#[test]
fn wgpu_pnp_abi_records_are_wgsl_aligned() {
    assert_eq!(std::mem::size_of::<GpuPnpImagePoint>(), 16);
    assert_eq!(std::mem::size_of::<GpuPnpObjectPoint>(), 16);
    assert_eq!(std::mem::size_of::<GpuPnpModel>(), 48);
}

#[test]
fn wgpu_pnp_scorer_matches_cpu_projection_and_mask() -> anyhow::Result<()> {
    let Some(context) = WgpuContext::try_new_optional()? else { return Ok(()); };
    let mut scorer = WgpuPnpModelScorer::from_context(context)?;
    let image = [[0.0, 0.0], [0.1, 0.0], [-0.2, 0.2], [0.0, 0.0]];
    let world = [[0.0, 0.0, 2.0], [0.2, 0.0, 2.0], [-0.4, 0.4, 2.0], [0.0, 0.0, -1.0]];
    scorer.prepare(&image, &world, 0.01)?;
    let supports = scorer.score_models(&[SE3::identity()])?;
    let mask = scorer.inlier_mask(&SE3::identity())?;
    assert_eq!(supports[0].inliers, 4);
    assert_eq!(mask, vec![true, true, true, true]);
    Ok(())
}
```

The test must fail because the scorer types and shader are not present.

- [x] **Step 2: Run the GPU scorer tests and verify RED**

Run: `cargo test -p rustsfm --lib wgpu_pnp_ --features gpu-wgpu -- --nocapture`

Expected: FAIL with missing scorer symbols before any adapter-dependent assertion.

- [x] **Step 3: Implement the owned scorer and WGSL pipeline**

Implement `WgpuPnpModelScorer` with `from_context`, `try_new`, `prepare`, `score_models`, and `inlier_mask`. The scorer owns its observation buffers, bind-group layout, support pipeline, and mask pipeline so `prepare` can replace only the two observation buffers while retaining the device and pipelines across mapper solves. Use a 64-lane workgroup and one workgroup per model. The shader reads `[R | t]`, projects each 3D point, uses `[0, 0]` for non-positive z, and reduces only finite residuals that are below `threshold^2`.

Validate matching nonzero lengths, finite f32 conversion, nonnegative finite threshold, buffer sizes against `max_buffer_size` and `max_storage_buffer_binding_size`, model count against `u32`, and mask dispatch count against `max_compute_workgroups_per_dimension`. Read back `GpuModelSupport` summaries in model order and verify the selected mask count equals the selected summary count. Return `anyhow::Error` with `gpu pnp` context for every device, readback, validation, or summary/mask mismatch.

Implement the RustSLAM trait in `pnp_scorer.rs`:

```rust
impl PnPModelScorer for WgpuPnpModelScorer {
    type Error = anyhow::Error;

    fn prepare(&mut self, points: &[[f32; 2]], objects: &[[f32; 3]], threshold: f32)
        -> Result<(), Self::Error>
    {
        WgpuPnpModelScorer::prepare(self, points, objects, threshold)
    }

    fn score_models(&mut self, models: &[SE3])
        -> Result<Vec<PnPModelSupport>, Self::Error>
    {
        WgpuPnpModelScorer::score_models(self, models)
    }

    fn inlier_mask(&mut self, model: &SE3) -> Result<Vec<bool>, Self::Error> {
        WgpuPnpModelScorer::inlier_mask(self, model)
    }
}
```

Add `#[cfg(feature = "gpu-wgpu")] mod pnp_scorer;` and re-export `WgpuPnpModelScorer` from `RustSFM/src/gpu/mod.rs`. In non-GPU builds, do not expose a fake scoring implementation; explicit mapper requests must fail at configuration/runtime routing.

- [x] **Step 4: Run scorer tests and GPU feature checks**

Run: `cargo test -p rustsfm --lib wgpu_pnp_ --features gpu-wgpu -- --nocapture`

Run: `cargo check -p rustsfm --no-default-features`

Expected: GPU residual/mask tests pass on Metal, and the CPU-only build has no missing wgpu-gated symbols.

```bash
git add RustSFM/src/gpu/pnp_scorer.rs RustSFM/src/gpu/shaders/pnp_scoring.wgsl RustSFM/src/gpu/mod.rs
git commit -m "feat(rustsfm): add wgpu PnP model scorer"
```

### Task 3: Mapper Backend Routing And Known-Intrinsics PnP

**Files:**
- Modify: `RustSFM/src/sfm/mapper/config.rs`
- Modify: `RustSFM/src/sfm/mapper.rs`
- Test: `RustSFM/src/sfm/mapper.rs` mapper tests

- [x] **Step 1: Write failing mapper routing/error tests**

Add tests for the default and explicit route:

```rust
#[test]
fn mapper_gpu_pnp_is_disabled_by_default() {
    assert!(!MapperConfig::default().use_gpu_pnp);
}

#[cfg(feature = "gpu-wgpu")]
#[test]
fn mapper_gpu_pnp_rejects_focal_estimation_route() {
    let error = validate_gpu_pnp_route(true, false, false).expect_err("focal route must reject");
    assert!(error.to_string().contains("focal"));
}
```

Add a deterministic mapper-level parity fixture that supplies known camera priors and enough structure-based observations for one candidate registration. It invokes the backend-aware absolute pose helper twice with the same seed and asserts equal masks and at least 90% of the CPU inlier count.

Run: `cargo test -p rustsfm --lib mapper::tests::mapper_gpu_pnp --features gpu-wgpu -- --nocapture`

Expected: FAIL because `use_gpu_pnp`, validation, and the backend-aware mapper helper are undefined.

- [x] **Step 2: Add config validation and persistent scorer creation**

Add `pub use_gpu_pnp: bool` to `MapperConfig`, defaulting to `false`. Add `validate_gpu_pnp_config(config: &MapperConfig, has_global_mapper: bool) -> anyhow::Result<()>` that rejects GPU PnP with the global mapper and returns a clear error when the crate lacks `gpu-wgpu`. Add `validate_gpu_pnp_route(estimate_focal: bool, generalized: bool, structureless: bool) -> anyhow::Result<()>` for route-specific rejection. Under `gpu-wgpu`, `run_reconstruction_impl` creates one `WgpuPnpModelScorer` after configuration validation and passes it only to the incremental pipeline. The global mapper branch never receives it.

Keep `run_reconstruction` and `run_incremental_pipeline` public signatures unchanged. Add private scorer-aware wrappers and let existing wrappers pass `None`:

```rust
fn incremental_pipeline_map_with_pnp_scorer(
    frames: &[ImageFrame],
    camera: CameraModel,
    reference: Option<&ReferenceCameraSetup>,
    pairs: &[PairGeometry],
    config: &MapperConfig,
    pnp_scorer: Option<&mut dyn PnPModelScorer<Error = anyhow::Error>>,
) -> Result<IncrementalPipelineResult>;
```

Thread the optional scorer through registration selection only; triangulation, BA, global mapper, and structureless registration remain unchanged.

- [x] **Step 3: Add fallible known-intrinsics absolute-pose backend**

Preserve existing `Option` helpers as CPU wrappers and add an internal fallible helper with an optional scorer. For the known-camera path, build the same normalized `PnPProblem` and call `PnPSolver::solve_with_model_scorer` when a scorer is supplied. For the CPU path call the existing `solve` method. Keep `evaluate_absolute_pose`, inlier-only reprojection refinement, camera parameter refinement, and acceptance thresholds shared.

When `use_gpu_pnp=true`, return an error instead of entering `solve_with_estimated_focal`, `solve_generalized_frame_absolute_pose`, or `solve_structureless_absolute_pose`. Do not convert any GPU scorer error to `None`.

Thread `Result<Option<AbsolutePose>>` through the scorer-aware registration choice and `mark_unregistered_images_with_no_absolute_pose` wrappers. Existing CPU test helpers continue to call the infallible wrappers and retain their signatures.

- [x] **Step 4: Run mapper parity and CPU regression tests**

Run: `cargo test -p rustsfm --lib mapper::tests --features gpu-wgpu -- --nocapture`

Run: `cargo test -p rustsfm --lib --no-default-features -- mapper::tests --nocapture`

Expected: mapper CPU tests pass in both configurations; GPU parity tests pass or skip only on missing adapter; unsupported GPU routes return errors.

```bash
git add RustSFM/src/sfm/mapper/config.rs RustSFM/src/sfm/mapper.rs
git commit -m "feat(rustsfm): route mapper PnP scoring through wgpu"
```

### Task 4: CLI Flags And Explicit Error Propagation

**Files:**
- Modify: `RustSFM/src/cli/mod.rs`
- Modify: `RustSFM/src/cli/project.rs`
- Modify: `RustSFM/src/cli/commands.rs`
- Test: `RustSFM/src/cli/mod.rs` and `RustSFM/src/cli/commands.rs` tests

- [x] **Step 1: Write failing CLI parsing tests**

Add parser tests for both native and COLMAP-compatible forms:

```rust
#[test]
fn native_reconstruct_parses_gpu_pnp() {
    let cli = Cli::try_parse_from(["rustsfm", "reconstruct", "--input", "in", "--output", "out", "--use-gpu-pnp"])
        .expect("native GPU PnP flag");
    let Commands::Reconstruct(args) = cli.command else { panic!("reconstruct command"); };
    assert!(args.use_gpu_pnp);
}

#[test]
fn colmap_mapper_parses_gpu_pnp() {
    let cli = Cli::try_parse_from(["rustsfm", "mapper", "--Mapper.use_gpu_pnp", "1"])
        .expect("COLMAP GPU PnP flag");
    let Commands::Mapper(args) = cli.command else { panic!("mapper command"); };
    assert_eq!(args.use_gpu_pnp, Some(1));
}
```

Run: `cargo test -p rustsfm --bin rustsfm cli::tests:: -- --nocapture`

Expected: FAIL because both fields are undefined.

- [x] **Step 2: Implement flag mapping and command validation**

Add `use_gpu_pnp: bool` to native `ReconstructArgs`, `use_gpu_pnp: Option<i32>` to `ColmapMapperArgs`, and the corresponding optional field to the resolved mapper project options. Map `--use-gpu-pnp` directly into `MapperConfig`. Parse `--Mapper.use_gpu_pnp` with the existing `colmap_optional_bool` helper and pass it through `run_colmap_mapper`.

The command must report the scorer's contextual error and exit nonzero when the feature is not compiled, no adapter exists, or an unsupported mapper route is selected. It must not print a successful reconstruction summary after a GPU PnP error.

- [x] **Step 3: Run CLI tests and CPU-only command compilation**

Run: `cargo test -p rustsfm --bin rustsfm --features gpu-wgpu -- --nocapture`

Run: `cargo check -p rustsfm --no-default-features`

Run: `cargo fmt -p rustsfm -- --check`

Expected: CLI parsing and existing command tests pass, CPU-only build succeeds, and formatting is clean.

```bash
git add RustSFM/src/cli/mod.rs RustSFM/src/cli/project.rs RustSFM/src/cli/commands.rs
git commit -m "feat(rustsfm): expose explicit GPU PnP mapper flag"
```

### Task 5: Verification, Review, And Benchmark

**Files:**
- Modify: `docs/superpowers/plans/2026-07-18-rustsfm-wgpu-pnp.md`
- Modify: mapper debug reporting only if timing fields are not already available.

- [x] **Step 1: Run focused GPU and CPU verification**

Run:

```bash
cargo test -p rustslam pnp_model_scorer -- --nocapture
cargo test -p rustsfm --lib wgpu_pnp_ --features gpu-wgpu -- --nocapture
cargo test -p rustsfm --lib mapper::tests::mapper_gpu_pnp --features gpu-wgpu -- --nocapture
cargo test -p rustsfm --bin rustsfm --features gpu-wgpu -- --nocapture
cargo check -p rustsfm --no-default-features
cargo fmt -p rustsfm -- --check
```

Expected: all focused tests pass or skip only for unavailable GPU adapters; CPU-only compile and formatting succeed.

- [x] **Step 2: Run the full non-fixture library suite and record known baseline failures**

Run: `cargo test -p rustsfm --lib --features gpu-wgpu -- --skip real_colmap_sparse`

Expected: the existing non-fixture suite passes. The local ignored `flowers2_colmap` symlink currently has zero image observations/point tracks, so the 19 `real_colmap_sparse` tests remain a data-fixture blocker documented by the previous plan.

- [x] **Step 3: Run static checks and review changed-file diagnostics**

Run: `cargo clippy -p rustslam --lib --no-deps -- -D warnings`

Run: `cargo clippy -p rustsfm --lib --features gpu-wgpu --no-deps --message-format short -- -D warnings`

Record repository baseline diagnostics, and fix any diagnostic introduced in the new or modified lines. Run `git diff --check` and inspect every staged hunk before committing.

Observed baseline: strict clippy reports 105 existing RustSLAM diagnostics and 138 existing RustSFM diagnostics. The new timing helper initially triggered `too_many_arguments`; its stage values were subsequently grouped into a fixed array, and the modified lines no longer add that diagnostic.

- [x] **Step 4: Benchmark the real dataset without changing user data**

Use separate output directories and fixed seeds for CPU and GPU mapper runs on `/Users/tfjiang/Projects/RustScan/test_data/flowers2`. Record the existing debug timing lines plus P3P candidate generation, GPU summary/mask readback, local optimization, registered image count, and point count. Compare GPU quality against the CPU run and require at least 95% of CPU registered images and points before calling the acceleration usable.

Observed results:

- CPU baseline on 16 consecutive images: `registered_images=16`, `points=4674`, `pairs=49`, `elapsed_ms=5567.68`; mapper timings were `extract=4591.70 ms`, `pairs=5.35 ms`, and `incremental=911.33 ms`.
- The same GPU command selected the Apple M5 Max Metal adapter and emitted `pnp_timing backend=batch64` for known-intrinsics PnP solves. Final review removed a redundant summary dispatch/readback from every mask request and reduced the unused score-only mask allocation to a four-byte binding placeholder; repeated timings were approximately `score_ms=2.6-2.7` and `mask_ms=1.3-1.5` for the representative solves. The run then reached the explicit unsupported `structureless absolute pose` route and returned an error without CPU fallback (`real=5.46 s`), so no GPU quality/count comparison is valid yet. End-to-end acceleration is therefore not called usable until that route is implemented or excluded by configuration.
- The 19 `real_colmap_sparse` tests remain a fixture blocker: `test_data/flowers2_colmap` points to a sparse export with zero image observations and zero point tracks.

- [x] **Step 5: Review and commit the completed plan**

After fresh verification, update all checkboxes and the observed baseline note in this plan. Use `superpowers:requesting-code-review` for a final review, then commit only the plan/status update if it is not already included in the implementation commits:

```bash
git add docs/superpowers/plans/2026-07-18-rustsfm-wgpu-pnp.md
git commit -m "docs(rustsfm): track wgpu PnP implementation"
```

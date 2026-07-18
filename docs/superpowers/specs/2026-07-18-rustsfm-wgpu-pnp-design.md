# RustSFM wgpu PnP Scoring Design

**Status:** Approved for planning

**Goal:** Accelerate the known-intrinsics central-camera P3P/PnP RANSAC path used by the
incremental mapper by batching model scoring on wgpu, while preserving the existing CPU
sampler, minimal solver, local optimization, final refinement, and mapper acceptance logic.

## Scope

This change covers only `PnPSolver::solve` for central cameras with known intrinsics. In the
RustSFM mapper this is the `solve_absolute_pose_for_camera` path after observations have been
converted to normalized camera coordinates.

The following remain CPU-only and are not silently substituted when GPU PnP is requested:

- focal-length estimation through `PnPSolver::solve_with_estimated_focal`;
- generalized rig absolute pose;
- structureless absolute pose;
- P3P hypothesis generation, EPnP local optimization, final pose refinement, camera
  refinement, bundle adjustment, and mapper control flow.

## Architecture

RustSLAM owns P3P/PnP sampling and optimization, so it also owns the backend-neutral scoring
contract. RustSLAM must not depend on wgpu or RustSFM. RustSFM implements the contract with a
wgpu scoring session and injects it only when `MapperConfig.use_gpu_pnp` is enabled.

The preferred interface is a small trait with an associated error type:

```rust
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

`PnPSolver::solve` keeps its current infallible `Option<(SE3, Vec<bool>)>` API and CPU
behavior. A new fallible `solve_with_model_scorer` entry point uses the same internal RANSAC
implementation and returns `Result<Option<(SE3, Vec<bool>)>, S::Error>`. A CPU scorer adapter
allows the shared implementation to preserve one control flow without introducing a GPU
dependency.

## RANSAC Data Flow

The PnP GPU path uses fixed 64-trial chunks. For each chunk:

1. The existing COLMAP-compatible CPU RNG samples three correspondences per trial.
2. The existing CPU P3P solver generates zero to four `SE3` models per sample in its current
   order.
3. The models are appended in trial and solver order and scored in one GPU dispatch.
4. Compact `(inlier_count, residual_sum)` summaries are processed in that same order.
5. A candidate that can beat the current best requests one mask from the GPU, then enters the
   existing CPU local optimization path.
6. The CPU-refined support updates the best model and dynamic trial bound.
7. A reduced dynamic bound takes effect at the next chunk boundary. The current chunk always
   completes, matching the already established wgpu two-view batching contract.

The effective chunk end is:

```text
min(iteration + 64,
    min(max_trials, max(dynamic_trials, min_trials) + 1))
```

The `+1` preserves the existing zero-based COLMAP abort gate. If P3P generates no model for a
sample, the trial still counts. Candidate ordering and tie-breaking remain stable.

The GPU never runs Rayon work for this path. CPU work is serial hypothesis generation and
optimization, not a parallel CPU substitute for model scoring.

## GPU ABI And Residual

The scorer uploads observations once per PnP solve:

```rust
#[repr(C)]
struct GpuPnpImagePoint {
    x: f32,
    y: f32,
    pad0: f32,
    pad1: f32,
}

#[repr(C)]
struct GpuPnpObjectPoint {
    x: f32,
    y: f32,
    z: f32,
    pad: f32,
}

#[repr(C)]
struct GpuPnpModel {
    row0: [f32; 4],
    row1: [f32; 4],
    row2: [f32; 4],
}
```

`GpuPnpModel` is the row-major world-to-camera transform `[R | t]` obtained from
`SE3::rotation_matrix()` and `SE3::translation()`. All records have 16-byte-aligned WGSL
storage layouts.

For observation `i`, the shader computes:

```text
camera = R * object[i] + t
projected = camera.z > 0 ? camera.xy / camera.z : [0, 0]
residual = squared_distance(image[i], projected)
inlier = residual <= threshold^2
```

The `z <= 0` behavior deliberately matches the current RustSLAM CPU implementation. Changing
positive-depth semantics is a separate correctness change, not part of GPU acceleration.

One 64-lane workgroup scores one model and reduces its inlier count and inlier residual sum.
The summary readback is one `GpuModelSupport` per model. Mask dispatch writes one `u32` per
observation only for a selected candidate.

All host inputs must have matching nonzero lengths, fit wgpu buffer and dispatch limits, and
remain finite after f32 conversion. Model matrices and thresholds receive the same checks.

## Mapper And CLI Integration

`MapperConfig` gains:

```rust
pub use_gpu_pnp: bool
```

It defaults to `false`. The native `reconstruct` command exposes `--use-gpu-pnp`; the
COLMAP-compatible mapper exposes `--Mapper.use_gpu_pnp 0|1`. These flags are independent of
SIFT extraction and matching backend selection.

When enabled, reconstruction creates one persistent `WgpuPnpModelScorer` from one
`Arc<WgpuContext>` and passes it through the incremental mapper absolute-pose call chain.
Individual PnP solves create lightweight prepared sessions that reuse the scorer pipelines
and upload only that image's current 2D-3D observations.

Known-intrinsics central-camera registration uses GPU scoring. If the mapper decides that
focal estimation is required, or selects a generalized/structureless path, it returns a
clear unsupported-configuration error for an explicit GPU request instead of executing CPU
model scoring. This prevents a report from claiming GPU PnP while silently using CPU RANSAC.

The global mapper does not use this incremental PnP path. `use_gpu_pnp=true` with
`global_mapper=true` is rejected during configuration validation.

## Error Handling

The current mapper absolute-pose functions return `Option`, so the GPU path requires a
fallible backend boundary. Internal absolute-pose helpers become `Result<Option<_>>`; CPU
wrappers retain current `Option` behavior where public compatibility requires it.

The following errors propagate to the CLI with context:

- RustSFM built without `gpu-wgpu` while GPU PnP is requested;
- no compatible wgpu adapter or device creation failure;
- shader or pipeline creation failure;
- buffer-size or dispatch-limit overflow;
- non-finite observations, models, threshold, support, or summary/mask disagreement;
- unsupported focal-estimation, generalized, structureless, or global-mapper route.

There is no automatic CPU fallback after explicit GPU selection.

## Testing

RustSLAM tests verify the backend-neutral contract without requiring a GPU:

- fixed 64-trial chunk boundaries preserve the zero-based abort gate;
- a recording scorer receives candidates in P3P trial/model order;
- only a promising candidate requests a mask;
- scorer errors propagate from `solve_with_model_scorer`;
- the existing CPU `solve` deterministic fixtures retain their masks and poses.

RustSFM GPU tests verify:

- WGSL ABI record sizes and buffer validation;
- single-model residuals and masks match CPU PnP evaluation, including `z <= 0` behavior;
- batched summaries preserve model order and support tie-breaking;
- repeated GPU solves with a fixed seed return the same mask;
- GPU inlier support is at least 90% of CPU support on a deterministic synthetic pose fixture;
- explicit unsupported routes and device/readback failures are returned rather than converted
  to missing registrations;
- CLI flags populate `MapperConfig.use_gpu_pnp`.

GPU tests skip only when `WgpuContext::try_new_optional` reports no compatible adapter.
CPU-only compilation with `--no-default-features`, formatting, focused mapper tests, and the
existing non-fixture RustSFM library suite remain required before commit.

## Performance Acceptance

The implementation is accepted only if instrumentation confirms that a PnP solve uploads its
observations once and scores multiple candidates per dispatch. The `flowers2` benchmark must
report separately:

- total mapper wall time;
- time in P3P candidate generation;
- time in GPU support and mask readback;
- time in CPU local optimization and final refinement;
- registered image and point counts relative to the CPU run.

Performance work must not reduce registration quality merely to lower wall time. A benchmark
run is considered comparable when registered images and sparse points remain within 5% of the
fixed-seed CPU result, with no new GPU or mapper errors.

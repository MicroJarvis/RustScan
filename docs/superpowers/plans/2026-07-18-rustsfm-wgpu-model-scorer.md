# RustSFM wgpu Model Scorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persistent wgpu service that scores batches of two-view 3x3 models and reads back compact support summaries or one selected inlier mask.

**Architecture:** The CPU keeps candidate generation and RANSAC control flow. `WgpuModelScorer` uploads row-major models and point pairs, dispatches one 64-lane workgroup per model, and reduces inlier count plus the sum of inlier residuals; a separate kernel entry point writes the selected model's mask. Homographies use forward projection error, while Essential and Fundamental matrices share squared Sampson error.

**Tech Stack:** Rust 2021, wgpu 29, WGSL compute shaders, bytemuck, anyhow.

---

## File Map

- `RustSFM/src/gpu/scorer.rs`: public scorer API, input validation, GPU buffers, dispatch, and readback conversion.
- `RustSFM/src/gpu/shaders/model_scoring.wgsl`: Homography and Sampson residuals, workgroup reduction, and mask output.
- `RustSFM/src/gpu/mod.rs`: module exports and hardware-backed contract tests.

### Task 1: Homography Support Summaries And Mask

**Files:**
- Create: `RustSFM/src/gpu/scorer.rs`
- Create: `RustSFM/src/gpu/shaders/model_scoring.wgsl`
- Modify: `RustSFM/src/gpu/mod.rs`

- [x] **Step 1: Write the failing Homography contract test**

Add `wgpu_model_scorer_scores_homographies_and_reads_mask` under `gpu::tests`. Score identity and translated row-major homographies against three identical point pairs, assert support `[3, 0]`, zero identity residual sum, and the identity mask `[true, true, true]`.

- [x] **Step 2: Run the test and verify RED**

Run: `cargo test -p rustsfm --lib wgpu_model_scorer_scores_homographies_and_reads_mask --features gpu-wgpu -- --nocapture`

Expected: FAIL to compile because `WgpuModelScorer` and `TwoViewModelKind` are undefined.

- [x] **Step 3: Implement the scorer service and Homography kernel**

Expose this API from `gpu/mod.rs`:

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuModelSupport {
    pub inliers: u32,
    pub residual_sum: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TwoViewModelKind {
    Sampson,
    HomographyForward,
}

pub struct WgpuModelScorer {
    context: Arc<WgpuContext>,
    bind_group_layout: wgpu::BindGroupLayout,
    score_pipeline: wgpu::ComputePipeline,
    mask_pipeline: wgpu::ComputePipeline,
}

impl WgpuModelScorer {
    pub fn try_new() -> Result<Self>;
    pub fn from_context(context: Arc<WgpuContext>) -> Result<Self>;
    pub fn score_two_view_models(
        &self,
        models: &[[f32; 9]],
        points1: &[[f32; 2]],
        points2: &[[f32; 2]],
        threshold: f32,
        kind: TwoViewModelKind,
    ) -> Result<Vec<GpuModelSupport>>;
    pub fn inlier_mask(
        &self,
        model: &[f32; 9],
        points1: &[[f32; 2]],
        points2: &[[f32; 2]],
        threshold: f32,
        kind: TwoViewModelKind,
    ) -> Result<Vec<bool>>;
}
```

Use a flat row-major `array<f32>` for models to avoid WGSL matrix padding. For Homography `H`, evaluate:

```wgsl
let z = h20 * x + h21 * y + h22;
let predicted = vec2<f32>(
    (h00 * x + h01 * y + h02) / z,
    (h10 * x + h11 * y + h12) / z,
);
let delta = point2 - predicted;
let residual = dot(delta, delta);
```

Treat a non-finite residual or `abs(z) <= 1e-12` as an outlier. The public API accepts the same unsquared threshold as the CPU scorer; the host sends `threshold.max(1e-12).powi(2)` to WGSL. Sum residuals only for observations satisfying `residual <= max_residual`, matching the CPU support definition.

- [x] **Step 4: Run the Homography test and verify GREEN**

Run: `cargo test -p rustsfm --lib wgpu_model_scorer_scores_homographies_and_reads_mask --features gpu-wgpu -- --nocapture`

Expected: PASS on Metal, or a clean skip when no compatible adapter exists.

### Task 2: Essential/Fundamental Sampson Scoring

**Files:**
- Modify: `RustSFM/src/gpu/mod.rs`
- Modify: `RustSFM/src/gpu/shaders/model_scoring.wgsl`

- [x] **Step 1: Write the failing Sampson contract test**

Use the row-major epipolar matrix below, which models horizontal epipolar lines. The first two pairs have equal `y` and zero residual; the third differs by one pixel and has squared Sampson residual `0.5`.

```rust
let model = [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0];
let points1 = [[0.0, 0.0], [1.0, 2.0], [-3.0, 4.0]];
let points2 = [[5.0, 0.0], [2.0, 2.0], [1.0, 5.0]];
```

At `threshold=0.1`, assert two inliers, zero inlier residual sum, and mask `[true, true, false]`.

- [x] **Step 2: Run the Sampson test and verify RED**

Run: `cargo test -p rustsfm --lib wgpu_model_scorer_matches_sampson_support --features gpu-wgpu -- --nocapture`

Expected: FAIL because the Sampson branch does not yet implement the CPU formula.

- [x] **Step 3: Implement squared Sampson residual**

For row-major `F`, compute `Fx1`, `F^T x2`, and:

```wgsl
let numerator = dot(point2_h, fx1);
let denominator = fx1.x * fx1.x + fx1.y * fx1.y
    + ftx2.x * ftx2.x + ftx2.y * ftx2.y;
let residual = numerator * numerator / denominator;
```

Treat a non-finite result or `denominator <= 1e-24` as an outlier, matching the CPU reference.

- [x] **Step 4: Run both scorer tests and verify GREEN**

Run: `cargo test -p rustsfm --lib wgpu_model_scorer_ --features gpu-wgpu -- --nocapture`

Expected: both scorer tests PASS.

### Task 3: Validation And Commit

**Files:**
- Modify: `RustSFM/src/gpu/scorer.rs`
- Modify: `RustSFM/src/gpu/mod.rs`
- Modify: `RustSFM/src/gpu/shaders/model_scoring.wgsl`
- Modify: `docs/superpowers/plans/2026-07-18-rustsfm-wgpu-model-scorer.md`

- [x] **Step 1: Validate host inputs**

Return contextual errors for mismatched point counts, non-finite or negative thresholds, counts exceeding `u32`, and dispatch counts exceeding the device limit. Empty model batches return an empty summary vector; empty observation batches return zero summaries and an empty mask without creating zero-sized buffers.

- [x] **Step 2: Run focused GPU and CPU-only checks**

Run: `cargo test -p rustsfm --lib wgpu_model_scorer_ --features gpu-wgpu -- --nocapture`

Run: `cargo test -p rustsfm --lib gpu_ --features gpu-wgpu -- --nocapture`

Run: `cargo check -p rustsfm --no-default-features`

Expected: scorer and GPU tests pass; the CPU-only crate compiles without references to wgpu-only symbols.

- [x] **Step 3: Run formatting and static checks**

Run: `cargo fmt -p rustsfm -- --check`

Run: `cargo clippy -p rustsfm --lib --features gpu-wgpu -- -D warnings`

Expected: both commands exit successfully with no new warnings.

Result: formatting passed. Strict clippy is blocked before completion by 84 existing
`RustSLAM` errors; `--no-deps` is blocked by 149 existing `RustSFM` errors. The one new
scorer lint was fixed, and filtered clippy output contains no remaining `gpu/scorer.rs`
diagnostic.

- [x] **Step 4: Review the diff and commit**

```bash
git add RustSFM/src/gpu/mod.rs RustSFM/src/gpu/scorer.rs \
  RustSFM/src/gpu/shaders/model_scoring.wgsl \
  docs/superpowers/plans/2026-07-18-rustsfm-wgpu-model-scorer.md
git commit -m "feat(rustsfm): score RANSAC models on wgpu"
```

The next plan integrates bounded candidate chunks into two-view RANSAC while preserving seeded candidate order, adaptive stopping, and LO-RANSAC semantics.

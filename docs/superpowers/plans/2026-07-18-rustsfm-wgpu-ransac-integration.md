# RustSFM wgpu RANSAC Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route explicitly requested GPU matching through batched wgpu scoring for Essential, Fundamental, and Homography RANSAC candidates without moving sampling, minimal solvers, LO-RANSAC, or refinement off CPU.

**Architecture:** Extend `WgpuModelScorer` with a prepared homogeneous-observation session so observations are uploaded once per model family. GPU RANSAC generates candidates in the existing seeded CPU order, scores bounded 64-trial chunks, and processes returned summaries in that same order; only a promising candidate's mask is read back before existing CPU LO-RANSAC. New `Result<Option<_>>` GPU entry points propagate device errors, while existing CPU `Option<_>` APIs remain unchanged.

**Tech Stack:** Rust 2021, wgpu 29, WGSL, nalgebra, anyhow, existing COLMAP-compatible RANSAC samplers and solvers.

---

## File Map

- `RustSFM/src/gpu/scorer.rs`: prepared homogeneous observation buffers and reusable scoring session.
- `RustSFM/src/gpu/shaders/model_scoring.wgsl`: homogeneous point ABI and residuals for arbitrary finite `z`.
- `RustSFM/src/gpu/mod.rs`: session exports and homogeneous residual contract test.
- `RustSFM/src/geometry/two_view.rs`: CPU/GPU scoring backend, chunk boundaries, and batched Essential/Fundamental/Homography loops.
- `RustSFM/src/geometry/geometry.rs`: GPU-specific fallible pair-estimation entry point with shared postprocessing.
- `RustSFM/src/feature/feature_matching_db.rs`: one shared context, matcher, and scorer per command; serial GPU verification for computed and existing matches.

### Task 1: Prepared Homogeneous Scoring Session

**Files:**
- Modify: `RustSFM/src/gpu/scorer.rs`
- Modify: `RustSFM/src/gpu/shaders/model_scoring.wgsl`
- Modify: `RustSFM/src/gpu/mod.rs`

- [x] **Step 1: Write the failing homogeneous Sampson test**

Add `wgpu_model_scorer_preserves_homogeneous_sampson_scaling`. Use
`F = [[0,0,0],[0,0,-1],[0,1,0]]`, `x1=[1,4,2]`, and `x2=[3,6,2]`.
The CPU formula yields residual `2.0`, whereas prematurely dehomogenizing both points yields
`0.5`. Assert zero inliers at threshold `1.0` and one inlier with residual sum `2.0` at
threshold `2.0`.

- [x] **Step 2: Run the test and verify RED**

Run: `cargo test -p rustsfm --lib wgpu_model_scorer_preserves_homogeneous_sampson_scaling --features gpu-wgpu -- --nocapture`

Expected: FAIL because no homogeneous scorer method exists.

- [x] **Step 3: Add the point ABI and prepared session**

Use a 16-byte storage record:

```rust
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuHomogeneousPoint {
    x: f32,
    y: f32,
    z: f32,
    pad: f32,
}

pub(crate) struct WgpuModelScoringSession<'a> {
    scorer: &'a WgpuModelScorer,
    points1: wgpu::Buffer,
    points2: wgpu::Buffer,
    observation_count: u32,
}
```

Add:

```rust
pub(crate) fn prepare_homogeneous_session(
    &self,
    points1: &[[f32; 3]],
    points2: &[[f32; 3]],
) -> Result<WgpuModelScoringSession<'_>>;
```

Move model upload, summary dispatch, and mask dispatch onto the session so both operations
reuse its two point buffers. Keep the existing 2D public methods by packing `[x,y,1]` and
delegating to the same session. Add crate-visible homogeneous convenience methods used by
tests and geometry integration.

Change WGSL bindings 1 and 2 to `array<HomogeneousPoint>`. Homography divides both the
predicted point and target point by their finite nonzero `z`; Sampson uses the full uploaded
homogeneous vectors without dehomogenizing.

- [x] **Step 4: Run scorer tests and verify GREEN**

Run: `cargo test -p rustsfm --lib wgpu_model_scorer_ --features gpu-wgpu -- --nocapture`

Expected: all scorer tests pass on Metal.

- [x] **Step 5: Commit the session extension**

```bash
git add RustSFM/src/gpu/scorer.rs RustSFM/src/gpu/shaders/model_scoring.wgsl \
  RustSFM/src/gpu/mod.rs docs/superpowers/plans/2026-07-18-rustsfm-wgpu-ransac-integration.md
git commit -m "feat(rustsfm): reuse GPU RANSAC observations"
```

### Task 2: Candidate Chunk Boundary And Homography RANSAC

**Files:**
- Modify: `RustSFM/src/geometry/two_view.rs`

- [ ] **Step 1: Write pure chunk-boundary tests**

Define `GPU_RANSAC_CHUNK_TRIALS = 64` and test:

```rust
assert_eq!(gpu_ransac_chunk_end(0, 10_000, 10_000, 100), 64);
assert_eq!(gpu_ransac_chunk_end(64, 10_000, 24, 100), 101);
assert_eq!(gpu_ransac_chunk_end(96, 10_000, 24, 100), 101);
assert_eq!(gpu_ransac_chunk_end(101, 10_000, 24, 100), 101);
assert_eq!(gpu_ransac_chunk_end(9_980, 10_000, usize::MAX, 100), 10_000);
```

The effective dynamic end is `max(dynamic_max_trials, min_num_trials) + 1`, clamped to
`max_num_trials`. This reproduces the existing zero-based abort gate at chunk boundaries.

- [ ] **Step 2: Run boundary tests and verify RED**

Run: `cargo test -p rustsfm --lib gpu_ransac_chunk_ --features gpu-wgpu -- --nocapture`

Expected: FAIL because the helper does not exist.

- [ ] **Step 3: Implement ordered Homography chunks**

Add an internal candidate record containing `trial`, `model_index`, `models_in_sample`,
`sample`, and row-major `Matrix3<f64>`. For each boundary:

1. Generate samples and CPU DLT models in existing order until the boundary end.
2. Convert candidate matrices to row-major `[f32;9]` and dispatch one session call.
3. Iterate zipped candidates/summaries in insertion order.
4. For a summary that can beat `best`, read that candidate's mask, expand it through
   `active_indices`, then run existing CPU `refine_homography_support_with_trace`.
5. Update `dynamic_max_trials` and `best` exactly as the CPU loop does.
6. Finish the current chunk even when a new dynamic limit is smaller; the next boundary
   applies that limit, so no selected candidate exceeds the budget known before its chunk.

Fallback DLT and final `refine_homography_support` remain CPU code. Any scorer error is
returned with model-family context.

- [ ] **Step 4: Add a GPU Homography RANSAC integration test**

Use a deterministic grid transformed by a translation homography plus fixed outliers.
Assert GPU RANSAC finds at least the exact inlier grid, returns the same inlier mask on two
runs with the same seed, and reports a finite residual sum.

- [ ] **Step 5: Run the Homography integration test**

Run: `cargo test -p rustsfm --lib gpu_homography_ransac_ --features gpu-wgpu -- --nocapture`

Expected: PASS on Metal.

### Task 3: Essential And Fundamental Batched Scoring

**Files:**
- Modify: `RustSFM/src/geometry/two_view.rs`

- [ ] **Step 1: Write an Essential/Fundamental GPU parity test**

Generate deterministic normalized correspondences from a known relative pose, append fixed
outliers, and use a fixed sampler seed. Run CPU and GPU two-view estimation with identical
options. Assert both select a valid calibrated configuration, GPU retains at least 90% of
the CPU inlier count, and repeated GPU runs return the same mask and matrices within f32
scoring tolerance.

- [ ] **Step 2: Run the parity test and verify RED**

Run: `cargo test -p rustsfm --lib gpu_batched_two_view_ --features gpu-wgpu -- --nocapture`

Expected: FAIL because the fallible GPU estimator is undefined.

- [ ] **Step 3: Extract the CPU Essential loop without behavior changes**

Move the current inline Essential RANSAC block into `estimate_essential_ransac`, preserving
the same sampler salt, trial increment, per-model abort check, fallback eight-point model,
and final CPU refinement. Keep existing CPU callers on this function and run the current
two-view test suite before adding GPU selection.

- [ ] **Step 4: Add Essential and Fundamental GPU loops**

Essential uses homogeneous unit bearing vectors and `TwoViewModelKind::Sampson`.
Fundamental uses pixel homogeneous points and the same residual kind. Both loops use the
Task 2 chunk contract, stable candidate order, compact summaries, selected-candidate mask
readback, existing CPU LO-RANSAC, CPU fallback, and CPU final refinement.

Add an internal backend enum:

```rust
enum TwoViewScoringBackend<'a> {
    Cpu,
    #[cfg(feature = "gpu-wgpu")]
    Wgpu(&'a WgpuModelScorer),
}
```

Refactor the shared estimator body to return `anyhow::Result<Option<TwoViewEstimate>>`.
The existing CPU functions call it with `Cpu` and retain their `Option` signatures. Add a
crate-visible GPU function returning `Result<Option<_>>`. Reject `multiple_models` and
`force_h_use` on the explicit GPU entry until their recursive/control-flow variants receive
the same chunk integration; do not execute CPU scoring for these explicit requests.

- [ ] **Step 5: Run CPU and GPU two-view tests**

Run: `cargo test -p rustsfm --lib two_view::tests --features gpu-wgpu -- --nocapture`

Expected: existing CPU tests and new GPU parity tests pass.

### Task 4: Fallible Pair Geometry And Command Routing

**Files:**
- Modify: `RustSFM/src/geometry/geometry.rs`
- Modify: `RustSFM/src/feature/feature_matching_db.rs`

- [ ] **Step 1: Write routing and error-contract tests**

Add a testable backend-name helper and assert `SiftMatchingOptions.use_gpu=true` selects
`"wgpu_match_and_score"`. Add a tiny database test that requests GPU matching, verifies at
least one geometry is persisted, and confirms the report does not identify CPU scoring.

- [ ] **Step 2: Run routing tests and verify RED**

Run: `cargo test -p rustsfm --lib gpu_matching_routes_model_scoring --features gpu-wgpu -- --nocapture`

Expected: FAIL because matching only creates `WgpuSiftMatcher` and pair estimation is
infallible CPU scoring.

- [ ] **Step 3: Add fallible GPU pair estimation**

Refactor the current pair-estimation body into a shared internal function returning
`Result<Option<PairGeometry>>`. Existing public CPU wrappers call it with the CPU backend
and retain signatures/behavior. Add a cfg-gated GPU wrapper taking `&WgpuModelScorer`; only
the two-view estimation call differs, while match preparation, pose refinement, dense inlier
expansion, and `PairGeometry` construction stay shared.

- [ ] **Step 4: Share context and propagate failures in matching**

For computed matches, create one `Arc<WgpuContext>`, then construct both
`WgpuSiftMatcher::from_context(context.clone())` and
`WgpuModelScorer::from_context(context)`. Run pairs serially and use `?` on both matching and
geometry estimation. For existing matches with GPU explicitly selected, create one scorer,
bypass Rayon/FIFO workers, and verify pairs serially. CPU branches remain unchanged.

- [ ] **Step 5: Run feature-matching and CLI tests**

Run: `cargo test -p rustsfm --lib feature_matching_db::tests --features gpu-wgpu -- --nocapture`

Run: `cargo test -p rustsfm --bin rustsfm --features gpu-wgpu -- --nocapture`

Expected: PASS, with explicit GPU errors propagated rather than converted to missing
geometries.

### Task 5: Verification And Commit

**Files:**
- Modify: all files listed above
- Modify: `docs/superpowers/plans/2026-07-18-rustsfm-wgpu-ransac-integration.md`

- [ ] **Step 1: Run focused and compatibility verification**

Run: `cargo test -p rustsfm --lib gpu_ --features gpu-wgpu -- --nocapture`

Run: `cargo test -p rustsfm --lib two_view::tests --features gpu-wgpu -- --nocapture`

Run: `cargo check -p rustsfm --no-default-features`

Run: `cargo fmt -p rustsfm -- --check`

Expected: GPU and CPU tests pass, CPU-only compilation succeeds, and formatting is clean.

- [ ] **Step 2: Run static checks and record baseline blockers**

Run: `cargo clippy -p rustsfm --lib --features gpu-wgpu --no-deps -- -D warnings`

Expected: no diagnostic in changed files. Existing repository-wide clippy failures are
recorded without unrelated cleanup.

- [ ] **Step 3: Review and commit**

```bash
git add RustSFM/src/gpu RustSFM/src/geometry/two_view.rs \
  RustSFM/src/geometry/geometry.rs RustSFM/src/feature/feature_matching_db.rs \
  docs/superpowers/plans/2026-07-18-rustsfm-wgpu-ransac-integration.md
git commit -m "feat(rustsfm): batch RANSAC scoring on wgpu"
```

PnP scoring remains a separate mapper-stage change. The full `flowers2` benchmark follows
after this two-view integration is verified.

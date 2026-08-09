# RustSFM GPU Geometry Observability Design

## Context

The flowers2 match-pair benchmark shows that a 2,890-pair cold run spends 251.470 seconds in
geometry validation and 196.765 seconds in descriptor matching. Descriptor telemetry already shows
that 189.677 seconds are GPU queue/readback wait. Geometry is now the largest remaining opaque
stage.

The current wgpu geometry path evaluates Essential, Fundamental, and Homography RANSAC in sequence.
Each estimator generates candidates on the CPU in 64-trial chunks, submits a GPU scoring dispatch,
reads back compact support summaries, and may submit another dispatch/readback for an inlier mask
when a candidate improves. These synchronization points are not currently counted or timed.

## Goals

- Measure GPU geometry session preparation, candidate generation, dispatch, readback, and CPU
  refinement without changing results.
- Count score-summary calls, inlier-mask calls, models scored, readback calls, and readback bytes.
- Attribute the counts and durations separately to Essential, Fundamental, and Homography RANSAC.
- Aggregate per-pair measurements into `MatchFeaturesTimingReport` and the existing
  `benchmark-match-pairs` JSON output.
- Preserve compatibility for callers that do not need profiling.
- Use the generic `gpu-wgpu` backend path on macOS; do not hard-code Metal or Vulkan.

## Non-Goals

- Do not change `GPU_RANSAC_CHUNK_TRIALS`, RANSAC thresholds, iteration limits, seeds, samplers,
  candidate ordering, local optimization, classification, or pose selection.
- Do not batch multiple image pairs or combine Essential, Fundamental, and Homography dispatches.
- Do not add a keyframe, image, pair, or benchmark limit.
- Do not modify RustViewer, `MapperConfig`, sequence pair generation, SQLite transactions, progress
  events, pause/cancel checkpoints, or RustGS handoff.
- Do not request timestamp-query device features. This stage uses host wall-clock measurements,
  matching the existing descriptor telemetry.

## Considered Approaches

### 1. Explicit profiled return values

Add profiled scorer and geometry APIs that return the normal result together with a timing value.
Existing APIs remain wrappers that discard timing. Timing is accumulated in the same stack frame as
the work it describes.

This is the selected approach. It keeps ownership explicit, has no global state, and remains correct
if pair scheduling becomes concurrent later.

### 2. Mutable counters stored in `WgpuModelScorer`

The scorer could retain cumulative counters and callers could snapshot differences around each
pair. This requires interior mutability, makes errors and overlapping work difficult to attribute,
and would complicate future concurrent batching. It is rejected.

### 3. Change chunking before adding telemetry

Increasing the 64-trial chunk size would likely reduce synchronization, but it can delay dynamic
RANSAC termination and change the candidate set evaluated before termination. It is rejected for
this stage because the current evidence cannot quantify the expected benefit or semantic risk.

## Timing Model

Always-available DTOs live in `RustSFM/src/gpu/mod.rs` so serialized report schemas do not depend on
whether the crate was compiled with `gpu-wgpu`:

- `WgpuModelScorerTiming` records buffer/bind-group preparation, submit, readback total,
  copy-submit, wait, map/decode, call counts, models scored, and bytes read.
- `WgpuRansacStageTiming` records session preparation, CPU candidate generation, CPU refinement,
  score-summary calls, inlier-mask calls, and a nested `WgpuModelScorerTiming`.
- `WgpuGeometryTiming` contains `essential`, `fundamental`, and `homography` stage values.

All timing types derive `Debug`, `Clone`, `Copy`, `Default`, `PartialEq`, `Serialize`, and
`Deserialize`, use `#[serde(default)]`, and implement saturating aggregation. Durations must remain
finite and non-negative. Counts use saturating addition. Empty operations report deterministic
zero values and perform no GPU work.

`MatchFeaturesTimingReport` gains a serde-defaulted `gpu_geometry_detail: WgpuGeometryTiming` field.
The existing `gpu_geometry_seconds` wall-clock field remains unchanged for compatibility and serves
as the enclosing total. Nested geometry durations are diagnostic components and are not added to
the top-level classified-time calculation a second time.

## API And Data Flow

`WgpuModelScoringSession` gains two internal profiled methods:

- `score_two_view_models_profiled` returns support summaries and `WgpuModelScorerTiming`.
- `inlier_mask_profiled` returns the decoded mask and `WgpuModelScorerTiming`.

The existing `score_two_view_models` and `inlier_mask` methods call the profiled variants and discard
the timing value. Both variants reuse `WgpuContext::read_buffer_profiled`; no additional dispatch or
readback is introduced.

Each GPU RANSAC estimator owns one `WgpuRansacStageTiming`. It measures homogeneous-point session
creation once, candidate generation around the existing CPU loops, scorer calls around the existing
GPU calls, and local refinement around the existing refinement calls. It returns its existing model
result together with the timing value. The calibrated two-view function combines the three stage
values into `WgpuGeometryTiming`.

The pair-geometry layer exposes an internal profiled variant returning
`(Option<PairGeometry>, WgpuGeometryTiming)`. Its existing GPU function remains a wrapper returning
only `Option<PairGeometry>`. The computed matching path calls the profiled variant and accumulates
the timing in `ComputedMatchPairBatch`; CPU and existing-match paths keep a zero timing value.

The flow is therefore:

```text
RANSAC scorer/readback
    -> per-stage timing
    -> per-pair geometry timing
    -> computed batch aggregation
    -> MatchFeaturesTimingReport
    -> benchmark-match-pairs JSON
```

## Error And Compatibility Behavior

Profiling must not turn successful work into an error. Existing validation errors, adapter errors,
and readback errors propagate unchanged. If an operation fails, no `MatchFeaturesReport` is returned,
matching current behavior; partial timing does not need a separate failure channel.

Legacy serialized reports without `gpu_geometry_detail` deserialize to zero. Existing API callers,
including RustViewer and sequence registration, keep their current signatures. Event contents and
database writes remain byte-for-byte governed by the existing matching and geometry results.

## Verification

Tests are added before production changes:

1. CPU-only aggregation tests prove exact totals for two synthetic score calls and one mask call,
   including call counts, model counts, and byte counts.
2. A serialization compatibility test proves an old `MatchFeaturesReport` without geometry detail
   deserializes to zero.
3. An adapter-optional scorer smoke test compares profiled and compatibility-wrapper results and
   asserts that profiling does not add dispatches.
4. An adapter-optional two-view test compares the profiled geometry result with the existing result
   for the same fixed seed and input.
5. A feature-matching aggregation test proves Essential, Fundamental, and Homography timing reaches
   `MatchFeaturesTimingReport` without changing existing descriptor counters.
6. Existing matcher, geometry, controlled matching, pause/cancel, and database regression suites
   remain green under `--no-default-features --features gpu-wgpu`.

After tests pass, run the release benchmark on flowers2 with 96 pairs and three repetitions. Each run
must retain 96 matched pairs, 96 verified pairs, and 62,409 matches. Compare the newly measured stage
counts and wait durations across repetitions without imposing a wall-clock pass threshold. Run the
2,890-pair benchmark only after the bounded results are stable, and never concurrently with another
flowers2 GPU benchmark.

## Decision Gate After Measurement

- If summary readback wait dominates and calls scale with 64-trial chunks, design a larger or
  adaptive chunk experiment with fixed-seed result parity tests.
- If mask readbacks dominate, design deferred or combined mask recovery while preserving the exact
  candidate-selection and local-refinement order.
- If per-pair session preparation dominates, design buffer residency or multi-pair batching.
- If CPU candidate generation/refinement dominates, optimize those CPU algorithms before changing
  GPU scheduling.

No optimization is selected until the bounded benchmark identifies which condition applies.

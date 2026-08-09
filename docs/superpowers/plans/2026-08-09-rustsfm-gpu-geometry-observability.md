# RustSFM GPU Geometry Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure Essential, Fundamental, and Homography GPU RANSAC synchronization costs without changing match-pair results or scheduling.

**Architecture:** Add always-available, serde-compatible timing DTOs in the GPU module. Profile the existing model-scorer calls by returning timing alongside results, propagate stage timing through two-view and pair geometry, and aggregate it into the existing match-feature and benchmark reports. Existing non-profiled APIs remain wrappers.

**Tech Stack:** Rust, serde, wgpu, existing `WgpuContext::read_buffer_profiled`, COLMAP-compatible two-view geometry, Cargo tests.

---

### Prerequisite 1: Consistent Benchmark Snapshots And Repetition State

**Files:**
- Modify: `RustSFM/Cargo.toml`
- Modify: `RustSFM/src/io/database.rs`
- Modify: `RustSFM/src/diagnostics/match_pair_benchmark.rs`
- Test: `RustSFM/src/io/database.rs`
- Test: `RustSFM/src/diagnostics/match_pair_benchmark.rs`

- [x] **Step 1: Write failing online-backup and repetition-isolation tests**

Enable a database test named `database_backup_captures_committed_wal_rows`. Create a database,
switch a separate writer connection to WAL mode, create and commit a marker row while keeping the
writer open, then call the not-yet-existing `ColmapDatabase::backup_to`. Open the backup and assert
the committed marker is present.

Replace `match_pair_benchmark_rejects_source_with_uncheckpointed_wal` with
`match_pair_benchmark_snapshots_source_with_uncheckpointed_wal`. Keep a WAL writer open after a
committed source change and assert the benchmark succeeds, selects the images contained in the
snapshot, and leaves the source unchanged.

Add a pure helper test for `copy_benchmark_snapshot_for_run(snapshot, work_dir, run_index)`. Create a
snapshot containing a distinctive match and geometry row, copy it for run indices 0 and 1, mutate
only run 0, and assert run 1 plus the snapshot retain the original rows. This proves each repetition
has an independent starting file.

- [x] **Step 2: Run focused tests and verify RED**

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib database_backup_captures_committed_wal_rows -- --nocapture
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib match_pair_benchmark -- --nocapture
```

Expected: compilation failure because the backup and per-run snapshot helpers do not exist, and the
old benchmark still rejects a non-empty WAL.

- [x] **Step 3: Enable and wrap SQLite online backup**

Change the dependency to:

```toml
rusqlite = { version = "0.32", features = ["backup", "bundled"] }
```

Add this method to `ColmapDatabase`:

```rust
pub fn backup_to(&self, destination: impl AsRef<Path>) -> Result<u64> {
    let destination = destination.as_ref();
    let mut target = Connection::open(destination)
        .with_context(|| format!("open SQLite backup destination {}", destination.display()))?;
    let backup = rusqlite::backup::Backup::new(&self.conn, &mut target)
        .context("initialize SQLite online backup")?;
    backup
        .run_to_completion(256, std::time::Duration::from_millis(1), None)
        .context("copy SQLite online backup")?;
    drop(backup);
    drop(target);
    Ok(std::fs::metadata(destination)?.len())
}
```

The destination is a new file inside a private `TempDir`. Do not expose the internal connection or
run schema migrations against the snapshot.

- [x] **Step 4: Snapshot once and isolate every repetition**

Remove the WAL rejection helper. In `benchmark_match_pairs`, open the source read-only and call
`backup_to` into `source-snapshot.db`, measuring this as `database_copy_seconds`. Close the source,
then open the completed snapshot read-only and sort its images by `(name, image_id)`.

Add:

```rust
fn copy_benchmark_snapshot_for_run(
    snapshot: &Path,
    work_dir: &Path,
    run_index: usize,
) -> Result<PathBuf> {
    let database = work_dir.join(format!("run-{run_index}.db"));
    std::fs::copy(snapshot, &database)
        .with_context(|| format!("copy benchmark snapshot for run {}", run_index + 1))?;
    Ok(database)
}
```

Inside the repetition loop, create a distinct run database from the stable snapshot and pass it to
`match_explicit_image_pairs_to_database_with_session`. Keep one `ExplicitPairMatchingSession`
outside the loop and keep `run_options.clear_existing=true`. Do not include per-run file-copy time in
`matching_seconds`.

- [x] **Step 5: Verify, format, and commit**

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib database_backup_captures_committed_wal_rows -- --nocapture
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib match_pair_benchmark -- --nocapture
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --bin rustsfm benchmark_match_pairs -- --nocapture
cargo fmt --all -- --check
git diff --check
```

Commit:

```bash
git add RustSFM/Cargo.toml RustSFM/src/io/database.rs \
  RustSFM/src/diagnostics/match_pair_benchmark.rs \
  docs/superpowers/plans/2026-08-09-rustsfm-gpu-geometry-observability.md
git commit -m "fix(rustsfm): isolate match-pair benchmark runs"
```

### Prerequisite 2: Complete Existing-Match FIFO Timing Semantics

**Files:**
- Modify: `RustSFM/src/feature/feature_matching_db.rs`
- Modify: `docs/superpowers/plans/2026-08-08-rustsfm-match-pair-observability.md`
- Test: `RustSFM/src/feature/feature_matching_db.rs`

- [ ] **Step 1: Write failing FIFO timing tests**

Extend `controlled_fifo_trace_is_independent_of_task_commit_batch_size` and the replay counterpart.
For each returned report assert exact attempted-pair, produced-report, and committed-batch counts;
assert `pair_compute_seconds`, `database_commit_seconds`, and `event_sink_seconds` are finite and
non-negative. Assert classified timings do not exceed `matching_seconds` except for floating-point
epsilon.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib controlled_fifo_ -- --nocapture
```

Expected: count assertions may pass, but commit/event timing assertions fail because FIFO currently
synthesizes only counts after the helper returns.

- [ ] **Step 3: Thread batch timing through live and replay FIFO helpers**

Pass `&mut MatchFeaturesTimingReport` through `run_controlled_colmap_fifo_batches`,
`run_controlled_colmap_replay_batches`, and `commit_ready_fifo_prefixes`. Capture the
`MatchPairBatchTiming` returned by `commit_and_emit_pair_batch` and call `record_batch` once per
committed prefix. Remove the outer FIFO block that manually assigns counts.

Measure wall time around the entire live/replay FIFO helper. After it returns, subtract the commit
and event deltas recorded during that call and add the remaining non-negative duration to
`pair_compute_seconds`. This includes FIFO scheduling and worker wait as part of pair computation
without summing per-worker durations or double-counting commit/event time.

- [ ] **Step 4: Document overlapping readback timing and API compatibility**

Update the completed observability plan to state that `gpu_readback_map_decode_seconds` is an
overlapping callback-to-decode latency containing the nested wait and must not be summed with
`gpu_readback_wait_seconds`. State that serde compatibility is preserved, while external Rust struct
literals are not a supported compatibility guarantee for this workspace-internal 0.1 API.

- [ ] **Step 5: Verify and commit**

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib controlled_fifo_ -- --nocapture
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib match_feature_timing -- --nocapture
cargo fmt --all -- --check
git diff --check
```

Commit:

```bash
git add RustSFM/src/feature/feature_matching_db.rs \
  docs/superpowers/plans/2026-08-08-rustsfm-match-pair-observability.md \
  docs/superpowers/plans/2026-08-09-rustsfm-gpu-geometry-observability.md
git commit -m "fix(rustsfm): complete fifo timing report"
```

### Task 1: Model-Scorer Timing Contract And Profiled Readbacks

**Files:**
- Modify: `RustSFM/src/gpu/mod.rs`
- Modify: `RustSFM/src/gpu/scorer.rs`
- Test: `RustSFM/src/gpu/mod.rs`

- [ ] **Step 1: Add failing CPU-only aggregation tests**

Add `gpu_geometry_timing_accumulates_model_scorer_work` in `gpu::tests`. Construct two
`WgpuModelScorerTiming` values representing one score call and one mask call, add them, and assert
exact totals:

```rust
let mut total = WgpuModelScorerTiming {
    buffer_prepare_seconds: 0.2,
    submit_seconds: 0.3,
    readback_total_seconds: 0.4,
    readback_copy_submit_seconds: 0.05,
    readback_wait_seconds: 0.25,
    readback_map_decode_seconds: 0.3,
    score_calls: 1,
    mask_calls: 0,
    models_scored: 64,
    readback_calls: 1,
    readback_bytes: 512,
};
total += WgpuModelScorerTiming {
    buffer_prepare_seconds: 1.0,
    submit_seconds: 2.0,
    readback_total_seconds: 3.0,
    readback_copy_submit_seconds: 0.5,
    readback_wait_seconds: 2.0,
    readback_map_decode_seconds: 2.5,
    score_calls: 0,
    mask_calls: 1,
    models_scored: 0,
    readback_calls: 1,
    readback_bytes: 1024,
};
assert_eq!(total.score_calls, 1);
assert_eq!(total.mask_calls, 1);
assert_eq!(total.models_scored, 64);
assert_eq!(total.readback_calls, 2);
assert_eq!(total.readback_bytes, 1536);
assert!((total.buffer_prepare_seconds - 1.2).abs() < 1.0e-12);
assert!((total.readback_wait_seconds - 2.25).abs() < 1.0e-12);
```

Also add a `WgpuGeometryTiming` aggregation test. Put distinct `score_calls` values in Essential,
Fundamental, and Homography and prove stage attribution survives `+=`.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu gpu_geometry_timing -- --nocapture
```

Expected: compilation failure because `WgpuModelScorerTiming`, `WgpuRansacStageTiming`, and
`WgpuGeometryTiming` do not exist.

- [ ] **Step 3: Add always-available timing DTOs**

In `gpu/mod.rs`, outside `#[cfg(feature = "gpu-wgpu")]`, add serde-compatible DTOs with these exact
fields:

```rust
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WgpuModelScorerTiming {
    pub buffer_prepare_seconds: f64,
    pub submit_seconds: f64,
    pub readback_total_seconds: f64,
    pub readback_copy_submit_seconds: f64,
    pub readback_wait_seconds: f64,
    pub readback_map_decode_seconds: f64,
    pub score_calls: usize,
    pub mask_calls: usize,
    pub models_scored: usize,
    pub readback_calls: usize,
    pub readback_bytes: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WgpuRansacStageTiming {
    pub session_prepare_seconds: f64,
    pub candidate_generation_seconds: f64,
    pub cpu_refinement_seconds: f64,
    pub scorer: WgpuModelScorerTiming,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WgpuGeometryTiming {
    pub essential: WgpuRansacStageTiming,
    pub fundamental: WgpuRansacStageTiming,
    pub homography: WgpuRansacStageTiming,
}
```

Implement `AddAssign` for all three types. Add durations normally and use `saturating_add` for every
integer count. Do not add global state or feature-gated fields.

- [ ] **Step 4: Add profiled scorer methods**

In `gpu/scorer.rs`, keep `score_two_view_models` and `inlier_mask` as wrappers. Move their bodies to
new methods with these signatures:

```rust
pub(crate) fn score_two_view_models_profiled(
    &self,
    models: &[[f32; 9]],
    threshold: f32,
    kind: TwoViewModelKind,
) -> Result<(Vec<GpuModelSupport>, WgpuModelScorerTiming)>;

pub(crate) fn inlier_mask_profiled(
    &self,
    model: &[f32; 9],
    threshold: f32,
    kind: TwoViewModelKind,
) -> Result<(Vec<bool>, WgpuModelScorerTiming)>;
```

Measure buffer/bind-group/encoder construction before submission and submission separately. Replace
`read_buffer` with `read_buffer_profiled` and copy all readback fields. A non-empty score call sets
`score_calls=1`, `models_scored=models.len()`, and `mask_calls=0`; a mask call sets `mask_calls=1`,
`score_calls=0`, and `models_scored=0`. Empty model input returns an empty result and zero timing.
The compatibility methods must be exactly:

```rust
self.score_two_view_models_profiled(models, threshold, kind)
    .map(|(supports, _)| supports)
```

and the equivalent mask wrapper, ensuring profiling performs no extra dispatch.

- [ ] **Step 5: Add adapter-optional result and accounting smoke tests**

Extend the existing model-scorer test fixture. When `WgpuContext::try_new_optional()` returns
`None`, print an explicit skip and return `Ok(())`. Otherwise call the profiled support method once
and the profiled mask method once. Assert:

```rust
assert_eq!(score_timing.score_calls, 1);
assert_eq!(score_timing.mask_calls, 0);
assert_eq!(score_timing.models_scored, models.len());
assert_eq!(score_timing.readback_calls, 1);
assert_eq!(mask_timing.score_calls, 0);
assert_eq!(mask_timing.mask_calls, 1);
assert_eq!(mask_timing.readback_calls, 1);
assert!(score_timing.readback_bytes > 0);
assert!(mask_timing.readback_bytes > 0);
assert!(score_timing.readback_map_decode_seconds >= score_timing.readback_wait_seconds);
```

Compare the returned support and mask values with the existing fixture expectations, not only their
lengths.

- [ ] **Step 6: Verify Task 1 and commit**

Run:

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu gpu_geometry_timing -- --nocapture
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu wgpu_model_scorer_ -- --nocapture
cargo fmt --all -- --check
git diff --check
```

Expected: all focused tests pass; adapter-dependent tests either pass or explicitly skip.

Commit:

```bash
git add RustSFM/src/gpu/mod.rs RustSFM/src/gpu/scorer.rs \
  docs/superpowers/plans/2026-08-09-rustsfm-gpu-geometry-observability.md
git commit -m "feat(rustsfm): profile gpu model scorer work"
```

### Task 2: Propagate Essential, Fundamental, And Homography Timing

**Files:**
- Modify: `RustSFM/src/geometry/two_view.rs`
- Modify: `RustSFM/src/geometry/geometry.rs`
- Test: `RustSFM/src/geometry/two_view.rs`
- Test: `RustSFM/src/geometry/geometry.rs`

- [ ] **Step 1: Write failing stage-attribution and parity tests**

Add an adapter-optional fixed-seed geometry test that calls the existing GPU entry point and the new
profiled entry point on identical synthetic correspondences and asserts their `Option<PairGeometry>`
values are equal. The test must also assert that any non-zero scorer calls appear in the expected
stage fields and that all durations are finite and non-negative. Stage aggregation itself is already
covered by Task 1's CPU-only test and must not be duplicated here.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu gpu_geometry_profiled -- --nocapture
```

Expected: compilation failure because the profiled two-view and pair-geometry functions do not
exist.

- [ ] **Step 3: Return timing from each GPU RANSAC estimator**

Change only the three private GPU RANSAC functions. Each returns its existing model result plus one
stage timing:

```rust
anyhow::Result<(
    Option<(Matrix3<f64>, ModelSupport, bool)>,
    WgpuRansacStageTiming,
)>
```

Create `let mut timing = WgpuRansacStageTiming::default()` before early validation. Measure session
preparation around point conversion plus `prepare_homogeneous_session`. Measure each existing
candidate-generation loop with `Instant`. Replace scorer calls with their profiled variants and add
returned values to `timing.scorer`. Measure existing local optimization/refinement calls with
`Instant`; do not move them or change their conditions. Every early return must include the current
timing value.

- [ ] **Step 4: Add a profiled calibrated two-view API**

Add:

```rust
pub(crate) fn estimate_calibrated_two_view_with_observations_rays_and_cameras_gpu_profiled(
    scorer: &WgpuModelScorer,
    pts1: &[[f32; 2]],
    pts2: &[[f32; 2]],
    obs1_px: &[[f32; 2]],
    obs2_px: &[[f32; 2]],
    rays1: Option<&[[f64; 3]]>,
    rays2: Option<&[[f64; 3]]>,
    camera1: CameraModel,
    camera2: CameraModel,
    options: &TwoViewOptions,
) -> anyhow::Result<(Option<TwoViewEstimate>, WgpuGeometryTiming)>;
```

Refactor the internal calibrated implementation so CPU callers keep the same return type while the
GPU profiled caller receives all three stage timing values. The existing GPU function calls the
profiled function and discards timing. Essential failure returns Essential timing and zero later
stages; Fundamental or Homography `None` results retain their measured stage timing.

- [ ] **Step 5: Add a profiled pair-geometry API**

In `geometry.rs`, add
`estimate_pair_geometry_with_options_and_cameras_gpu_profiled` with the current GPU arguments and
return type:

```rust
Result<(Option<PairGeometry>, WgpuGeometryTiming)>
```

Keep `estimate_pair_geometry_with_options_and_cameras_gpu` as a wrapper that calls the profiled
variant and drops timing. For pre-estimation early exits caused by too few matches, return
`(None, WgpuGeometryTiming::default())`. Do not change match selection, normalization, camera math,
inlier expansion, triangulation, or report construction.

- [ ] **Step 6: Verify parity and commit**

Run:

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu gpu_geometry_profiled -- --nocapture
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu two_view::tests -- --nocapture
cargo fmt --all -- --check
git diff --check
```

Expected: fixed-seed profiled/non-profiled outputs are identical and existing two-view tests pass.

Commit:

```bash
git add RustSFM/src/geometry/two_view.rs RustSFM/src/geometry/geometry.rs \
  docs/superpowers/plans/2026-08-09-rustsfm-gpu-geometry-observability.md
git commit -m "feat(rustsfm): attribute gpu ransac timing"
```

### Task 3: Aggregate Geometry Timing Into Match Reports

**Files:**
- Modify: `RustSFM/src/feature/feature_matching_db.rs`
- Modify: `RustSFM/src/lib.rs`
- Test: `RustSFM/src/feature/feature_matching_db.rs`

- [ ] **Step 1: Add failing serialization and aggregation tests**

Extend `match_feature_timing_defaults_when_deserializing_legacy_report` to assert
`gpu_geometry_detail == WgpuGeometryTiming::default()`. Add
`gpu_geometry_timing_folds_into_match_feature_report`, creating a synthetic
`ComputedMatchPairBatch` whose Essential, Fundamental, and Homography fields contain different
counts and durations. Call `record_computed_batch` and assert exact stage values. Preserve the
existing descriptor timing assertions in the same test module.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu gpu_geometry_timing_folds -- --nocapture
```

Expected: compilation or assertion failure because the match report does not carry geometry detail.

- [ ] **Step 3: Add the backward-compatible report field**

Add to `MatchFeaturesTimingReport`:

```rust
#[serde(default)]
pub gpu_geometry_detail: WgpuGeometryTiming,
```

Re-export `WgpuGeometryTiming`, `WgpuModelScorerTiming`, and `WgpuRansacStageTiming` from `lib.rs` so
benchmark/report consumers can deserialize the nested field without private-type leakage. Keep
`gpu_geometry_seconds` and all descriptor timing fields unchanged.

- [ ] **Step 4: Collect and aggregate per-pair timing**

Add `gpu_geometry_timing: WgpuGeometryTiming` to `ComputedMatchPairBatch`. In the GPU loop, call
`estimate_pair_geometry_with_options_and_cameras_gpu_profiled`, add its timing even when the geometry
result is `None`, and construct the same pair report as before. CPU paths initialize the field to
zero. `record_computed_batch` adds the nested value to `MatchFeaturesTimingReport`.

Do not alter pair order, transaction batching, `task_pair_batch_size`, progress events,
pause/cancel checks, or result filtering.

- [ ] **Step 5: Run matching regressions**

Run:

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu match_feature_timing -- --nocapture
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu controlled_computed_matching_ -- --nocapture
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu controlled_matching_ -- --nocapture
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo check -p rustsfm \
  --no-default-features --features gpu-wgpu
cargo fmt --all -- --check
git diff --check
```

Expected: all report, transaction, progress, pause, cancellation, and compile checks pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add RustSFM/src/feature/feature_matching_db.rs RustSFM/src/lib.rs \
  docs/superpowers/plans/2026-08-09-rustsfm-gpu-geometry-observability.md
git commit -m "feat(rustsfm): report gpu geometry timing"
```

### Task 4: Release Verification And flowers2 Measurement

**Files:**
- Modify: `docs/superpowers/plans/2026-08-09-rustsfm-gpu-geometry-observability.md`

- [ ] **Step 1: Run the complete determinable regression suite**

Run the RustSFM library suite with the wgpu feature. Skip only the repository's already documented
external-fixture test and adapter-required legacy SIFT benchmark when their prerequisites are absent:

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu -- --nocapture
cargo fmt --all -- --check
git diff --check
```

Record exact passed, failed, and skipped counts. Any new failure blocks benchmarking.

- [ ] **Step 2: Build the release CLI**

Run:

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo build -p rustsfm \
  --release --no-default-features --features gpu-wgpu
```

Expected: `target/release/rustsfm` is produced successfully.

- [ ] **Step 3: Run the bounded flowers2 benchmark**

First confirm no benchmark process is running. Then run exactly one GPU benchmark:

```bash
target/release/rustsfm benchmark-match-pairs \
  --database test_data/flowers2/out9/Untitled.rustscanproject/Cache/.staging/keyframe_sfm-1/rustsfm/Cache/database.db \
  --window 5 --pair-limit 96 --repetitions 3 --use-gpu --random-seed 0 \
  --output-json /tmp/rustsfm-gpu-geometry-96x3.json
```

Expected for every repetition: `pair_count=96`, `matched_pairs=96`, `verified_pairs=96`, and
`total_matches=62409`. Verify that each stage's call/model/byte counts are deterministic across the
three runs and that every duration is finite and non-negative. Do not assert a speed threshold.

- [ ] **Step 4: Decide whether the full benchmark is warranted**

Run the 2,890-pair benchmark only if bounded result parity and timing accounting pass:

```bash
target/release/rustsfm benchmark-match-pairs \
  --database test_data/flowers2/out9/Untitled.rustscanproject/Cache/.staging/keyframe_sfm-1/rustsfm/Cache/database.db \
  --window 5 --pair-limit 2890 --repetitions 1 --use-gpu --random-seed 0 \
  --output-json /tmp/rustsfm-gpu-geometry-2890.json
```

Never run it concurrently with another flowers2 GPU benchmark. Compare against the baseline of
2,890 matched/verified pairs, 2,958,062 matches, and 452.045 seconds without treating wall time as a
hard pass criterion.

- [ ] **Step 5: Record evidence and commit**

Append exact bounded and, when run, full results to this plan. State which decision-gate condition
from the design applies and identify the next optimization to design. Do not commit `/tmp` JSON or
logs.

Run:

```bash
git diff --check
git status --short
```

Commit:

```bash
git add docs/superpowers/plans/2026-08-09-rustsfm-gpu-geometry-observability.md
git commit -m "docs(rustsfm): record gpu geometry profile"
```

### Final Review And Integration Gate

- [ ] Dispatch a final specification reviewer across the prerequisite and Task 1-4 commits.
- [ ] Dispatch a final code-quality reviewer across the prerequisite and Task 1-4 commits.
- [ ] Resolve every Critical or Important finding and re-run affected tests.
- [ ] Sync `codex/match-pair-telemetry` with current `main`.
- [ ] Run the complete affected RustSFM and RustViewer regression suites after synchronization.
- [ ] Merge into `main` only when tests pass.
- [ ] Remove the worktree and delete the merged feature branch.

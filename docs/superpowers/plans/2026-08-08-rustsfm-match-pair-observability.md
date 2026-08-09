# RustSFM Match-Pair Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add structured, low-overhead timing and count telemetry that explains where `MatchPairBatch` time is spent without changing matching, persistence, progress, pause, or cancellation behavior.

**Architecture:** `MatchFeaturesReport` receives a backward-compatible `timings` value populated by both database-wide and explicit-pair entry points. Batch compute, SQLite commit, event delivery, preparation, and backend initialization are measured with wall-clock `Instant` values and accumulated in memory; no per-pair logging is added. GPU packing, dispatch, and readback detail is a follow-up task after this report contract is verified.

**Tech Stack:** Rust, serde, rusqlite, Rayon, wgpu (compile coverage only for this task).

---

### Task 1: Structured Match-Feature Timing Summary

**Files:**
- Modify: `RustSFM/src/feature/feature_matching_db.rs`
- Modify: `RustSFM/src/lib.rs`
- Test: `RustSFM/src/feature/feature_matching_db.rs`

- [x] **Step 1: Write failing compatibility and accounting tests**

Add tests proving that an old serialized `MatchFeaturesReport` without `timings` deserializes with a default timing summary, and that controlled computed matching reports the exact attempted-pair and committed-batch counts for batch sizes 1 and 2. Assert every seconds value is finite and non-negative. Do not assert a wall-clock performance threshold.

- [x] **Step 2: Run the focused tests and verify RED**

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu match_feature_timing -- --nocapture
```

Expected: compilation or assertion failure because `MatchFeaturesTimingReport` and `MatchFeaturesReport.timings` do not exist.

- [x] **Step 3: Add the report contract**

Add a public `MatchFeaturesTimingReport` with `Debug + Clone + Default + Serialize + Deserialize`. It must contain:

```rust
pub backend_initialization_seconds: f64,
pub database_prepare_seconds: f64,
pub pair_compute_seconds: f64,
pub database_commit_seconds: f64,
pub event_sink_seconds: f64,
pub unclassified_seconds: f64,
pub attempted_pairs: usize,
pub produced_pair_reports: usize,
pub committed_batches: usize,
```

Add `#[serde(default)] pub timings: MatchFeaturesTimingReport` to `MatchFeaturesReport`. Preserve the existing `matching_seconds` field for API compatibility.

- [x] **Step 4: Instrument both matching entry points**

Measure backend construction, database/frame preparation, pair computation, transaction persistence, and event delivery. `commit_and_emit_pair_batch` must return or accumulate commit and event timings without changing event contents or checkpoint placement. Count attempted pairs from input batches, not only successful reports. Compute `unclassified_seconds` with saturating floating-point subtraction from the total so nesting or timer resolution cannot produce a negative value.

`ExplicitPairMatchingSession` should store its one-time initialization duration so every explicit report identifies cold-start cost without recreating the backend. Do not change `task_pair_batch_size`, pair order, GPU selection, database schema, or matching thresholds.

- [x] **Step 5: Run focused tests and verify GREEN**

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu match_feature_timing -- --nocapture
```

Expected: all timing compatibility and accounting tests pass.

- [x] **Step 6: Run behavioral regression tests**

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu controlled_matching_ -- --nocapture
```

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu controlled_computed_matching_ -- --nocapture
```

Expected: progress, rollback, pause, and cancellation tests remain unchanged and pass.

- [x] **Step 7: Compile the macOS wgpu configuration**

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo check -p rustsfm \
  --no-default-features --features gpu-wgpu
```

Expected: exit code 0. This command must not run GPU-heavy tests.

- [x] **Step 8: Format, self-review, and commit**

```bash
cargo fmt --check
git diff --check
git add RustSFM/src/feature/feature_matching_db.rs RustSFM/src/lib.rs docs/superpowers/plans/2026-08-08-rustsfm-match-pair-observability.md
git commit -m "feat(rustsfm): report match-pair timing breakdown"
```

Self-review must explicitly confirm that matching results, event sequence, transaction boundaries, and pause/cancel checkpoints are unchanged.

### Task 2: GPU Matcher And Readback Timing Breakdown

**Files:**
- Modify: `RustSFM/src/gpu/context.rs`
- Modify: `RustSFM/src/gpu/matcher.rs`
- Modify: `RustSFM/src/gpu/mod.rs`
- Modify: `RustSFM/src/feature/feature_matching_db.rs`
- Test: `RustSFM/src/gpu/mod.rs`
- Test: `RustSFM/src/feature/feature_matching_db.rs`

- [x] **Step 1: Write failing timing aggregation tests**

Add CPU-only tests for a `WgpuSiftMatcherTiming` value that accumulates two one-way calls without losing byte or call counts, and for folding a synthetic matcher timing into `MatchFeaturesTimingReport`. Add an adapter-optional smoke test that calls the profiled matcher with cross-check enabled and, when an adapter exists, asserts two direction/readback calls, non-zero readback bytes, and finite non-negative durations.

- [x] **Step 2: Run focused tests and verify RED**

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu gpu_match_timing -- --nocapture
```

Expected: compilation failure because the profiled matcher API and timing records do not exist.

- [x] **Step 3: Add a profiled readback helper without changing existing callers**

In `WgpuContext`, add an internal profiled readback helper returning the decoded values plus timings for staging/copy submission, device wait, callback/map/decode, total duration, call count, and byte count. Keep `read_buffer` as a compatibility wrapper over the profiled helper. Empty reads must return zero timings and no GPU work.

- [x] **Step 4: Add a profiled SIFT matcher API**

Add `match_descriptors_profiled` returning matches plus a `WgpuSiftMatcherTiming`. Keep `match_descriptors` as a wrapper returning only matches. Measure descriptor packing, buffer/bind-group/encoder preparation, compute submission, profiled readback, and CPU candidate/cross-check/sort processing. Cross-check must still execute forward and reverse matching exactly once each.

All timing structs need deterministic zero defaults, checked/saturating count accumulation, and finite non-negative wall-clock values. Do not request timestamp-query features or change device limits.

- [x] **Step 5: Fold GPU detail into `MatchFeaturesTimingReport`**

Add backward-compatible, serde-defaulted fields for GPU descriptor matching, geometry validation, descriptor packing, buffer preparation, submission, readback total/copy/wait/map-decode, CPU postprocessing, direction calls, readback calls, and readback bytes. The computed GPU path must collect one matcher timing per pair and aggregate in memory; CPU and existing-match paths keep zero GPU-specific values. Do not emit per-pair logs.

- [x] **Step 6: Verify GREEN and existing matcher behavior**

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu gpu_match_timing -- --nocapture
```

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu gpu::tests::wgpu_sift_matcher_applies_ratio_distance_and_cross_check -- --exact --nocapture
```

Expected: timing tests and the existing result-equivalence matcher test pass, or adapter-dependent tests explicitly skip when no adapter is available.

- [x] **Step 7: Run matching regressions and compile checks**

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu feature_matching_db::tests -- --nocapture
```

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo check -p rustsfm \
  --no-default-features --features gpu-wgpu
```

Expected: all feature matching tests pass and the macOS wgpu configuration compiles.

- [x] **Step 8: Format, self-review, and commit**

```bash
cargo fmt --check
git diff --check
git add RustSFM/src/gpu/context.rs RustSFM/src/gpu/matcher.rs RustSFM/src/gpu/mod.rs RustSFM/src/feature/feature_matching_db.rs docs/superpowers/plans/2026-08-08-rustsfm-match-pair-observability.md
git commit -m "feat(rustsfm): profile gpu match-pair work"
```

Self-review must confirm byte/call accounting, timer nesting, empty-input behavior, and exact result equivalence. No shader, pair scheduling, database, event, pause/cancel, backend selection, or keyframe-count behavior may change.

### Task 3: Repeatable Match-Pair Benchmark Harness

**Files:**
- Create: `RustSFM/src/diagnostics/match_pair_benchmark.rs`
- Modify: `RustSFM/src/feature/feature_matching_db.rs`
- Modify: `RustSFM/src/lib.rs`
- Modify: `RustSFM/src/cli/mod.rs`
- Modify: `RustSFM/src/cli/commands.rs`
- Test: `RustSFM/src/diagnostics/match_pair_benchmark.rs`
- Test: `RustSFM/src/cli/mod.rs`

- [x] **Step 1: Write failing benchmark-contract tests**

Add CPU-only tests for a deterministic helper that selects local-window pairs from sorted image IDs. Prove `pair_limit=Some(96)` returns exactly 96 pairs, while `pair_limit=None` preserves a generated set larger than 2,890 pairs. Add a tiny-database test proving two repetitions return two structured run summaries and leave the source database's matches and two-view geometries unchanged. Add CLI parsing tests for `benchmark-match-pairs --window 5 --pair-limit 96 --repetitions 3 --use-gpu --output-json report.json`.

- [x] **Step 2: Run focused tests and verify RED**

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu match_pair_benchmark -- --nocapture
```

Expected: compilation failure because the benchmark types, pair selector, and CLI command do not exist.

- [x] **Step 3: Add the isolated benchmark API**

Create serializable `MatchPairBenchmarkReport` and `MatchPairBenchmarkRun` records. The top-level report must contain the source database path, copied byte count, database-copy duration, requested pair limit, actual pair count, repetition count, one-time backend initialization duration, and one summary per run. Each run contains aggregate match counts, `matching_seconds`, and `MatchFeaturesTimingReport`; it must not duplicate every per-pair report.

Add:

```rust
pub fn benchmark_match_pairs(
    source_database: &Path,
    window: usize,
    pair_limit: Option<usize>,
    repetitions: usize,
    options: &MatchFeaturesOptions,
) -> Result<MatchPairBenchmarkReport>
```

Reject zero window, zero pair limit, and zero repetitions before creating GPU state. Copy the source database once into a `tempfile::TempDir`, generate local-window pairs from image IDs sorted by `(name, image_id)`, and reuse one `ExplicitPairMatchingSession` for every repetition. Force `clear_existing=true` only on the temporary working database so every run starts from the same persistence state. Keep source database contents unchanged. Report session initialization once; per-run timings must set backend initialization to zero and recompute unclassified time against that run's `matching_seconds`.

- [x] **Step 4: Add the CLI command and structured output**

Add `benchmark-match-pairs` with `--database`, `--window` (default 5), optional `--pair-limit`, `--repetitions` (default 1), `--use-gpu`, optional `--output-json`, `--random-seed` (default 0), and `--log-level`. Build matching options from RustSFM defaults with `task_pair_batch_size=1`, matching RustViewer's current pair commit cadence. Print a one-line aggregate and write pretty JSON when requested.

The command must not add a default pair or image cap. `--pair-limit` is an explicit benchmark-only request and must not be threaded into RustViewer or `MapperConfig`.

- [x] **Step 5: Verify focused behavior and CLI parsing**

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu match_pair_benchmark -- --nocapture
```

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --bin rustsfm --no-default-features --features gpu-wgpu benchmark_match_pairs -- --nocapture
```

Expected: all benchmark and parsing tests pass without requiring a GPU adapter.

- [x] **Step 6: Compile, format, review, and commit**

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo check -p rustsfm \
  --no-default-features --features gpu-wgpu
cargo fmt --all -- --check
git diff --check
git add RustSFM/src/diagnostics/match_pair_benchmark.rs RustSFM/src/feature/feature_matching_db.rs RustSFM/src/lib.rs RustSFM/src/cli/mod.rs RustSFM/src/cli/commands.rs docs/superpowers/plans/2026-08-08-rustsfm-match-pair-observability.md
git commit -m "feat(rustsfm): add match-pair benchmark harness"
```

Self-review must confirm the source database is never opened for writes, session initialization is not counted once per repetition, pair selection has no implicit cap, and benchmark-only options do not affect RustViewer behavior.

## flowers2 Baseline Evidence

The release benchmark used the 960-image database at
`test_data/flowers2/out9/Untitled.rustscanproject/Cache/.staging/keyframe_sfm-1/rustsfm/Cache/database.db`
with the generic `gpu-wgpu` backend, local window 5, random seed 0, and one-pair commit cadence.
The benchmark source was opened read-only and copied to a temporary working database. No benchmark
pair or image cap is enabled unless `--pair-limit` is explicitly supplied.

The bounded 96-pair run repeated three times in 17.098, 17.064, and 17.100 seconds. Every run
matched and verified all 96 pairs with 62,409 matches. Descriptor matching consumed 6.38-6.42
seconds and geometry validation consumed 10.32-10.50 seconds. The stable repetitions show that the
measurement is repeatable and that backend initialization, reported separately at 0.017 seconds,
does not explain the sustained delay.

The 2,890-pair run completed in 452.045 seconds (7 minutes 32 seconds), matched and verified every
pair, and produced 2,958,062 matches. Its timing breakdown was:

- pair computation: 448.238 seconds;
- geometry validation: 251.470 seconds (55.6 percent of total);
- descriptor matching: 196.765 seconds (43.5 percent of total);
- descriptor readback total: 189.940 seconds, including 189.677 seconds waiting for GPU work;
- descriptor packing and buffer preparation: 6.327 seconds combined;
- SQLite commits: 3.246 seconds (0.7 percent of total);
- event delivery: 0.001 seconds.

Cross-check issued 5,780 one-way descriptor dispatch/readback calls and copied 756,094,400 bytes.
The low CPU utilization observed in RustViewer is therefore consistent with repeated GPU queue and
readback waits, not SQLite persistence. Geometry is now the largest unclassified sub-pipeline; the
next measurement must count its RANSAC summary and inlier-mask dispatch/readback synchronization
before changing RANSAC chunking or result semantics.

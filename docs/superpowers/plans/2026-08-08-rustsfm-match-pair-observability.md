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

- [ ] **Step 1: Write failing compatibility and accounting tests**

Add tests proving that an old serialized `MatchFeaturesReport` without `timings` deserializes with a default timing summary, and that controlled computed matching reports the exact attempted-pair and committed-batch counts for batch sizes 1 and 2. Assert every seconds value is finite and non-negative. Do not assert a wall-clock performance threshold.

- [ ] **Step 2: Run the focused tests and verify RED**

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu match_feature_timing -- --nocapture
```

Expected: compilation or assertion failure because `MatchFeaturesTimingReport` and `MatchFeaturesReport.timings` do not exist.

- [ ] **Step 3: Add the report contract**

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

- [ ] **Step 4: Instrument both matching entry points**

Measure backend construction, database/frame preparation, pair computation, transaction persistence, and event delivery. `commit_and_emit_pair_batch` must return or accumulate commit and event timings without changing event contents or checkpoint placement. Count attempted pairs from input batches, not only successful reports. Compute `unclassified_seconds` with saturating floating-point subtraction from the total so nesting or timer resolution cannot produce a negative value.

`ExplicitPairMatchingSession` should store its one-time initialization duration so every explicit report identifies cold-start cost without recreating the backend. Do not change `task_pair_batch_size`, pair order, GPU selection, database schema, or matching thresholds.

- [ ] **Step 5: Run focused tests and verify GREEN**

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu match_feature_timing -- --nocapture
```

Expected: all timing compatibility and accounting tests pass.

- [ ] **Step 6: Run behavioral regression tests**

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu controlled_matching_ -- --nocapture
```

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu controlled_computed_matching_ -- --nocapture
```

Expected: progress, rollback, pause, and cancellation tests remain unchanged and pass.

- [ ] **Step 7: Compile the macOS wgpu configuration**

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo check -p rustsfm \
  --no-default-features --features gpu-wgpu
```

Expected: exit code 0. This command must not run GPU-heavy tests.

- [ ] **Step 8: Format, self-review, and commit**

```bash
cargo fmt --check
git diff --check
git add RustSFM/src/feature/feature_matching_db.rs RustSFM/src/lib.rs docs/superpowers/plans/2026-08-08-rustsfm-match-pair-observability.md
git commit -m "feat(rustsfm): report match-pair timing breakdown"
```

Self-review must explicitly confirm that matching results, event sequence, transaction boundaries, and pause/cancel checkpoints are unchanged.

# RustSFM Incremental Registration Hot-Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove measured incremental-mapper memory copies and quadratic bookkeeping, then replace eager all-image PnP sweeps with deterministic support-aware lazy retries while preserving exhaustive registration fallback.

**Architecture:** `ObservationManager` will share its immutable correspondence graph through `Arc` and own a monotonic external point-ID allocator. The mapper will retain vector-backed reconstruction and existing candidate ranking, but track the support level of failed registration units and invoke PnP lazily. A final fallback pass ignores exhausted trial counters before model termination. Session telemetry will expose candidate, pose, and triangulation costs.

**Tech Stack:** Rust 2021, `std::sync::Arc`, existing RustSFM incremental mapper and unit-test fixtures, COLMAP-compatible sparse output, fixed-seed `flowers2` benchmarks.

---

### Task 1: Shared Immutable Correspondence Graph

**Files:**
- Modify: `RustSFM/src/sfm/observation_manager.rs`
- Test: `RustSFM/src/sfm/observation_manager.rs`

- [x] **Step 1: Write the failing shared-allocation test**

Add a module-private test that clones the installed handle and requires `Arc::ptr_eq`:

```rust
#[test]
fn correspondence_graph_clones_share_one_allocation() {
    let frames = vec![frame(0, 100, 100), frame(1, 100, 100)];
    let pairs = vec![pair(0, 1, &[(0, 0)])];
    let reconstruction = reconstruction(&frames);
    let manager = ObservationManager::new(&frames, &pairs, &reconstruction);

    let first = manager.correspondence_graph.clone().expect("graph");
    let second = first.clone();
    assert!(Arc::ptr_eq(&first, &second));
}
```

- [x] **Step 2: Run the focused test and verify RED**

Run:

```bash
cargo test -p rustsfm correspondence_graph_clones_share_one_allocation
```

Expected: compilation fails because the field contains `CorrespondenceGraph`, not `Arc<CorrespondenceGraph>`.

- [x] **Step 3: Store the graph in `Arc`**

Import `Arc`, change the field and installation path, and preserve the public borrowed accessor:

```rust
use std::sync::Arc;

pub struct ObservationManager {
    image_stats: Vec<ImageStat>,
    image_pair_stats: HashMap<(usize, usize), ImagePairStat>,
    point3d_correspondence_counts: Vec<Vec<usize>>,
    modified_point3d_ids: HashSet<usize>,
    correspondence_graph: Option<Arc<CorrespondenceGraph>>,
}

pub fn install_correspondence_graph(&mut self, graph: CorrespondenceGraph) {
    self.correspondence_graph = Some(Arc::new(graph));
}

pub fn correspondence_graph(&self) -> Option<&CorrespondenceGraph> {
    self.correspondence_graph.as_deref()
}
```

Existing `self.correspondence_graph.clone()` mutation paths remain structurally unchanged but now clone only an `Arc`.

- [x] **Step 4: Verify graph behavior and manager regressions**

Run:

```bash
cargo test -p rustsfm observation_manager
cargo test -p rustsfm incremental_triangulator
```

Expected: PASS, including existing visibility, add, delete, merge, register, and rollback tests.

- [x] **Step 5: Commit Task 1**

```bash
git add RustSFM/src/sfm/observation_manager.rs
git commit -m "perf(rustsfm): share mapper correspondence graph"
```

### Task 2: Monotonic O(1) Point-ID Allocation

**Files:**
- Modify: `RustSFM/src/sfm/observation_manager.rs`
- Test: `RustSFM/src/sfm/observation_manager.rs`

- [x] **Step 1: Write failing point-ID behavior tests**

Add one test that creates a point with external ID 41, deletes it, and adds a new valid point. Assert the new external ID is 42 rather than reusing 41. Add a second fixture with two points but only one `point_ids` entry and assert the cold repair path produces unique IDs before appending a new point.

```rust
assert!(manager.delete_point3d(&frames, &pairs, &mut reconstruction, highest_index));
let new_index = manager
    .add_point3d(&frames, &pairs, &mut reconstruction, replacement)
    .expect("replacement point");
assert_eq!(reconstruction.point_ids[new_index], 42);
assert_eq!(reconstruction.point_ids.len(), reconstruction.points.len());
assert_eq!(reconstruction.point_ids.iter().copied().collect::<HashSet<_>>().len(), reconstruction.point_ids.len());
```

- [x] **Step 2: Run the focused tests and verify RED**

Run:

```bash
cargo test -p rustsfm point3d_id_allocator
```

Expected: the delete/add test fails because the current maximum scan reuses the deleted highest ID, and the allocator API does not exist.

- [x] **Step 3: Add the monotonic allocator state**

Add an internal allocator whose normal path does not inspect `point_ids`:

```rust
#[derive(Debug, Clone)]
struct Point3DIdAllocator {
    next: Option<u64>,
}

impl Point3DIdAllocator {
    fn from_reconstruction(reconstruction: &Reconstruction) -> Self {
        Self {
            next: reconstruction
                .point_ids
                .iter()
                .copied()
                .max()
                .map_or(Some(1), |max_id| max_id.checked_add(1)),
        }
    }

    fn allocate(&mut self) -> Option<u64> {
        let id = self.next?;
        self.next = id.checked_add(1);
        Some(id)
    }

    fn observe(&mut self, id: u64) {
        let observed_next = id.checked_add(1);
        self.next = match (self.next, observed_next) {
            (Some(current), Some(observed)) => Some(current.max(observed)),
            _ => None,
        };
    }
}
```

Store it in `ObservationManager`, initialize it in `new`, and preserve its monotonic value across `rebuild` and rollback by only increasing it when imported IDs are larger.

- [x] **Step 4: Make legacy repair cold and allocation constant-time**

Replace unconditional table scans in `add_point3d` with:

```rust
if reconstruction.point_ids.len() < reconstruction.points.len() {
    repair_point_id_table(reconstruction, &mut self.point3d_id_allocator)?;
}
let external_id = self.point3d_id_allocator.allocate()?;
reconstruction.point_ids.push(external_id);
reconstruction.points.push(point);
```

`repair_point_id_table` may scan existing IDs once only while the vectors are mismatched. Remove `next_point3d_id` and keep delete/merge external-ID semantics unchanged.

- [x] **Step 5: Verify point and triangulation suites**

Run:

```bash
cargo test -p rustsfm point3d_id_allocator
cargo test -p rustsfm observation_manager
cargo test -p rustsfm incremental_triangulator
```

Expected: PASS, with unique non-reused external IDs and unchanged track statistics.

- [x] **Step 6: Commit Task 2**

```bash
git add RustSFM/src/sfm/observation_manager.rs
git commit -m "perf(rustsfm): allocate point ids in constant time"
```

### Task 3: Support-Aware Lazy Registration Retries

**Files:**
- Modify: `RustSFM/src/sfm/mapper.rs`
- Test: `RustSFM/src/sfm/mapper.rs`

- [x] **Step 1: Write failing retry-state tests**

Add tests around a new `RegistrationRetryState` with separate structure-based and structureless support. Cover unchanged suppression, support growth, rig-unit propagation, and fallback:

```rust
let mut state = RegistrationRetryState::new(reconstruction.poses.len());
state.record_failure(&reconstruction, 2, NextImageRegistrationMode::StructureBased, 30);
state.record_failure(&reconstruction, 2, NextImageRegistrationMode::StructureBased, 30);
state.record_failure(&reconstruction, 2, NextImageRegistrationMode::StructureBased, 30);
assert!(!state.is_eligible(&reconstruction, 2, NextImageRegistrationMode::StructureBased, 30, 3, false));
assert!(state.is_eligible(&reconstruction, 2, NextImageRegistrationMode::StructureBased, 31, 3, false));
assert!(state.is_eligible(&reconstruction, 2, NextImageRegistrationMode::StructureBased, 30, 3, true));
```

- [x] **Step 2: Run focused tests and verify RED**

Run:

```bash
cargo test -p rustsfm registration_retry_state
```

Expected: compilation fails because `RegistrationRetryState` does not exist.

- [x] **Step 3: Implement unit-aware retry state**

Add:

```rust
#[derive(Debug, Clone, Copy, Default)]
struct RegistrationModeAttempt {
    trials: usize,
    last_support: usize,
}

#[derive(Debug, Clone)]
struct RegistrationRetryState {
    structure_based: Vec<RegistrationModeAttempt>,
    structureless: Vec<RegistrationModeAttempt>,
}
```

Methods use `image_indices_for_registration_unit` to read and update all rig siblings. A support increase resets trials before eligibility is evaluated. `record_failure` stores the current support and increments trials for the whole unit. Success clears the chosen mode for the whole unit.

- [x] **Step 4: Route candidate selection through retry state**

Replace the two raw trial vectors in `incremental_map_single_attempt_with_pnp_scorer` and candidate helpers with `RegistrationRetryState`. `find_next_registration_images` receives a pass enum:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegistrationPass {
    Normal,
    ExhaustiveFallback,
}
```

Normal selection respects support-aware `max_reg_trials`. Fallback ignores exhausted trials but retains minimum support, filtered-unit ordering, deterministic rank ordering, and registration-unit deduplication.

- [x] **Step 5: Remove the eager all-image PnP sweep**

Delete the production call sequence:

```rust
reset_unregistered_registration_trials(...);
mark_unregistered_images_with_no_absolute_pose_and_pnp_scorer(...)?;
```

When normal selection returns no choice, immediately run one `ExhaustiveFallback` selection. If it also returns no choice, terminate the model. If it succeeds, register the image and return to normal selection. Failed attempts from either pass are recorded with the support used for that attempt.

- [x] **Step 6: Verify retry and mapper registration regressions**

Run:

```bash
cargo test -p rustsfm registration_retry_state
cargo test -p rustsfm find_next_registration_images
cargo test -p rustsfm choose_next_registration
cargo test -p rustsfm mapper_pnp
```

Expected: PASS. Existing deterministic queue ordering remains unchanged, and the fallback test proves a bridge candidate is attempted before termination.

- [x] **Step 7: Commit Task 3**

```bash
git add RustSFM/src/sfm/mapper.rs
git commit -m "perf(rustsfm): retry registration candidates lazily"
```

### Task 4: Incremental Registration Telemetry

**Files:**
- Modify: `RustSFM/src/sfm/mapper.rs`
- Test: `RustSFM/src/sfm/mapper.rs`

- [x] **Step 1: Write the failing telemetry-format test**

Define the expected stable keys before implementation:

```rust
#[test]
fn incremental_registration_telemetry_reports_hot_path_stages() {
    let telemetry = IncrementalRegistrationTelemetry {
        candidate_units: 7,
        skipped_unchanged: 3,
        structure_based_attempts: 4,
        structureless_attempts: 2,
        fallback_epochs: 1,
        collect_observations_ms: 1.25,
        pose_solve_refine_ms: 2.5,
        observation_update_ms: 3.75,
        triangulation_ms: 4.5,
    };
    let line = telemetry.format_log();
    for key in ["candidate_units=7", "skipped_unchanged=3", "collect_observations_ms=1.25", "triangulation_ms=4.50"] {
        assert!(line.contains(key), "missing {key}: {line}");
    }
}
```

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
cargo test -p rustsfm incremental_registration_telemetry_reports_hot_path_stages
```

Expected: compilation fails because the telemetry type is missing.

- [x] **Step 3: Add session-scoped counters and timers**

Create `IncrementalRegistrationTelemetry` in `mapper.rs`. Pass one mutable instance through candidate selection. Time observation collection inside `solve_absolute_pose_with_pnp_scorer`, time the remaining pose solve/refinement separately, and time observation updates and triangulation blocks in the registration loop with `Instant`.

The aggregate line must be:

```text
incremental_registration candidate_units=... skipped_unchanged=... structure_based_attempts=... structureless_attempts=... fallback_epochs=... collect_observations_ms=... pose_solve_refine_ms=... observation_update_ms=... triangulation_ms=...
```

Append it to `debug_log` once when the incremental mapping attempt exits. Do not alter existing `pnp_timing` output.

- [x] **Step 4: Verify telemetry and full RustSFM library tests**

Run:

```bash
cargo test -p rustsfm incremental_registration_telemetry
cargo test -p rustsfm --lib -- --skip real_colmap_sparse
```

Expected: PASS with the repository's known real-fixture exclusions only.

- [x] **Step 5: Commit Task 4**

```bash
git add RustSFM/src/sfm/mapper.rs
git commit -m "perf(rustsfm): report registration hot-path timings"
```

### Task 5: Formatting, Bounded Benchmark, and RustGS Compatibility

**Files:**
- Modify: `docs/superpowers/plans/2026-07-19-rustsfm-incremental-registration-hot-path.md`

- [x] **Step 1: Run static and regression verification**

Run:

```bash
cargo fmt --all -- --check
cargo check -p rustsfm
cargo test -p rustsfm --lib -- --skip real_colmap_sparse
cargo check -p rustgs
```

Expected: all commands pass; the RustSFM test count is at least the pre-change 562 passing tests with only 19 known fixture tests filtered.

- [x] **Step 2: Run a fixed-seed 200-image benchmark**

Build release and run against the existing `flowers2` database in a new `/tmp` output directory, preserving single-thread mapper behavior:

```bash
cargo build --release -p rustsfm
target/release/rustsfm reconstruct \
  --input /Users/tfjiang/Projects/RustScan/test_data/flowers2/images \
  --output /tmp/rustsfm-registration-hot-path-20260719/optimized200 \
  --database /Users/tfjiang/Projects/RustScan/test_data/flowers2/rustsfm_full_work/database.db \
  --max-images 200 \
  --random-seed 0 \
  --threads 1 \
  --log-level debug \
  --summary-json /tmp/rustsfm-registration-hot-path-20260719/optimized200/summary.json
```

Expected: 200/200 images register, poses are finite, output contains `sparse/0`, and telemetry shows materially fewer pose attempts than the old eager sweep.

- [x] **Step 3: Probe RustGS input compatibility**

Run a one-frame, one-iteration training probe against the optimized output:

```bash
cargo run -p rustgs --release --bin rustgs -- train \
  --input /tmp/rustsfm-registration-hot-path-20260719/optimized200 \
  --image-root /Users/tfjiang/Projects/RustScan/test_data/flowers2/images \
  --output /tmp/rustsfm-registration-hot-path-20260719/rustgs-probe.ply \
  --iterations 1 \
  --max-frames 1 \
  --max-initial-gaussians 1000
```

Expected: RustGS resolves `sparse/0`, reports zero missing registered images, initializes from a non-empty sparse point cloud, and writes the probe PLY.

- [x] **Step 4: Record benchmark evidence in this plan**

Append the before/after wall time, registered images, points, pose-attempt count, fallback epochs, and hot-path timing totals. Do not commit `/tmp` outputs or logs.

- [x] **Step 5: Final review and commit**

Review `git diff`, run `git diff --check`, and commit only the benchmark record if all correctness gates pass:

```bash
git add docs/superpowers/plans/2026-07-19-rustsfm-incremental-registration-hot-path.md
git commit -m "docs(rustsfm): record registration hot-path benchmark"
```

## Execution Record (2026-07-19)

All commands ran from the `feat/rustsfm-wgpu` linked worktree on an Apple M5 Max. The mapper
benchmark used `--threads 1`, fixed random seed `0`, and the existing `flowers2` database.

The pre-change profile covered all 960 images rather than this bounded 200-image sample, so its
approximately 4.5 hour wall time and 42,697 PnP calls are retained as motivation only. No direct
wall-time speedup ratio is reported across the different input sizes. The old PnP internals
accounted for 57.3 seconds, or less than 0.4 percent of that run.

The bounded post-change benchmark produced:

- wall time: 362.31 seconds (`summary.elapsed_ms=361673.86`);
- registered images: 200/200, with 200 finite exported poses and no non-finite pose values;
- sparse points and pairs: 91,894 points and 993 pairs;
- pose work: 5,542 ranked candidate-unit entries, 200 structure-based attempts, 0
  structureless attempts, and 0 fallback epochs;
- retry behavior: 2 failed attempts, 198 successful registrations after the initial pair, no
  registration rollback, and no unchanged-support skips;
- hot-path totals: 54.89 ms collecting observations, 2,251.72 ms solving/refining poses,
  31.57 ms updating observations, and 39,203.18 ms in per-registration triangulation;
- outputs: a COLMAP-compatible `sparse/0` text model plus the 200 selected images.

The bounded run therefore used approximately one PnP attempt per input image. Observation
collection plus pose solve/refinement consumed about 0.64 percent of wall time, while measured
per-registration triangulation consumed about 10.84 percent. The remaining runtime is dominated
by local and scheduled/final global BA and their postprocessing rather than PnP.

The RustGS compatibility probe resolved `sparse/0`, reported zero missing images, loaded 91,894
initialization points, and trained one selected frame for one iteration through wgpu/Metal. It
exported 1,000 Gaussians to `rustgs-probe.ply` in 0.81 seconds, and the parity report passed its
no-NaN, no-OOM, and export-roundtrip gates. Benchmark artifacts and logs remain under
`/tmp/rustsfm-registration-hot-path-20260719` and are not committed.

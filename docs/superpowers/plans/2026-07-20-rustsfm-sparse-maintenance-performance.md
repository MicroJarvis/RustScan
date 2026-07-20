# RustSFM Sparse Maintenance Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace superlinear point deletion and cumulative mapper maintenance with O(track-length) swap deletion, bounded dirty frontiers, and geometrically identical subset filtering.

**Architecture:** Keep `Reconstruction` vectors and stable external point IDs, but replace shifting deletion with one tail-to-hole remap. Store session telemetry in `ObservationManager`, instrument track completion and merging through `IncrementalTriangulator`, and make `mapper.rs` select either full filtering at global boundaries or dirty-frontier filtering for registration and local BA.

**Tech Stack:** Rust 2021, existing RustSFM observation/triangulation/mapper modules, `HashSet`, `Instant`, Ceres-backed BA, COLMAP-compatible text export, RustGS Metal training probe.

---

## File Map

- Modify `RustSFM/src/sfm/observation_manager.rs`: swap deletion, dirty-index remapping,
  consuming frontier, and session-scoped sparse-maintenance counters.
- Modify `RustSFM/src/sfm/incremental_triangulator.rs`: merge-trial remapping, frontier API,
  rollback cleanup, and complete/merge timing.
- Modify `RustSFM/src/sfm/mapper.rs`: shared point-filter kernel, full/subset traversal,
  maintenance scheduling, and aggregate diagnostic output.
- Create `docs/superpowers/plans/2026-07-20-rustsfm-sparse-maintenance-performance-benchmark.md`:
  exact 96/200/960-image results and RustGS compatibility evidence.

No new runtime module is needed: the behavior belongs to the existing ownership boundaries, and
splitting it out would introduce circular access to `Reconstruction` and `ObservationManager`.

### Task 1: Swap-Based Point Deletion

**Files:**
- Modify: `RustSFM/src/sfm/observation_manager.rs:86-94`
- Modify: `RustSFM/src/sfm/observation_manager.rs:473-505`
- Modify: `RustSFM/src/sfm/observation_manager.rs:817-862`
- Test: `RustSFM/src/sfm/observation_manager.rs:980-1840`

- [ ] **Step 1: Write the failing middle-deletion test**

Add a test that constructs four independently tracked points, marks only the tail point dirty,
deletes point index 1, and requires the tail point rather than every trailing point to move:

```rust
#[test]
fn delete_middle_point_swap_removes_tail_and_remaps_tracks() {
    let frames = (0..8)
        .map(|id| frame(id, 100, 100))
        .collect::<Vec<_>>();
    let mut reconstruction = reconstruction(&frames);
    reconstruction.poses.fill(Some(SE3::identity()));
    for point_id in 0..4 {
        let track = vec![
            TrackObservation {
                image: point_id * 2,
                feature: 0,
            },
            TrackObservation {
                image: point_id * 2 + 1,
                feature: 0,
            },
        ];
        for obs in &track {
            reconstruction.observations[obs.image][obs.feature] = Some(point_id);
        }
        reconstruction.point_ids.push(100 + point_id as u64);
        reconstruction.points.push(Point3D {
            xyz: [point_id as f32, 0.0, 2.0],
            color: [point_id as u8, 0, 0],
            error: 0.0,
            track,
        });
    }
    let mut manager = ObservationManager::new(&frames, &[], &reconstruction);
    manager.mark_point3d_modified(3);

    assert!(manager.delete_point3d(&frames, &[], &mut reconstruction, 1));

    assert_eq!(reconstruction.point_ids, vec![100, 103, 102]);
    assert_eq!(reconstruction.points[1].color, [3, 0, 0]);
    assert_eq!(reconstruction.points[2].color, [2, 0, 0]);
    assert_eq!(reconstruction.observations[2][0], None);
    assert_eq!(reconstruction.observations[3][0], None);
    assert_eq!(reconstruction.observations[6][0], Some(1));
    assert_eq!(reconstruction.observations[7][0], Some(1));
    assert_eq!(reconstruction.observations[4][0], Some(2));
    assert_eq!(reconstruction.observations[5][0], Some(2));
    assert_eq!(manager.modified_point3d_ids(), &HashSet::from([1]));
}
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
cargo test -p rustsfm --lib delete_middle_point_swap_removes_tail_and_remaps_tracks
```

Expected: FAIL because shifting deletion leaves external IDs ordered as `[100, 102, 103]`, moves
both trailing points, and maps the dirty tail from index 3 to 2.

- [ ] **Step 3: Implement the single-index remap and swap deletion**

Add the private remap type next to `ObservationManager`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Point3DIndexRemap {
    removed: usize,
    moved_from: Option<usize>,
}

impl Point3DIndexRemap {
    fn remap_existing(self, point_id: usize) -> Option<usize> {
        if point_id == self.removed {
            None
        } else if self.moved_from == Some(point_id) {
            Some(self.removed)
        } else {
            Some(point_id)
        }
    }
}
```

Change `delete_point3d_internal` to return `Option<Point3DIndexRemap>`. Keep the public
`delete_point3d` return type as `bool` by replacing boolean negation with `.is_none()` checks.
Use this complete deletion body:

```rust
fn delete_point3d_internal(
    &mut self,
    reconstruction: &mut Reconstruction,
    point_id: usize,
) -> Option<Point3DIndexRemap> {
    if point_id >= reconstruction.points.len()
        || !repair_point_id_table(reconstruction, &mut self.point3d_id_allocator)
    {
        return None;
    }

    let last_point_id = reconstruction.points.len() - 1;
    let remap = Point3DIndexRemap {
        removed: point_id,
        moved_from: (point_id != last_point_id).then_some(last_point_id),
    };
    let removed_point = reconstruction.points.swap_remove(point_id);
    reconstruction.point_ids.swap_remove(point_id);

    for obs in removed_point.track {
        let slot = &mut reconstruction.observations[obs.image][obs.feature];
        if *slot == Some(point_id) {
            *slot = None;
        }
    }
    if let Some(moved_from) = remap.moved_from {
        for obs in &reconstruction.points[point_id].track {
            let slot = &mut reconstruction.observations[obs.image][obs.feature];
            if *slot == Some(moved_from) {
                *slot = Some(point_id);
            }
        }
    }

    self.modified_point3d_ids = self
        .modified_point3d_ids
        .iter()
        .filter_map(|&id| remap.remap_existing(id))
        .collect();
    Some(remap)
}
```

In `merge_points3d`, use `self.delete_point3d_internal(reconstruction, remove_id)?;` and retain
`Some(keep_id)` as the public result.

- [ ] **Step 4: Run observation-manager tests and verify GREEN**

Run:

```bash
cargo test -p rustsfm --lib observation_manager::tests
```

Expected: PASS, including the new swap-remap test and all existing allocator/statistics tests.

- [ ] **Step 5: Commit Task 1**

```bash
git add RustSFM/src/sfm/observation_manager.rs
git commit -m "perf(rustsfm): swap-remove deleted points"
```

### Task 2: Merge-Trial Cache Remapping

**Files:**
- Modify: `RustSFM/src/sfm/incremental_triangulator.rs:176-190`
- Modify: `RustSFM/src/sfm/incremental_triangulator.rs:918-979`
- Test: `RustSFM/src/sfm/incremental_triangulator.rs:1200-2090`

- [ ] **Step 1: Write the failing cache-remap test**

```rust
#[test]
fn merge_trial_cache_remaps_only_the_swapped_tail_point() {
    let frames = vec![frame(0), frame(1)];
    let reconstruction = reconstruction(&frames);
    let mut state = IncrementalTriangulatorState::new(&frames, &[], &reconstruction);
    state.merge_trials = HashSet::from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]);

    state.sync_merge_trials_after_point_merge(0, 2, Some(5));

    assert_eq!(state.merge_trials, HashSet::from([(2, 4), (3, 4)]));
}
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
cargo test -p rustsfm --lib merge_trial_cache_remaps_only_the_swapped_tail_point
```

Expected: FAIL to compile because the current synchronization helper accepts only the retained
and removed indices and shifts every larger index.

- [ ] **Step 3: Implement cache remapping and pass the old tail index**

Replace the helper with:

```rust
fn sync_merge_trials_after_point_merge(
    &mut self,
    keep_id: usize,
    remove_id: usize,
    moved_from: Option<usize>,
) {
    self.merge_trials = self
        .merge_trials
        .iter()
        .filter_map(|&(left, right)| {
            if left == keep_id || right == keep_id || left == remove_id || right == remove_id {
                return None;
            }
            let remap = |point_id| {
                if moved_from == Some(point_id) {
                    remove_id
                } else {
                    point_id
                }
            };
            let left = remap(left);
            let right = remap(right);
            (left != right).then_some(ordered_point_pair(left, right))
        })
        .collect();
}
```

In `try_merge_pair`, capture the tail before calling `merge_points3d`:

```rust
let last_point_id = self.reconstruction.points.len() - 1;
let moved_from = (remove_id != last_point_id).then_some(last_point_id);
let merged = self
    .state
    .observation_manager_mut()
    .merge_points3d(
        self.frames,
        self.pairs,
        self.reconstruction,
        keep_id,
        remove_id,
        Point3D {
            xyz: merged_xyz,
            color: merged_color,
            error: merged_error,
            track: merged_track,
        },
    )
    .is_some();
if merged {
    self.state
        .sync_merge_trials_after_point_merge(keep_id, remove_id, moved_from);
}
```

- [ ] **Step 4: Run merge and observation tests**

Run:

```bash
cargo test -p rustsfm --lib merge_tracks
cargo test -p rustsfm --lib merge_trial_cache
cargo test -p rustsfm --lib observation_manager::tests
```

Expected: PASS with no stale or out-of-range cached point pair.

- [ ] **Step 5: Commit Task 2**

```bash
git add RustSFM/src/sfm/incremental_triangulator.rs
git commit -m "fix(rustsfm): remap merge cache after point swaps"
```

### Task 3: Consumable Dirty Frontier and Maintenance Telemetry

**Files:**
- Modify: `RustSFM/src/sfm/observation_manager.rs:1-100`
- Modify: `RustSFM/src/sfm/observation_manager.rs:758-768`
- Modify: `RustSFM/src/sfm/incremental_triangulator.rs:109-190`
- Modify: `RustSFM/src/sfm/incremental_triangulator.rs:362-400`
- Test: `RustSFM/src/sfm/observation_manager.rs:980-1840`
- Test: `RustSFM/src/sfm/incremental_triangulator.rs:1200-2090`

- [ ] **Step 1: Write the failing frontier-consumption test**

```rust
#[test]
fn take_modified_points_consumes_only_the_current_frontier() {
    let frames = vec![frame(0, 100, 100), frame(1, 100, 100)];
    let reconstruction = reconstruction(&frames);
    let mut manager = ObservationManager::new(&frames, &[], &reconstruction);
    manager.mark_point3d_modified(3);
    manager.mark_point3d_modified(7);

    assert_eq!(manager.take_modified_point3d_ids(), HashSet::from([3, 7]));
    assert!(manager.modified_point3d_ids().is_empty());

    manager.mark_point3d_modified(9);
    assert_eq!(manager.take_modified_point3d_ids(), HashSet::from([9]));
}
```

Add a second failing test for tail deletion and deletion-work telemetry:

```rust
#[test]
fn delete_tail_records_no_move_and_preserves_unrelated_tracks() {
    let frames = (0..4)
        .map(|id| frame(id, 100, 100))
        .collect::<Vec<_>>();
    let mut reconstruction = reconstruction(&frames);
    reconstruction.poses.fill(Some(SE3::identity()));
    let mut manager = ObservationManager::new(&frames, &[], &reconstruction);
    for image_pair in [[0, 1], [2, 3]] {
        manager
            .add_point3d(
                &frames,
                &[],
                &mut reconstruction,
                Point3D {
                    xyz: [0.0, 0.0, 2.0],
                    color: [0, 0, 0],
                    error: 0.0,
                    track: image_pair
                        .into_iter()
                        .map(|image| TrackObservation { image, feature: 0 })
                        .collect(),
                },
            )
            .expect("point");
    }
    manager.clear_modified_point3d_ids();

    assert!(manager.delete_point3d(&frames, &[], &mut reconstruction, 1));

    assert_eq!(reconstruction.points.len(), 1);
    assert_eq!(reconstruction.observations[0][0], Some(0));
    assert_eq!(reconstruction.observations[1][0], Some(0));
    assert_eq!(reconstruction.observations[2][0], None);
    assert_eq!(reconstruction.observations[3][0], None);
    let log = manager.sparse_maintenance_log();
    assert!(log.contains("point_deletes=1"), "{log}");
    assert!(log.contains("moved_points=0"), "{log}");
    assert!(log.contains("rewritten_observations=2"), "{log}");
}
```

- [ ] **Step 2: Run the frontier test and verify RED**

```bash
cargo test -p rustsfm --lib take_modified_points_consumes_only_the_current_frontier
cargo test -p rustsfm --lib delete_tail_records_no_move_and_preserves_unrelated_tracks
```

Expected: FAIL to compile because there is no consuming frontier API.

- [ ] **Step 3: Add the telemetry record and consuming API**

Import `std::time::Instant` and add this record before `ObservationManager`:

```rust
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SparseMaintenanceTelemetry {
    complete_calls: usize,
    complete_frontier_points: usize,
    completed_observations: usize,
    complete_ms: f64,
    merge_calls: usize,
    merge_frontier_points: usize,
    merged_observations: usize,
    merge_ms: f64,
    full_filter_calls: usize,
    subset_filter_calls: usize,
    filter_points: usize,
    filter_observations: usize,
    filtered_observations: usize,
    filter_ms: f64,
    point_deletes: usize,
    moved_points: usize,
    rewritten_observations: usize,
    delete_ms: f64,
    frontier_peak: usize,
    frontier_cycles: usize,
    frontier_points_consumed: usize,
}
```

Store it as `maintenance: SparseMaintenanceTelemetry` in `ObservationManager`. Implement these
methods for complete, merge, filter, deletion, and frontier totals:

```rust
pub fn record_complete(&mut self, frontier: usize, changed: usize, elapsed_ms: f64) {
    self.maintenance.complete_calls += 1;
    self.maintenance.complete_frontier_points += frontier;
    self.maintenance.completed_observations += changed;
    self.maintenance.complete_ms += elapsed_ms;
}

pub fn record_merge(&mut self, frontier: usize, changed: usize, elapsed_ms: f64) {
    self.maintenance.merge_calls += 1;
    self.maintenance.merge_frontier_points += frontier;
    self.maintenance.merged_observations += changed;
    self.maintenance.merge_ms += elapsed_ms;
}

pub fn record_filter(
    &mut self,
    is_subset: bool,
    points: usize,
    observations: usize,
    removed: usize,
    elapsed_ms: f64,
) {
    if is_subset {
        self.maintenance.subset_filter_calls += 1;
    } else {
        self.maintenance.full_filter_calls += 1;
    }
    self.maintenance.filter_points += points;
    self.maintenance.filter_observations += observations;
    self.maintenance.filtered_observations += removed;
    self.maintenance.filter_ms += elapsed_ms;
}

fn record_delete(&mut self, moved: bool, rewritten: usize, elapsed_ms: f64) {
    self.maintenance.point_deletes += 1;
    self.maintenance.moved_points += if moved { 1 } else { 0 };
    self.maintenance.rewritten_observations += rewritten;
    self.maintenance.delete_ms += elapsed_ms;
}

pub fn take_modified_point3d_ids(&mut self) -> HashSet<usize> {
    let frontier = std::mem::take(&mut self.modified_point3d_ids);
    self.maintenance.frontier_cycles += 1;
    self.maintenance.frontier_points_consumed += frontier.len();
    frontier
}

pub fn sparse_maintenance_log(&self) -> String {
    let m = &self.maintenance;
    format!(
        "sparse_maintenance complete_calls={} complete_frontier_points={} completed_observations={} complete_ms={:.2} merge_calls={} merge_frontier_points={} merged_observations={} merge_ms={:.2} full_filter_calls={} subset_filter_calls={} filter_points={} filter_observations={} filtered_observations={} filter_ms={:.2} point_deletes={} moved_points={} rewritten_observations={} delete_ms={:.2} frontier_peak={} frontier_cycles={} frontier_points_consumed={}",
        m.complete_calls,
        m.complete_frontier_points,
        m.completed_observations,
        m.complete_ms,
        m.merge_calls,
        m.merge_frontier_points,
        m.merged_observations,
        m.merge_ms,
        m.full_filter_calls,
        m.subset_filter_calls,
        m.filter_points,
        m.filter_observations,
        m.filtered_observations,
        m.filter_ms,
        m.point_deletes,
        m.moved_points,
        m.rewritten_observations,
        m.delete_ms,
        m.frontier_peak,
        m.frontier_cycles,
        m.frontier_points_consumed,
    )
}
```

Update `mark_point3d_modified` after insertion:

```rust
self.maintenance.frontier_peak = self
    .maintenance
    .frontier_peak
    .max(self.modified_point3d_ids.len());
```

Time `delete_point3d_internal` with `Instant::now()`. Count both cleared deleted-track slots and
rewritten moved-track slots with this exact pattern:

```rust
let started = Instant::now();
let mut rewritten = 0usize;
// In each successful `*slot = None` or `*slot = Some(point_id)` branch:
rewritten += 1;
// After dirty-set remapping:
self.record_delete(
    remap.moved_from.is_some(),
    rewritten,
    started.elapsed().as_secs_f64() * 1_000.0,
);
```

- [ ] **Step 4: Write the failing complete/merge timing test**

```rust
#[test]
fn complete_and_merge_record_frontier_work() {
    let frames = vec![frame(0), frame(1)];
    let pairs = vec![pair(0, 1, &[])];
    let mut reconstruction = reconstruction(&frames);
    let mut state = IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
    let mut triangulator =
        IncrementalTriangulator::new(&frames, &pairs, &mut reconstruction, &mut state);

    triangulator.complete_tracks(&IncrementalTriangulatorOptions::default(), &HashSet::new());
    triangulator.merge_tracks(&IncrementalTriangulatorOptions::default(), &HashSet::new());

    let log = triangulator
        .state
        .observation_manager()
        .sparse_maintenance_log();
    assert!(log.contains("complete_calls=1"), "{log}");
    assert!(log.contains("merge_calls=1"), "{log}");
}
```

- [ ] **Step 5: Run the timing test and verify RED**

```bash
cargo test -p rustsfm --lib complete_and_merge_record_frontier_work
```

Expected: FAIL because `complete_tracks` and `merge_tracks` do not record work.

- [ ] **Step 6: Instrument complete/merge and clear rollback frontiers**

In both track methods, measure the existing operation and call the observation-manager record
method after the iterator completes:

```rust
let started = Instant::now();
let frontier_points = point_ids.len();
let changed = point_ids
    .iter()
    .copied()
    .map(|point_id| self.complete_track(options, point_id))
    .sum();
self.state.observation_manager_mut().record_complete(
    frontier_points,
    changed,
    started.elapsed().as_secs_f64() * 1_000.0,
);
changed
```

Import `std::time::Instant` in `incremental_triangulator.rs`. Implement `merge_tracks` with the
same record boundary and `record_merge(frontier_points, changed, elapsed_ms)`. Add this
pass-through to `IncrementalTriangulator`:

```rust
pub fn take_modified_points3d(&mut self) -> HashSet<usize> {
    self.state
        .observation_manager_mut()
        .take_modified_point3d_ids()
}
```

At the start of
`sync_after_reconstruction_rollback`, call `clear_modified_point3d_ids()` before rebuilding so
rolled-back indices cannot survive. Extend
`rollback_sync_preserves_retriangulation_trials_and_refreshes_stats` with:

```rust
assert!(tri_state
    .observation_manager()
    .modified_point3d_ids()
    .is_empty());
```

- [ ] **Step 7: Run frontier, rollback, and telemetry tests**

```bash
cargo test -p rustsfm --lib take_modified_points
cargo test -p rustsfm --lib rollback_sync
cargo test -p rustsfm --lib complete_and_merge_record_frontier_work
cargo test -p rustsfm --lib observation_manager::tests
```

Expected: PASS. The log contains finite non-negative timing values, and rollback leaves an empty
dirty set.

- [ ] **Step 8: Commit Task 3**

```bash
git add RustSFM/src/sfm/observation_manager.rs RustSFM/src/sfm/incremental_triangulator.rs
git commit -m "perf(rustsfm): bound sparse maintenance frontiers"
```

### Task 4: Shared Full and Subset Reprojection Filtering

**Files:**
- Modify: `RustSFM/src/sfm/mapper.rs:8681-8830`
- Test: `RustSFM/src/sfm/mapper.rs:18650-19120`

- [ ] **Step 1: Write the failing subset-filter test**

Create three two-view points: selected-invalid, unselected-invalid, and valid. The selected
deletion swaps the valid tail into index 0; the unselected invalid point must remain:

```rust
#[test]
fn subset_track_filter_leaves_unselected_invalid_points_for_full_boundary() {
    let mut frames = (0..6)
        .map(|id| minimal_frame(id, &format!("{id}.jpg")))
        .collect::<Vec<_>>();
    for frame in &mut frames {
        frame.keypoints[0] = rustslam::KeyPoint::new(50.0, 50.0);
    }
    let mut reconstruction = test_reconstruction(&frames);
    reconstruction.poses.fill(Some(SE3::identity()));
    let tracks = [
        vec![
            TrackObservation { image: 0, feature: 0 },
            TrackObservation { image: 1, feature: 0 },
        ],
        vec![
            TrackObservation { image: 2, feature: 0 },
            TrackObservation { image: 3, feature: 0 },
        ],
        vec![
            TrackObservation { image: 4, feature: 0 },
            TrackObservation { image: 5, feature: 0 },
        ],
    ];
    add_test_point3d(&mut reconstruction, 11, tracks[0].clone());
    add_test_point3d(&mut reconstruction, 22, tracks[1].clone());
    add_test_point3d(&mut reconstruction, 33, tracks[2].clone());
    reconstruction.points[0].xyz = [0.0, 0.0, -1.0];
    reconstruction.points[1].xyz = [0.0, 0.0, -1.0];
    reconstruction.points[2].xyz = [0.0, 0.0, 2.0];
    let mut state = IncrementalTriangulatorState::new(&frames, &[], &reconstruction);

    let removed = filter_reprojection_tracks_subset_with_state(
        &frames,
        &[],
        &mut reconstruction,
        &MapperConfig::default(),
        &mut state,
        &HashSet::from([0]),
    );

    assert_eq!(removed, 2);
    assert_eq!(reconstruction.point_ids, vec![33, 22]);
    assert!(reconstruction.points[1].xyz[2] < 0.0);
    assert_eq!(filter_reprojection_tracks_with_state(
        &frames,
        &[],
        &mut reconstruction,
        &MapperConfig::default(),
        &mut state,
    ), 2);
    assert_eq!(reconstruction.point_ids, vec![33]);
}
```

- [ ] **Step 2: Run the subset test and verify RED**

```bash
cargo test -p rustsfm --lib subset_track_filter_leaves_unselected_invalid_points_for_full_boundary
```

Expected: FAIL to compile because subset filtering does not exist.

- [ ] **Step 3: Extract one-point filtering and add two traversals**

Add these private types:

```rust
#[derive(Debug, Clone, Copy, Default)]
struct TrackFilterReport {
    removed_observations: usize,
    examined_points: usize,
    examined_observations: usize,
}

#[derive(Debug, Clone, Copy)]
enum TrackFilterScope<'a> {
    Full,
    Subset(&'a HashSet<usize>),
}
```

Move the body that currently filters one `point_id` into:

```rust
fn filter_reprojection_point(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    config: &MapperConfig,
    observation_manager: &mut ObservationManager,
    image_cameras: &[CameraModel],
    point_id: usize,
    max_error: f32,
    min_tri_angle: f32,
    min_track_length: usize,
) -> (usize, usize, bool) {
    if point_id >= reconstruction.points.len() {
        return (0, 0, false);
    }
    let point_xyz = reconstruction.points[point_id].xyz;
    let track = reconstruction.points[point_id].track.clone();
    let examined = track.len();
    let observations_to_delete = track
        .iter()
        .filter(|obs| {
            let Some(pose) = reconstruction.poses.get(obs.image).copied().flatten() else {
                return true;
            };
            let Some(kp) = frames
                .get(obs.image)
                .and_then(|frame| frame.keypoints.get(obs.feature))
            else {
                return true;
            };
            if !point_has_positive_depth(point_xyz, pose) {
                return true;
            }
            let Some(camera) = image_cameras.get(obs.image).copied() else {
                return true;
            };
            if camera_has_bogus_params(camera, config) {
                return true;
            }
            let err = crate::geometry::reprojection_error_px(
                point_xyz,
                pose,
                [kp.x(), kp.y()],
                camera,
            );
            !err.is_finite() || err > max_error
        })
        .cloned()
        .collect::<Vec<_>>();

    if observations_to_delete.len() >= track.len().saturating_sub(1) {
        let removed = track.len();
        observation_manager.delete_point3d(frames, pairs, reconstruction, point_id);
        return (removed, examined, true);
    }
    let mut removed = 0;
    for obs in observations_to_delete {
        if observation_manager.delete_observation(
            frames,
            pairs,
            reconstruction,
            obs.image,
            obs.feature,
        ) {
            removed += 1;
        }
    }
    if point_id >= reconstruction.points.len() {
        return (removed, examined, true);
    }
    let track = reconstruction.points[point_id].track.clone();
    if track.len() < min_track_length
        || !track_has_min_triangulation_angle(
            reconstruction.points[point_id].xyz,
            &track,
            reconstruction,
            min_tri_angle,
        )
    {
        removed += track.len();
        observation_manager.delete_point3d(frames, pairs, reconstruction, point_id);
        return (removed, examined, true);
    }
    if let Some(error) = mean_track_reprojection_error(
        reconstruction.points[point_id].xyz,
        &track,
        frames,
        reconstruction,
    ) {
        reconstruction.points[point_id].error = error;
        (removed, examined, false)
    } else {
        removed += track.len();
        observation_manager.delete_point3d(frames, pairs, reconstruction, point_id);
        (removed, examined, true)
    }
}
```

Create `filter_reprojection_tracks_with_policy_in_scope` around that helper with this signature:

```rust
fn filter_reprojection_tracks_with_policy_in_scope(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    config: &MapperConfig,
    observation_manager: Option<&mut ObservationManager>,
    scope: TrackFilterScope<'_>,
    max_error: f32,
    min_tri_angle: f32,
    min_track_length: usize,
) -> usize
```

Use this traversal body:

```rust
let started = Instant::now();
let image_cameras = (0..reconstruction.poses.len())
    .map(|image| reconstruction.camera_for_image(image))
    .collect::<Vec<_>>();
let mut temporary_manager;
let observation_manager = if let Some(manager) = observation_manager {
    manager
} else {
    temporary_manager = ObservationManager::new(frames, pairs, reconstruction);
    &mut temporary_manager
};
let mut report = TrackFilterReport::default();
match scope {
    TrackFilterScope::Full => {
        let mut point_id = 0usize;
        while point_id < reconstruction.points.len() {
            let (removed, examined, deleted) = filter_reprojection_point(
                frames,
                pairs,
                reconstruction,
                config,
                observation_manager,
                &image_cameras,
                point_id,
                max_error,
                min_tri_angle,
                min_track_length,
            );
            report.removed_observations += removed;
            report.examined_points += 1;
            report.examined_observations += examined;
            if !deleted {
                point_id += 1;
            }
        }
    }
    TrackFilterScope::Subset(point_ids) => {
        let mut point_ids = point_ids
            .iter()
            .copied()
            .filter(|&point_id| point_id < reconstruction.points.len())
            .collect::<Vec<_>>();
        point_ids.sort_unstable_by(|left, right| right.cmp(left));
        for point_id in point_ids {
            if point_id >= reconstruction.points.len() {
                continue;
            }
            let (removed, examined, _) = filter_reprojection_point(
                frames,
                pairs,
                reconstruction,
                config,
                observation_manager,
                &image_cameras,
                point_id,
                max_error,
                min_tri_angle,
                min_track_length,
            );
            report.removed_observations += removed;
            report.examined_points += 1;
            report.examined_observations += examined;
        }
    }
}
observation_manager.record_filter(
    matches!(scope, TrackFilterScope::Subset(_)),
    report.examined_points,
    report.examined_observations,
    report.removed_observations,
    started.elapsed().as_secs_f64() * 1_000.0,
);
report.removed_observations
```

Keep `filter_reprojection_tracks_with_policy` as a full-scope wrapper so existing tests and
callers retain their signature. Implement the subset state wrapper as:

```rust
fn filter_reprojection_tracks_subset_with_state(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    config: &MapperConfig,
    triangulation_state: &mut IncrementalTriangulatorState,
    point_ids: &HashSet<usize>,
) -> usize {
    filter_reprojection_tracks_with_policy_in_scope(
        frames,
        pairs,
        reconstruction,
        config,
        Some(triangulation_state.observation_manager_mut()),
        TrackFilterScope::Subset(point_ids),
        track_filter_max_error_px(config),
        track_filter_min_tri_angle_deg(config),
        track_filter_min_track_length(),
    )
}
```

- [ ] **Step 4: Run full and subset filter tests**

```bash
cargo test -p rustsfm --lib subset_track_filter
cargo test -p rustsfm --lib track_filter
cargo test -p rustsfm --lib stateful_track_filter_updates_candidate_visibility_stats
```

Expected: PASS. Full and subset traversals use identical geometry rules, and state statistics
match a fresh rebuild.

- [ ] **Step 5: Commit Task 4**

```bash
git add RustSFM/src/sfm/mapper.rs RustSFM/src/sfm/observation_manager.rs
git commit -m "perf(rustsfm): filter only affected sparse points"
```

### Task 5: Integrate Bounded Maintenance into the Mapper

**Files:**
- Modify: `RustSFM/src/sfm/mapper.rs:2440-2450`
- Modify: `RustSFM/src/sfm/mapper.rs:2650-2680`
- Modify: `RustSFM/src/sfm/mapper.rs:4110-4120`
- Modify: `RustSFM/src/sfm/mapper.rs:4310-4325`
- Modify: `RustSFM/src/sfm/mapper.rs:5030-5108`
- Modify: `RustSFM/src/sfm/mapper.rs:8681-8720`
- Test: `RustSFM/src/sfm/mapper.rs:18650-20190`

- [ ] **Step 1: Write the failing dirty-filter lifecycle test**

```rust
#[test]
fn modified_track_filter_consumes_frontier_without_full_scan() {
    let mut frames = (0..4)
        .map(|id| minimal_frame(id, &format!("{id}.jpg")))
        .collect::<Vec<_>>();
    for frame in &mut frames {
        frame.keypoints[0] = rustslam::KeyPoint::new(50.0, 50.0);
    }
    let mut reconstruction = test_reconstruction(&frames);
    reconstruction.poses.fill(Some(SE3::identity()));
    add_test_point3d(
        &mut reconstruction,
        11,
        vec![
            TrackObservation { image: 0, feature: 0 },
            TrackObservation { image: 1, feature: 0 },
        ],
    );
    add_test_point3d(
        &mut reconstruction,
        22,
        vec![
            TrackObservation { image: 2, feature: 0 },
            TrackObservation { image: 3, feature: 0 },
        ],
    );
    reconstruction.points[0].xyz = [0.0, 0.0, -1.0];
    reconstruction.points[1].xyz = [0.0, 0.0, -1.0];
    let mut state = IncrementalTriangulatorState::new(&frames, &[], &reconstruction);
    state.observation_manager_mut().mark_point3d_modified(0);

    assert_eq!(filter_modified_reprojection_tracks_with_state(
        &frames,
        &[],
        &mut reconstruction,
        &MapperConfig::default(),
        &mut state,
    ), 2);

    assert_eq!(reconstruction.point_ids, vec![22]);
    assert!(state
        .observation_manager()
        .modified_point3d_ids()
        .is_empty());
    let log = state.observation_manager().sparse_maintenance_log();
    assert!(log.contains("full_filter_calls=0"), "{log}");
    assert!(log.contains("subset_filter_calls=1"), "{log}");
    assert!(log.contains("frontier_cycles=1"), "{log}");
}
```

- [ ] **Step 2: Run the lifecycle test and verify RED**

```bash
cargo test -p rustsfm --lib modified_track_filter_consumes_frontier_without_full_scan
```

Expected: FAIL because there is no helper that filters and consumes the dirty frontier.

- [ ] **Step 3: Add full-boundary and dirty-frontier wrappers**

Change `filter_reprojection_tracks_with_state` to run full policy and consume the manager's
modified set:

```rust
let removed = filter_reprojection_tracks_with_policy(
    frames,
    pairs,
    reconstruction,
    config,
    Some(triangulation_state.observation_manager_mut()),
    max_error,
    min_tri_angle,
    min_track_length,
);
let _ = triangulation_state
    .observation_manager_mut()
    .take_modified_point3d_ids();
removed
```

Add:

```rust
fn filter_modified_reprojection_tracks_with_state(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    config: &MapperConfig,
    triangulation_state: &mut IncrementalTriangulatorState,
) -> usize {
    let point_ids = triangulation_state
        .observation_manager()
        .modified_point3d_ids()
        .clone();
    let removed = filter_reprojection_tracks_subset_with_state(
        frames,
        pairs,
        reconstruction,
        config,
        triangulation_state,
        &point_ids,
    );
    let _ = triangulation_state
        .observation_manager_mut()
        .take_modified_point3d_ids();
    removed
}
```

The full wrapper performs the same final `take_modified_point3d_ids()` after full filtering.

- [ ] **Step 4: Replace registration-time full filtering**

Keep the complete/merge/retriangulate order at `mapper.rs:2650-2672`. Replace only the call at
`mapper.rs:2674` with:

```rust
filter_modified_reprojection_tracks_with_state(
    frames,
    pairs,
    &mut reconstruction,
    config,
    &mut triangulation_state,
);
```

The manager's dirty set remains live through all three triangulator stages, so point swaps remap
the actual frontier before subset filtering consumes it.

- [ ] **Step 5: Replace local-BA full filtering and remove duplicate stable-ID scans**

After resolving `post_ba_stable_point_ids` once, mark those indices before merging:

```rust
let post_ba_point_ids =
    point_indices_for_stable_point_ids(reconstruction, &post_ba_stable_point_ids);
for &point_id in &post_ba_point_ids {
    triangulation_state
        .observation_manager_mut()
        .mark_point3d_modified(point_id);
}
let (merged_observations, completed_observations, completed_image_observations) = {
    let mut triangulator =
        IncrementalTriangulator::new(frames, pairs, reconstruction, triangulation_state);
    let modified = triangulator.get_modified_points3d().clone();
    let merged = triangulator.merge_tracks(tri_options, &modified);
    let modified = triangulator.get_modified_points3d().clone();
    let completed = triangulator.complete_tracks(tri_options, &modified);
    let complete_report = triangulator.complete_image(tri_options, registered_image);
    (merged, completed, complete_report.total_observations())
};
let filtered_observations = filter_modified_reprojection_tracks_with_state(
    frames,
    pairs,
    reconstruction,
    config,
    triangulation_state,
);
```

Delete the second `point_indices_for_stable_point_ids` scan and the cumulative
`modified_after_merge` clone. Preserve `LocalBundleReport` and change-ratio accounting.

- [ ] **Step 6: Preserve full filters at global boundaries and emit telemetry**

Keep calls in initial/global BA paths using `filter_reprojection_tracks_with_state`; its updated
wrapper consumes the boundary frontier after full traversal. Before the existing registration
telemetry line at `mapper.rs:2846`, add:

```rust
debug_log.push(
    triangulation_state
        .observation_manager()
        .sparse_maintenance_log(),
);
```

- [ ] **Step 7: Run mapper maintenance tests**

```bash
cargo test -p rustsfm --lib modified_track_filter_consumes_frontier_without_full_scan
cargo test -p rustsfm --lib local_ba_refinement_change_ratio
cargo test -p rustsfm --lib registration_rollback
cargo test -p rustsfm --lib track_filter
cargo test -p rustsfm --lib incremental_registration_telemetry_reports_hot_path_stages
```

Expected: PASS, with full-filter counts unchanged at global boundaries and subset-filter calls in
registration/local maintenance.

- [ ] **Step 8: Commit Task 5**

```bash
git add RustSFM/src/sfm/mapper.rs
git commit -m "perf(rustsfm): bound incremental sparse maintenance"
```

### Task 6: Full Static and Regression Verification

**Files:**
- Modify only if a verification failure identifies a regression in the three implementation files.

- [ ] **Step 1: Format the implementation**

```bash
cargo fmt --all
```

- [ ] **Step 2: Run focused module suites**

```bash
cargo test -p rustsfm --lib observation_manager::tests
cargo test -p rustsfm --lib incremental_triangulator::tests
cargo test -p rustsfm --lib track_filter
cargo test -p rustsfm --lib modified_track_filter
```

Expected: every command exits zero.

- [ ] **Step 3: Run the non-stale full library suite and builds**

```bash
cargo test -p rustsfm --lib -- --skip real_colmap_sparse
cargo check -p rustsfm
cargo check -p rustgs
cargo build -p rustsfm --release
cargo fmt --all -- --check
git diff --check
```

Expected: all commands exit zero. The `real_colmap_sparse` exclusion remains necessary because the
local fixture no longer contains the documented 24-image/256-point model; no other test may be
excluded.

- [ ] **Step 4: Review the implementation against the design invariants**

Check the diff and confirm:

```text
points.len() == point_ids.len()
all observation point indices are in range
all track observations point back to their owning index
rollback clears abandoned dirty indices
local maintenance calls subset filtering
initial/global/final boundaries call full filtering
no CPU parallelism, threshold, BA, RANSAC, or output-format change
```

- [ ] **Step 5: Commit formatting or verification-only corrections**

If formatting changed tracked code, commit it separately:

```bash
git add RustSFM/src/sfm/observation_manager.rs RustSFM/src/sfm/incremental_triangulator.rs RustSFM/src/sfm/mapper.rs
git commit -m "test(rustsfm): verify sparse maintenance invariants"
```

If there is no diff, do not create an empty commit.

### Task 7: Staged flowers2 and RustGS Acceptance

**Files:**
- Create: `docs/superpowers/plans/2026-07-20-rustsfm-sparse-maintenance-performance-benchmark.md`
- Read: `/Users/tfjiang/Projects/RustScan/test_data/flowers2/images`
- Read: `/Users/tfjiang/Projects/RustScan/test_data/flowers2/rustsfm_full_work/database.db`
- Create output outside git: `/tmp/rustsfm-sparse-maintenance-20260720`

- [ ] **Step 1: Record the build and baseline**

Create the benchmark document with commit hash, hardware, fixed seed, database path, and the
interrupted baseline:

```text
all960 baseline: terminated at 19,509.85 s, no sparse export
profile: filter 62%, complete 30%, merge 8%
observation slots: 6,612,183
largest earlier model: 900 images, 420,294 points
```

- [ ] **Step 2: Run the 96-image preflight**

```bash
mkdir -p /tmp/rustsfm-sparse-maintenance-20260720/optimized96
/usr/bin/time -p target/release/rustsfm reconstruct \
  --input /Users/tfjiang/Projects/RustScan/test_data/flowers2/images \
  --output /tmp/rustsfm-sparse-maintenance-20260720/optimized96 \
  --database /Users/tfjiang/Projects/RustScan/test_data/flowers2/rustsfm_full_work/database.db \
  --max-images 96 \
  --random-seed 0 \
  --threads 1 \
  --log-level debug \
  --summary-json /tmp/rustsfm-sparse-maintenance-20260720/optimized96/summary.json \
  > /tmp/rustsfm-sparse-maintenance-20260720/optimized96/run.log 2>&1
```

Require `registered_images == 96`, finite output, `sparse/0`, and one `sparse_maintenance` line whose
subset calls are nonzero and frontier peak is well below total reconstruction points.

- [ ] **Step 3: Run the 96-image RustGS probe**

```bash
cargo run -p rustgs --release --bin rustgs -- train \
  --input /tmp/rustsfm-sparse-maintenance-20260720/optimized96 \
  --image-root /Users/tfjiang/Projects/RustScan/test_data/flowers2/images \
  --output /tmp/rustsfm-sparse-maintenance-20260720/rustgs96.ply \
  --iterations 1 \
  --max-frames 1 \
  --max-initial-gaussians 1000
```

Expected: RustGS resolves `sparse/0`, reports zero missing registered images, runs one Metal
iteration, and writes a non-empty PLY.

- [ ] **Step 4: Run and gate the 200-image benchmark**

```bash
mkdir -p /tmp/rustsfm-sparse-maintenance-20260720/optimized200
/usr/bin/time -p target/release/rustsfm reconstruct \
  --input /Users/tfjiang/Projects/RustScan/test_data/flowers2/images \
  --output /tmp/rustsfm-sparse-maintenance-20260720/optimized200 \
  --database /Users/tfjiang/Projects/RustScan/test_data/flowers2/rustsfm_full_work/database.db \
  --max-images 200 \
  --random-seed 0 \
  --threads 1 \
  --log-level debug \
  --summary-json /tmp/rustsfm-sparse-maintenance-20260720/optimized200/summary.json \
  > /tmp/rustsfm-sparse-maintenance-20260720/optimized200/run.log 2>&1
cargo run -p rustgs --release --bin rustgs -- train \
  --input /tmp/rustsfm-sparse-maintenance-20260720/optimized200 \
  --image-root /Users/tfjiang/Projects/RustScan/test_data/flowers2/images \
  --output /tmp/rustsfm-sparse-maintenance-20260720/rustgs200.ply \
  --iterations 1 \
  --max-frames 1 \
  --max-initial-gaussians 1000
```

Require 200/200 finite poses, zero RustGS missing images, and materially lower
`filter_ms + complete_ms + merge_ms` per registered image than the interrupted full-run profile
indicates. Record wall time and compare with the previous 200-image result of 362.31 seconds and
91,894 points; investigate any image or major point coverage regression before proceeding.

- [ ] **Step 5: Run all 960 images without sampling**

```bash
mkdir -p /tmp/rustsfm-sparse-maintenance-20260720/optimized960
/usr/bin/time -p target/release/rustsfm reconstruct \
  --input /Users/tfjiang/Projects/RustScan/test_data/flowers2/images \
  --output /tmp/rustsfm-sparse-maintenance-20260720/optimized960 \
  --database /Users/tfjiang/Projects/RustScan/test_data/flowers2/rustsfm_full_work/database.db \
  --random-seed 0 \
  --threads 1 \
  --log-level debug \
  --summary-json /tmp/rustsfm-sparse-maintenance-20260720/optimized960/summary.json \
  > /tmp/rustsfm-sparse-maintenance-20260720/optimized960/run.log 2>&1
```

Do not stop while CPU time, RSS, or log timestamps are progressing. Require the summary and
exported model to contain finite poses for 960/960 images.

- [ ] **Step 6: Run the final RustGS compatibility probe**

```bash
cargo run -p rustgs --release --bin rustgs -- train \
  --input /tmp/rustsfm-sparse-maintenance-20260720/optimized960 \
  --image-root /Users/tfjiang/Projects/RustScan/test_data/flowers2/images \
  --output /tmp/rustsfm-sparse-maintenance-20260720/rustgs960.ply \
  --iterations 1 \
  --max-frames 1 \
  --max-initial-gaussians 1000
```

Expected: RustGS reports 960 resolved images, zero missing images, a non-empty initialization
cloud, one successful training iteration, and a non-empty PLY.

- [ ] **Step 7: Record evidence and commit the benchmark report**

The report must contain exact commands and these measured fields for every stage:

```text
wall seconds
input and registered images
point count
complete/merge/full-filter/subset-filter calls and milliseconds
point deletions and rewritten track observations
frontier peak and consumed cycles
finite-pose result
RustGS resolved/missing image result
```

Then run:

```bash
cargo fmt --all -- --check
git diff --check
git status --short
git add docs/superpowers/plans/2026-07-20-rustsfm-sparse-maintenance-performance-benchmark.md
git commit -m "docs(rustsfm): record sparse maintenance benchmark"
```

Commit only the benchmark document. `/tmp` models, logs, summaries, and PLY files remain outside
the repository.

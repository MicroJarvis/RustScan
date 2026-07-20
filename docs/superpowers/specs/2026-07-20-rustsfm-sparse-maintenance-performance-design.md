# RustSFM Sparse Maintenance Performance Design

**Date:** 2026-07-20

**Goal:** Remove the measured superlinear point-maintenance cost from the incremental mapper
without adding CPU parallelism, weakening geometric checks, reducing image coverage, or changing
the COLMAP-compatible output consumed by RustGS.

## Evidence

The interrupted `flowers2` run was launched with all 960 input images and ran for 19,509.85
seconds (5 hours 25 minutes) before receiving `SIGINT`. It remained at one fully utilized CPU
core with about 5 GiB resident memory and had not exported a sparse model, so it provides no
claim about final registered-image coverage.

A three-second stack sample attributed approximately 62 percent of samples to
`filter_reprojection_tracks_with_policy`, 30 percent to `complete_tracks`, and 8 percent to
`merge_tracks`. Most filtering samples were below `ObservationManager::delete_point3d_internal`.
The reconstruction contains 6,612,183 observation slots and has reached roughly 420,000 points.

Each current point deletion calls `Vec::remove`, scans every observation slot to clear or shift
point indices, and rebuilds the complete modified-point set. This makes deletion proportional to
the full reconstruction rather than to the deleted and moved tracks. In addition,
`modified_point3d_ids` is never cleared by a production caller, and full-model reprojection
filtering runs after every accepted registration and every local BA refinement. The baseline
performed 1,978 local refinements but removed only 4,177 observations, about 2.1 removals per
refinement.

## Scope

This change includes:

1. O(track length) point deletion using `swap_remove`;
2. exact remapping of moved observations, dirty point IDs, and merge-trial cache entries;
3. a bounded, consumable dirty frontier for registration maintenance;
4. subset reprojection filtering for registration and local-BA maintenance;
5. full-model filtering only at initial, global-BA, and final reconstruction boundaries;
6. complete, merge, filter, and delete timing plus frontier-size telemetry;
7. deterministic unit tests and staged 96-, 200-, and 960-image `flowers2` validation.

This change does not alter feature extraction, pair matching, PnP, structureless PoseLib solvers,
RANSAC budgets, BA residuals or solver policy, triangulation thresholds, output layout, or RustGS
loading behavior. It does not add CPU parallelism or wgpu work. Structureless RANSAC budgeting is
a separate follow-up after this maintenance change is measured.

## Alternatives

### A. Stable arena slots with tombstones

Stable internal point handles would eliminate index remapping and provide the highest long-term
ceiling. It would also require changing nearly every point loop, BA input builder, serializer,
rollback path, and track invariant. Tombstones would retain memory and make dense solver inputs
require a separate compaction layer. This is too broad for the measured bottleneck.

### B. Batch compaction after each filter pass

A filter could first mark all rejected points and then compact points and observations once in
linear time. This is attractive for full-model filtering, but it does not fix deletions caused by
track merging or two-view observation removal. It also preserves the repeated full-model passes
that dominate the current local-refinement schedule.

### C. Swap deletion plus bounded maintenance frontiers

This is the selected approach. `swap_remove` limits a deletion to the deleted track and, when
needed, the one point moved from the vector tail. A per-maintenance-cycle dirty frontier and
subset filtering remove repeated work on historical points. Existing dense vectors, stable
external point IDs, BA interfaces, and serialization remain intact.

## Design

### Swap-Based Point Deletion

`ObservationManager::delete_point3d_internal` will repair the external ID table as today and
then remove matching entries from `Reconstruction::points` and `point_ids` with `swap_remove`.
The removed point's track will clear only observations that still refer to the removed internal
index. This conditional clear is required by merge: the observations from the removed track may
already refer to the retained merged point.

When the deleted point was not the vector tail, the former tail point moves to the deleted
index. Only observations in that moved point's track will be rewritten from the old tail index
to the new index. Its external `u64` point ID moves with it, so surviving point identity is stable
and exported IDs are unchanged.

The dirty set will apply the same single-index remap: remove the deleted index and replace the
old tail index with the deleted index when the tail point moved. No other point index changes.

### Merge Cache Remapping

`IncrementalTriangulatorState::merge_trials` stores internal point-index pairs. After a successful
merge, entries touching the retained or removed point remain invalidated because the retained
track and geometry changed. Entries touching the former tail point are remapped to the removed
index when that point moved. All other cache entries remain unchanged.

The merge path will capture the vector-tail index before deletion and pass the resulting
single-index remap to the cache synchronization helper. `retriangulation_trials` needs no point
remap because it is keyed by image pairs.

Track merge and subset-filter worklists will continue processing internal indices in descending
order. Under `swap_remove`, deleting the current entry cannot invalidate any lower index still in
the worklist. A selected tail point has already been processed; an unselected tail point does not
belong to the current subset.

### Dirty Frontier Lifecycle

The observation manager will expose a consuming operation for modified point indices in addition
to read-only inspection. A registration maintenance cycle keeps the frontier live while it:

1. triangulates the newly registered unit;
2. completes tracks from the current modified set;
3. merges tracks from the updated modified set;
4. retriangulates under-reconstructed pairs;
5. subset-filters the final remapped modified set.

The frontier is consumed or cleared after that cycle, rather than accumulating for the full
mapper attempt. Mutations caused by completion, merging, retriangulation, and deletion continue
to mark points, so the set always expands to cover the current cycle before it is consumed.

Local BA does not depend on the ambient frontier to identify optimized points. It starts from the
bundle's existing stable external point IDs, resolves their current indices after merging, unions
points modified by merge and completion, subset-filters that union, and then clears the cycle.

Rollback discards the dirty frontier from the abandoned reconstruction before rebuilding
observation statistics. Global BA postprocessing runs complete-all, merge-all, and full filtering,
then clears the frontier at the boundary.

### Scoped Reprojection Filtering

The existing full filtering policy remains the single source of geometric acceptance thresholds.
Its point evaluation will be shared by two traversal modes:

- full traversal for initial validation, global BA postprocessing, and final export cleanup;
- descending subset traversal for registration and local BA maintenance.

Both modes apply the same positive-depth, finite reprojection, maximum-error, minimum-track-length,
minimum-triangulation-angle, and mean-error rules. Subset filtering changes only which points are
examined, not whether an examined observation or track is accepted.

The full traversal continues at the same internal index after a deletion so that the point moved
from the tail is also examined. The subset traversal uses descending indices, which makes the
single swap remap safe without a global observation scan or a persistent index map.

### Telemetry

Session-scoped sparse-maintenance telemetry will report:

- complete and merge call counts, input frontier sizes, changed-observation counts, and elapsed
  milliseconds;
- full and subset filter call counts, examined point/observation counts, removed observations,
  and elapsed milliseconds;
- point deletion count, moved-point count, rewritten track-observation count, and deletion
  milliseconds;
- dirty frontier peak size and the number of consumed maintenance cycles.

The mapper will append one aggregate `sparse_maintenance` diagnostic line next to the existing
`incremental_registration` telemetry. Counters must use the normal mutation paths and must not
scan or clone the full reconstruction solely for measurement.

## Correctness Invariants

- Every non-`None` reconstruction observation refers to an existing point whose track contains
  that image-feature pair.
- Every track observation refers back to the point's current internal index.
- `points.len()` equals `point_ids.len()`, and an external point ID remains attached to the same
  surviving 3D point across swaps.
- Dirty point indices and merge-trial pairs contain no removed or out-of-range indices.
- Subset filtering applies exactly the same geometric policy as full filtering.
- Registration rollback cannot retain dirty indices created only by the abandoned state.
- A fixed random seed remains deterministic within the optimized implementation.
- All accepted poses and 3D coordinates are finite.
- COLMAP directory structure, camera/image records, and RustGS-readable output are unchanged.

Point-vector ordering is explicitly not an external contract. It may differ from the shifting
deletion implementation, while external IDs and reconstruction quality remain protected.

## Testing

Tests will be written and observed failing before production changes. Focused coverage will prove:

- deleting a middle point moves only the former tail point, preserves its external ID, rewrites
  its track references, and clears the deleted track references;
- deleting the tail does not rewrite unrelated observations;
- dirty IDs apply the same swap remap, including the case where the moved point is dirty;
- merging with a third tail point preserves merged and moved tracks and remaps merge-trial cache
  entries without stale indices;
- incremental observation and image-pair statistics still match a fresh manager rebuild;
- consuming or clearing a frontier prevents historical points from being reprocessed;
- subset filtering removes invalid affected points while leaving an invalid unselected point for
  a later full boundary pass;
- full filtering still examines a swapped-in tail point and removes it when invalid;
- rollback clears abandoned dirty state;
- telemetry reports the selected traversal, frontier size, deletion work, and elapsed stages.

Verification will run formatting, the focused observation/triangulator/mapper tests, the complete
RustSFM library suite with the documented stale fixture tests excluded, and `cargo check -p
rustgs`.

## flowers2 Acceptance

Performance validation uses the existing database and original-resolution images from
`/Users/tfjiang/Projects/RustScan/test_data/flowers2`; it does not sample the final dataset.
Fixed-seed runs proceed only when the previous stage passes:

1. 96 contiguous images for rapid correctness and telemetry checks;
2. 200 contiguous images for runtime and reconstruction-quality comparison;
3. all 960 images for the final RustGS training input.

Each stage must register every image supplied to that stage, export a valid COLMAP sparse model,
contain only finite poses and points, and pass the RustGS loader probe. The final model's image
records must contain finite poses for all 960 input images, and the RustGS loader must report
960 resolved images with zero missing images. The output must provide the `sparse/0` plus image
layout used by RustGS training. Benchmark notes will record wall time, registered images, points,
complete/merge/filter/delete timings, peak dirty frontier, and RustGS probe result.

## Deferred Work

After this change is validated, the next independent optimization is structureless registration:
a separate RANSAC budget, staged hypothesis trials, correspondence deduplication and spatial
coverage, precomputed rig rays, and one final GR8P refinement. wgpu remains a candidate only for
large batched hypothesis scoring. It is not suitable for point deletion, hash-set mutation, graph
traversal, or the GR6P polynomial solver.

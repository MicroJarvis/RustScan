# RustSFM Incremental Registration Hot-Path Design

**Date:** 2026-07-19

**Goal:** Reduce incremental mapper runtime on large reconstructions without reducing the
number of images that can be registered or changing accepted camera poses and tracks.

## Evidence

The 960-image `flowers2` run issued 42,697 PnP solves in about 4.5 hours. Timings reported
inside the PnP solver totaled 57.3 seconds, less than 0.4 percent of elapsed time. A five-second
sample placed about 71 percent of samples under `IncrementalTriangulator::triangulate_image`.
The dominant stacks cloned the full `CorrespondenceGraph` from
`ObservationManager::add_observation`; the point-creation branch repeatedly rebuilt point-ID
sets and scanned all existing point IDs.

The current registration loop also resets every unregistered image's trial counter and then
eagerly runs absolute-pose estimation for all eligible unregistered images after each successful
registration. This creates quadratic candidate probing even though the next-image selection loop
can perform the same probes lazily.

## Scope

This change includes:

1. shared immutable ownership for the correspondence graph;
2. amortized O(1) stable point-ID allocation;
3. lazy candidate PnP retries with support-aware failure state;
4. counters and timings that separate candidate preparation, PnP, and triangulation work;
5. deterministic regression and bounded `flowers2` benchmark coverage.

This change does not alter P3P, RANSAC scoring, pose refinement, acceptance thresholds, local or
global BA policy, triangulation geometry, or output formats. It does not add CPU parallelism.

## Alternatives

### A. Optimize only the PnP solver or move more PnP work to wgpu

This is low value at the current profile. Even deleting all measured PnP solver time would save
less than one percent of the full run. Per-candidate GPU submission and readback can also cost
more than CPU scoring for the common 100-trial solve.

### B. Replace mapper state with stable arenas and a fully incremental priority graph

This offers the highest long-term ceiling, including O(track length) point deletion and exact
candidate invalidation. It is too broad for the first performance correction because it changes
point indexing, rollback, filtering, triangulation, and serialization together.

### C. Remove measured hot-path work while preserving mapper decisions

This is the selected approach. Shared graph ownership and monotonic IDs remove proven costs.
Lazy, support-aware retries avoid eager quadratic probing while retaining an exhaustive fallback
when normal selection stalls. The existing vector-based reconstruction and output contracts stay
unchanged.

## Design

### Shared Correspondence Graph

`ObservationManager` will store `Option<Arc<CorrespondenceGraph>>`. Graph construction and
installation remain explicit, and `correspondence_graph()` continues returning
`Option<&CorrespondenceGraph>` so callers do not depend on `Arc`.

Mutation paths that need both `&mut self` and a graph handle clone the `Arc`, not the graph. The
graph is immutable after installation, so this changes ownership cost without changing graph
contents or synchronization behavior. No lock is required.

### Stable Point-ID Allocator

`ObservationManager` will own the next external `u64` point ID. Construction initializes it to
one greater than the maximum imported ID. Allocation increments the counter and never reuses an
ID deleted during the current mapping session.

Malformed or legacy reconstructions whose `point_ids` vector is shorter than `points` retain a
cold repair path. Repair runs only while the lengths differ, fills missing IDs above the current
maximum, and advances the allocator. Normal `add_point3d` no longer builds a `HashSet` or scans
the full ID vector.

Allocator state is intentionally monotonic across registration rollback. A rollback may leave a
gap in exported point IDs, which is valid in COLMAP data and avoids duplicate IDs without an
expensive rescan.

### Lazy Registration Attempts

The mapper will stop calling the eager all-image
`mark_unregistered_images_with_no_absolute_pose` sweep after every successful registration.
`choose_next_registration` remains the only normal path that invokes PnP for candidate images.

Per registration unit and per mode, retry state records:

- failed attempts at the last observed support level;
- the visible 3D-point count for structure-based registration;
- the visible correspondence count for structureless registration.

A failed unit is retried when its applicable support count increases. An unchanged unit keeps its
failure count, preventing the same PnP problem from being solved again after an unrelated image
registers. Existing `max_reg_trials` semantics apply at each support level.

When normal selection has no successful candidate, the mapper performs one exhaustive fallback
epoch that permits unchanged units to retry. If fallback registers an image, normal support-aware
selection resumes because the map may now expose new points. If fallback registers nothing, the
current model terminates as before. This fallback protects bridge images whose PnP becomes viable
because BA improved geometry without increasing correspondence counts.

Rig siblings continue sharing one registration-unit state through the existing
`RegistrationUnitKey` helpers.

### Telemetry

The incremental mapper will report one aggregate diagnostic line containing at least:

- candidate units considered;
- candidates skipped because support was unchanged;
- structure-based and structureless pose attempts;
- exhaustive fallback epochs;
- time collecting candidate observations;
- time inside pose solving and refinement;
- time in per-registration triangulation and observation updates.

The existing `pnp_timing` line remains unchanged. New timings must not allocate or clone the
reconstruction merely to measure a stage.

## Correctness Invariants

- Candidate ordering among eligible units remains deterministic for a fixed random seed.
- PnP input observations, thresholds, random seeds, and acceptance tests remain unchanged.
- A candidate is never permanently suppressed solely because an earlier PnP attempt failed.
- The exhaustive fallback runs before declaring that no further image can be registered.
- Point IDs remain unique and stable for the lifetime of each surviving point.
- Correspondence statistics after add, merge, delete, register, and rollback match the current
  implementation.
- RustGS-readable COLMAP output structure remains unchanged.

## Testing

Unit tests will first fail against the current implementation and then cover:

- cloned correspondence-graph handles share one allocation;
- adding observations and points preserves all visibility and pair statistics;
- deleting the highest external point ID and adding a new point does not reuse the ID;
- legacy short `point_ids` tables are repaired once and remain unique;
- unchanged failed registration units are not retried during normal selection;
- increased structure-based or structureless support makes the unit eligible again;
- an exhaustive fallback can recover an unchanged bridge candidate;
- rig siblings share retry state and deterministic ordering.

Verification uses the existing RustSFM library and CLI suites. A fixed-seed 200-image benchmark
must preserve registered-image count, finite poses, and RustGS-readable output while reducing
pose-attempt count and wall time. The full 960-image run follows only after the bounded benchmark
passes.

## Deferred Work

After this change is measured, later work may add `swap_remove`-based point deletion, an
incremental candidate heap, ranked PROSAC sampling, prior-seeded robust pose refinement, or
cross-candidate wgpu scoring. Each requires a new profile showing that its target remains material.

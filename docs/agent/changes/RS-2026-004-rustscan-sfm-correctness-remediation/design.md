# RS-2026-004 Design

## 1. Preserve Image Identity Through Database Filtering

The mapper currently represents the same image set with several parallel
vectors (`paths`, `names`, decoded `frames`, image IDs, camera indices, and
frame indices). `database_frames` can remove an entry, after which the original
vector index no longer identifies the same image.

Introduce one resolved input record for each retained image. The exact name is
an implementation choice, but it must own or reference at least:

- the original source index;
- normalized image name and path;
- decoded/database-backed `ImageFrame`;
- database image ID;
- camera ID or resolved camera index;
- optional rig frame index;
- optional reference-image index.

Filtering produces a new ordered collection of these records. Every later
frame, camera, seed, target, support, and pair lookup must derive from that
collection. A target is located by its stable name or ID, never by
`names.len() - 1`. Missing requested targets and support images return errors
that name the missing image.

`ReferenceCameraSetup` may remain temporarily for compatibility, but its
construction must consume the retained image collection and validate that all
per-image arrays have exactly the retained frame count.

## 2. Define Reconstruction Invariants Once

Add a structured validation error and central validation API. Keep the checks
in a dedicated module if `core/types.rs` would otherwise grow further.

`validate_structure` must check:

- equal lengths for image names, paths, IDs, camera indices, frame indices,
  poses, keypoints, and observations;
- camera metadata lengths and camera-index bounds;
- uniqueness and legal range of camera, image, point, rig, and frame IDs;
- observation count equal to keypoint count for each image;
- every observation points to an existing sparse point;
- every track image and feature index is valid;
- each `(image, feature)` occurs in at most one point track;
- observation-to-track and track-to-observation agreement;
- rig, frame, sensor, and data-ID references are resolvable;
- finite camera parameters, poses, points, keypoints, and point errors.

`validate_for_colmap_export` calls the structural validator and additionally
requires every exported track to reference an exported registered image and a
valid feature. Sequence-specific minimum counts stay in the sequence layer but
must call the shared structural validator first.

Persistence code must use checked ID and camera accessors. Existing fallback
accessors may be retained only as deprecated compatibility helpers; export,
database, and reconstruction code must not call them.

## 3. Allocate IDs From Occupied Domains

ID allocation is based on the occupied IDs, not vector position. The allocator
collects existing IDs from the reference model and database, starts after the
maximum, uses checked arithmetic, respects the COLMAP database image-ID range,
and confirms uniqueness before returning the completed setup.

The same policy applies when an internal point needs a new persistent point ID.
ID zero and overflow are explicit errors where the corresponding COLMAP domain
does not permit them.

## 4. Keep Raw COLMAP Records At The Boundary

Text and binary readers may initially decode into raw `Colmap*` records with
arrays. Before creating a `Reconstruction`, validate the complete raw model:

- IDs are unique;
- dimensions and focal lengths are positive and finite;
- all camera parameters, poses, points, observations, and errors are finite;
- pose quaternions have a finite norm above an explicit epsilon;
- every image camera exists;
- every image-side point reference exists;
- every track image and feature exists;
- image-side and point-side references agree exactly;
- rig and frame metadata contain unique IDs and valid sensor/data references.

Small quaternion norm drift may be normalized after validation. A zero or
non-finite quaternion is rejected; it is never converted to identity. Error
messages include the record type, ID, referenced ID, feature index where
applicable, and source file when known.

## 5. Give CameraModel One Source Of Truth

COLMAP parameter storage is the canonical camera state. Focal length and
principal point are read through accessors derived from canonical parameters.
Mutations use checked methods such as `set_focal_lengths`,
`set_principal_point`, `set_param`, and `scale_focal`; each method updates one
representation atomically and validates the result.

If removing serialized `fx/fy/cx/cy` fields changes an existing first-party
format, use an explicit serde proxy to read the previous shape and reject
disagreement instead of maintaining two mutable in-memory sources of truth.

View-graph focal refinement must score genuinely different canonical cameras.
Tests use a synthetic scene whose optimum is neither 0.9 nor 1.0, then assert
that the selected parameters, accessors, projection, and COLMAP export agree.

## 6. Commit BA Results Only When Usable

Ceres already optimizes detached parameter storage. Map and inspect the solver
summary before mutating the caller's `Reconstruction`.

- A usable solution is written back once, followed by point-error refresh and
  optional covariance work.
- A failure, user failure, cancellation, or non-finite solution returns a
  report/error while leaving the caller's cameras, poses, points, and errors
  byte-for-byte equivalent to the pre-call state.
- `global_mapper` sets success only from `report.is_solution_usable()` and
  stops the refinement round when BA did not produce a usable update.
- Mapper-level camera plausibility rollback remains a second policy gate after
  solver usability; it must not compensate for the public BA API mutating on
  failure.

Use a deterministic test seam or extracted commit-decision function so failure
and cancellation behavior can be tested without relying on Ceres to fail
randomly.

## 7. Make Database Batches Atomic

Wrap each logical mutation in the existing `ColmapDatabase::with_transaction`:

- local matching database population;
- a batch of pair-geometry writes;
- complete database merge into the target.

The transaction owns all related rows. Tests inject a deterministic failure
after earlier writes and then assert that row counts, IDs, existing payloads,
and the database-entry-deleted bookkeeping match the pre-operation snapshot.

## 8. Finish CPU nalgebra Ownership

Replace the known hand-written `Mat3d` and `Vec3d` implementation with
`Matrix3<f64>` and `Vector3<f64>`. Migrate general CPU owning values such as
`Point3D.xyz` to `Point3<f32>` and internal rays/directions to the appropriate
`Vector*` type.

Raw `[w, x, y, z]`, `[x, y, z]`, flat matrices, and row-major arrays stay only
in COLMAP, database, FFI, GPU, or serialization adapters. Conversion helpers
must state component order, scalar type, shape, layout, and pose direction.

Do not add another pose implementation. Remove the CPU role of `Rigid3`; keep
raw COLMAP rigid records in the IO adapter and convert them to the shared
`rustscan-types` pose at the boundary. If precision requirements expose a gap
in the shared pose API, extend that API in `rustscan-types` rather than defining
a RustSFM-local replacement.

Extend the existing CPU-math CI check so it detects known hand-written math
owners and public/internal floating-point vector or matrix arrays outside an
explicit boundary allowlist. The allowlist entry must include a reason.

## 9. Integration Strategy

Use one commit per task below. Tasks with disjoint scopes may run in parallel
worktrees, but integration follows dependency order. After each cherry-pick,
run that task's focused tests before accepting the next commit. Resolve
behavioral conflicts in the owning task rather than weakening acceptance or
adding fallback behavior during integration.


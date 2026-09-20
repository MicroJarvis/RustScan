# RS-2026-004 RustScan SfM Correctness Remediation

## Goal

Make `rustscan-sfm` reject inconsistent inputs, preserve reconstruction state on
failed optimization, export only structurally valid COLMAP models, and comply
with the repository's CPU `nalgebra` ownership rules.

The change is complete when the default-feature sequence registration suite no
longer panics, every identified corruption path has a focused regression test,
and the declared CPU, native, GPU, cross-crate, lint, and export checks pass.

## Why This Is A Change Package

The work changes numerical code, mapper indexing, public data invariants,
COLMAP import/export behavior, Ceres write-back semantics, database transaction
boundaries, and CPU mathematical types. It therefore uses the change-package
workflow required by `docs/agent/task-protocol.md` instead of a single task
YAML.

## Scope

- Database-backed mapper input identity and filtering.
- `Reconstruction` structural and COLMAP-export validation.
- Collision-free camera, image, frame, rig, and point ID handling.
- Strict COLMAP numeric and bidirectional observation/track validation.
- `CameraModel` source-of-truth cleanup and view-graph focal refinement.
- Atomic bundle-adjustment state updates.
- Transactions for multi-row/multi-table COLMAP database operations.
- Remaining first-party RustSFM CPU point, vector, quaternion, pose, and matrix
  ownership migrations to `nalgebra`.
- Focused RustSFM and affected RustViewer tests, CI checks, and task evidence.

## Out Of Scope

- New SfM algorithms, reconstruction-quality tuning, or performance work that
  is not required to preserve behavior while fixing a finding.
- GPU shader algorithm changes, new GPU backends, or treating a skipped GPU
  test as passing evidence.
- Changes under `third_party/`.
- A permissive repair mode for corrupted COLMAP models. Strict rejection is
  sufficient for this change; repair can be proposed separately.
- Renaming the `rustsfm` compatibility binary or changing unrelated persisted
  formats.
- The pre-existing dirty changes in `README.md`, `ROADMAP.md`,
  `docs/current-project-status.md`, and `docs/index.md`.

## Compatibility Policy

- Valid COLMAP models and databases must continue to load and round-trip.
- Corrupted, ambiguous, duplicate-ID, or numerically invalid models may now
  return structured errors; accepting those models is not a compatibility
  requirement.
- Existing constructors should remain when they can enforce the new
  invariants. Direct mutation of duplicated camera fields may be replaced by
  checked accessors and mutation methods. If a public compatibility wrapper is
  retained, internal persistence and numerical paths must use strict APIs.
- COLMAP wire arrays remain boundary representations. They must be converted to
  `nalgebra` owning types before general CPU numerical work.

## Acceptance Summary

1. Database filtering cannot change image identity or leave parallel metadata
   arrays misaligned.
2. No public export path fabricates IDs or writes a structurally invalid model.
3. Strict COLMAP import rejects duplicate IDs, invalid numerics, missing
   references, and observation/track disagreement with actionable errors.
4. View-graph focal candidates change the actual projection parameters and the
   selected camera has one internally consistent representation.
5. An unusable BA result leaves cameras, poses, points, and point errors
   unchanged; global mapping does not report it as successful.
6. Multi-table database operations either commit completely or leave the
   database unchanged.
7. RustSFM general CPU math uses `nalgebra`; remaining arrays are documented
   boundaries with explicit conversions and layout/component order.
8. All verification commands in `tasks.yaml` have recorded results in
   `verification.md`.


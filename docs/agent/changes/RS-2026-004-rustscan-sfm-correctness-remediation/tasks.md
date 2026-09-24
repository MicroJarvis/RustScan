# RS-2026-004 Detailed Execution Plan

Unchecked boxes describe required work. A task is checked only after its code,
focused tests, commit, and verification evidence are recorded in
`verification.md`.

## Execution Waves

- **Wave 0:** T0 only.
- **Wave 1, parallel-capable:** T1, T2, T5, and T6 in separate worktrees.
- **Wave 2:** T4 after T2; T3 after T2 and T4.
- **Wave 3:** T7 after T1 through T6 are integrated.
- **Wave 4:** T8 integration, full verification, independent review, handoff.

Do not assign two Agents to the same task or let parallel tasks edit files
outside their declared scope without updating `tasks.yaml` first.

## T0 — Isolate Work And Freeze The Baseline

**Dependencies:** none  
**Primary scope:** this change package and test commands only

- [x] Create branch `agent/RS-2026-004/rustscan-sfm-correctness-remediation`
      and the worktree declared in `tasks.yaml` from the recorded base commit.
- [x] Confirm the implementation worktree does not contain the unrelated dirty
      documentation changes listed in `proposal.md`.
- [x] Re-run the baseline sequence, no-default-feature, COLMAP IO, view-graph,
      BA, and GPU-focused commands. Record exact command, commit, OS, native
      dependencies, adapter, result, duration, and skipped tests.
- [x] Add focused failing regression tests before implementation when the
      trigger is deterministic. A panic is not the expected post-fix error.
- [x] Record any baseline drift. Do not rewrite acceptance conditions to match
      new failures.

**Exit condition:** baseline evidence is reproducible and each later task has a
named owner, branch/worktree, and file scope.

## T1 — Replace Filtered Parallel Indices With Stable Image Identity

**Dependencies:** T0  
**Primary scope:**
`rustscan-sfm/src/sfm/mapper.rs`,
`rustscan-sfm/src/sfm/mapper/reconstruction_input.rs`, and mapper/sequence tests

- [x] Introduce the resolved retained-image record described in `design.md`.
- [x] Make database frame loading return retained identity and source mapping,
      rather than a frame-only vector whose index is ambiguous.
- [x] Build reference/database camera setup and seed reconstruction from the
      retained images in their final mapper order.
- [x] Locate single-registration target and support images by stable name/ID.
      Return a contextual missing/disconnected-target error.
- [x] Validate all per-image setup lengths before starting pair estimation or
      constructing a `Reconstruction`.
- [x] Add regressions for a dropped leading image, dropped middle image,
      disconnected target, filtered support image, multi-camera metadata, and
      reference seed alignment.
- [x] Re-run every sequence test that previously panicked and the complete
      `sequence_registration` integration test serially.
- [x] Review follow-up: when both a reference model and a database are present,
      retained images that are not in the reference keep their database image,
      camera, and frame identity, merged by stable ID. Database frame indexes
      are not copied into the reference frame list. A support that is in the
      database and the registered reference but absent from the match-connected
      cache returns an error that names the support.
- [x] Review P1: sequence registration must not silently drop an unconnected
      support. Controlled degradation requires a diagnostic note, at least one
      remaining match-connected support, and a contextual error otherwise.
- [x] Review P1: overlapping reference/database same-name images must agree on
      image_id, camera_id, and frame/rig identity when both sides provide them.

**Exit condition:** no retained frame can inherit metadata from a different
source path, and the ten mapper index panics become passing tests or intentional
contextual errors.

## T2 — Centralize Reconstruction Validation And ID Allocation

**Dependencies:** T0  
**Primary scope:** `rustscan-sfm/src/core/`, reconstruction construction helpers,
and focused unit tests

- [x] Add a structured `ReconstructionValidationError` and the structural and
      export validators specified in `design.md`.
- [x] Add strict camera/image/point ID and camera lookup APIs. Migrate internal
      persistence callers away from fabricated fallback IDs and cameras.
- [x] Add a checked occupied-ID allocator for reference/database/new images and
      points, including overflow and COLMAP-domain limits.
- [x] Validate uniqueness, parallel metadata lengths, camera indices,
      observation/track agreement, feature bounds, and rig/frame references.
- [x] Add table-driven tests covering every invariant and error payload.
- [x] Preserve compatibility wrappers only when necessary and mark them
      deprecated. T2-owned construction and the mapper pre-export gate do not
      call them. Remaining production fallback calls are listed in
      `verification.md` for T3, T4, and T5; they are not removed in T2.

**Exit condition:** an inconsistent `Reconstruction` cannot pass strict export
validation, and ID allocation is collision-free for sparse and non-contiguous
existing IDs.

## T3 — Make COLMAP Import And Export Strict

**Dependencies:** T2 and T4  
**Primary scope:** `rustscan-sfm/src/io/colmap.rs` and COLMAP IO tests

- [x] Validate complete raw text and binary models before internal conversion.
- [x] Reject duplicate camera, image, point, rig, and frame IDs instead of
      allowing map collection to overwrite an earlier record.
- [x] Reject missing camera/point/image/feature references and conflicting
      image-side versus point-track observations. Remove the implicit
      `ensure_*` repair behavior from normal import.
- [x] Reject zero/non-finite quaternions, non-finite translations/points/errors,
      zero dimensions, and invalid/non-finite focal parameters with record IDs.
- [x] Call strict reconstruction/export validation from every public COLMAP
      export and sparse snapshot writer before creating or truncating files.
- [x] Add equivalent malformed text and binary fixtures for duplicate IDs,
      unknown references, conflicting tracks, zero quaternion, NaN/Inf, and
      feature-index overflow.
- [x] Add a non-symmetric rotation, non-zero translation, multiple-camera,
      non-contiguous-ID round-trip test.

**Exit condition:** valid models round-trip with stable IDs and geometry;
invalid models fail before producing a partial internal reconstruction or
truncating an export file.

## T4 — Remove CameraModel's Split State And Fix Focal Refinement

**Status:** reviewed at tip `be27ce41843823810c2c4a2f06a0a7e5e2ee6dbf`
(`agent/RS-2026-004/t4-camera-invariants`).

**Dependencies:** T2  
**Primary scope:** `rustscan-sfm/src/core/types.rs`,
`rustscan-sfm/src/sfm/view_graph_calibration.rs`, direct camera callers, and tests

- [x] Make COLMAP parameters the canonical camera state and provide checked
      derived accessors/mutators.
- [x] Migrate every direct write to `params`, `fx`, `fy`, `cx`, or `cy` to an
      invariant-preserving API.
- [x] Preserve a required serialized schema through an explicit validated
      proxy; do not retain two mutable runtime sources of truth.
- [x] Change focal refinement so each candidate changes canonical projection
      parameters before scoring.
- [x] Reject a non-finite scale and invalid resulting focal length.
- [x] Add a synthetic focal-search test with an optimum away from the first
      grid element, plus projection/export consistency assertions.
- [x] Add constructor and deserialization tests for zero, negative, NaN, and
      infinite focal parameters.
- [x] Review P1: checked mutators commit only after candidate validation so
      failures leave the camera unchanged.
- [x] Review P1: mapper/config intrinsics overrides collect fx/fy once via
      `apply_optional_intrinsics` (single-focal mean is order-independent) and
      return contextual errors instead of panicking.
- [x] Review P1: production BA/mapper camera writes go through checked APIs;
      `params` is private.
- [x] Review P1: `apply_optional_intrinsics` is fully atomic (candidate params,
      one validation, one commit); illegal principal-point updates leave focals
      unchanged.
- [x] Independent review accepted tip `be27ce4` with focused fmt/check/types/
      view-graph/colmap/no-default/`git diff --check` evidence. Full
      `--all-targets` was not completed for that tip. Clippy remains blocked by
      pre-existing `rustscan-slam` lints and is not a T4 defect.

**Exit condition:** camera accessors, projection, BA, calibration, and export
cannot observe different intrinsics for the same `CameraModel`.

## T5 — Make Bundle Adjustment State Updates Atomic

**Status:** reviewed and approved on `agent/RS-2026-004/t5-atomic-ba` (base
`701d051814a29ab3ee4fbefc01355112f71b5a47`; prior CHANGES_REQUESTED reviews
and the accepted distortion-classification re-review are retained).

**Dependencies:** T0  
**Primary scope:** `rustscan-sfm/src/ba/`,
`rustscan-sfm/src/sfm/mapper/bundle_adjustment.rs`,
`rustscan-sfm/src/sfm/global_mapper.rs`, and BA/global-mapper tests

- [x] Inspect the Ceres summary and all solution parameters before write-back.
- [x] Commit camera, pose, point, and point-error changes only for a usable,
      finite solution.
- [x] Ensure cancellation and all failure termination types leave the caller's
      state unchanged.
- [x] Make `global_mapper` derive success from `is_solution_usable()` and stop
      the affected refinement round after an unusable result.
- [x] Keep mapper camera-plausibility rollback as a separate post-success gate.
- [x] Add deterministic convergence, no-convergence, failure, user-failure,
      cancellation, and non-finite-solution tests. Compare the complete mutable
      BA state before and after rejected solutions.
- [x] Review P1: stage full candidate (cameras, derived poses, points, and
      checked point errors including f64→f32 / accumulation) before any live
      commit; reject without mutating on non-finite derived error (focal `3e38`
      repro). Keep usable `NoConvergence` commit semantics.
- [x] Review P2: skip trailing track filter after unusable/absent/cancelled BA
      (`filter_min_track_length=3` two-view repro); preserve prior accepted work.
- [x] Review P2: move fault injection to `cfg(test)` `commit_test_hooks`; remove
      production `BaCommitTestOverride` / options fields; shrink
      `should_commit_ba_solution` to `pub(crate)`.
- [x] Review 2026-09-24 P1: validate composed image/frame/sensor poses after
      write-back (finite translation/rotation, valid normalized quaternion);
      distinguish behind-camera geometric skips from non-finite/overflow
      projection failures in `refresh_point_errors_checked`; composed-pose and
      projection regressions on the real candidate/commit path.
- [x] Review 2026-09-24-final P1: model-aware `CameraProjectionOutcome` /
      `classify_img_from_cam_unchecked` so SIMPLE_RADIAL / OPENCV / fisheye
      distortion Inf/NaN is `NonFiniteProjection` (candidate reject), while
      finite behind-camera and division/EUCM domain rejection stay
      `FiniteGeometricDomainSkip`; real commit-path radial overflow regression.

**Exit condition:** every rejected BA result is observationally atomic and is
never reported as a successful global BA round.

## T6 — Add Transactions Around Logical Database Batches

**Dependencies:** T0  
**Primary scope:** `rustscan-sfm/src/io/database.rs`,
`rustscan-sfm/src/sfm/mapper/database_io.rs`, and database tests

- [ ] Wrap local database population in one transaction.
- [ ] Wrap each pair-geometry batch in one transaction.
- [ ] Wrap complete target database merge in one transaction.
- [ ] Preserve and restore deletion/vacuum bookkeeping on rollback.
- [ ] Add deterministic mid-batch failure tests for population and merge.
- [ ] Assert pre-existing target rows and counts are unchanged after rollback,
      and successful retry commits exactly once.

**Exit condition:** each public logical operation either commits all related
rows or leaves the target database unchanged.

## T7 — Complete RustSFM nalgebra Ownership

**Dependencies:** T1 through T6 integrated  
**Primary scope:** RustSFM CPU math, required `rustscan-types` shared pose work,
affected RustViewer tests, math-check script, and CI

- [ ] Replace `generalized_pose` hand-written `Mat3d`/`Vec3d` with
      `Matrix3<f64>`/`Vector3<f64>` and preserve PoseLib FFI flat buffers only
      inside the adapter.
- [ ] Migrate `Point3D.xyz` to `Point3<f32>` and audit rays, positions,
      translations, rotations, and matrices for the correct nalgebra owner.
- [ ] Remove the CPU pose role of `Rigid3`; use the shared pose internally and
      keep raw COLMAP component arrays in IO/database records.
- [ ] Convert internal `PairGeometry` matrices/vectors to nalgebra where they
      are used numerically; convert to flat arrays only in the database/COLMAP
      adapter with named row-major/component-order helpers.
- [ ] Audit every remaining floating-point `[T; 2/3/4/9/16]` and `[[T; N]; M]`
      in RustSFM. Document and allowlist only real GPU, FFI, serialization, or
      wire-format boundaries.
- [ ] Extend the CPU-math check and CI so known hand-written owning types and
      unapproved public/internal array owners fail the check.
- [ ] Add non-symmetric rotation, translation, point-versus-vector, matrix
      layout, quaternion order, and FFI/COLMAP round-trip tests.

**Exit condition:** general CPU RustSFM ownership uses nalgebra, boundary arrays
are explicit and tested, and the automated repository check enforces the rule.

## T8 — Integrate, Verify, Review, And Hand Off

**Dependencies:** T1 through T7  
**Primary scope:** integration fixes, this change package, and no unrelated work

- [ ] Rebase or merge task commits in dependency order and resolve conflicts
      without weakening tests or adding silent fallback behavior.
- [ ] Run focused checks after each task integration, then the complete matrix
      from `tasks.yaml`.
- [ ] Remove RustSFM warnings in changed/default/all-feature targets or add only
      narrow, documented boundary allowances permitted by `rust-style.md`.
- [ ] Record native Ceres/PoseLib and GPU environment details. A known macOS
      AGX skip remains an unavailable result until run on a working adapter.
- [ ] Have a reviewer independently inspect mapper identity, reconstruction
      invariants, public API compatibility, numerical conventions, rollback,
      transaction coverage, and the final diff.
- [ ] Update `verification.md` with commits, exact outputs, limitations, and
      next action. Use the repository handoff template.
- [ ] Correct the completion status in `docs/nalgebra-unification-todo.md` if
      T7 proves any previously checked assertion was incomplete.

**Exit condition:** all acceptance conditions have evidence, the reviewer has
no unresolved P1/P2 findings, and integration does not include the unrelated
dirty documentation changes.

# COLMAP Module Parity Matrix

This document maps COLMAP's source modules to the current RustSFM modules and
tracks module-level reproduction progress. Percentages are engineering
estimates based on current code and `COLMAP_COMPAT_TODO.md`; they are not a
claim of bit-for-bit parity. A module is only marked 100% when there is direct
evidence that all relevant COLMAP behavior is implemented and covered by parity
tests.

Last re-evaluated: 2026-06-20, with `cargo test -p rustsfm --lib` passing
263 tests and `cargo check -p rustsfm --bin rustsfm` passing with two existing
dead-code warnings. The optional `poselib` feature was also checked locally, but
cannot currently build without `POSELIB_ROOT` or `third_party/PoseLib`; PoseLib
solver paths are therefore counted as optional, not default-verified behavior.
Current narrow 100% parity boundaries remain limited to sparse model I/O and
database/cache/correspondence graph behavior.

## Summary

| Priority | COLMAP module | RustSFM module(s) | Estimated parity | 100% replicated? | Notes |
| --- | --- | --- | ---: | --- | --- |
| P0 | `scene`, `sensor` core data and camera semantics | `types.rs`, `colmap.rs`, `database.rs`, `mapper.rs`, `ba.rs` | 63% | No | Camera model ids and sparse camera I/O are complete as narrow subfeatures, and all official COLMAP camera model projection/unprojection paths are represented. Initial-pair gating and next-frame registration respect non-trivial rig requirements with known `sensor_from_rig` poses in the covered paths. Full `scene`/`sensor` behavior remains broader than the current RustSFM model: reconstruction manager behavior, visibility pyramids, reconstruction clustering/pruning, exact camera sharing/reset scheduling, pose priors, covariance, and Ceres-equivalent rig/sensor refinement are still partial or missing. |
| P0 | Sparse model I/O codec | `colmap.rs`, `types.rs` | 100% | Yes | Raw COLMAP sparse `cameras/images/points3D/rigs/frames` text and binary codecs are implemented with COLMAP-style file selection, little-endian layouts, 17-digit text precision, ids, tracks, rigs/frames, optional invalid point ids, empty rigs/frames files, and text/bin round-trip tests. Exporting from RustSFM's internal `Reconstruction` remains precision-limited by its `f32` pose/keypoint/point storage, so this 100% mark is for the file-format codec boundary, not all `scene` reconstruction semantics. |
| P1 | `feature` SIFT extraction | `sift.rs`, `wide.rs` | 18% | No | COLMAP-style option defaults exist, but the backend is still `lowe-sift` best-effort rather than COLMAP VLFeat/SiftGPU/feature-module parity. Affine/DSP/multi-orientation/exact descriptor normalization and newer COLMAP feature backends are not replicated. |
| P1 | `feature` matching | `sift.rs`, `database.rs`, `correspondence_graph.rs` | 30% | No | Ratio/distance/cross-check/max-match controls exist and verified geometry can be persisted through the COLMAP database path. Guided matching, full strategy selection, exhaustive/sequential/vocab-tree pairing, GPU/FAISS behavior, and ONNX matcher parity are missing. |
| P1 | Database/cache and correspondence graph | `database.rs`, `correspondence_graph.rs`, `parity.rs` | 100% | Yes | COLMAP SQLite database schema/migration/API codec, pair-id and blob direction semantics, raw keypoint/match overloads, descriptors, matches, two-view geometry payload storage, rigs/frames/pose-priors, database merge, close-time vacuum, `LoadRandomDatabaseDescriptors`, `DatabaseCache::Load/CreateFromCache`, cache add/find/count helpers, legacy trivial rigs/frames, ENU pose-prior conversion, and `CorrespondenceGraph` behavior are reproduced for the database/cache/correspondence-graph boundary. This does not include generating exact two-view geometry via COLMAP estimators or mapper use of every geometry config. |
| P2 | `estimators` / `geometry` two-view geometry | `two_view.rs`, `five_point.rs`, `polynomial.rs`, `generalized_pose.rs` | 80% | No | E/F/H estimation, homography pose, support scoring, metadata reporting/write-back, generalized relative pose preparation, and panoramic-rig relative-pose fallback are covered well in the Rust paths. Optional PoseLib GR6P/GR8P support exists in code but is not default-build verified locally. Remaining gaps include exact estimator stack, byte-level Eigen/Solver parity, exact metadata semantics for every config, and complete mapper/refinement integration. |
| P2 | `optim` RANSAC/LORANSAC | `two_view.rs`, `mapper.rs`, `generalized_pose.rs` | 58% | No | Sampling shape, support ordering, dynamic stopping, MT19937, without-replacement trial counts, and E/F/H local refit are aligned for covered paths. COLMAP's full sampler family, SPRT/progressive behavior, library-level fixed-seed behavior, parallel random seeding, and bit-level LORANSAC behavior are not complete. |
| P2/P3 | Absolute/generalized pose solvers | `mapper.rs`, `generalized_pose.rs`, `geometry.rs` | 61% | No | Central absolute-pose paths are COLMAP/PoseLib-shaped for covered mapper cases, including P3P/EPNP/unknown-focal scheduling and inlier-only refinement. Generalized relative/absolute pose input preparation and scoring exist, but the actual GR6P/GR8P/GP3P solver paths require the optional `poselib` feature, which did not build in this local checkout due to missing PoseLib sources. Exact RANSAC/LORANSAC parity, covariance, Ceres-equivalent refinement, and full camera reset/refinement scheduling remain missing. |
| P3 | `sfm` incremental mapper state machine | `mapper.rs`, `parity.rs` | 71% | No | Database-first flow, COLMAP-style initial-image/second-image ordering, initial-pair gates, initialization trials/relaxation, bad-initial-pair retry, first multi-model keep/discard behavior, current-submodel overlap accounting, snapshot-frequency sparse exports, COLMAP-style per-registration and final all-image `extract_colors` behavior, callback timing for initial/next/last registration events, reference sparse-model seed continuation without initial-pair reselection, covered reconstruction-manager index-0 continuation semantics, covered `fix_existing_frames` behavior for local/global BA and registered-frame filtering including non-trivial rig-frame sparse fixtures, registration rollback, BA scheduling hooks, structure-less boundaries, COLMAP-shaped `FindNextImages` two-bucket candidate queue/ranking, failed-candidate continuation and trial recording for the covered next-image path, frame-aware trial gating, and 20-frame registered-frame filtering are implemented for covered paths. Official COLMAP-output / real-dataset rig-frame continuation fixtures, exact generalized-rig retry/reset semantics, and Ceres-equivalent solver summaries remain partial. |
| P3/P6 | `sfm` global mapper / rotation averaging / pose graph | `pose_graph.rs`, `mapper.rs` | 12% | No | `pose_graph.rs` is a RustSFM-specific pose-graph initializer with rotation/translation averaging heuristics and periodic-scene regularization. It is not yet a COLMAP `GlobalMapper` reproduction. COLMAP's global mapper orchestration, pose graph ownership, track establishment, rotation averaging pipeline, global positioning, iterative BA/retriangulation stages, options, and tests are largely missing. |
| P4 | `sfm` observation management | `observation_manager.rs`, `mapper.rs` | 55% | No | Add/delete/merge ownership, visible correspondence stats, modified points, frame-aware hooks, and filtering-time deregistration event rollback exist. Long-lived mapper-owned manager semantics, every cleanup event path, and exact frame/rig counter behavior remain incomplete. |
| P4 | `sfm` triangulation and filtering | `incremental_triangulator.rs`, `mapper.rs`, `geometry.rs` | 45% | No | Create/continue/merge/complete/retriangulate and filtering are partially reproduced, including COLMAP-style deregistration of registered frames with bogus cameras or zero point3D observations after the 20-frame threshold. Transitivity queues, trial limits, exact defaults, scheduling, and full interaction with observation manager/BA still differ. |
| P5 | Bundle adjustment / `optim` Ceres layer | `ba.rs`, `mapper.rs` | 38% | No | Parameter block scheduling, analytic Jacobians, gauges, reports, and some convergence knobs are substantial. The backend remains a Rust hand-rolled LM implementation, not Ceres-equivalent; covariance, robust loss details, sparse linear solver behavior, threading, and solver summaries are not COLMAP-complete. |
| P5 | Local/global BA orchestration | `mapper.rs`, `ba.rs`, `observation_manager.rs` | 48% | No | Local/global BA scheduling, gauge handling, frame-aware options, post-registration constant-camera decisions, and post-BA filtering are partial. Exact image/point selection, robust losses, normalization, pose priors, covariance, and Ceres behavior remain incomplete. |
| P6 | Controllers / end-to-end pipeline | `main.rs`, `mapper.rs`, `parity.rs` | 42% | No | RustSFM has a reconstruction entry point, reports, a first COLMAP-shaped multi-model pipeline slice, current-submodel overlap accounting, snapshot-frequency sparse exports, an `extract_colors` controller switch with per-registration and final all-image timing, COLMAP-shaped initial/next/last registration callback event boundaries in the pipeline log, and a lightweight public callback sink API exposed through `run_reconstruction_with_callbacks` with payload tests. It also has `fix_existing_frames` and reference-model continuation into the mapper with index-0 seed reuse only on the first continuation trial, including non-trivial rig-frame coverage through a COLMAP sparse-text fixture. Full controller-level orchestration, deterministic parallel initial-pair probing, and full parity harness are not complete. |
| P6 | `mvs` dense reconstruction | none | 0% | No | COLMAP MVS, PatchMatch, fusion, meshing, and dense workspace behavior are not replicated. |
| P6 | `ui`, `exe`, `tools` | `main.rs` only | 5% | No | RustSFM has a CLI binary, not COLMAP GUI/tools parity. |
| P6 | `retrieval` | none | 0% | No | Vocab-tree/image retrieval pipeline is not replicated. |
| Support | `math`, `util`, `image` helpers | scattered: `polynomial.rs`, `types.rs`, `geometry.rs`, `pose_graph.rs` | 30% | No | Individual math routines are ported where needed, but COLMAP helper module parity is not a goal yet. Logging/options/timers/random utilities, image/bitmap behavior, and many math helpers remain partial or absent. |

## COLMAP Source Module Shape

The official COLMAP source tree is organized roughly as follows:

- `sensor` and `scene`: camera models, images, points, rigs, frames, tracks,
  pose priors, reconstruction ownership, and sparse model state.
- `feature`: SIFT extraction, descriptor matching, guided matching, and
  feature/match option surfaces.
- `retrieval`: vocabulary-tree and image-retrieval infrastructure.
- `estimators` and `geometry`: minimal solvers, two-view geometry,
  homography/essential/fundamental estimation, pose estimation, triangulation,
  and geometric predicates.
- `optim`: RANSAC/LORANSAC, bundle adjustment, Ceres cost functions,
  parameterizations, gauges, and solver reports.
- `sfm`: incremental mapper, observation manager, triangulator, registration,
  local/global BA orchestration, reconstruction management, and filtering.
- `image`, `math`, and `util`: supporting image I/O, numerical helpers,
  logging, random sampling, timing, options, and general utilities.
- `controllers`, `exe`, `tools`, and `ui`: application orchestration, command
  line tools, GUI, and higher-level workflows.
- `mvs`: dense reconstruction, PatchMatch stereo, fusion, and meshing.

## Current Overall Read

- Full COLMAP parity, including dense MVS, GUI/tools, retrieval, controllers,
  and global mapper, remains low because several entire COLMAP modules have no
  RustSFM counterpart yet.
- Sparse SfM parity is meaningfully higher than full-COLMAP parity: database
  loading, sparse model codecs, two-view geometry metadata, generalized pose
  bridges, registration scheduling, triangulation, observation bookkeeping, and
  BA orchestration all have partial implementations.
- The default build does not currently include the PoseLib solver bridge. The
  code path exists behind `--features poselib`, but this checkout is missing
  the PoseLib source tree required by `build.rs`, so generalized rig solver
  claims are treated as optional and not part of default verified parity.
- The highest-confidence completed areas are file/database compatibility
  boundaries, not numerical reconstruction behavior. Mapper, solver, and BA
  percentages should stay conservative until they are backed by COLMAP parity
  fixtures or exact library-level behavior.

## Narrow Subfeatures That Are Currently Complete

These are smaller than COLMAP modules, but have enough local evidence to treat
as complete for now:

- COLMAP sparse camera model id/name/parameter count/order preservation in
  camera sparse I/O.
- Raw COLMAP sparse model file-format codec for `cameras/images/points3D`,
  plus optional `rigs/frames`, in both text and binary forms. This includes
  COLMAP file-selection precedence, little-endian binary fields, invalid
  point3D sentinels, image names through end-of-line, deterministic id-based
  write ordering, empty rig/frame sidecar files, and text/binary round-trip
  coverage.
- COLMAP SQLite database/cache/correspondence-graph boundary. Current
  `cameras/images/rigs/rig_sensors/frames/frame_data/pose_priors` and
  feature/match/two-view tables are created with COLMAP-compatible schema;
  old `inlier_matches`, legacy image pose priors, missing descriptor `type`,
  and old zero/identity two-view sentinels are migrated; core
  read/write/update/delete/clear/count/exists/transaction APIs are covered by
  tests. `Database::Merge`, raw keypoint and match blob overloads,
  close-time vacuum, `LoadRandomDatabaseDescriptors`,
  `DatabaseCache::Load/CreateFromCache`, cache add/find/count helpers,
  legacy trivial rigs/frames, WGS84-to-ENU pose-prior conversion, and
  `CorrespondenceGraph` flatten/extract/update semantics are reproduced. This
  is the second explicitly marked 100% parity boundary. It does not imply exact
  COLMAP estimator parity for producing two-view geometry; that remains under
  P2/P3.
- Removal of RustSFM-specific local-window / 192-frame sequence heuristics from
  the default mapper path; those behaviors are now explicit experimental
  options only.
- COLMAP-shaped optional two-view geometry blobs are represented in SQLite
  read/write and now flow through Rust-estimated `TwoViewEstimate` /
  `PairGeometry` metadata for F/E/H/qvec/tvec reporting and opt-in database
  write-back.

## Highest-Priority Replication Targets

1. `scene`/`sensor` core data and camera semantics (`P0`)
   - Finish exact generalized pose reset/refinement scheduling for non-trivial rigs.
   - Tighten image-only camera sharing/grouping.
   - Keep all sparse/database ids and frame/rig ownership stable through every
     mapper cleanup path.
2. `estimators`/`geometry` two-view geometry (`P2`)
   - Finish full `TwoViewGeometry` pose/config metadata parity.
   - Make the optional PoseLib GR6P/GR8P/GP3P bridge reproducibly buildable in
     the default development environment, then finish COLMAP's exact
     `CALIBRATED_RIG` LORANSAC behavior on top of that bridge.
   - Continue reducing solver/RANSAC differences using existing linear algebra
     crates rather than recreating Eigen internals.
3. `sfm` incremental mapper (`P3`)
   - Expand the new COLMAP sparse-text rig-frame continuation and
     `fix_existing_frames` fixtures to official COLMAP-output / real-dataset
     cases.
   - Exact initial-image and next-image priority queues.
   - Registration retry/reset semantics around generalized rig registration.
4. Bundle adjustment (`P5`)
   - Replace or wrap the hand-rolled LM backend with Ceres-equivalent behavior.

## Next Implementation Choice

The absolute highest priority remains P0, but its remaining generalized rig
work depends directly on P2/P3 solver and mapper behavior. The next actionable
step should be one of two tightly scoped paths:

1. Make the PoseLib bridge reproducibly build in this checkout, then verify the
   GR6P/GR8P/GP3P tests under `--features poselib`.
2. Continue the P3 controller/mapper slice by expanding the new COLMAP
   sparse-text rig-frame continuation and `fix_existing_frames` fixtures to
   official COLMAP-output / real-dataset cases, which do not depend on the
   missing PoseLib source tree.

Given the current local build state, the lower-risk next implementation step is
to extend the covered `IncrementalPipeline` behavior around priority queues,
a real user-extensible callback API, and official COLMAP-output fixtures while
the external PoseLib source tree is still unavailable.

# COLMAP Module Parity Matrix

This document maps COLMAP's source modules to the current RustSFM modules and
tracks module-level reproduction progress. Percentages are engineering
estimates based on current code and `COLMAP_COMPAT_TODO.md`; they are not a
claim of bit-for-bit parity. A module is only marked 100% when there is direct
evidence that all relevant COLMAP behavior is implemented and covered by parity
tests.

Last re-evaluated: 2026-06-20, with `cargo test -p rustsfm --lib`
passing 304 tests (Ceres BA enabled by default via `ceres-ba` feature),
`cargo test -p rustsfm --features poselib --lib` passing 309 tests, and
`cargo test -p rustsfm --no-default-features --lib` passing 301 tests (native
BA only). Ceres BA (`ceres_problem.rs`) now supports rig/frame/sensor pose
blocks, intrinsics refinement, and gauge policies, and reuses native analytic
reprojection Jacobians in the Ceres cost callback (numeric fallback retained).
A new `geometry/triangulation` primitive module (`triangulation.rs`)
faithfully ports COLMAP's `geometry/triangulation.{h,cc}` (two-view DLT,
midpoint, multi-view DLT, Lindstrom optimal triangulation, and triangulation
angles), and is now the shared backend for two-view DLT (`two_view.rs`) and
multi-view track triangulation (`incremental_triangulator.rs`). A new
`estimators/triangulation` module (`triangulation_estimator.rs`) ports the
`TriangulationEstimator` (two-view/multi-view model estimation with cheirality
+ triangulation-angle gating and angular/reprojection residuals) and the
`EstimateTriangulation` LORANSAC entry point (structurally faithful; it now
uses COLMAP's deterministic `CombinationSampler` order for the triangulation
estimator, while exact random-sampler and parallel LORANSAC parity remain under
the broader `optim` RANSAC item).
`EstimateTriangulation` is now wired into the incremental triangulator's
track-creation path (`IncrementalTriangulator::create_pair_track`), which
gathers the seed pair plus transitively-corresponding registered observations
(bounded by `max_transitivity`, COLMAP `Create`-style) and triangulates them in
one robust multi-view step using angular residuals.
PoseLib v2.0.5 is now vendored under `third_party/PoseLib` and the
`poselib` feature builds and is test-verified locally, so GR6P/GR8P/GP3P solver
paths are now optional-but-default-buildable rather than unbuildable. This pass
also reflects new work: shared COLMAP MT19937 fixed-seed RNG across crates
(`RustSLAM/src/colmap_rng.rs`), SIFT L1_ROOT/upright/scale-limit extraction,
matching-strategy selection (exhaustive/sequential/local-window) plus guided
matching, a `--write-database` local-matching database population path,
per-image-pair retriangulation trial counting, and a Cholesky-based reduced
camera matrix (Schur complement) solve in bundle adjustment matching Ceres'
`DENSE_SCHUR`/`SPARSE_SCHUR` linear solvers. Current narrow 100% parity
boundaries remain limited to sparse model I/O and database/cache/correspondence
graph behavior.

## Summary

| Priority | COLMAP module | RustSFM module(s) | Estimated parity | 100% replicated? | Notes |
| --- | --- | --- | ---: | --- | --- |
| P0 | `scene`, `sensor` core data and camera semantics | `types.rs`, `colmap.rs`, `database.rs`, `mapper.rs`, `ba.rs` | 63% | No | Camera model ids, sparse camera I/O, and COLMAP `VisibilityPyramid` scoring are complete as narrow subfeatures, and all official COLMAP camera model projection/unprojection paths are represented. Initial-pair gating and next-frame registration respect non-trivial rig requirements with known `sensor_from_rig` poses in the covered paths. Full `scene`/`sensor` behavior remains broader than the current RustSFM model: reconstruction manager behavior, reconstruction clustering/pruning, exact camera sharing/reset scheduling, pose priors, covariance, and Ceres-equivalent rig/sensor refinement are still partial or missing. |
| P0 | Sparse model I/O codec | `colmap.rs`, `types.rs` | 100% | Yes | Raw COLMAP sparse `cameras/images/points3D/rigs/frames` text and binary codecs are implemented with COLMAP-style file selection, little-endian layouts, 17-digit text precision, ids, tracks, rigs/frames, optional invalid point ids, empty rigs/frames files, and text/bin round-trip tests. Exporting from RustSFM's internal `Reconstruction` remains precision-limited by its `f32` pose/keypoint/point storage, so this 100% mark is for the file-format codec boundary, not all `scene` reconstruction semantics. |
| P1 | `feature` SIFT extraction | `sift.rs`, `wide.rs` | 30% | No | COLMAP-style option defaults exist plus L1_ROOT descriptor normalization, upright mode, and feature limiting by scale (`size * 2^octave`). The backend is still `lowe-sift` best-effort rather than COLMAP VLFeat/SiftGPU/feature-module parity: `estimate_affine_shape`, `domain_size_pooling`, and `max_num_orientations > 2` are exposed as options but not actually realized in the backend, and exact VLFeat/SiftGPU descriptor parity is missing. |
| P1 | `feature` matching | `sift.rs`, `database.rs`, `feature_matching.rs`, `correspondence_graph.rs` | 45% | No | Ratio/distance/cross-check/max-match controls, guided matching (epipolar-line filter + post-estimation geometry refit), and matching-strategy selection (exhaustive/sequential with overlap/quadratic-overlap/loop-detection/local-window) are implemented and CLI-wired. Verified geometry can be persisted through the COLMAP database path, and `--write-database` populates a full COLMAP SQLite database (cameras/images/keypoints/descriptors/matches/two-view geometries) for the local-matching fallback. Vocab-tree pairing, GPU/FAISS behavior, and ONNX matcher parity are missing. |
| P1 | Database/cache and correspondence graph | `database.rs`, `correspondence_graph.rs`, `parity.rs` | 100% | Yes | COLMAP SQLite database schema/migration/API codec, pair-id and blob direction semantics, raw keypoint/match overloads, descriptors, matches, two-view geometry payload storage, rigs/frames/pose-priors, database merge, close-time vacuum, `LoadRandomDatabaseDescriptors`, `DatabaseCache::Load/CreateFromCache`, cache add/find/count helpers, legacy trivial rigs/frames, ENU pose-prior conversion, and `CorrespondenceGraph` behavior are reproduced for the database/cache/correspondence-graph boundary. This does not include generating exact two-view geometry via COLMAP estimators or mapper use of every geometry config. |
| P2 | `estimators` / `geometry` two-view geometry | `two_view.rs`, `five_point.rs`, `polynomial.rs`, `generalized_pose.rs` | 80% | No | E/F/H estimation, homography pose, support scoring, metadata reporting/write-back, generalized relative pose preparation, and panoramic-rig relative-pose fallback are covered well in the Rust paths. PoseLib GR6P/GR8P support now builds and is test-verified under `--features poselib` (vendored `third_party/PoseLib`). Remaining gaps include exact estimator stack, byte-level Eigen/Solver parity, exact metadata semantics for every config, and complete mapper/refinement integration. |
| P2 | `geometry` triangulation primitives | `triangulation.rs` | 100% | Yes | Faithful port of COLMAP `geometry/triangulation.{h,cc}`: `TriangulatePoint` (two-view DLT via SVD), `TriangulateMidPoint` (with cheirality guard), `TriangulateMultiViewPoint` (smallest-eigenvector DLT), `TriangulateOptimalPoint` (Lindstrom optimal correction using `EssentialMatrixFromPose` + `FindOptimalImageObservations` from `essential_matrix.cc`), and `CalculateTriangulationAngle(s)` / `CalculateAngleBetweenVectors`, all in `f64` to match COLMAP's `double`. Covered by synthetic recovery/consistency tests, and wired as the shared backend for `two_view.rs` two-view DLT and `incremental_triangulator.rs` multi-view track triangulation. This 100% mark is for the triangulation primitive boundary; the `estimators/triangulation.cc` RANSAC `EstimateTriangulation` wrapper is tracked separately under P4. |
| P2 | `optim` RANSAC/LORANSAC | `two_view.rs`, `mapper.rs`, `generalized_pose.rs`, `support_measurement.rs`, `RustSLAM/src/colmap_rng.rs` | 69% | No | Sampling shape, support ordering, dynamic stopping, MT19937, without-replacement trial counts, COLMAP `NChooseK`, deterministic `CombinationSampler`, `ProgressiveSampler`/PROSAC state progression, COLMAP `InlierSupportMeasurer`/`UniqueInlierSupportMeasurer`/`MEstimatorSupportMeasurer`, and E/F/H local refit are aligned for covered paths. A shared COLMAP-compatible MT19937 + libc++ `uniform_int_distribution` fixed-seed sampler now backs PnP/essential/two-view/generalized paths across crates. COLMAP's SPRT, parallel random seeding, and bit-level LORANSAC behavior are not complete. |
| P2/P3 | Absolute/generalized pose solvers | `mapper.rs`, `generalized_pose.rs`, `geometry.rs` | 63% | No | Central absolute-pose paths are COLMAP/PoseLib-shaped for covered mapper cases, including P3P/EPNP/unknown-focal scheduling and inlier-only refinement. Generalized relative/absolute pose input preparation and scoring exist, and the GR6P/GR8P/GP3P solver paths now build and pass tests under the optional `poselib` feature (vendored `third_party/PoseLib`). Exact RANSAC/LORANSAC parity, covariance, Ceres-equivalent refinement, and full camera reset/refinement scheduling remain missing. |
| P3 | `sfm` incremental mapper state machine | `mapper.rs`, `parity.rs` | **~85%** ↑ | No | Database-first flow, COLMAP-style initial/next-image ordering, initialization trials/relaxation, multi-model control, snapshots, color extraction, callbacks, reference-model continuation, `fix_existing_frames`, registration rollback, structure-based/structure-less two-bucket **`FindNextImages`** queue with **separate `structureless_reg_trials`**, **`CorrespondenceGraph`-based 2D-3D/2D-2D registration collection** (PnP/GP3P/structureless), and rig **`MinUncertainty` max-sibling scoring. Remaining: parallel initial-pair probing, full controller API, real-dataset rig continuation fixtures, generalized rig camera reset/refinement schedule parity. |
| P3/P6 | `sfm` global mapper / rotation averaging / pose graph | `pose_graph.rs`, `mapper.rs` | 12% | No | `pose_graph.rs` is a RustSFM-specific pose-graph initializer with rotation/translation averaging heuristics and periodic-scene regularization. It is not yet a COLMAP `GlobalMapper` reproduction. COLMAP's global mapper orchestration, pose graph ownership, track establishment, rotation averaging pipeline, global positioning, iterative BA/retriangulation stages, options, and tests are largely missing. |
| P4 | `sfm` observation management | `observation_manager.rs`, `visibility_pyramid.rs`, `mapper.rs` | 100% | Yes | COLMAP-style **`ObservationManager`** with embedded **`CorrespondenceGraph`**, incremental **`SetObservationAsTriangulated`** / **`ResetTriObservations`**, **`IncrementCorrespondenceHasPoint3D`** / **`DecrementCorrespondenceHasPoint3D`**, register/deregister visible-correspondence propagation, add/delete/merge point3D ownership, 6-level **`VisibilityPyramid`** (`SetPoint`/`ResetPoint`, weighted `Score`, `MaxScore`, metadata accessors), modified-point tracking, and frame/rig register/deregister hooks. Session-scoped state lives in **`IncrementalTriangulatorState`**. Covered by incremental-vs-rebuild stat parity tests. |
| P4 | `sfm` triangulation and filtering | `incremental_triangulator.rs`, `mapper.rs`, `geometry.rs`, `triangulation.rs`, `triangulation_estimator.rs`, `correspondence_graph.rs` | 100% | Yes | **`TriangulateImage`** per point2D (`Find`/`Continue`/`Create`) and **`CompleteImage`** per point2D (complete existing tracks + create orphan clusters with reprojection RANSAC) use **`CorrespondenceGraph::extract_transitive_correspondences`**. **`Complete`** uses direct-graph BFS with squared reprojection gating. Session-scoped trial counters persist. **`EstimateTriangulation`** uses COLMAP `CombinationSampler` deterministic two-view combination order. Covered by triangulation/complete-image and combination-sampler tests. |
| P5 | Bundle adjustment / `optim` Ceres layer | `ba/` | **~72%** ↑ | No | Parameter block scheduling, analytic Jacobians (native + **Ceres path now reuses native analytic projection/frame/sensor/camera Jacobians** with numeric fallback), gauges, reports, convergence knobs, and Ceres-equivalent robust loss family. The `ba/` module splits into `mod.rs`, `shared.rs`, `native.rs`, `ceres.rs`, and `ceres_problem.rs` (full Ceres problem: image/frame/sensor pose blocks, intrinsics, gauges, fixed poses). `ceres-ba` is **enabled by default**; no fallback to native. Remaining gaps: quaternion manifold parameterization in the Rust Ceres binding, Ceres solver summary fields, covariance, threading, true sparse Schur. |
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
- The PoseLib solver bridge now builds. PoseLib v2.0.5 is vendored under
  `third_party/PoseLib` and `build.rs` resolves it automatically, so the
  `--features poselib` GR6P/GR8P/GP3P paths are test-verified (309 tests).
  PoseLib remains an optional feature (not in default build), but when enabled
  COLMAP structureless registration is active via `cfg!(feature = "poselib")`
  without requiring the experimental pair-pose fallback flag.
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
- COLMAP `scene/VisibilityPyramid` scoring boundary. `visibility_pyramid.rs`
  mirrors COLMAP's level dimensions (2, 4, ..., 64 for the default 6-level
  mapper pyramid), finest-to-coarsest cell propagation, weighted score
  contribution by `level.size()`, `MaxScore`, and width/height metadata
  accessors. This is a narrow scene subfeature and does not imply full `scene`
  reconstruction ownership parity.
- Removal of RustSFM-specific local-window / 192-frame sequence heuristics from
  the default mapper path; those behaviors are now explicit experimental
  options only.
- COLMAP-shaped optional two-view geometry blobs are represented in SQLite
  read/write and now flow through Rust-estimated `TwoViewEstimate` /
  `PairGeometry` metadata for F/E/H/qvec/tvec reporting and opt-in database
  write-back.
- COLMAP `geometry/triangulation` primitive module (`triangulation.rs`): the
  two-view DLT, midpoint, multi-view DLT, Lindstrom optimal triangulation, and
  triangulation-angle functions are reproduced in `f64` and used as the shared
  triangulation backend. This is the third explicitly marked 100% parity
  boundary; it is the geometric primitive only and excludes the
  `estimators/triangulation.cc` RANSAC `EstimateTriangulation` wrapper.

## Highest-Priority Replication Targets (ROI-ordered, 2026-06-20 PM)

1. Bundle adjustment / Ceres layer (`P5`, ~72%) — largest numerical-parity
   bottleneck; caps final reconstruction accuracy.
   - Done: Ceres robust loss family (Trivial/Huber/SoftL1/Cauchy) with COLMAP's
     Cauchy default; Cholesky reduced-camera-matrix solve (LU fallback)
     matching Ceres' `DENSE_SCHUR`/`SPARSE_SCHUR` linear solvers; `ba/` split
     with Ceres default path, rig/frame/sensor/intrinsics blocks, gauges, and
     analytic Jacobians reused from native projection code.
   - Next: quaternion manifold parameterization, jacobian-scaled LM damping /
     Ceres trust-region update, true sparse Schur storage, covariance,
     normalization, and exact Ceres solver summaries.
2. `sfm` triangulation + observation management lifecycle (`P4`, 100%)
   - Done: **`ObservationManager`** incremental event paths,
     **`VisibilityPyramid`** weighted COLMAP score/max-score semantics,
     **`TriangulateImage`** / **`CompleteImage`**
     per-point2D paths, COLMAP **`Complete`** BFS, **`EstimateTriangulation`**
     deterministic `CombinationSampler` order, mapper wiring.
3. `sfm` incremental mapper (`P3`, ~85%)
   - Done: separate structureless trial counters, correspondence-graph registration collection.
   - Next: parallel initial-pair probing, real-dataset rig continuation fixtures, generalized rig camera scheduling.
4. `estimators`/`geometry` two-view geometry on the PoseLib bridge (`P2`, ~80%)
   - Finish COLMAP's exact `CALIBRATED_RIG` LORANSAC behavior now that the
     PoseLib GR6P/GR8P/GP3P bridge builds.
   - Full `TwoViewGeometry` pose/config metadata parity.
5. `feature` SIFT VLFeat-equivalent backend (`P1`, ~30%)
   - Realize affine shape, DSP, multi-orientation (>2), and exact descriptor
     parity instead of the current `lowe-sift` best-effort backend.
6. Long-lead whole modules: `retrieval` vocab-tree (0%, also unlocks
   vocab-tree matching), global mapper / rotation averaging (~12%), `mvs`
   dense (0%), and GUI/tools (~5%).

## Next Implementation Choice

Starting with the highest-ROI target: P5 Bundle Adjustment toward
Ceres-equivalent behavior (trust-region damping, sparse Schur, solver summary
parity). P3's next slice is parallel initial-pair probing and generalized rig
camera reset/refinement scheduling.

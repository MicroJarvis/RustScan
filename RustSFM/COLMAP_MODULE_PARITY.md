# COLMAP Module Parity Matrix

This document maps COLMAP's source modules to the current RustSFM modules and
tracks module-level reproduction progress. Percentages are engineering
estimates based on current code and `COLMAP_COMPAT_TODO.md`; they are not a
claim of bit-for-bit parity. A module is only marked 100% when there is direct
evidence that all relevant COLMAP behavior is implemented and covered by parity
tests.

Last re-evaluated: 2026-06-20 (late night), with `cargo test -p rustsfm --lib`
passing 294 tests (Ceres BA enabled by default via `ceres-ba` feature),
`cargo test -p rustsfm --features poselib --lib` passing 299 tests, and
`cargo test -p rustsfm --no-default-features --lib` passing 293 tests (native
BA only). A new `geometry/triangulation` primitive module (`triangulation.rs`)
faithfully ports COLMAP's `geometry/triangulation.{h,cc}` (two-view DLT,
midpoint, multi-view DLT, Lindstrom optimal triangulation, and triangulation
angles), and is now the shared backend for two-view DLT (`two_view.rs`) and
multi-view track triangulation (`incremental_triangulator.rs`). A new
`estimators/triangulation` module (`triangulation_estimator.rs`) ports the
`TriangulationEstimator` (two-view/multi-view model estimation with cheirality
+ triangulation-angle gating and angular/reprojection residuals) and the
`EstimateTriangulation` LORANSAC entry point (structurally faithful; exact
RANSAC RNG/sampler-shuffle parity remains under the `optim` RANSAC item).
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
| P0 | `scene`, `sensor` core data and camera semantics | `types.rs`, `colmap.rs`, `database.rs`, `mapper.rs`, `ba.rs` | 63% | No | Camera model ids and sparse camera I/O are complete as narrow subfeatures, and all official COLMAP camera model projection/unprojection paths are represented. Initial-pair gating and next-frame registration respect non-trivial rig requirements with known `sensor_from_rig` poses in the covered paths. Full `scene`/`sensor` behavior remains broader than the current RustSFM model: reconstruction manager behavior, visibility pyramids, reconstruction clustering/pruning, exact camera sharing/reset scheduling, pose priors, covariance, and Ceres-equivalent rig/sensor refinement are still partial or missing. |
| P0 | Sparse model I/O codec | `colmap.rs`, `types.rs` | 100% | Yes | Raw COLMAP sparse `cameras/images/points3D/rigs/frames` text and binary codecs are implemented with COLMAP-style file selection, little-endian layouts, 17-digit text precision, ids, tracks, rigs/frames, optional invalid point ids, empty rigs/frames files, and text/bin round-trip tests. Exporting from RustSFM's internal `Reconstruction` remains precision-limited by its `f32` pose/keypoint/point storage, so this 100% mark is for the file-format codec boundary, not all `scene` reconstruction semantics. |
| P1 | `feature` SIFT extraction | `sift.rs`, `wide.rs` | 30% | No | COLMAP-style option defaults exist plus L1_ROOT descriptor normalization, upright mode, and feature limiting by scale (`size * 2^octave`). The backend is still `lowe-sift` best-effort rather than COLMAP VLFeat/SiftGPU/feature-module parity: `estimate_affine_shape`, `domain_size_pooling`, and `max_num_orientations > 2` are exposed as options but not actually realized in the backend, and exact VLFeat/SiftGPU descriptor parity is missing. |
| P1 | `feature` matching | `sift.rs`, `database.rs`, `feature_matching.rs`, `correspondence_graph.rs` | 45% | No | Ratio/distance/cross-check/max-match controls, guided matching (epipolar-line filter + post-estimation geometry refit), and matching-strategy selection (exhaustive/sequential with overlap/quadratic-overlap/loop-detection/local-window) are implemented and CLI-wired. Verified geometry can be persisted through the COLMAP database path, and `--write-database` populates a full COLMAP SQLite database (cameras/images/keypoints/descriptors/matches/two-view geometries) for the local-matching fallback. Vocab-tree pairing, GPU/FAISS behavior, and ONNX matcher parity are missing. |
| P1 | Database/cache and correspondence graph | `database.rs`, `correspondence_graph.rs`, `parity.rs` | 100% | Yes | COLMAP SQLite database schema/migration/API codec, pair-id and blob direction semantics, raw keypoint/match overloads, descriptors, matches, two-view geometry payload storage, rigs/frames/pose-priors, database merge, close-time vacuum, `LoadRandomDatabaseDescriptors`, `DatabaseCache::Load/CreateFromCache`, cache add/find/count helpers, legacy trivial rigs/frames, ENU pose-prior conversion, and `CorrespondenceGraph` behavior are reproduced for the database/cache/correspondence-graph boundary. This does not include generating exact two-view geometry via COLMAP estimators or mapper use of every geometry config. |
| P2 | `estimators` / `geometry` two-view geometry | `two_view.rs`, `five_point.rs`, `polynomial.rs`, `generalized_pose.rs` | 80% | No | E/F/H estimation, homography pose, support scoring, metadata reporting/write-back, generalized relative pose preparation, and panoramic-rig relative-pose fallback are covered well in the Rust paths. PoseLib GR6P/GR8P support now builds and is test-verified under `--features poselib` (vendored `third_party/PoseLib`). Remaining gaps include exact estimator stack, byte-level Eigen/Solver parity, exact metadata semantics for every config, and complete mapper/refinement integration. |
| P2 | `geometry` triangulation primitives | `triangulation.rs` | 100% | Yes | Faithful port of COLMAP `geometry/triangulation.{h,cc}`: `TriangulatePoint` (two-view DLT via SVD), `TriangulateMidPoint` (with cheirality guard), `TriangulateMultiViewPoint` (smallest-eigenvector DLT), `TriangulateOptimalPoint` (Lindstrom optimal correction using `EssentialMatrixFromPose` + `FindOptimalImageObservations` from `essential_matrix.cc`), and `CalculateTriangulationAngle(s)` / `CalculateAngleBetweenVectors`, all in `f64` to match COLMAP's `double`. Covered by synthetic recovery/consistency tests, and wired as the shared backend for `two_view.rs` two-view DLT and `incremental_triangulator.rs` multi-view track triangulation. This 100% mark is for the triangulation primitive boundary; the `estimators/triangulation.cc` RANSAC `EstimateTriangulation` wrapper is tracked separately under P4. |
| P2 | `optim` RANSAC/LORANSAC | `two_view.rs`, `mapper.rs`, `generalized_pose.rs`, `RustSLAM/src/colmap_rng.rs` | 62% | No | Sampling shape, support ordering, dynamic stopping, MT19937, without-replacement trial counts, and E/F/H local refit are aligned for covered paths. A shared COLMAP-compatible MT19937 + libc++ `uniform_int_distribution` fixed-seed sampler now backs PnP/essential/two-view/generalized paths across crates. COLMAP's full sampler family, SPRT/progressive behavior, parallel random seeding, and bit-level LORANSAC behavior are not complete. |
| P2/P3 | Absolute/generalized pose solvers | `mapper.rs`, `generalized_pose.rs`, `geometry.rs` | 63% | No | Central absolute-pose paths are COLMAP/PoseLib-shaped for covered mapper cases, including P3P/EPNP/unknown-focal scheduling and inlier-only refinement. Generalized relative/absolute pose input preparation and scoring exist, and the GR6P/GR8P/GP3P solver paths now build and pass tests under the optional `poselib` feature (vendored `third_party/PoseLib`). Exact RANSAC/LORANSAC parity, covariance, Ceres-equivalent refinement, and full camera reset/refinement scheduling remain missing. |
| P3 | `sfm` incremental mapper state machine | `mapper.rs`, `parity.rs` | 71% | No | Database-first flow, COLMAP-style initial-image/second-image ordering, initial-pair gates, initialization trials/relaxation, bad-initial-pair retry, first multi-model keep/discard behavior, current-submodel overlap accounting, snapshot-frequency sparse exports, COLMAP-style per-registration and final all-image `extract_colors` behavior, callback timing for initial/next/last registration events, reference sparse-model seed continuation without initial-pair reselection, covered reconstruction-manager index-0 continuation semantics, covered `fix_existing_frames` behavior for local/global BA and registered-frame filtering including non-trivial rig-frame sparse fixtures, registration rollback, BA scheduling hooks, structure-less boundaries, COLMAP-shaped `FindNextImages` two-bucket candidate queue/ranking, failed-candidate continuation and trial recording for the covered next-image path, frame-aware trial gating, and 20-frame registered-frame filtering are implemented for covered paths. Official COLMAP-output / real-dataset rig-frame continuation fixtures, exact generalized-rig retry/reset semantics, and Ceres-equivalent solver summaries remain partial. |
| P3/P6 | `sfm` global mapper / rotation averaging / pose graph | `pose_graph.rs`, `mapper.rs` | 12% | No | `pose_graph.rs` is a RustSFM-specific pose-graph initializer with rotation/translation averaging heuristics and periodic-scene regularization. It is not yet a COLMAP `GlobalMapper` reproduction. COLMAP's global mapper orchestration, pose graph ownership, track establishment, rotation averaging pipeline, global positioning, iterative BA/retriangulation stages, options, and tests are largely missing. |
| P4 | `sfm` observation management | `observation_manager.rs`, `mapper.rs` | 55% | No | Add/delete/merge ownership, visible correspondence stats, modified points, frame-aware hooks, and filtering-time deregistration event rollback exist. Long-lived mapper-owned manager semantics, every cleanup event path, and exact frame/rig counter behavior remain incomplete. |
| P4 | `sfm` triangulation and filtering | `incremental_triangulator.rs`, `mapper.rs`, `geometry.rs`, `triangulation.rs`, `triangulation_estimator.rs` | 64% | No | Create/continue/merge/complete/retriangulate and filtering are partially reproduced, including COLMAP-style deregistration of registered frames with bogus cameras or zero point3D observations after the 20-frame threshold. Retriangulation now tracks a per-image-pair trial counter limited by `re_max_trials` (matching COLMAP's `re_num_trials_` map). Track point estimation uses the COLMAP-faithful multi-view DLT (`TriangulateMultiViewPoint`). Track creation now runs the ported `estimators/triangulation` `EstimateTriangulation` LORANSAC path: `create_pair_track` gathers the seed pair plus transitively-corresponding registered observations (`max_transitivity`-bounded, COLMAP `Create`-style) and robustly triangulates them in one multi-view step with angular residuals + cheirality + `min_angle` gating, emitting a multi-view track from the inlier set (newly created points are left uncolored for the separate color-extraction stage). Exact RANSAC RNG/sampler-shuffle parity is still pending (tracked under `optim` RANSAC). Exact transitivity queues, official defaults, and a long-lived mapper-owned triangulator (the mapper currently rebuilds a triangulator per registration step, resetting merge/retriangulation trial state across steps) still differ. |
| P5 | Bundle adjustment / `optim` Ceres layer | `ba/`, `mapper.rs` | 52% | No | Parameter block scheduling, analytic Jacobians, gauges, reports, convergence knobs, and a Ceres-equivalent robust loss family (Trivial/Huber/SoftL1/Cauchy with Ceres `rho(s)`/`rho'(s)` formulas, defaulting local/global mapper BA to COLMAP's Cauchy scale 1.0) are substantial. The `ba/` module is split into `mod.rs` (types + dispatch), `shared.rs` (observations/gauge helpers), `native.rs` (hand-rolled LM reference backend), and `ceres.rs` (default Ceres backend). The `ceres-ba` feature is **enabled by default**; `refine_bundle_adjustment` always uses Ceres with **no fallback** to native. Unsupported configs return `None`. Build native-only with `--no-default-features`. Remaining gaps: extend Ceres to rig/sensor/intrinsics, manifold/quaternion parameterization in the Rust Ceres binding, covariance, threading, and exact solver summaries. |
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
  `--features poselib` GR6P/GR8P/GP3P paths are test-verified (280 tests). The
  default build still excludes PoseLib, so generalized rig solver claims remain
  optional-but-buildable rather than default-on parity.
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
- COLMAP `geometry/triangulation` primitive module (`triangulation.rs`): the
  two-view DLT, midpoint, multi-view DLT, Lindstrom optimal triangulation, and
  triangulation-angle functions are reproduced in `f64` and used as the shared
  triangulation backend. This is the third explicitly marked 100% parity
  boundary; it is the geometric primitive only and excludes the
  `estimators/triangulation.cc` RANSAC `EstimateTriangulation` wrapper.

## Highest-Priority Replication Targets (ROI-ordered, 2026-06-20 PM)

1. Bundle adjustment / Ceres layer (`P5`, ~47%) — largest numerical-parity
   bottleneck; caps final reconstruction accuracy.
   - Done: Ceres robust loss family (Trivial/Huber/SoftL1/Cauchy) with COLMAP's
     Cauchy default, and a Cholesky reduced-camera-matrix solve (LU fallback)
     matching Ceres' `DENSE_SCHUR`/`SPARSE_SCHUR` linear solvers.
   - Next: switch LM damping from `mu*I` to Ceres' jacobian-scaled
     `mu*clamp(diag(JᵀJ), 1e-6, 1e32)` together with Ceres' radius-based
     trust-region update (needed so weakly-constrained rig poses stay robust;
     a naive switch under the current `damping*=10` recovery regressed the
     generalized-rig refinement test).
   - Then: true sparse Schur storage, covariance, normalization, and exact
     Ceres solver summaries.
2. `sfm` triangulation + observation management lifecycle (`P4`, ~64% / ~55%)
   - Done: the `geometry/triangulation` primitive module is fully ported
     (`triangulation.rs`, 100%) and now backs two-view + multi-view track
     triangulation; the `estimators/triangulation` `TriangulationEstimator` +
     `EstimateTriangulation` LORANSAC path is ported (`triangulation_estimator.rs`,
     pending only exact RANSAC RNG parity) and is now wired into the incremental
     triangulator's `create_pair_track` track-creation path (COLMAP `Create`-style
     transitive correspondence gather + robust multi-view triangulation).
   - Next: finish exact RANSAC sampler/RNG parity in `optim`, and restructure
     `triangulate_image` to iterate per-point2D like COLMAP's `TriangulateImage`
     instead of the current pairwise loop.
   - Introduce a long-lived mapper-owned triangulator + observation manager so
     merge/retriangulation trial state and visibility stats persist across
     registration steps (6 creation sites in `mapper.rs` today).
   - Finish exact transitivity queues and official option defaults.
3. `sfm` incremental mapper (`P3`, ~71%)
   - Official COLMAP-output / real-dataset rig-frame continuation fixtures.
   - Exact initial-image and next-image priority queues.
   - Registration retry/reset semantics around generalized rig registration.
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
Ceres-equivalent behavior, since it is the dominant numerical-parity bottleneck
and is independent of the remaining optional PoseLib work. The P4 long-lived
triangulator/observation-manager lifecycle refactor is the planned follow-up.

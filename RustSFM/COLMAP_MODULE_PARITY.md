# COLMAP Module Parity Matrix

This document maps COLMAP's source modules to the current RustSFM modules and
tracks module-level reproduction progress. Percentages are engineering
estimates based on current code and `COLMAP_COMPAT_TODO.md`; they are not a
claim of bit-for-bit parity. A module is only marked 100% when there is direct
evidence that all relevant COLMAP behavior is implemented and covered by parity
tests.

Last re-evaluated: 2026-06-27, with `cargo test -p rustsfm --lib`
passing 483 tests (Ceres BA enabled by default via `ceres-ba` feature).

**flowers2 end-to-end parity re-measured 2026-06-27** (see
`PARITY_ROADMAP.md` for the full table): features exact (118006/118006),
matches exact (89/89), registration exact (24/24), BA poses within rot
0.054°/trans RMSE 0.0067. The residual sparse-SfM differences (two-view
config 0.943 agreement / 5 boundary pairs; tracks 6190 vs 6449, len-2 77 vs
240) were traced to **byte-level numerical divergence in the minimal
solvers / SVD** (`nalgebra` ≠ bit-for-bit Eigen), not logic bugs: the
incremental triangulator, `estimate_triangulation` LORANSAC, and the
calibrated two-view classifier were all line-verified against COLMAP source.
Closing them requires bit-exact Eigen-equivalent linear algebra. Ceres
BA (`ceres_problem.rs`) now supports rig/frame/sensor pose blocks, intrinsics
refinement, gauge policies, and COLMAP-style CPU linear-solver auto-selection,
and reuses native analytic reprojection Jacobians in the Ceres cost callback
(numeric fallback retained).
This pass also adds COLMAP prior-position global BA control-flow parity:
mapper global BA uses the pose-prior path without the normal two-camera gauge,
skips incremental reconstruction normalization when pose priors are active, and
is covered by a real COLMAP sparse-text scheduled global BA fixture.
A new `geometry/triangulation` primitive module (`triangulation.rs`)
faithfully ports COLMAP's `geometry/triangulation.{h,cc}` (two-view DLT,
midpoint, multi-view DLT, Lindstrom optimal triangulation, and triangulation
angles), and is now the shared backend for two-view DLT (`two_view.rs`) and
multi-view track triangulation (`incremental_triangulator.rs`). A new
`estimators/triangulation` module (`triangulation_estimator.rs`) ports the
`TriangulationEstimator` (two-view/multi-view model estimation with cheirality
+ triangulation-angle gating and angular/reprojection residuals) and the
`EstimateTriangulation` LORANSAC entry point (structurally faithful; it now
uses COLMAP's deterministic `CombinationSampler` order and recursive
LORANSAC local optimization for the triangulation estimator, and rejects
non-serial thread counts because COLMAP parallel LORANSAC only supports
`RandomSampler`, not `CombinationSampler`). Its `RANSACOptions::num_threads`
surface is still validated through the shared RANSAC option path, while mapper
triangulation options pin this path to one thread.
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
`DENSE_SCHUR`/`SPARSE_SCHUR` linear solvers. This pass also fixes mapper-wide
track filtering to reuse the long-lived `IncrementalTriangulatorState`
`ObservationManager`, so initial-pair, post-registration, local-BA, and
global-BA cleanup keep next-image visible-point statistics synchronized without
clearing triangulation trial state. Registration rollback now also refreshes
the long-lived observation manager against the restored reconstruction while
preserving merge/retriangulation trial history, and registered-frame filtering
now has real 24-image `flowers2_colmap` coverage proving it uses frame-aware
deregistration without clearing retriangulation trials while keeping long-lived
observation-manager statistics identical to a fresh rebuild after the COLMAP
20-frame filtering threshold. Real `flowers2_colmap` local/global BA
post-filter fixtures now also verify that post-BA track cleanup mutates the
same session-scoped observation manager and still matches a fresh rebuild; a
real local-BA post-merge fixture covers split-track merging in the same
long-lived state. Real `flowers2_colmap` global-BA prepare fixtures now verify
`complete_all_tracks`, `merge_all_tracks`, and pair `retriangulate` on true
shared tracks before BA while keeping the session-scoped manager in sync, and
a public `run_incremental_pipeline` fixture now verifies the seeded controller
entry reaches the initial global-BA prepare completion path. Real
`flowers2_colmap` registration rollback coverage now also exercises the
register -> retriangulate -> rollback sync path for failed local BA and
bogus-camera rejection, preserves pair retriangulation trial history, and
verifies the session-scoped observation manager matches a fresh rebuild before
and after rollback. Real post-registration filtering now
also registers a candidate through the observation-manager hook, retriangulates
true shared COLMAP tracks, corrupts a continued candidate track, and verifies
the filtering path preserves trial history while matching a fresh manager
rebuild. This pass also fixes incremental `ObservationManager::register_image`
propagation so registering a candidate does not double-count existing registered
3D-point correspondences already present in the rebuild baseline. The 100%
replication goal is
now tracked as a non-GUI COLMAP parity program, not only sparse SfM. Official COLMAP 4.x
capabilities such as GLOMAP, ALIKED, LightGlue, vocabulary-tree retrieval,
dense MVS, controller/tool coverage, and Python bindings remain in target
scope; the Qt GUI is explicitly out of scope for this target. Existing 100%
marks below remain narrow boundary claims and must not be read as
whole-application completion.

## Summary

| Priority | COLMAP module | RustSFM module(s) | Estimated parity | 100% replicated? | Notes |
| --- | --- | --- | ---: | --- | --- |
| P0 | `scene`, `sensor` core data and camera semantics | `types.rs`, `colmap.rs`, `database.rs`, `mapper.rs`, `ba.rs` | 63% | No | Camera model ids, sparse camera I/O, and COLMAP `VisibilityPyramid` scoring are complete as narrow subfeatures, and all official COLMAP camera model projection/unprojection paths are represented. Initial-pair gating and next-frame registration respect non-trivial rig requirements with known `sensor_from_rig` poses in the covered paths. Full `scene`/`sensor` behavior remains broader than the current RustSFM model: reconstruction manager behavior, reconstruction clustering/pruning, exact camera sharing/reset scheduling, pose priors, covariance, and Ceres-equivalent rig/sensor refinement are still partial or missing. |
| P0 | Sparse model I/O codec | `colmap.rs`, `types.rs` | 100% | Yes | Raw COLMAP sparse `cameras/images/points3D/rigs/frames` text and binary codecs are implemented with COLMAP-style file selection, little-endian layouts, 17-digit text precision, ids, tracks, rigs/frames, optional invalid point ids, empty rigs/frames files, and text/bin round-trip tests. Exporting from RustSFM's internal `Reconstruction` remains precision-limited by its `f32` pose/keypoint/point storage, so this 100% mark is for the file-format codec boundary, not all `scene` reconstruction semantics. |
| P1 | `feature` SIFT extraction | `sift.rs`, `wide.rs`, `vlfeat_sift.c` | 80% | No | COLMAP-style option defaults plus vendored VLFeat CPU backends: standard `SiftCPUFeatureExtractor` and covariant `CovariantSiftCPUFeatureExtractor` (affine shape + domain-size pooling via `vl_covdet`, `force_covariant_extractor`). Matching uses COLMAP uint8 L2 thresholds with indexed or brute-force CPU paths (`cpu_brute_force_matcher`). Affine keypoints are stored in `SiftFeatures.colmap_keypoints`. `benchmark-sift` CLI reports per-image feature counts. SiftGPU extraction is rejected explicitly (`use_gpu`). |
| P1 | `feature` matching | `sift.rs`, `sift_index.rs`, `database.rs`, `feature_matching.rs`, `correspondence_graph.rs` | 55% | No | Ratio/distance/cross-check/max-match controls, guided matching, COLMAP uint8 L2 distance, indexed CPU matcher (`SiftDescriptorIndex`, default) and brute-force fallback (`cpu_brute_force_matcher`), and matching-strategy selection are implemented. Verified geometry persistence and `--write-database` work. Vocab-tree pairing (`vocab_tree` strategy) is wired end-to-end. Remaining gaps: FAISS IVF acceleration for large descriptor sets, GPU matcher parity, and ONNX matcher parity. |
| P1 | Database/cache and correspondence graph | `database.rs`, `correspondence_graph.rs`, `parity.rs` | 100% | Yes | COLMAP SQLite database schema/migration/API codec, pair-id and blob direction semantics, raw keypoint/match overloads, descriptors, matches, two-view geometry payload storage, rigs/frames/pose-priors, database merge, close-time vacuum, `LoadRandomDatabaseDescriptors`, `DatabaseCache::Load/CreateFromCache`, cache add/find/count helpers, legacy trivial rigs/frames, ENU pose-prior conversion, and `CorrespondenceGraph` behavior are reproduced for the database/cache/correspondence-graph boundary. This does not include generating exact two-view geometry via COLMAP estimators or mapper use of every geometry config. |
| P2 | `estimators` / `geometry` two-view geometry | `two_view.rs`, `five_point.rs`, `polynomial.rs`, `generalized_pose.rs` | 80% | No | E/F/H estimation, homography pose, support scoring, metadata reporting/write-back, generalized relative pose preparation, and panoramic-rig relative-pose fallback are covered well in the Rust paths. PoseLib GR6P/GR8P support now builds and is test-verified under `--features poselib` (vendored `third_party/PoseLib`). Remaining gaps include exact estimator stack, byte-level Eigen/Solver parity, exact metadata semantics for every config, and complete mapper/refinement integration. |
| P2 | `geometry` triangulation primitives | `triangulation.rs` | 100% | Yes | Faithful port of COLMAP `geometry/triangulation.{h,cc}`: `TriangulatePoint` (two-view DLT via SVD), `TriangulateMidPoint` (with cheirality guard), `TriangulateMultiViewPoint` (smallest-eigenvector DLT), `TriangulateOptimalPoint` (Lindstrom optimal correction using `EssentialMatrixFromPose` + `FindOptimalImageObservations` from `essential_matrix.cc`), and `CalculateTriangulationAngle(s)` / `CalculateAngleBetweenVectors`, all in `f64` to match COLMAP's `double`. Covered by synthetic recovery/consistency tests, and wired as the shared backend for `two_view.rs` two-view DLT and `incremental_triangulator.rs` multi-view track triangulation. This 100% mark is for the triangulation primitive boundary; the `estimators/triangulation.cc` RANSAC `EstimateTriangulation` wrapper is tracked separately under P4. |
| P2 | `optim` RANSAC/LORANSAC and small solvers | `two_view.rs`, `mapper.rs`, `generalized_pose.rs`, `sprt.rs`, `support_measurement.rs`, `least_absolute_deviations.rs`, `sparse_cholesky.rs`, `RustSLAM/src/colmap_rng.rs` | 82% | No | Sampling shape, support ordering, dynamic stopping, MT19937, without-replacement trial counts, shared COLMAP `RANSACOptions` defaults/checks, signed `random_seed` option plumbing for mapper two-view E/F/H estimation, fixed-seed two-view sampler reset semantics for E/F/H RANSAC instances, constructor-time initial trial clamp, two-view prior-inlier initial trial clamping, raw `RANSAC::ComputeNumTrials` dynamic caps plus post-model zero-based abort gates for covered two-view/ray/generalized/triangulation/PnP/PNPF loops, `RANSAC::Report` shape, and `RANSAC::ComputeNumTrials` examples, COLMAP `NChooseK`, deterministic `CombinationSampler`, official `RandomSampler` initialize/max/sample API shape, `ProgressiveSampler`/PROSAC state progression, COLMAP `InlierSupportMeasurer`/`UniqueInlierSupportMeasurer`/`MEstimatorSupportMeasurer`, SPRT thresholding/evaluation, E/F/H local refit, the COLMAP LAD ADMM update/options/convergence behavior, and the `SparseCholeskyWithFallbackSolver` analyze/factorize/solve/fallback state machine are aligned for covered paths. A shared COLMAP-compatible MT19937 + libc++ `uniform_int_distribution` fixed-seed sampler now backs PnP/essential/two-view/generalized paths across crates. Sparse Cholesky currently uses RustSFM's dense `nalgebra` Cholesky + LU fallback backend rather than Eigen sparse matrices or CHOLMOD, so true sparse storage/factorization, parallel random seeding, exact nondeterministic source parity, and bit-level LORANSAC behavior are not complete. |
| P2/P3 | Absolute/generalized pose solvers | `mapper.rs`, `generalized_pose.rs`, `geometry.rs`, `colmap.rs` | 65% | No | Central absolute-pose paths are COLMAP/PoseLib-shaped for covered mapper cases, including P3P/EPNP/unknown-focal scheduling, inlier-only refinement, signed `random_seed` handling, and COLMAP post-model dynamic trial abort gates for both known-focal PnP and unknown-focal PNPF. A real COLMAP sparse-text track fixture now verifies that registered 2D-3D observations from `images.txt`/`points3D.txt` recover the COLMAP image pose through RustSFM camera unprojection plus RustSLAM PnP; a mapper-level continuation fixture additionally seeds `frame_0002.jpg` and registers `frame_0003.jpg` via `source=pnp` using true COLMAP track matches. Generalized relative/absolute pose input preparation and scoring exist, and the GR6P/GR8P/GP3P solver paths now build and pass tests under the optional `poselib` feature (vendored `third_party/PoseLib`), with the covered generalized RANSAC samplers also honoring non-negative fixed seeds versus `-1` non-fixed sampling; generalized relative now returns the LORANSAC report boundary without a RustSFM-only final GR8P refit and uses COLMAP `InlierSupportMeasurer` tie-break semantics by summing only inlier residuals, while generalized absolute keeps COLMAP's unique-inlier support ordering, uses the generic RANSAC total-inlier success gate internally, and reports `num_inliers` as unique 3D-point inliers like COLMAP. Both generalized absolute and generalized relative now apply COLMAP's post-model, zero-based dynamic trial abort gate and replace the dynamic limit from the latest best support instead of enforcing a RustSFM-only monotonic cutoff. Exact byte-level RANSAC/LORANSAC parity, covariance, Ceres-equivalent refinement, and full camera reset/refinement scheduling remain missing. |
| P3 | `sfm` incremental mapper state machine | `mapper.rs`, `parity.rs` | **89%** | No | Database-first flow, COLMAP-style initial/next-image ordering, initialization trials/relaxation, multi-model control, snapshots, color extraction, callbacks, reference-model continuation, `fix_existing_frames`, registration rollback, structure-based/structure-less two-bucket **`FindNextImages`** queue with **separate `structureless_reg_trials`**, **`CorrespondenceGraph`-based 2D-3D/2D-2D registration collection** (PnP/GP3P/structureless), rig **`MinUncertainty` max-sibling scoring**, **deterministic parallel initial-pair probing** (`threads > 1`, with order-preserving state replay), **bogus rig-frame camera reset before registration probing**, **whole registration-unit initial-pair triangulation**, **registration-commit probe camera reset**, **database pose-prior handoff into mapper BA options**, **public controller API** (`run_incremental_pipeline`, status/results, `reference_camera_setup`), **bad-initial-pair guard scoped to initial-pair path only**, **COLMAP sparse-text rig continuation fixture**, and **real COLMAP sparse-text central PnP continuation fixture** exist. The default next-image method is now back to COLMAP-shaped `MinUncertainty` and exposed through `--image-selection-method`. Remaining gaps: exact generalized-pose RANSAC byte parity, full reconstruction-manager retry semantics, and controller-level parity fixtures. |
| P3/P6 | `sfm` global mapper / rotation averaging / pose graph | `global_mapper.rs`, `view_graph_calibration.rs`, `view_graph_splitting.rs`, `joint_global_positioning.rs`, `joint_global_positioning_ceres.rs`, `rotation_averaging.rs`, `global_positioning.rs`, `track_establishment.rs`, `track_triangulation.rs`, `incremental_triangulator.rs`, `pose_graph.rs`, `mapper.rs`, `main.rs` | 85% | No | GLOMAP-shaped global pipeline through sparse reconstruction. **Orchestrator**: view-graph calibration → connected-component splitting → rotation averaging → track establishment → joint positioning (default Ceres LM) → multi-pass global BA with **`IncrementalTriangulator` complete/merge/retriangulate** before and after each BA round (`GlobalStructureRefinementStats`), reprojection filtering, observation-manager sync; `--global-mapper` CLI. **Joint positioning** (6 tests): GLOMAP ray constraints, BATA warm start, **Ceres LM** on centers+points with analytic Jacobians (`joint_global_positioning_ceres.rs`), alternating fallback. **View-graph calibration** (2 tests), **component splitting** (4 tests), rotation averaging (5), BATA (5), tracks (6), DLT triangulation (2), retriangulation integration (1), multi-component integration (1). Still missing: full controller callback parity. |
| P4 | `sfm` observation management | `observation_manager.rs`, `visibility_pyramid.rs`, `mapper.rs` | 99% | No | COLMAP-style **`ObservationManager`** with embedded **`CorrespondenceGraph`**, incremental **`SetObservationAsTriangulated`** / **`ResetTriObservations`**, **`IncrementCorrespondenceHasPoint3D`** / **`DecrementCorrespondenceHasPoint3D`**, register/deregister visible-correspondence propagation, add/delete/merge point3D ownership, 6-level **`VisibilityPyramid`** (`SetPoint`/`ResetPoint`, weighted `Score`, `MaxScore`, metadata accessors), modified-point tracking, and frame/rig register/deregister hooks are present. Session-scoped state lives in **`IncrementalTriangulatorState`**; mapper track filtering mutates that long-lived manager for covered cleanup phases, registration rollback refreshes manager stats against the restored reconstruction without clearing trial history, failed local-BA and bogus-camera registration rollback have mapper-level fresh-rebuild stats parity coverage, real COLMAP sparse-track filtering, real post-registration filtering after candidate register/retriangulate, real registration rollback after candidate register/retriangulate, real registered-frame filtering at the 24-image `flowers2_colmap` boundary, real global-BA prepare track completion/merge/retriangulation, a controller-level seeded global-BA prepare completion fixture, real local-BA post-merge, and local/global BA post-filtering have long-lived-vs-fresh observation-manager parity coverage. Remaining gap is broader real-pipeline fixture coverage for every mapper cleanup/rollback branch before this should be marked 100%. |
| P4 | `sfm` triangulation and filtering | `incremental_triangulator.rs`, `mapper.rs`, `geometry.rs`, `triangulation.rs`, `triangulation_estimator.rs`, `correspondence_graph.rs` | 99% | No | **`TriangulateImage`** per point2D (`Find`/`Continue`/`Create`) and **`CompleteImage`** per point2D (complete existing tracks + create orphan clusters with reprojection RANSAC) use **`CorrespondenceGraph::extract_transitive_correspondences`**. **`Complete`** uses direct-graph BFS with squared reprojection gating. Session-scoped trial counters persist across triangulator instances, failed local-BA and bogus-camera registration rollback, generic registration rollback, and registered-frame filtering. **`EstimateTriangulation`** uses COLMAP `CombinationSampler` deterministic two-view combination order, recursive `LORANSAC` local optimization capped at 10 inlier-expansion trials, the COLMAP post-model zero-based dynamic abort gate, and the shared `RANSACOptions::num_threads` validation surface while enforcing COLMAP's serial-only `CombinationSampler` constraint. Mapper triangulation pins `num_threads=1` even when global mapper/BA threads are configured. Initial-pair, post-registration, real post-registration filtering after candidate retriangulation, real global-BA prepare completion/merge/retriangulation, controller-level seeded global-BA prepare completion, real local-BA post-merge, real registration rollback after candidate retriangulation including bogus-camera rejection, local/global BA post-filtering, rollback, real registered-frame cleanup, and real COLMAP sparse-track filtering now preserve session-scoped visibility stats and retriangulation/merge trial state. Remaining gap is broader mapper cleanup parity fixture coverage for every real-pipeline cleanup branch before this should be marked 100%. |
| P5 | Bundle adjustment / `optim` Ceres layer | `ba/` | **88%** | No | Parameter block scheduling, analytic Jacobians (native + Ceres path reuses native projection/frame/sensor/camera Jacobians), gauges, structured Ceres iteration-summary/termination/reduced-count fields, convergence knobs, COLMAP CPU threading gate, COLMAP CPU linear-solver auto-selection (`DENSE_SCHUR` ≤50 / `SPARSE_SCHUR` ≤1000 / `ITERATIVE_SCHUR` + `SCHUR_JACOBI` above), Ceres 7D pose manifolds, COLMAP raw Eigen-quaternion residuals with exact 2x7 ambient Jacobians, Ceres-equivalent robust loss family, native Ceres-style LM trust-region damping, native PCG iterative Schur, image/frame/sensor pose-prior residual primitives with f32-stable camera-center finite differences, covariance, and solver summary fields are present. Remaining gaps: true Eigen/CHOLMOD sparse backend parity, exact Ceres summary/termination parity, broader COLMAP prior-position numerical comparisons, and large-real-reconstruction fixtures. |
| P5 | Local/global BA orchestration | `mapper.rs`, `ba.rs`, `observation_manager.rs` | 70% | No | Local/global BA scheduling, gauge handling, frame-aware options, post-registration constant-camera decisions, COLMAP incremental local/global Ceres loss and iteration defaults, local iterative refinement count/change defaults, robust-to-trivial local refinement loss handoff, COLMAP small-reconstruction global BA convergence tightening, incremental global-BA reconstruction normalization, final-all global BA no-normalization handoff, optional COLMAP redundant-point ignore/re-optimize global BA passes, mapper-level database pose priors for local/global BA, and post-BA filtering are partial. Prior-position global BA now matches COLMAP's control-flow boundary by using no explicit gauge and skipping incremental normalization when pose priors are active. Real COLMAP sparse-text continuation fixtures verify mapper PnP registration through local BA, scheduled global BA normalization, and prior-position scheduled global BA no-normalization behavior while preserving the covered COLMAP pose/track boundaries. A controller-level synthetic pipeline fixture verifies final `global_ba reason=final` runs after post-schedule reconstruction changes while skipping final-all normalization, and a real `flowers2_colmap` final-BA prepare fixture verifies final all-track completion plus session-scoped observation-manager parity. Exact image/point selection, local filtering scope, broader final-all/retriangulate-all parity, and large COLMAP-vs-RustSFM BA fixtures remain incomplete. |
| P6 | Controllers / end-to-end pipeline | `main.rs`, `mapper.rs`, `parity.rs` | 42% | No | RustSFM has a reconstruction entry point, reports, a first COLMAP-shaped multi-model pipeline slice, current-submodel overlap accounting, snapshot-frequency sparse exports, an `extract_colors` controller switch with per-registration and final all-image timing, COLMAP-shaped initial/next/last registration callback event boundaries in the pipeline log, and a lightweight public callback sink API exposed through `run_reconstruction_with_callbacks` with payload tests. It also has `fix_existing_frames` and reference-model continuation into the mapper with index-0 seed reuse only on the first continuation trial, including non-trivial rig-frame coverage through a COLMAP sparse-text fixture. Full controller-level orchestration and full parity harness are not complete. |
| P6 | `mvs` dense reconstruction | none | 0% | No | COLMAP MVS, PatchMatch, fusion, meshing, and dense workspace behavior are not replicated. |
| P6 | `exe`, `tools` (GUI excluded) | `main.rs` only | 8% | No | RustSFM has a CLI binary, but not COLMAP command/tool parity. COLMAP's Qt GUI is out of scope for this replication target. |
| P6 | `retrieval` | `retrieval.rs`, `feature_matching.rs`, `feature_matching_db.rs`, `mapper.rs`, `main.rs` | 40% | No | Pure-Rust classical vocabulary tree (hierarchical k-means + TF-IDF inverted index + L2-normalized BoW cosine query) with deterministic COLMAP MT19937 clustering, `vocab_tree`-style candidate-pair generation (`build_vocab_tree_pairs` / `generate_vocab_tree_pairs`), on-disk JSON (de)serialization (`VocabTree::save`/`load`), and end-to-end `vocab_tree_matcher` wiring into `MatchFeatures`/mapper pair generation + CLI (`MatchingPairStrategy::VocabTree`, `--matching-strategy vocab-tree`, `--vocab-tree-num-images`). Covered by 13 unit tests. Remaining gaps vs COLMAP `retrieval::VisualIndex`: FAISS-backed IVF index, Hamming embedding, and vote-and-verify spatial re-ranking. |
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
- `controllers`, `exe`, and `tools`: application orchestration, command line
  tools, and higher-level workflows. The Qt `ui` module is tracked separately
  as out of scope for the current target.
- `mvs`: dense reconstruction, PatchMatch stereo, fusion, and meshing.

## Current Overall Read

- Full non-GUI COLMAP parity, including dense MVS, tools, retrieval,
  controllers, and global mapper, remains low because several entire COLMAP
  modules have no RustSFM counterpart yet.
- Sparse SfM parity is meaningfully higher than full-COLMAP parity: database
  loading, sparse model codecs, two-view geometry metadata, generalized pose
  bridges, registration scheduling, triangulation, observation bookkeeping, and
  BA orchestration all have partial implementations.
- The PoseLib solver bridge now builds. PoseLib v2.0.5 is vendored under
  `third_party/PoseLib` and `build.rs` resolves it automatically, so the
  `--features poselib` GR6P/GR8P/GP3P paths are test-verified (414 tests).
  PoseLib remains an optional feature (not in default build), but when enabled
  COLMAP structureless registration is active via `cfg!(feature = "poselib")`
  without requiring the experimental pair-pose fallback flag.
- The highest-confidence completed areas are file/database compatibility
  boundaries, not numerical reconstruction behavior. Mapper, solver, and BA
  percentages should stay conservative until they are backed by COLMAP parity
  fixtures or exact library-level behavior.

## 100% Non-GUI Replication Program

The target is full COLMAP application parity excluding the Qt GUI, not only a
Rust sparse SfM core. Treat each phase as complete only when it has both code
parity and COLMAP-vs-RustSFM fixture coverage.

### Platform decisions (2026-06-24)

- **Python / pycolmap bindings:** out of scope.
- **CUDA / SiftGPU / COLMAP GPU matcher:** out of scope. GPU acceleration
  targets **wgpu** (Vulkan/Metal/DX12) for SIFT extraction, descriptor matching,
  PatchMatch stereo, and other compute-heavy stages.
- **Qt GUI:** out of scope.

1. Stabilize the parity baseline.
   - Keep these required test configurations green: default Ceres build,
     `--features poselib`, `--no-default-features`, CLI smoke tests, and
     database/sparse-model round trips.
   - Keep public defaults aligned with COLMAP. The mapper default next-image
     strategy is `MinUncertainty`; `--image-selection-method` exists for
     explicit overrides.
2. Finish sparse SfM parity.
   - Database pose priors now feed local/global mapper BA options for registered
     variable poses; the covered prior-position global BA path now skips the
     normal explicit gauge and incremental normalization like COLMAP.
   - Finish exact absolute/generalized pose RANSAC/LORANSAC behavior,
     metadata, and refinement scheduling.
   - Add real COLMAP reconstruction fixtures for initial-pair, registration,
     filtering, local BA, global BA, retriangulation, and multi-model flows.
3. Finish feature and retrieval parity.
   - Add vocabulary-tree retrieval and vocab-tree matching.
   - Add FAISS CPU matcher parity; add **wgpu** brute-force / IVF matcher where
     COLMAP uses GPU matching.
   - Add ALIKED and LightGlue-compatible extraction/matching paths or
     documented bindings with equivalent database outputs.
   - Implement `use_gpu` via a **wgpu** SIFT/matching backend instead of
     SiftGPU/CUDA.
4. Finish global reconstruction parity.
   - Reproduce COLMAP/GLOMAP global mapper orchestration, pose-graph ownership,
     rotation averaging, global positioning, iterative refinement, and exports.
   - Replace RustSFM-specific `pose_graph.rs` behavior with a COLMAP-compatible
     global pipeline or isolate it as non-default experimental functionality.
5. Finish dense reconstruction parity.
   - Implement COLMAP workspace layout, undistortion, **wgpu** PatchMatch stereo,
     stereo fusion, meshing, and dense output formats.
   - Add fixture-based checks against COLMAP dense outputs on small scenes.
6. Finish application-surface parity.
   - Mirror COLMAP CLI command coverage and option names where practical.
   - Add controller-level orchestration for feature extraction, matching,
     mapping, undistortion, dense reconstruction, model conversion, alignment,
     and analysis tools.
   - Qt GUI and Python/pycolmap bindings are out of scope for this target.

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

1. Bundle adjustment / Ceres layer (`P5`, ~88%) — numerical-parity backbone is strong but not yet a 100% COLMAP boundary.
   - Done: Ceres robust loss family; Cholesky/LU reduced-camera solve; `ba/` split with Ceres default path; COLMAP 7D pose manifolds; rig/frame/sensor/intrinsics blocks; gauges; analytic Jacobians; structured Ceres termination and reduced-count fields; COLMAP CPU solver auto-selection; **native Ceres-style LM damping**; **sparse Schur CSC + simplicial Cholesky**; **native ITERATIVE_SCHUR + Schur-Jacobi PCG**; **pose-prior BA** with f32-stable camera-center finite differences; **prior-position global BA gauge/normalization control-flow fixture**; **post-BA covariance**; **full solver-summary fields** (`linear_solver`, `preconditioner`, trust-region radius).
   - Next to reach 100%: exact Ceres/Eigen sparse backend behavior, broader COLMAP prior-position numerical comparisons, and large-reconstruction solver-summary parity.
2. `sfm` triangulation + observation management lifecycle (`P4`, ~99%)
   - Done: **`ObservationManager`** incremental event paths,
     **`VisibilityPyramid`** weighted COLMAP score/max-score semantics,
     **`TriangulateImage`** / **`CompleteImage`**
     per-point2D paths, COLMAP **`Complete`** BFS, **`EstimateTriangulation`**
     deterministic `CombinationSampler` order and recursive LORANSAC local
     optimization, triangulation `num_threads` validation with COLMAP's
     serial-only `CombinationSampler` constraint, mapper wiring, stateful mapper track filtering after
     initial-pair/registration/local-BA/global-BA cleanup, rollback sync
     that preserves long-lived triangulation trial history, and
     real registered-frame filtering, real global-BA prepare completion/merge/retriangulation, controller-level seeded global-BA prepare completion, real local-BA post-merge, real post-registration filtering, real registration rollback including bogus-camera rejection, and real local/global BA post-filter coverage that preserves retriangulation trials
     while matching fresh-rebuild observation-manager stats.
3. `sfm` incremental mapper (`P3`, ~89%)
   - Done: separate structureless trial counters, correspondence-graph registration collection, parallel deterministic initial-pair probing, bogus rig-frame camera reset before registration probing, whole registration-unit initial-pair triangulation, registration-commit probe camera reset, public `run_incremental_pipeline` controller API, bad-initial-pair guard scoped to initial-pair path, COLMAP sparse-text rig continuation fixture.
4. `estimators`/`geometry` two-view geometry on the PoseLib bridge (`P2`, ~80%)
   - Finish COLMAP's exact `CALIBRATED_RIG` LORANSAC behavior now that the
     PoseLib GR6P/GR8P/GP3P bridge builds.
   - Full `TwoViewGeometry` pose/config metadata parity.
5. `feature` SIFT VLFeat-equivalent backend (`P1`, ~80%)
   - Finish SiftGPU behavior, exact descriptor/orientation edge cases, and
     COLMAP-vs-RustSFM feature-count/descriptor fixtures.
6. Long-lead whole modules: `retrieval` vocab-tree (0%, also unlocks
   vocab-tree matching), global mapper / rotation averaging (~70%), `mvs`
   dense (0%), and CLI tools (~8%, GUI excluded).

## Next Implementation Choice

Next highest-ROI target: P2 two-view geometry LORANSAC parity, then P5
local/global BA orchestration fixtures for broader final-all/retriangulate-all
and large-reconstruction behavior.

# COLMAP Compatibility ToDo

Goal: reimplement COLMAP in Rust with behavior-compatible algorithms and model
semantics. The items below are ordered by dependency and expected impact on
reconstruction parity.

## P0 - Core Data And Camera Semantics

1. [done] Parse and preserve COLMAP camera model ids, names, parameter counts,
   and parameter ordering for text/binary sparse models.
2. [partial] Replace pinhole-only normalization/projection with
   COLMAP-equivalent `CamFromImg`, `CamRayFromImg`, `ImgFromCam`, and
   pixel-threshold conversion for every official camera model.
   - Implemented official projection/unprojection formulas for all 17 COLMAP
     camera models and wired reprojection/threshold paths through them.
   - Iterative undistortion now uses analytic/Jet-equivalent distortion
     Jacobians for the COLMAP distortion models instead of finite-difference
     derivatives.
   - Downstream pair geometry, stored-pose metrics, incremental triangulation,
     track triangulation, and mapper DLT reconstruction now use failable
     `CamFromImg` lifting instead of the legacy silent pinhole fallback.
   - Removed the legacy `CameraModel::normalize` helper so failed
     undistortion can no longer silently fall back to pinhole normalization.
   - Remaining strict-parity work: replace the local normalized-coordinate
     PnP/two-view/triangulation solver stack with COLMAP-equivalent estimators
     and bearing/projection semantics, including wide-FOV/fisheye edge cases.
3. [partial] Replace the single-camera reconstruction with COLMAP-style
   camera/image ownership so images can reference distinct shared cameras.
   - Added reconstruction camera table, COLMAP camera ids, image ids, and
     per-image camera references.
   - Text/binary camera reading now preserves all cameras; text export writes
     camera ids and per-image camera ownership instead of hard-coding camera 1.
   - Reference-model initialization now keeps COLMAP camera/image ids and maps
     images back to their shared cameras.
   - Database-driven mapper initialization keeps COLMAP camera/image ids and
     maps images back to database camera ownership.
   - Image-only local matching fallback now assigns a distinct camera id and
     per-image camera ownership instead of forcing all images through camera 1.
   - Reference-model two-view estimation now passes each image's camera and
     averages the normalized RANSAC threshold like COLMAP.
   - Remaining work: camera sharing/grouping for image-only local matching is
     still heuristic instead of database-driven.
4. [partial] Introduce COLMAP-compatible identifiers for cameras, images,
   points, frames, rigs, and sensors instead of relying on contiguous vector
   indices.
   - Added camera/image ids to reconstruction and image pose I/O.
   - Added point3D ids to reconstruction export so observations and points can
     preserve non-contiguous COLMAP point ids instead of hard-coding index + 1.
   - Added text/binary COLMAP image reading with 2D keypoints and optional
     point3D associations while preserving the existing pose reader API.
   - Added text/binary COLMAP `points3D` reading with point ids, RGB, error,
     and track image/feature ids.
   - Added text/binary COLMAP `rigs` and `frames` reading with sensor/data ids and
     rig/frame poses.
   - Added full text/binary sparse model loading into `Reconstruction`, preserving
     camera ids, image ids, point3D ids, image camera ownership, keypoints,
     point observations, and point tracks, with text export/read round-trip
     coverage.
   - Added a full sparse-model container that reads/writes `rigs` and `frames`
     alongside the reconstruction, with text round-trip coverage for
     rig/frame ownership data.
   - Added mapper-level rig/frame metadata and per-image frame ownership, wired
     through reference sparse models, database cache initialization, and
     COLMAP text export.
   - Added frame-aware mapper registration helpers: registering or
     deregistering one image now operates on its full COLMAP frame, preserves
     `rig_from_world`, derives sibling camera poses from known
     `sensor_from_rig` transforms, and avoids selecting same-frame images as
     initial-pair seeds.
   - Added COLMAP-shaped registration event counters for registered frames per
     rig, registered images per camera, total registration counts, and shared
     registration counts, plus parity reporting for frame ownership and
     `frame_data` coverage.
   - Local BA option construction now expands selected images to their full
     COLMAP registration frames, and global BA scheduling counts registered
     frames rather than individual frame images.
   - Registered frame poses are now synchronized from their current image poses
     after BA/pose refinement and before export, keeping `frames.txt`
     `rig_from_world` consistent with `images.txt` poses for known
     `sensor_from_rig` setups.
   - Mapper BA options now carry COLMAP-style constant rig and
     `sensor_from_rig` scheduling: explicit constant rig ids are exposed on the
     CLI, and local BA fixes non-reference `sensor_from_rig` poses when the
     local bundle does not contain every registered frame for that rig.
   - Bundle adjustment now creates a shared frame `rig_from_world` pose block
     for registered multi-image COLMAP frames instead of optimizing each frame
     image as an independent pose, and propagates the optimized frame pose back
     through fixed `sensor_from_rig` transforms.
   - Non-reference rig `sensor_from_rig` poses are now first-class BA parameter
     blocks with COLMAP-style constant-sensor scheduling, while reference
     sensors remain fixed to identity.
   - Local BA now requests a COLMAP-style `THREE_POINTS` gauge, promoting up
     to three linearly independent 3D points to constant point blocks, and
     global BA gauge-image selection now counts distinct COLMAP registration
     units instead of accidentally choosing two images from the same frame.
   - Global BA now requests COLMAP-style `TWO_CAMS_FROM_WORLD` gauge handling:
     the first observed registration unit is fixed fully, the second fixes only
     the largest-baseline translation coordinate, and degenerate cases fall
     back to `THREE_POINTS`.
   - Remaining work: generalized pose registration for non-trivial rigs,
     exact filtered-frame event counters in all mapper cleanup paths, and
     Ceres-equivalent rig/sensor solver behavior.

## P1 - Feature Extraction, Matching, And Database Graph

5. [partial] Port COLMAP SIFT extraction options and descriptor normalization
   behavior, including first octave, octave resolution, peak/edge thresholds,
   affine shape, DSP, upright mode, and max orientation handling.
   - Added COLMAP-style SIFT extraction option struct and defaults, with a
     best-effort mapping into the current `lowe-sift` backend.
   - Remaining work: replace or extend `lowe-sift` to match COLMAP
     VLFeat/SiftGPU exactly, including feature limiting by scale, affine shape,
     DSP, upright orientation, descriptor normalization, and multi-orientation
     handling.
6. [partial] Port COLMAP matching options: max ratio, max distance, cross
   check, max matches, guided matching, matching strategy selection, and
   geometric verification persistence.
   - Added COLMAP-style SIFT matching defaults and wired max ratio, max
     distance, cross check, and max matches into the current matcher.
   - Remaining work: guided matching, full matching strategy selection,
     COLMAP database persistence, and exact FAISS/GPU matcher parity.
7. [partial] Build a persistent COLMAP-style database/cache layer with
   keypoints, descriptors, matches, two-view geometries, and correspondence
   graph.
   - Added a COLMAP-style in-memory correspondence graph foundation, including
     image-pair ids, duplicate/out-of-bounds match filtering, bidirectional
     observation correspondences, finalize-time flattening, transitive
     correspondence extraction, match extraction between image pairs, and
     two-view observation checks.
   - Added SQLite COLMAP feature-table read/write support for keypoints,
     descriptors, raw matches, and two-view geometries, including COLMAP
     image-pair ordering, matrix blob encoding, optional F/E/H/qvec/tvec blobs,
     and legacy 2/4-column keypoint blobs.
   - Added a database-to-correspondence-graph construction path from stored
     keypoint counts and two-view geometries, matching the core
     `DatabaseCache` flow before mapper integration.
   - Added SQLite cameras/images table read-write support with COLMAP camera
     parameter blobs, prior-focal flags, explicit or autoincrement ids, and
     image-name lookup.
   - Added SQLite rigs/rig_sensors/frames/frame_data read-write support for
     COLMAP rig sensor ids, optional sensor-from-rig poses, and frame data ids.
   - Added SQLite pose_priors read-write support with correlated sensor/data
     ids, position/covariance/gravity blobs, and coordinate-system metadata.
   - Added a lightweight database cache loader that collects rigs, cameras,
     frames, images, pose priors, and a filtered correspondence graph using
     COLMAP-style `min_num_matches`, watermark, and `load_all_images` controls.
   - Added COLMAP-style image-name filtering at frame granularity, including
     loading all images in a selected frame and reading image frame ownership
     via `frame_data`.
   - Added legacy database-cache compatibility for databases without frames by
     creating COLMAP-style trivial frames per image.
   - Added a mapper bridge that maps database-cache image names and the
     correspondence graph into frame-indexed pair matches, with an opt-in
     two-view estimation path for database-derived candidate pairs.
   - Made the mapper database-first by default: it auto-discovers
     `database.db` next to the input images/root, loads COLMAP database cache,
     replaces frame keypoints with database keypoints, preserves database
     camera/image ownership, and estimates pairs from verified database
     correspondences. Image-only local matching now requires the explicit
     `--local-matching` fallback.
   - Database-driven pair estimation now reuses stored `qvec/tvec` two-view
     geometry poses when present and valid, falling back to local two-view
     estimation otherwise.
   - Added official COLMAP two-view geometry configuration constants and made
     database-cache edge loading reject `UNDEFINED`/`DEGENERATE` geometries
     while preserving watermark filtering semantics.
   - `PairGeometry` now carries the COLMAP two-view configuration, preserving
     database configs and marking Rust-estimated calibrated pairs explicitly.
   - Reconstruction summaries now report pair counts by COLMAP two-view
     configuration to support stage-by-stage parity checks.
   - Remaining work: SQLite COLMAP database schema/read-write parity, database
     cache ENU pose-prior conversion, and complete pose/config semantics for
     all two-view geometry cases.
8. [done] Remove sequence-specific local-window and 192-frame ring heuristics
   from the default mapper path; keep them only behind explicit experimental
   options.
   - Default mapper path now uses database-derived pair candidates when a
     COLMAP database is available.
   - Image-only local matching requires `--local-matching`; 192-frame segment
     bridge, ring closure, translation continuity, and low-parallax
     regularization require explicit experimental switches.

## P2 - Two-View Geometry Parity

9. [partial] Port COLMAP `TwoViewGeometry` configuration logic: calibrated,
   uncalibrated, homography, planar, panoramic, watermark, multiple models, and
   rig verification.
   - Added calibrated E/F/H configuration selection with COLMAP-style
     E/F/H inlier-ratio gates, H-dominant planar/panoramic classification,
     explicit forced-H estimation, watermark config preservation, and repeated
     multi-model extraction.
   - Mapper pair filtering now treats WATERMARK and MULTIPLE as verified
     geometry classes that are not used as default incremental reconstruction
     edges.
   - Remaining work: exact homography pose decomposition, exact official
     planar-vs-panoramic disambiguation, and `CALIBRATED_RIG` generalized
     relative pose.
10. [partial] Replace the local RANSAC implementation with COLMAP-equivalent
    RANSAC / LORANSAC support scoring, stopping criteria, random seeding, and
    solver selection.
    - Added configurable random seed, adaptive stopping, full-support scoring
      by default, and LORANSAC-style inlier refitting for E/F/H models.
    - Two-view E/F/H RANSAC dynamic stopping now uses COLMAP's current
      without-replacement success probability and 3x dynamic trial multiplier,
      replacing the older independent inlier-ratio power approximation.
    - Two-view E/F/H random sampling now follows COLMAP `RandomSampler`'s
      stateful partial Fisher-Yates shape over a persistent index pool instead
      of repeatedly drawing unique indices by rejection.
    - Two-view E/F/H support ordering now matches COLMAP's default
      `InlierSupportMeasurer`: more inliers win, ties are broken by smaller
      summed inlier residuals instead of median/mean residual heuristics.
    - Two-view E/F/H fixed-seed sampling now uses COLMAP's MT19937-32 random
      source plus the local libc++ `std::uniform_int_distribution<uint32_t>`
      bit-extraction behavior for `RandomSampler::Shuffle`.
    - Fundamental-matrix LORANSAC now uses a COLMAP-style seven-point minimal
      estimator with multiple rank-2 hypotheses, while keeping the eight-point
      estimator for local inlier refitting.
    - Homography RANSAC support now uses COLMAP's one-way squared forward
      projection residual (`H * x1 -> x2`) instead of the previous symmetric
      bidirectional transfer error.
    - Homography estimation now follows COLMAP's official estimator shape:
      four-point samples use the unnormalized 8x8 partial-pivot LU solve,
      larger inlier sets use the unnormalized 2N x 9 SVD nullspace with
      Eigen-style rank rejection, and singular homographies are rejected by
      the official determinant threshold.
    - Fundamental seven-point minimal estimation now uses COLMAP's raw
      unnormalized sample coordinates, explicit determinant cubic
      coefficients, and COLMAP's cubic root polishing path; it is pinned by
      COLMAP's official seven-point reference case.
    - Remaining work: replace the lightweight samplers/solvers with COLMAP's
      exact estimator stack, including byte-level official seven-point
      nullspace extraction, official eight-point fundamental parity, and
      official essential estimators.
11. [partial] Match COLMAP initial-pair checks: min inliers, max forward
    motion, triangulation-angle threshold, and generalized relative pose for
    rigs.
    - Initial pair selection now requires the strict COLMAP-style checks by
      default instead of falling back to weak candidates when no pair passes.
    - Remaining work: generalized rig initial-pair estimation and COLMAP's full
      initial-image bookkeeping/ranking.

## P3 - Incremental Mapper State Machine

12. [partial] Port COLMAP initial image selection using prior focal length and
    correspondence counts, including registration trial bookkeeping.
   - Added registration trial bookkeeping for unregistered images and stopped
     using relative-pose-only registration as a default mapper path.
   - Remaining work: exact COLMAP initial-image priority queue, prior focal
     length handling, and retry/reset semantics.
13. [partial] Port next-image ranking by visible points count, visible points
    ratio, and uncertainty score.
   - Registration candidates are now ranked primarily by visible 3D points,
     PnP inliers, visible-points ratio, and image coverage score.
   - Remaining work: exact uncertainty score and COLMAP's registration queue
     ordering.
14. [partial] Port absolute pose estimation and refinement, including focal/extra
    parameter estimation, bogus camera reset, RANSAC scheduling, and
    generalized absolute pose for rigs.
   - Partial: RustSFM now lifts absolute-pose 2D observations through the
     per-image `CameraModel::CamFromImg` before PnP, so distorted/non-pinhole
     cameras no longer enter PnP through raw pinhole intrinsics; PnP thresholds
     are also created from the registering image camera.
   - Added absolute-pose minimum inlier count checks and pose-only reprojection
     refinement before accepting a registration.
   - Added absolute-pose camera-parameter refinement for the registering
     image's focal length, principal point, and extra camera parameters using
     the mapper BA refinement flags.
   - Absolute pose now uses COLMAP's default 12px max-error threshold for PnP
     RANSAC/inlier acceptance, and skips 2D-3D correspondences contributed by
     registered images whose cameras have bogus parameters.
   - Absolute-pose camera-parameter refinement now follows COLMAP's shared
     camera scheduling: constant cameras and already-registered healthy shared
     cameras are kept fixed, while new or bogus cameras may still be refined.
   - Registering images now keep a mapper-level copy of input camera priors and
     reset a bogus candidate camera to its healthy prior before PnP or
     structure-less registration attempts, matching the core COLMAP database
     camera reset behavior.
   - Candidate images whose shared camera is not yet used by any registered
     image now start registration from the healthy input prior even if the
     current mutable camera block has drifted but is not yet formally bogus;
     already-registered healthy shared cameras remain fixed for later images.
   - Absolute-pose refinement now uses only the RANSAC inlier observations,
     matching COLMAP's inlier-mask refinement path, and camera-parameter
     refinement failures or bogus refined cameras reject the registration
     instead of silently accepting the unrefined pose/camera.
   - COLMAP database `prior_focal_length` is now preserved in the mapper camera
     setup and used in absolute-pose focal-estimation scheduling; cameras
     without prior focal length and without registered shared-camera support
     now use an in-solver unknown-focal RANSAC path instead of the old coarse
     mapper-level focal hypothesis set before PnP.
   - Absolute-pose PnP now receives COLMAP-style RANSAC option plumbing for
     `abs_pose_min_inlier_ratio`, 0.99999 confidence, 100/10000 trial bounds,
     COLMAP's dynamic trial-count formula, and explicit non-negative
     `random_seed` values from the mapper CLI/config; the inlier-ratio option
     is used for RANSAC trial budgeting rather than as an extra final
     registration gate.
   - Replaced the local absolute-pose non-minimal DLT rescue/refit path with a
     COLMAP/PoseLib-style P3P minimal estimator, EPNP local estimator,
     LORANSAC-style recursive inlier re-estimation, and COLMAP
     inlier-support ordering by inlier count followed by squared residual sum.
   - Unknown-focal absolute pose now has a COLMAP-shaped RANSAC path that
     samples four correspondences, scores centered-pixel residuals, and returns
     pose plus a shared focal length. Its model generator now uses a
     COLMAP/PoseLib-style P4PF algebraic solver with `re3q3`/Sturm roots first,
     then falls back to the earlier P3P/focal-update bridge for numerical
     coverage.
   - Absolute-pose RANSAC sampling now uses COLMAP's partial Fisher-Yates
     sampler shape instead of sorted unique indices, and mapper-level
     `random_seed = -1` no longer collapses to the solver's deterministic
     input-hash seed.
   - The PnP RANSAC path now uses an internal MT19937-32 generator with
     rejection-based uniform integer sampling instead of the earlier LCG/modulo
     sampler.
   - Post-registration writes are now guarded by a reconstruction snapshot:
     failed required local BA or bogus registered cameras roll the candidate
     image/camera/track changes back and count as a failed registration trial.
   - Remaining work: validate exact sample sequences against COLMAP's target
     standard-library `std::uniform_int_distribution` behavior if byte-for-byte
     fixed-seed parity is required, then finish full frame/rig-level
     refinement/reset scheduling, deregistration event counters, and
     generalized rig pose.
15. [partial] Port structure-less registration fallback.
   - The earlier pair-pose-derived structure-less registration path is now
     explicit experimental behavior only
     (`--experimental-structureless-pair-pose-fallback`) and is not used by the
     default COLMAP-compatibility path.
   - The default mapper path now follows the COLMAP `RegisterNextStructureLessImage`
     boundary: require at least two registered images, enforce the
     `2 * abs_pose_min_num_inliers` 2D-2D correspondence gate, skip bogus
     registered-neighbor cameras, collect query/world 2D correspondences with
     world camera indices, and call a Rust `EstimateStructureLessAbsolutePose`
     entry point.
   - Structure-less registration now carries the supporting 2D-2D inlier
     correspondences into the registration step and uses them to first continue
     existing 3D tracks, then group remaining inliers by query feature and
     triangulate new tracks, matching COLMAP's post-estimation
     continue-or-triangulate flow at the mapper level.
   - Structure-less support is now filtered per 2D-2D correspondence with a
     final-pose normalized Sampson gate before those correspondences are used
     for track continuation/triangulation, reducing the previous whole-pair
     inlier-mask approximation.
   - Added a conservative fixed-center Sampson rotation refinement for the
     structure-less fallback. It lowers 2D-2D Sampson cost while preserving the
     pair-pose-derived camera center and rejecting updates that degrade
     verified-pair rotation consistency.
   - Removed the remaining default adjacent-pair-only rotation-consistency
     check in next-image registration so database/non-local verified pairs can
     participate like COLMAP correspondences.
   - Remaining work: finish the real COLMAP `EstimateStructureLessAbsolutePose`
     implementation by porting `LORANSAC<GR6PEstimator, GR8PEstimator>`,
     including PoseLib's GR6P solver, COLMAP's GR8P local estimator, Sampson
     residual support scoring, random sampler semantics, and the official
     absolute-pose camera reset/refinement schedule.

## P4 - Triangulation And Observation Management

16. [partial] Port `ObservationManager` for adding/removing/merging observations and
    tracking modified points.
   - Added visible-point/correspondence statistics used by the mapper
     registration ranking.
   - Added COLMAP-style point/observation add-delete-merge ownership, including
     image-pair stat refresh, visible point/correspondence counters,
     modified-point tracking, stable point-id maintenance, and whole-point
     deletion for two-view observations.
   - Routed mapper batch track reconstruction, reprojection/depth/triangulation
     filtering, and legacy pair triangulation through the observation manager
     instead of rewriting reconstruction point/observation tables directly.
   - Added register/deregister image bookkeeping, including visible
     correspondence refresh and deregistration-time observation deletion.
   - Extended registration/deregistration hooks to operate at COLMAP frame
     granularity, deleting observations and refreshing correspondence stats
     for every image in the frame, with rig sensor pose propagation for known
     `sensor_from_rig` setups.
   - Remaining work: keep a longer-lived mapper-level observation manager and
     finish exact COLMAP frame/rig semantics for generalized pose, BA
     scheduling, and all filtering/deregistration cleanup events.
17. [partial] Port `IncrementalTriangulator` create/continue/merge/complete/retriangulate
    behavior, transitivity, angular/reprojection thresholds, and two-view-track
    handling.
   - Added create/continue angular error gates, explicit two-view-track
     suppression in default triangulator options, track re-triangulation after
     continue/merge, and best-baseline track re-estimation instead of
     averaging merged points.
   - The mapper now filters reprojection outliers after each incremental
     triangulation step.
   - Remaining work: exact COLMAP transitivity queues, retriangulation trial
     limits per image pair, and official point creation/continuation option
     defaults.
18. [partial] Port filtering by negative depth, reprojection error, triangulation angle,
    short tracks, and bogus camera parameters.
   - Reprojection filtering is now part of the default incremental loop.
   - Added COLMAP-style observation filtering for negative depth, whole-point
     filtering for small triangulation angles, configurable short-track
     pruning, and post-filter point re-triangulation/error refresh.
   - Added COLMAP-style bogus camera parameter checks for focal ratio,
     principal point bounds, and extra parameter magnitude; mapper filtering
     removes observations from cameras that violate these limits.
   - Absolute-pose registration now rejects images with bogus cameras before
     PnP, and mapper BA rejects or rolls back runs that start/end with bogus
     registered cameras.
   - Remaining work: exact COLMAP frame/rig-level deregistration counters after
     failed refinements and exact filtering schedules for local/global BA
     phases.

## P5 - Bundle Adjustment And Refinement

19. Replace the experimental BA path with a Ceres-equivalent optimizer layer or
    a numerically equivalent Rust implementation, including analytic residuals.
    - Added a scoped BA interface that can optimize selected registered images
      and selected 3D points, enabling local bundle adjustment without
      perturbing the full reconstruction on every image registration.
    - BA configuration now supports explicit constant images, so local BA can
      keep the mapper gauge image fixed instead of relying only on image index
      conventions.
    - BA configuration now supports explicit variable/constant camera parameter
      blocks for focal length, principal point, and extra camera model
      parameters; the mapper now exposes COLMAP-style BA intrinsic refinement
      defaults and `--ba-constant-camera-id` CLI controls.
    - BA now supports constant 3D point ids: fixed points still contribute
      residuals to pose/camera updates but their coordinates are not optimized.
    - BA reports now expose COLMAP-style solver-agnostic termination type,
      termination reason, reduced residual count, attempted/successful/
      unsuccessful iteration counts, and failure counters. Mapper local/global
      BA logs use these fields and reject non-usable failure summaries.
    - BA options now mirror COLMAP/Ceres default convergence controls for
      maximum iterations, function/gradient/parameter tolerances, maximum
      linear solver iterations, consecutive invalid steps, and consecutive
      nonmonotonic steps. The Rust solver uses these controls to classify
      convergence, no-convergence, and failure summaries.
    - BA now uses analytic pose/point Jacobians for SIMPLE_PINHOLE, PINHOLE,
      SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV, FOV, SIMPLE_FISHEYE,
      FISHEYE, SIMPLE_RADIAL_FISHEYE, RADIAL_FISHEYE, OPENCV_FISHEYE,
      THIN_PRISM_FISHEYE, RAD_TAN_THIN_PRISM_FISHEYE, SIMPLE_DIVISION,
      DIVISION, and EUCM reprojection residuals.
    - BA now uses analytic camera-intrinsic Jacobians for SIMPLE_PINHOLE,
      PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV, FOV,
      SIMPLE_FISHEYE, FISHEYE, SIMPLE_RADIAL_FISHEYE, RADIAL_FISHEYE, and
      OPENCV_FISHEYE, THIN_PRISM_FISHEYE, RAD_TAN_THIN_PRISM_FISHEYE,
      SIMPLE_DIVISION, DIVISION, and EUCM focal/principal-point/distortion
      parameters.
    - BA now uses analytic chain-rule Jacobians for COLMAP frame
      `rig_from_world` and non-reference `sensor_from_rig` pose blocks,
      replacing the finite-difference rig/sensor pose derivatives in the
      active frame-aware BA path.
    - BA pose updates now follow COLMAP/Ceres parameter-block semantics:
      quaternion manifold updates for rotation and direct Euclidean updates for
      translation, instead of coupled SE(3) exponential updates.
    - BA termination now separates Ceres-style gradient and parameter tolerance
      exits and reports reduced effective parameter counts, gradient max-norm,
      and step norm in the solver summary.
    - BA now applies the configured linear-solver iteration budget to its
      solver wrapper and reports accumulated linear-solver iterations.
    - BA step acceptance now computes a trust-region-style actual/predicted
      decrease ratio, exposes it in the solver summary, and adjusts damping
      from step quality instead of using a fixed reduction on every accepted
      step.
    - Mapper local/global BA option construction now applies COLMAP
      incremental-pipeline Ceres convergence settings: local BA uses gradient
      tolerance 10 and global BA uses gradient tolerance 1, both with 100
      linear-solver iterations and zero parameter tolerance.
    - Remaining work: replace hand-rolled LM with Ceres-equivalent solver
      behavior, robust trust-region/linear-solver behavior, and full backend
      solver-summary parity.
20. Match COLMAP local BA image selection, gauge fixing, robust losses,
    constant camera/rig controls, and short-track point selection.
    - Added a first COLMAP-style local BA pass after each successful
      registration: select images by shared 3D points with the newly
      registered image, keep the initial gauge image fixed, optimize local
      points, then run track filtering.
    - Local bundle selection now ranks shared-point neighbors, applies a
      COLMAP-like triangulation-angle threshold relaxation schedule, and limits
      variable local points to new/short-track points by default.
    - After local BA, the mapper now runs variable-point merge, track
      completion, a new-image completion pass, and then track filtering, closer
      to COLMAP's local BA post-processing order.
    - Long-track local points are now passed to local BA as constant points
      instead of being dropped from the local BA residual set.
    - Local BA now adds all images belonging to selected local COLMAP frames and
      fixes shared camera intrinsics when the local image set does not contain
      every currently registered image for that camera, matching COLMAP's
      per-camera local BA scheduling boundary.
    - Frame `rig_from_world` metadata is refreshed after successful BA so
      exported sparse models preserve the current frame pose instead of the
      original registration pose.
    - Local BA now applies COLMAP's rig coverage boundary for
      `sensor_from_rig`: explicit constant rigs and rigs only partially covered
      by the local frame set keep all non-reference rig sensor poses fixed.
    - BA now uses shared frame pose blocks for registered COLMAP frames, keeping
      all images in the same frame tied to `sensor_from_rig * rig_from_world`
      during optimization instead of repairing frame consistency only after BA.
    - Non-reference `sensor_from_rig` poses are optimized as dedicated BA
      parameter blocks unless marked constant by COLMAP's constant-rig or
      partial-rig local BA rules.
    - Local BA now uses a `THREE_POINTS` gauge policy that fixes three
      linearly independent observed 3D points when possible.
    - Remaining work: exact COLMAP local bundle selection, formal gauge
      strategies, and Ceres-equivalent backend behavior.
21. Match COLMAP global BA scheduling, convergence settings, optional redundant
    point handling, pose priors, normalization, and iterative refinement loops.
    - Added COLMAP-style global BA scheduling after initialization, during
      incremental registration when registered image/point growth crosses
      frequency or ratio thresholds, and at finalization if the reconstruction
      changed since the last global pass.
    - Global BA now uses an explicit gauge by fixing the first two registered
      images and runs global track completion, merge, retriangulation prep, and
      post-BA filtering/refinement-change loops with debug reporting.
    - CLI options now expose global BA enablement, iteration count, image/point
      frequency and ratio triggers, maximum refinements, and refinement-change
      threshold.
    - Global BA image-growth scheduling now uses registered frame counts so
      multi-camera frames do not prematurely trigger global BA merely because
      they contain multiple images.
    - Global BA options now preserve explicit constant rig ids and their
      non-reference `sensor_from_rig` constant set for the future
      rig-parameterized BA backend.
    - Global BA gauge selection now chooses images from distinct registration
      frames/units before fixing pose blocks, avoiding same-frame multi-camera
      images as duplicate gauge anchors.
    - Global BA now uses COLMAP-style `TWO_CAMS_FROM_WORLD` gauge handling
      rather than fixing two full image poses: the first observed registration
      unit is fixed, the second keeps rotation and two translation axes
      variable, and insufficient-baseline cases fall back to `THREE_POINTS`.
    - Remaining work: reconstruction normalization, pose priors, redundant 3D
      point pruning, and Ceres-equivalent convergence settings.

## P6 - I/O, Multi-Model Pipeline, And Parity Tests

22. Read/write full COLMAP sparse text and binary models, including cameras,
    images, points, rigs, frames, and database-derived data where applicable.
23. Port COLMAP incremental pipeline behavior for sub-model creation, overlap
    limits, min model size, snapshots, final refinement, and color extraction.
24. Build parity test fixtures that compare RustSFM against official COLMAP on
    synthetic and real datasets at each stage: camera I/O, feature counts,
    matches, two-view geometry, registration order, tracks, BA cost, and final
    sparse model.

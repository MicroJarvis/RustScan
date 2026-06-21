# COLMAP Compatibility ToDo

Goal: reimplement COLMAP in Rust with behavior-compatible algorithms and model
semantics. The items below are ordered by dependency and expected impact on
reconstruction parity.

## P0 - Core Data And Camera Semantics

1. [done] Parse and preserve COLMAP camera model ids, names, parameter counts,
   and parameter ordering for text/binary sparse models.
   - Added a raw COLMAP sparse file codec for `cameras/images/points3D` plus
     optional `rigs/frames` in both text and binary formats, matching COLMAP
     file-selection precedence, little-endian field layouts, 17-digit text
     precision, invalid point3D sentinels, image names through end-of-line,
     deterministic id-based write ordering, empty rig/frame sidecar files, and
     text/binary round-trip coverage.
   - This is the first explicitly marked 100% parity boundary:
     `Sparse model I/O codec`. It does not imply full `scene`/`sensor`
     semantic parity, because exporting through RustSFM's internal
     `Reconstruction` still inherits existing `f32` pose/keypoint/point
     precision limits.
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
   - Added generalized absolute-pose registration for non-trivial COLMAP rig
     frames with known `sensor_from_rig` poses: mapper candidates now collect
     whole-frame 2D-3D correspondences, estimate `rig_from_world`, expand the
     pose back to every frame image, and continue inlier 3D tracks.
   - Generalized frame registration now runs a BA-backed pose-only refinement
     after GP3P, using the RANSAC inlier mask with fixed 3D points, fixed
     cameras, and fixed `sensor_from_rig` poses before expanding the refined
     `rig_from_world` back to all frame images.
   - Generalized frame registration now follows COLMAP's good-focal gate: a
     non-trivial rig only uses generalized absolute pose when every frame
     camera has prior focal length or has already been registered through the
     same shared camera. Otherwise the mapper falls back to central absolute
     pose so unknown focal length can be estimated first.
   - Central registration now resets bogus cameras in the same COLMAP frame
     from healthy database/reference priors, matching COLMAP's recovery path
     for non-trivial rigs with bad sibling camera parameters.
   - Mapper filtering now follows COLMAP's registered-frame filtering gate for
     the covered cleanup path: once at least 20 registration units are
     registered, frames/images with bogus cameras or zero point3D observations
     are deregistered at full-frame granularity and the registered-frame,
     per-camera, total-registration, and shared-registration counters are
     rolled back through the same event path.
   - Next-image selection now follows COLMAP's two-bucket retry ordering for
     the covered mapper path: clean never-tried registration units are ranked
     first, while filtered or previously failed units are ranked only after the
     clean bucket.
   - Candidate ranking now exposes COLMAP's next-image selection methods
     internally and defaults to `MIN_UNCERTAINTY`, i.e. the mapper ranks by
     `Point3DVisibilityScore`; the `MAX_VISIBLE_POINTS_NUM` and
     `MAX_VISIBLE_POINTS_RATIO` formulas are also implemented.
   - The covered next-image path now has a COLMAP-shaped `FindNextImages`
     queue split: structure-based candidates are selected by visible 3D-point
     threshold, sorted into clean and filtered/failed buckets, and only then
     consumed by the registration solver; structure-less candidates use the
     visible-correspondence ranking path when that fallback is enabled.
     A parity test verifies the bucketed queue independently of PnP success.
   - The covered next-image queue consumer now matches COLMAP's core
     "could not register, trying another image" behavior: when a queued
     candidate reaches the registration solver but fails pose estimation, the
     candidate is recorded as a failed attempt, its full registration unit
     trial count is incremented by the pipeline, and later candidates in the
     same ranked queue can still register successfully.
   - Frame-level retry gating now reads the maximum trial count over the full
     registration unit, so any sibling image in a non-trivial frame reaching
     `max_reg_trials` blocks the whole frame consistently.
   - Remaining work: exact generalized absolute-pose reset/refinement
     scheduling, full priority-queue orchestration and COLMAP
     initialization/reconstruction retry bookkeeping, event-counter coverage
     for every failed-refinement cleanup path, and Ceres-equivalent rig/sensor
     solver behavior.

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
   - Added guided SIFT matching with epipolar-line filtering and
     post-estimation geometry refinement when more inliers are recovered.
   - Added COLMAP-style matching pair strategies: exhaustive, sequential
     (overlap/quadratic overlap/loop detection), and local window, wired
     through mapper pair-graph construction and CLI flags.
   - Added `--write-database` for local-matching fallback: creates/populates
     a COLMAP SQLite database with cameras, images, keypoints, descriptors,
     raw matches, and verified two-view geometries.
   - Remaining work: vocab-tree pairing, exact FAISS/GPU matcher parity, and
     camera sharing heuristics for image-only local matching.
7. [done] Build a persistent COLMAP-style database/cache layer with
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
   - Added COLMAP-style SQLite open-time schema migration:
     `inlier_matches` is renamed to `two_view_geometries`, legacy
     `pose_priors(image_id, ...)` rows are migrated to correlated
     sensor/data pose priors, missing two-view `F/E/H/qvec/tvec` and
     descriptor `type` columns are added, old identity/zero pose sentinels and
     zero F/E/H matrices are normalized to NULL, and `PRAGMA user_version` is
     updated to the current COLMAP database version.
   - Added COLMAP database API coverage for exists/count/max/rows-for-image,
     `ReadRigWithSensor`, `ReadNumMatches`, update methods for
     rigs/cameras/frames/images/pose_priors/keypoints/two-view geometries,
     pair deletion, inlier-match deletion, clear methods, and transactions.
   - Added COLMAP-style `Database::Merge`, including camera/rig/image/frame
     id remapping, duplicate image-name rejection, pose-prior data-id remapping,
     keypoint/descriptor transfer, match transfer through COLMAP's `rows > 0`
     all-matches API, and two-view geometry transfer through remapped image
     pairs.
   - Added raw keypoint and match blob overloads for COLMAP-compatible
     `Read/Write/UpdateKeypointsBlob`, `ReadMatchesBlob`, and
     `ReadAllMatchesBlob` behavior, preserving matrix rows/cols/data and
     match-pair direction swapping.
   - Added COLMAP-style close-time vacuum behavior: delete/clear methods mark
     the database as having removed entries, and explicit close/drop runs
     `VACUUM` to release SQLite freelist pages.
   - Added COLMAP-style `LoadRandomDatabaseDescriptors` helper behavior for
     all-descriptor and bounded random-subset loads, including feature-type and
     descriptor-dimensionality consistency checks.
   - `WriteTwoViewGeometry` now follows COLMAP's insert-only behavior, while
     mapper write-back uses explicit update semantics when replacing an
     existing verified pair.
   - Added a lightweight database cache loader that collects rigs, cameras,
     frames, images, pose priors, and a filtered correspondence graph using
     COLMAP-style `min_num_matches`, watermark, and `load_all_images` controls.
   - Added `DatabaseCache::CreateFromCache` parity, copying all images in
     selected frames, filtered rigs/cameras/frames, pose priors, and full
     two-view metadata/inlier matches into a rebuilt correspondence graph.
   - Added COLMAP-style `DatabaseCache` API helpers for count, add, lookup,
     existence checks, image-name search, and correspondence-graph access.
   - Added COLMAP-style opt-in database-cache pose-prior conversion from WGS84
     latitude/longitude/altitude to local ENU Cartesian coordinates, including
     consistent coordinate-system validation and first-WGS84-prior reference
     origin behavior.
   - Added COLMAP-style image-name filtering at frame granularity, including
     loading all images in a selected frame and reading image frame ownership
     via `frame_data`.
   - Added legacy database-cache compatibility for databases without frames by
     creating COLMAP-style trivial rigs per camera and trivial frames per image.
   - The correspondence graph now preserves full two-view geometry metadata
     (`config`, optional F/E/H/qvec/tvec) separately from inlier matches, with
     COLMAP-style inversion/update/extraction semantics.
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
   - Added official COLMAP two-view geometry configuration constants. Database
     cache filtering now matches COLMAP `UseInlierMatchesCheck`: only
     `min_num_matches` and optional watermark skipping are applied, so
     `UNDEFINED`/`DEGENERATE` rows with enough inlier matches are still loaded.
   - `PairGeometry` now carries the COLMAP two-view configuration, preserving
     database configs and marking Rust-estimated calibrated pairs explicitly.
   - Reconstruction summaries now report pair counts by COLMAP two-view
     configuration to support stage-by-stage parity checks.
   - This is the second explicitly marked 100% parity boundary:
     `Database/cache and correspondence graph`. The boundary covers storage,
     cache construction/filtering, and graph behavior; exact generation of
     two-view geometry by COLMAP's estimator stack remains in P2.
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
   - Homography-dominant planar/panoramic disambiguation now follows COLMAP's
     homography-pose path: decompose normalized H, choose the cheirality/bearing
     reprojection winner with midpoint triangulation, and classify
     `PLANAR_OR_PANORAMIC` by the selected pose translation norm instead of
     the previous closest-rotation residual heuristic.
   - Homography decomposition and pose selection are now pinned by COLMAP's
     official nominal `DecomposeHomographyMatrix` and `PoseFromHomographyMatrix`
     reference fixtures.
   - Planar and panoramic two-view estimates now hand off the selected
     homography pose into the returned relative pose instead of falling back to
     essential/fundamental pose selection, including the forced-homography path.
   - Rust-estimated pairs now carry COLMAP-shaped optional F/E/H/qvec/tvec
     metadata from `TwoViewEstimate` into `PairGeometry`, and mapper summaries
     report verified-pair metadata coverage for parity checks.
   - Added opt-in `--write-two-view-geometries` write-back for Rust-estimated
     verified pair metadata, including COLMAP-style replacement and
     `TwoViewGeometry::Invert` behavior for swapped image ids.
   - Added the COLMAP-style generalized relative pose preparation bridge:
     RANSAC defaults, camera/ray packing, panoramic-rig ray handling, original
     camera pose recomposition from rig-relative pose, and mapper initial-pair
     gating for non-trivial rigs.
   - Added an optional PoseLib-backed GR6P generalized relative pose solver
     bridge behind the `poselib` Cargo feature. The Rust side packs
     COLMAP-style rig observations into PoseLib's 6-point solver, scores
     candidate rig poses with generalized Sampson residuals, and returns a
     rig-relative pose plus inlier mask for non-panoramic rigs.
   - Added a COLMAP-derived GR8P local estimator bridge behind the same
     feature. Current generalized relative pose RANSAC now refits the best
     inlier sets with GR8P, re-scores the returned rig poses, and keeps the
     stronger support, matching the main shape of COLMAP's
     `LORANSAC<GR6PEstimator, GR8PEstimator>` path without reimplementing
     the solver from scratch in Rust.
   - Tightened generalized relative pose parity further: normalized RANSAC
     threshold is now averaged over all rig cameras like COLMAP, dynamic
     stopping uses COLMAP's current without-replacement trial formula, GR8P
     local optimization can recursively expand the inlier set up to COLMAP's
     10 local trials, and the panoramic-rig branch now falls back to ordinary
     relative pose and returns `pano2_from_pano1`.
   - `EstimateStructureLessAbsolutePose` is now wired through the same
     generalized relative pose solver path, matching COLMAP's world-rig to
     query-camera formulation for non-panoramic world rigs.
   - Remaining work: finish exact metadata semantics for every geometry config
     and close the remaining bit-level solver/random perturbation semantics,
     generalized absolute pose registration/refinement, and mapper
     rig-registration integration behind `CALIBRATED_RIG`.
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
    - Shared `ColmapRandomSampler` now exposes the official COLMAP
      constructor/`Initialize`/`MaxNumSamples`/`Sample` shape in addition to
      the existing RustSFM indexed-pool helper API, with official less-samples
      and equal-samples uniqueness tests plus oversized-initialize rejection.
    - Shared `colmap_ransac_num_trials` now ports COLMAP
      `RANSAC::ComputeNumTrials` with explicit minimal sample size and the full
      official `ransac_test.cc` numeric examples. RustSLAM's PnP/essential
      solver wrapper plus RustSFM's two-view, triangulation-estimator, and
      generalized-pose RANSAC paths now use this shared helper instead of
      separate local formulas.
    - Added shared `ColmapRansacOptions` with COLMAP's default surface and
      validation bounds (`max_error`, inlier ratio, confidence,
      min/max trials, random seed, and thread count). Generalized pose
      RANSAC options now derive their defaults from the shared COLMAP surface.
      The shared default `max_num_trials` now follows COLMAP's signed
      `int` default (`2147483647`) rather than Rust's platform `usize` max.
    - Added shared `ColmapRansacReport` and the COLMAP constructor-time
      `max_num_trials` clamp from `min_inlier_ratio`; generalized absolute and
      generalized relative pose RANSAC now use that shared initialization path.
      `EstimateTriangulation` also maps its flattened options through the
      shared COLMAP `RANSACOptions` initialization before enumerating
      `CombinationSampler` trials.
      The panoramic/generalized two-view ray relative-pose fallback now uses
      the same shared initialization instead of a local pre-clamp formula.
      RustSLAM's PnP/PnPF RANSAC initial trial budget now maps through the
      shared COLMAP options initialization as well.
      Two-view E/F/H RANSAC now maps the COLMAP
      `TwoViewGeometryOptions` inlier prior (`ransac_options.min_inlier_ratio`
      default 0.25, overridden by top-level `min_inlier_ratio` when enabled)
      through the same constructor-time initial trial clamp.
      Shared `ColmapRansacOptions::dynamic_max_num_trials` now centralizes the
      post-best-model dynamic trial cap (`ComputeNumTrials`, min-trial floor,
      max-trial ceiling); two-view E/F/H and the ray relative-pose fallback use
      that shared COLMAP option path instead of local count wrappers.
      Two-view E/F/H dynamic stopping now honors COLMAP's
      `TwoViewGeometryOptions` `ransac_options.min_num_trials = 100` floor,
      clamped to the configured maximum trial budget, so high-support samples
      cannot collapse the loop below COLMAP's default minimum trial count.
      Two-view pair estimation now accepts and propagates COLMAP's signed
      `RANSACOptions::random_seed` semantics from the mapper config: `-1`
      keeps the default non-fixed sampling behavior, while non-negative seeds
      use a reproducible fixed sampler seed for the covered E/F/H paths.
    - Two-view E/F/H support ordering now matches COLMAP's default
      `InlierSupportMeasurer`: more inliers win, ties are broken by smaller
      summed inlier residuals instead of median/mean residual heuristics.
    - Ported COLMAP `optim/support_measurement` as shared Rust infrastructure:
      `InlierSupportMeasurer`, `UniqueInlierSupportMeasurer`, and
      `MEstimatorSupportMeasurer` now have COLMAP nominal tests, and
      `EstimateTriangulation` uses the shared inlier support measurer.
    - Ported COLMAP `optim/sprt` as `sprt.rs`: options, decision-threshold
      recurrence, absolute-residual early rejection, inlier/evaluation counters,
      and official all-inlier/all-outlier/mixed/empty tests are covered.
    - Ported COLMAP `optim/least_absolute_deviations` as
      `least_absolute_deviations.rs`: LAD ADMM options, validity checks,
      shrinkage update, primal/dual convergence thresholds, ridge
      regularization behavior, and the official overdetermined,
      well-determined, underdetermined, diagonal, identity, outlier, and ridge
      tests are covered. The Rust implementation currently uses a dense
      `nalgebra` Cholesky backend for both COLMAP solver-type variants.
    - Ported COLMAP `optim/sparse_cholesky` as a dense-backed compatibility
      wrapper in `sparse_cholesky.rs`: `Compute`, `AnalyzePattern`,
      `Factorize`, `Solve`, Cholesky-first state, fallback state, diagonal,
      chain Laplacian, reused-pattern, singular, ridge, indefinite fallback,
      and ill-conditioned-chain tests are covered. LAD's
      `SupernodalCholmodLlt` variant now uses this wrapper, so it has a real
      fallback path distinct from strict `SimplicialLlt`.
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
    - Fundamental/essential eight-point support now uses COLMAP's RMS
      point-normalization convention for F, COLMAP-shaped raw camera-ray
      estimation for E, and official COLMAP reference coverage for the
      eight-point F solver.
    - Five-point essential estimation now keeps COLMAP's generated polynomial
      expressions and filters companion-matrix roots with the official
      imaginary-part threshold instead of Rust-only sorting/deduplication and
      residual heuristics.
    - Five-point nullspace extraction now follows COLMAP's solver branch
      structure: exactly five camera rays use a Householder-Q nullspace from
      `Q^T`, while larger support sets use right-singular vectors with an
      explicit full-`V^T` completion for wide matrices instead of the previous
      `Q^T Q` eigenvector basis.
    - Fundamental/essential eight-point minimal nullspace extraction now
      reconstructs COLMAP's full Householder-Q column for exactly eight
      correspondences instead of using the previous explicit free-variable
      nullspace solve.
    - Fundamental seven-point minimal nullspace extraction now also uses a
      full Householder-Q reconstruction for the two-dimensional nullspace
      instead of the previous `A^T A` eigenvector basis.
    - Remaining work: replace the lightweight samplers/solvers with COLMAP's
      exact estimator stack, including byte-level full-pivot Householder
      parity for minimal nullspaces, byte-level Eigen JacobiSVD / companion
      root parity, and replacing the current dense `nalgebra` sparse-Cholesky
      compatibility wrapper with true sparse matrix storage, sparse
      factorization, and CHOLMOD-backed fallback behavior.
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
   - Initial-pair selection now follows COLMAP's two-stage ordering for the
     covered mapper path: first images are sorted by prior focal length and
     total correspondence count; second images are sorted by prior focal
     length and pair correspondence count; selected pair geometry is oriented
     to the chosen first/second image order.
   - Added COLMAP-style initialization bookkeeping for the covered single-model
     mapper path: `init_max_reg_trials` defaults to 2, first initial images are
     skipped after reaching the initialization trial limit, images already
     counted as registered are excluded from initial-pair seeding, tried
     initial pairs are suppressed by COLMAP image-pair id, and the
     corresponding initialization thresholds are exposed on the CLI.
   - Initialization bookkeeping now lives in a mapper-session state that can
     persist across reconstruction attempts: tried initial pairs and
     initialization trial counts survive failed attempts, successful kept
     reconstructions add COLMAP-style registration counts, discarded
     reconstructions roll those counts back, and `ResetInitializationStats`
     semantics are available for COLMAP's later relaxation passes.
   - Added COLMAP-shaped `init_num_trials` and initialization relaxation
     scheduling: strict initialization uses the configured thresholds, then
     two relaxation rounds reset initialization stats and alternately halve
     `init_min_num_inliers` and `init_min_tri_angle_deg`, matching the
     controller-level order in COLMAP.
   - Added COLMAP-style bad-initial-pair retry behavior for the covered
     initialization path: initial reconstructions that produce no/too few 3D
     points after triangulation or post-BA are rejected as `BAD_INITIAL_PAIR`,
     keep the failed image pair marked as tried, and continue with the next
     trial in the same relaxation stage. This is covered by a synthetic mapper
     parity test that recovers from a bad first pair to a valid second pair.
   - Added the first COLMAP-style multi-model pipeline control slice:
     `multiple_models`, `max_num_models`, `max_model_overlap`, and
     `min_model_size` defaults are exposed, the pipeline can keep multiple
     sub-models through a shared mapper session, the first reconstruction is
     kept independent of size, later reconstructions below the customized
     `min_model_size = min(min_model_size, num_images / 2)` threshold are
     discarded with registration counts rolled back, and retained sub-models
     are exported as `sparse/0`, `sparse/1`, ... when more than one model is
     kept. A synthetic parity test covers the first-small-kept /
     later-small-discarded rule.
   - Multi-model registration statistics now follow COLMAP's
     `NumSharedRegImages` boundary for the covered mapper-session path:
     shared-image counts are scoped to the current sub-model instead of being
     accumulated across all previous sub-models, while discarded uncommitted
     reconstructions no longer roll back global registration counts from
     earlier kept models.
   - Added COLMAP-shaped sparse snapshot exports for the covered incremental
     pipeline path: `snapshot_path` and `snapshot_frames_freq` are exposed on
     the CLI/config, snapshots are written only after registered-frame growth
     crosses the configured frequency after initialization, and each snapshot
     contains COLMAP sparse reconstruction text files.
   - Added the COLMAP controller-level color extraction switch for the covered
     mapper path: `extract_colors` defaults to true, `--no-extract-colors`
     disables color propagation into new 3D points by zeroing sampled keypoint
     colors, and parity tests cover the black-point behavior when disabled.
   - Color extraction now follows COLMAP's per-registration timing for the
     covered incremental pipeline path: new triangulated points start black,
     the initial image pair is colorized after initial global BA/filtering,
     each next registered frame is colorized after local/scheduled global
     refinement and before snapshots, and the per-image extraction only fills
     still-black 3D points instead of overwriting already-colored points.
   - Final color extraction now follows COLMAP's all-image averaging behavior
     for the covered output path: after the final global BA/refinement stage,
     all registered track observations are averaged per 3D point, existing
     non-black colors can be overwritten by the track mean, and unobserved
     points fall back to black when color extraction is enabled.
   - Added COLMAP-shaped controller callback event boundaries to the covered
     pipeline log: `INITIAL_IMAGE_PAIR_REG_CALLBACK` fires after initial-pair
     color extraction, `NEXT_IMAGE_REG_CALLBACK` fires after next-image
     color extraction and snapshot handling, and `LAST_IMAGE_REG_CALLBACK`
     fires after the reconstruction keep/discard decision.
   - Added a lightweight public callback sink for the covered reconstruction
     path: `run_reconstruction_with_callbacks` exposes the same
     initial/next/last registration event types with model index, registered
     image/frame counts, and point counts, and parity coverage verifies both
     callback ordering and sink payloads.
   - Added the first continue-existing reconstruction slice: reference sparse
     models now seed current mapper attempts with existing registered poses,
     point3D ids, points, and 2D-3D observations mapped by image name; seeded
     reconstructions skip initial-pair selection and enter next-image
     registration directly, preserving existing points while registering new
     images. Synthetic parity coverage verifies both sparse-model seed loading
     and the no-`initial_pair` continuation path.
   - Added COLMAP-style `fix_existing_frames` for the covered continuation
     path: the CLI/config exposes the switch, mapper attempts record the
     registration units that were already registered at
     `BeginReconstruction` time, local/global BA mark those existing units as
     constant, and registered-frame filtering skips existing units instead of
     deregistering them. Tests cover the default-off option, local-BA constant
     image scheduling, and filtering protection.
   - Added the covered `Reconstruct(..., continue_reconstruction)`
     reconstruction-manager index-0 control flow: when a reference sparse
     model seeds the pipeline, only the first strict trial reuses the seeded
     reconstruction; later trials keep the same camera/rig metadata but start
     from an empty reconstruction, matching COLMAP's index-0 reuse followed by
     new sub-model creation. A parity test verifies one continuation attempt
     followed by a new initial-pair sub-model.
   - Broadened the continuation/fixed-existing coverage to non-trivial
     COLMAP frames: tests now verify that a seeded multi-image rig frame is
     reused only on the first continuation trial, that `fix_existing_frames`
     marks the full seeded frame constant for local BA, and that registered
     frame filtering protects the full seeded frame instead of deregistering
     either sibling image.
   - Added a real COLMAP sparse-text fixture path for this continuation slice:
     tests now write `cameras/images/points3D/rigs/frames` files with
     non-contiguous image ids, load them through `reference_camera_setup`,
     verify `frames.txt` data-id ownership, confirm index-0 seed reuse only on
     the first continuation trial, and verify `fix_existing_frames` protects
     the seeded rig frame in local-BA scheduling and registered-frame
     filtering.
   - Remaining work: full `IncrementalPipeline` multi-model orchestration with
     deterministic parallel initial-pair probing, a real user-extensible
     callback API, and official COLMAP-output / real-dataset rig-frame
     continuation parity fixtures.
13. [partial] Port next-image ranking by visible points count, visible points
    ratio, and uncertainty score.
   - Registration candidates are now ranked primarily by visible 3D points,
     PnP inliers, visible-points ratio, and image coverage score.
   - Remaining work: exact uncertainty score and COLMAP's registration queue
     ordering.
   - **Update 2026-06-20:** `FindNextImages` now uses separate
     `structureless_reg_trials` (COLMAP `num_structure_less_reg_trials`), ranks
     rig frames with max sibling `MinUncertainty` score, and collects PnP/GP3P/
     structureless correspondences from the embedded `CorrespondenceGraph` when
     available (not only verified pair inliers).
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
   - Local BA after a new registration now builds its constant-camera schedule
     from COLMAP-style post-registration counters, so the just-registered
     image/frame is included when deciding whether a shared camera is only
     partially covered by the local bundle.
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
   - Failed registration trials now increment at COLMAP registration-unit
     granularity: a failed or impossible non-trivial frame attempt marks every
     image in that frame once, and successful registration resets the full
     frame's trial counters.
   - Added a COLMAP-shaped generalized absolute-pose path for non-trivial rig
     frames. It reuses PoseLib's GP3P minimal solver behind the existing
     `poselib` feature, converts pixel thresholds per observed rig camera,
     scores candidate poses with COLMAP-style unique-3D-point support ordering,
     handles panoramic three-camera samples with PoseLib P3P fallback, and
     registers/continues tracks for the full frame in the mapper.
   - Added COLMAP-shaped generalized absolute-pose refinement after GP3P:
     the mapper reuses RustSFM's frame-aware BA path on a scratch
     reconstruction, keeps points/cameras/rig sensor poses fixed, and only
     refines `rig_from_world` from the RANSAC inlier residuals.
   - Added COLMAP's generalized-registration scheduling gate: non-trivial rigs
     with unknown/unregistered focal lengths fall back to central absolute pose
     before generalized GP3P is attempted, and central registration resets any
     bogus sibling frame cameras from priors.
   - Generalized absolute and generalized relative pose sampling now honor the
     same signed COLMAP seed semantics as mapper PnP: non-negative
     `random_seed` values are fixed, while `-1` no longer collapses to a
     hard-coded deterministic seed in the covered PoseLib-backed RANSAC paths.
   - Remaining work: validate exact sample sequences against COLMAP's target
     standard-library `std::uniform_int_distribution` behavior if byte-for-byte
     fixed-seed parity is required, then finish Ceres-equivalent generalized
     refinement/covariance behavior, frame/rig-level camera reset/refinement
     scheduling, and deregistration event counters.
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
   - `EstimateStructureLessAbsolutePose` now reuses the generalized
     relative-pose GR6P/GR8P path for non-panoramic world rigs.
   - **Update 2026-06-20:** structure-less 2D-2D correspondences are now
     collected from the embedded `CorrespondenceGraph` (via
     `ObservationManager`) when available, not only from verified pair inlier
     matches; poselib builds exercise this COLMAP path without the experimental
     pair-pose fallback flag.
   - Remaining work: finish exact control-flow/random sampler semantics and
     the official absolute-pose camera reset/refinement schedule around
     registered non-trivial rigs.

## P4 - Triangulation And Observation Management

15b. [done] Port COLMAP's `geometry/triangulation` primitive module
    (`triangulation.rs`).
   - Faithful `f64` port of `TriangulatePoint` (two-view DLT via SVD),
     `TriangulateMidPoint` (with cheirality guard), `TriangulateMultiViewPoint`
     (smallest-eigenvector DLT), `TriangulateOptimalPoint` (Lindstrom optimal
     correction), and `CalculateTriangulationAngle(s)` /
     `CalculateAngleBetweenVectors`, plus the `EssentialMatrixFromPose` and
     `FindOptimalImageObservations` helpers from `essential_matrix.cc` that the
     optimal path depends on.
   - Covered by synthetic recovery/consistency unit tests (exact noise-free
     recovery, multi-view vs two-view agreement, optimal-vs-DLT under noise,
     midpoint recovery, and triangulation-angle min/supplement behavior).
   - Wired as the shared backend: `two_view.rs::triangulate_normalized_pair`
     delegates the two-view DLT, and
     `incremental_triangulator.rs::triangulate_track_observations` now runs the
     multi-view DLT over all track observations.
   - This is the triangulation primitive boundary marked 100% in
     `COLMAP_MODULE_PARITY.md`; the `estimators/triangulation.cc` RANSAC
     `EstimateTriangulation` wrapper remains to be ported (see item 17).

16. [done] Port `ObservationManager` for adding/removing/merging observations and
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
   - Added mapper-level registered-frame filtering that calls the frame-aware
     deregistration hook and registration-stat rollback together, matching
     COLMAP's `DeRegisterFrame` / `DeRegisterFrameEvent` pairing for bogus or
     zero-observation registered frames after the 20-frame threshold.
   - Remaining work: keep a longer-lived mapper-level observation manager and
     finish exact COLMAP frame/rig semantics for generalized pose, BA
     scheduling, retry bookkeeping, and all cleanup events.
   - **Update 2026-06-20:** incremental COLMAP event paths are now implemented:
     embedded `CorrespondenceGraph`, `SetObservationAsTriangulated` /
     `ResetTriObservations`, increment/decrement correspondence-has-point3D
     counters, register/deregister visible-correspondence propagation, and
     6-level `VisibilityPyramid` incremental scoring (`visibility_pyramid.rs`)
     with COLMAP's weighted `Score`, `MaxScore`, and metadata accessors.
     Session-scoped `IncrementalTriangulatorState` owns the manager+graph.
     Incremental-vs-rebuild stat parity tests pass.
17. [done] Port `IncrementalTriangulator` create/continue/merge/complete/retriangulate
    behavior, transitivity, angular/reprojection thresholds, and two-view-track
    handling.
   - Added create/continue angular error gates, explicit two-view-track
     suppression in default triangulator options, and track re-triangulation
     after continue/merge. Track point estimation now uses the COLMAP multi-view
     DLT (`TriangulateMultiViewPoint`) over all observations instead of the
     earlier widest-baseline pair / averaged-merge heuristics.
   - The mapper now filters reprojection outliers after each incremental
     triangulation step.
   - Retriangulation now tracks a per-image-pair trial counter and skips pairs
     once `re_max_trials` is reached, matching COLMAP's `re_num_trials_` map
     behavior instead of the previous one-shot boolean guard.
   - Ported the `estimators/triangulation.cc` `TriangulationEstimator` and
     `EstimateTriangulation` LORANSAC entry point (`triangulation_estimator.rs`):
     two-view/multi-view model estimation with cheirality + triangulation-angle
     gating, angular and squared-reprojection residuals (matching
     `scene/projection.cc`), `InlierSupportMeasurer` support comparison, the
     COLMAP dynamic-stopping trial count, and a final inlier multi-view refit.
     The 2-view sampling uses a shared Rust port of COLMAP's deterministic
     `CombinationSampler`; broader random/progressive sampler infrastructure is
     tracked under the `optim` RANSAC item.
   - Wired `estimate_triangulation` into the incremental triangulator's
     track-creation path: `IncrementalTriangulator::create_pair_track` now
     gathers the seed observation pair plus transitively-corresponding
     observations in registered images (bounded by `max_transitivity`, one
     observation per image, COLMAP `Create`-style), maps the
     `create_max_angle_error`/`min_angle` options into
     `EstimateTriangulationOptions` (angular residual), and emits a multi-view
     track from the inlier set in one robust step. Newly created points are left
     uncolored ([0,0,0]) so the dedicated color-extraction stage assigns colors,
     matching COLMAP. Verified by a three-view track-creation test plus the
     existing pair/retriangulate/color-extraction suite (293 default / 298
     poselib lib tests green).
   - Remaining work: finish parallel random seeding and bit-level LORANSAC
     parity under the `optim` RANSAC item.
   - **Update 2026-06-20:** `CompleteImage` per point2D (complete tracks +
     orphan-cluster reprojection RANSAC), `Complete` direct-graph BFS, and
     `EstimateTriangulation` deterministic `CombinationSampler` order are
     implemented. P4 triangulation + observation manager marked 100% in
     `COLMAP_MODULE_PARITY.md` (303 lib tests).
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
   - Registered-frame filtering now deregisters full frames/images with bogus
     cameras or zero point3D observations after COLMAP's 20-frame threshold,
     and rolls back frame/rig/camera registration counters.
   - Remaining work: exact COLMAP deregistration counters after every failed
     refinement path and exact filtering schedules for local/global BA phases.

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
    - Ceres-backed BA now forwards `max_linear_solver_iterations` through the
      Rust Ceres binding instead of leaving Ceres on the binding default.
    - Ceres-backed BA now reads final gradient norm, step norm, trust-region
      decrease ratio, and LM damping from Ceres iteration summaries through
      structured bindings instead of relying on report-text parsing.
    - Ceres-backed BA now reads Ceres termination type and message through
      structured bindings and maps the termination type directly like
      COLMAP's `CeresTerminationTypeToTerminationType`, keeping report-text
      parsing only as a fallback for reason classification.
    - Ceres-backed BA now reports reduced residual and effective-parameter
      counts directly from `ceres::Solver::Summary`, matching the fields
      COLMAP uses in `CeresBundleAdjustmentSummary` and `PrintSolverSummary`.
    - Ceres-backed BA now mirrors COLMAP's CPU threading gate: mapper
      `--threads` is forwarded to Ceres, while problems below 50,000 residuals
      are forced to one Ceres thread to avoid small-problem threading overhead.
    - Ceres-backed BA now mirrors COLMAP's CPU linear-solver auto-selection
      thresholds: up to 50 pose entities use `DENSE_SCHUR`, 51 through 1000
      use `SPARSE_SCHUR`, and larger problems use `ITERATIVE_SCHUR` with
      `SCHUR_JACOBI` preconditioning.
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
    - The vendored Rust Ceres binding now exposes COLMAP-compatible
      `EigenQuaternionManifold` support for 4D `[x,y,z,w]` rotation blocks and
      7D `[x,y,z,w,tx,ty,tz]` pose product manifolds, including fixed
      translation-axis subsets used by `TWO_CAMS_FROM_WORLD` gauges.
    - The active Ceres BA problem now stores image/frame/sensor poses as
      COLMAP-style 7D quaternion+translation parameter blocks and attaches the
      Ceres pose manifold, including constant block handling and the
      `TWO_CAMS_FROM_WORLD` fixed-translation-axis gauge.
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
      tolerance 10 and 25 iterations; global BA uses gradient tolerance 1 and
      50 iterations; both use 100 linear-solver iterations and zero parameter
      tolerance.
      Local BA now also follows COLMAP's iterative refinement defaults
      (`ba_local_max_refinements = 2`, `ba_local_max_refinement_change = 0.001`)
      and switches refinement rounds after the first from Soft-L1 to Trivial
      loss.
    - BA now provides a Ceres-equivalent robust loss family
      (`BundleAdjustmentLoss`: Trivial, Huber, SoftL1, Cauchy) whose `rho(s)`
      cost and `rho'(s)` IRLS weight match Ceres' `LossFunction` formulas for
      the squared residual `s = ||r||^2`, replacing the previous Huber-only
      scalar. The mapper BA loss defaults now match COLMAP's incremental
      pipeline split: local BA uses Soft-L1 with scale 1.0 and global BA uses
      a trivial loss (overridable via `RUSTSFM_BA_LOSS` /
      `RUSTSFM_BA_LOSS_SCALE`).
      Ceres-backed BA now also creates an explicit `TrivialLoss` for
      non-robust residuals, matching COLMAP's `CreateLossFunction(TRIVIAL)`
      path instead of omitting the loss pointer.
      BA entrypoints now reject negative, infinite, or NaN robust loss scales
      before solving, matching COLMAP's Ceres option validation gate.
      The Ceres solver auto-selection path now also reads Ceres' configured
      sparse linear algebra backend and skips `SPARSE_SCHUR` when it is
      `NO_SPARSE`, matching COLMAP's `has_sparse` gate.
      Ceres integer option forwarding now rejects `usize` values that cannot
      be represented by Ceres' signed `int` fields before they can wrap into
      apparently valid solver settings.
    - The reduced camera matrix (Schur complement) is now solved with a
      Cholesky factorization and an LU fallback (`ba.rs::solve_linear_system`),
      matching Ceres' `DENSE_SCHUR`/`SPARSE_SCHUR` linear solvers that rely on
      the damped reduced system being symmetric positive definite.
    - Added `ceres-ba` feature (`ceres-solver` 0.5 with bundled source build),
      **enabled by default**. The `ba/` module splits types/dispatch (`mod.rs`),
      shared observation/gauge helpers (`shared.rs`), the hand-rolled native LM
      reference backend (`native.rs`), and the Ceres backend (`ceres.rs`).
      `refine_bundle_adjustment` always uses Ceres with **no fallback** to
      native. Unsupported configs return `None`. Native BA remains for
      `--no-default-features` builds and native-specific unit tests.
    - Ceres backend (`ceres_problem.rs`) now builds full problems with
      image/frame/sensor pose blocks, intrinsics refinement, fixed poses, and
      gauge policies (`THREE_POINTS`, `TWO_CAMS_FROM_WORLD`). Ceres cost
      callbacks reuse native analytic projection/frame/sensor/camera
      Jacobians with numeric fallback.
    - Ceres image-pose cost callbacks now evaluate COLMAP's raw Eigen
      quaternion rotation formula and fill exact 2x7 ambient Jacobians for
      ordinary image pose blocks, matching
      `AnalyticalReprojErrorCostFunction`.
    - Ceres frame/sensor rig cost callbacks now evaluate COLMAP's raw
      Eigen-quaternion `RigReprojErrorCostFunctor` composition and fill exact
      2x7 ambient Jacobians for both `rig_from_world` and `sensor_from_rig`
      pose blocks, replacing the numeric ambient bridge for active Ceres rig
      BA.
    - Remaining work: switch native LM damping from a fixed `mu*I` to Ceres'
      jacobian-scaled
      `mu*clamp(diag(JᵀJ), 1e-6, 1e32)` diagonal together with Ceres'
      radius-based trust-region update (a naive switch under the current
      `damping*=10` recovery regressed the generalized-rig refinement test, so
      both must land together); then true sparse Schur storage, covariance,
      and full backend solver-summary parity.
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
    - The mapper now mirrors COLMAP's ordering more closely by including the
      just-registered image/frame in the registration counters used to build
      local BA constant-camera decisions, while still only committing those
      counters permanently after rollback checks pass.
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

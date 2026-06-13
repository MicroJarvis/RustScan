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
   - Remaining strict-parity work: replace numerical distortion Jacobians in
     iterative undistortion with analytic/Jet-equivalent derivatives and port
     all downstream pose solvers to use camera-model bearing/projection APIs.
3. [partial] Replace the single-camera reconstruction with COLMAP-style
   camera/image ownership so images can reference distinct shared cameras.
   - Added reconstruction camera table, COLMAP camera ids, image ids, and
     per-image camera references.
   - Text/binary camera reading now preserves all cameras; text export writes
     camera ids and per-image camera ownership instead of hard-coding camera 1.
   - Reference-model initialization now keeps COLMAP camera/image ids and maps
     images back to their shared cameras.
   - Reference-model two-view estimation now passes each image's camera and
     averages the normalized RANSAC threshold like COLMAP.
   - Remaining work: non-reference mapper input still initializes a single
     shared camera.
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
   - Remaining work: integrating read point ids into full reconstruction
     loading, database ids, and full sparse model round-trip.

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
   - Added a `--database` mapper path that loads COLMAP database cache,
     replaces frame keypoints with database keypoints, preserves database
     camera/image ownership, and estimates pairs from verified database
     correspondences instead of local matching when provided.
   - Database-driven pair estimation now reuses stored `qvec/tvec` two-view
     geometry poses when present and valid, falling back to local two-view
     estimation otherwise.
   - Added official COLMAP two-view geometry configuration constants and made
     database-cache edge loading reject `UNDEFINED`/`DEGENERATE` geometries
     while preserving watermark filtering semantics.
   - Remaining work: SQLite COLMAP database schema/read-write parity, database
     cache ENU pose-prior conversion, complete pose/config semantics for all
     two-view geometry cases, and making database-driven reconstruction the
     main COLMAP-style mapper path.
8. Remove sequence-specific local-window and 192-frame ring heuristics from the
   default mapper path; keep them only behind explicit experimental options.

## P2 - Two-View Geometry Parity

9. Port COLMAP `TwoViewGeometry` configuration logic: calibrated,
   uncalibrated, homography, planar, panoramic, watermark, multiple models, and
   rig verification.
10. Replace the local RANSAC implementation with COLMAP-equivalent RANSAC /
    LORANSAC support scoring, stopping criteria, random seeding, and solver
    selection.
11. Match COLMAP initial-pair checks: min inliers, max forward motion,
    triangulation-angle threshold, and generalized relative pose for rigs.

## P3 - Incremental Mapper State Machine

12. Port COLMAP initial image selection using prior focal length and
    correspondence counts, including registration trial bookkeeping.
13. Port next-image ranking by visible points count, visible points ratio, and
    uncertainty score.
14. Port absolute pose estimation and refinement, including focal/extra
    parameter estimation, bogus camera reset, inlier ratio checks, and
    generalized absolute pose for rigs.
   - Partial: RustSFM now lifts absolute-pose 2D observations through the
     per-image `CameraModel::CamFromImg` before PnP, so distorted/non-pinhole
     cameras no longer enter PnP through raw pinhole intrinsics; PnP thresholds
     are also created from the registering image camera.
   - Remaining work: replace the local P3P/DLT-RANSAC/refinement path with
     COLMAP-equivalent absolute pose estimation, focal/extra-parameter
     refinement, bogus camera reset, and generalized rig pose.
15. Port structure-less registration fallback.

## P4 - Triangulation And Observation Management

16. Port `ObservationManager` for adding/removing/merging observations and
    tracking modified points.
17. Port `IncrementalTriangulator` create/continue/merge/complete/retriangulate
    behavior, transitivity, angular/reprojection thresholds, and two-view-track
    handling.
18. Port filtering by negative depth, reprojection error, triangulation angle,
    short tracks, and bogus camera parameters.

## P5 - Bundle Adjustment And Refinement

19. Replace the experimental BA path with a Ceres-equivalent optimizer layer or
    a numerically equivalent Rust implementation, including analytic residuals.
20. Match COLMAP local BA image selection, gauge fixing, robust losses,
    constant camera/rig controls, and short-track point selection.
21. Match COLMAP global BA scheduling, convergence settings, optional redundant
    point handling, pose priors, normalization, and iterative refinement loops.

## P6 - I/O, Multi-Model Pipeline, And Parity Tests

22. Read/write full COLMAP sparse text and binary models, including cameras,
    images, points, rigs, frames, and database-derived data where applicable.
23. Port COLMAP incremental pipeline behavior for sub-model creation, overlap
    limits, min model size, snapshots, final refinement, and color extraction.
24. Build parity test fixtures that compare RustSFM against official COLMAP on
    synthetic and real datasets at each stage: camera I/O, feature counts,
    matches, two-view geometry, registration order, tracks, BA cost, and final
    sparse model.

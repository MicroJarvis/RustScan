# RustSFM

Pure-Rust COLMAP-style incremental SfM experiment for RustScan.

This crate intentionally does not call the external `colmap` executable. The
current implementation follows the COLMAP mapper shape: database/cache loading,
feature extraction, geometric verification, initial pair selection,
incremental PnP registration, track triangulation, and COLMAP text export.

By default the mapper now behaves like a COLMAP-style database-first pipeline:
it auto-discovers `database.db` next to the image root or input directory and
only falls back to local matching when `--local-matching` is set explicitly.
Two-view verification now preserves COLMAP-style geometry configs for
calibrated, uncalibrated, planar/panoramic, watermark, and multiple-model cases
while filtering ambiguous watermark/multiple pairs out of the default mapper
graph.
Rust-estimated verified pair geometry can be written back to the COLMAP
`two_view_geometries` table with `--write-two-view-geometries`; this is opt-in
so the default reconstruction path does not mutate the input database.
The local-matching fallback can also create a full COLMAP-style SQLite database
(cameras, images, keypoints, descriptors, matches, two-view geometries) with
`--local-matching --write-database [--database path/to/database.db]`.
Generalized rig relative/absolute pose now follows COLMAP's panoramic-rig
branches and uses PoseLib's GR6P/GP3P minimal solvers plus a COLMAP-derived
GR8P local refit bridge for non-panoramic rigs in default builds, with
BA-backed pose-only generalized absolute-pose refinement for rig frames and
COLMAP-style fallback to central PnP when a rig camera still needs focal-length
estimation. PoseLib v2.0.5 is pinned as the `third_party/PoseLib` submodule.
Initialize dependencies and run the default solver tests with:

```bash
git submodule update --init --recursive
cargo test -p rustsfm --lib
```

Existing clones can also run `./scripts/setup_rustsfm_deps.sh` to bootstrap
RustSFM's native dependencies. The intentional dependency-minimal build keeps
the explicit missing-solver fallback available:

```bash
cargo test -p rustsfm --lib --no-default-features
```
Incremental registration is absolute-pose driven with COLMAP-style next-image
ranking methods, registration trial bookkeeping, inlier-ratio checks, and
pose-only reprojection refinement before accepting new images. Filtered or
previously failed registration units are retried from a lower-priority bucket
like COLMAP, and max-trial checks are applied over the full frame registration
unit.
Successful registrations now trigger a local bundle-adjustment pass over the
new image, its strongest shared-point neighbors selected with
triangulation-angle checks, and new or short-track local 3D points; local BA is
followed by track merge/completion, new-image completion, and track filtering.
Global bundle adjustment now runs after initialization, on COLMAP-style
registered-image/point growth triggers, and at finalization when the model has
changed since the last global pass; each pass fixes two registered images as the
global gauge, follows COLMAP's focal/principal-point/extra-parameter refinement
defaults, accepts `--ba-constant-camera-id`, and performs track
completion/merge/filter post-processing.
Triangulation applies angular/reprojection gates, re-estimates track geometry
after continuation/merge, and filters negative-depth, reprojection,
triangulation-angle, bogus-camera, and short-track outliers during the
incremental loop. After COLMAP's 20-registration-unit warm-up threshold, mapper
cleanup also deregisters full frames/images whose registered cameras are bogus
or whose frame has no remaining point3D observations, keeping the frame/rig and
per-camera registration counters in sync with the reconstruction state.

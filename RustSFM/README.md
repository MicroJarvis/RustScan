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
Incremental registration is absolute-pose driven with visible-point ranking,
registration trial bookkeeping, inlier-ratio checks, and pose-only
reprojection refinement before accepting new images. Successful registrations
now trigger a local bundle-adjustment pass over the new image, its strongest
shared-point neighbors selected with triangulation-angle checks, and new or
short-track local 3D points; local BA is followed by track merge/completion,
new-image completion, and track filtering.
Global bundle adjustment now runs after initialization, on COLMAP-style
registered-image/point growth triggers, and at finalization when the model has
changed since the last global pass; each pass fixes two registered images as the
global gauge, follows COLMAP's focal/principal-point/extra-parameter refinement
defaults, accepts `--ba-constant-camera-id`, and performs track
completion/merge/filter post-processing.
Triangulation applies angular/reprojection gates, re-estimates track geometry
after continuation/merge, and filters negative-depth, reprojection,
triangulation-angle, bogus-camera, and short-track outliers during the
incremental loop.

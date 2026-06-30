# COLMAP trace probe

Offline diagnostic tool for comparing RustSFM two-view RANSAC traces against
COLMAP 3.13. It reads a COLMAP database pair, runs COLMAP estimators through a
local traced copy of `LORANSAC`, and emits JSON with best-support updates,
LO-internal updates, final supports, and residuals nearest the inlier threshold.

This is intentionally not part of the default Cargo build.

## Build

```bash
c++ -std=c++17 -O2 -DNDEBUG \
  -I/opt/homebrew/Cellar/colmap/3.13.0_3/include \
  -I/opt/homebrew/include/eigen3 \
  -I/opt/homebrew/opt/ceres-solver/include \
  -I/opt/homebrew/opt/glog/include \
  -I/opt/homebrew/opt/gflags/include \
  -I/opt/homebrew/opt/boost/include \
  RustSFM/tools/colmap_trace_probe/colmap_trace_probe.cpp \
  -L/opt/homebrew/Cellar/colmap/3.13.0_3/lib \
  -L/opt/homebrew/opt/glog/lib \
  -L/opt/homebrew/opt/gflags/lib \
  -L/opt/homebrew/opt/ceres-solver/lib \
  -lcolmap_estimators -lcolmap_optim -lcolmap_math \
  -lcolmap_scene -lcolmap_sensor -lcolmap_util \
  -lsqlite3 -lglog -lgflags -lceres \
  -o /tmp/colmap_trace_probe
```

## Example

```bash
/tmp/colmap_trace_probe \
  --database output/rustsfm_flowers2_cam_rays_full_lo/database.db \
  --image1 frame_0002.jpg \
  --image2 frame_0018.jpg \
  --random-seed 0 \
  > output/rustsfm_flowers2_colmap_probe/colmap_0002_0018_seed0.json
```

Use `--random-seed 0` for direct single-pair sampler comparison. Without an
explicit seed, COLMAP's default thread-local PRNG starts from seed 0 and advances
across E/F/H in a single process, but this still does not reproduce full mapper
parallel scheduling.

Homebrew COLMAP 3.13's SQLite backend prepares legacy pose-prior SQL against
`pose_priors.image_id`. RustSFM now adds a nullable compatibility column with
that name when opening newer sensor-pose-prior databases, which is enough for
empty-pose-prior verifier diagnostics. It does not make non-empty newer
pose-prior rows fully readable by old COLMAP 3.13 binaries.

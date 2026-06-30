# COLMAP verifier trace

Offline diagnostic tool for comparing RustSFM's fixed raw-match verifier
scheduling against COLMAP's verifier shape. It reads existing raw matches from a
COLMAP database in `ReadNumMatches()` order, keeps verifier worker threads alive
across batches, runs COLMAP `EstimateTwoViewGeometry`, and emits worker/dequeue/
completion/config/inlier events as JSON.

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
  -I/opt/homebrew/include \
  RustSFM/tools/colmap_verifier_trace/colmap_verifier_trace.cpp \
  -L/opt/homebrew/Cellar/colmap/3.13.0_3/lib \
  -L/opt/homebrew/lib \
  -L/opt/homebrew/opt/glog/lib \
  -L/opt/homebrew/opt/gflags/lib \
  -L/opt/homebrew/opt/ceres-solver/lib \
  -L/opt/homebrew/opt/openssl@3/lib \
  -L/opt/homebrew/opt/libomp/lib \
  -lcolmap_estimators -lcolmap_optim -lcolmap_math \
  -lcolmap_scene -lcolmap_sensor -lcolmap_geometry \
  -lcolmap_image -lcolmap_feature_types -lcolmap_feature \
  -lcolmap_util -lcolmap_vlfeat -lPoseLib \
  -lsqlite3 -lglog -lgflags -lceres -lfreeimage \
  -lcurl -lssl -lcrypto -lomp \
  -o /tmp/colmap_verifier_trace
```

## Example

```bash
/tmp/colmap_verifier_trace \
  --database output/rustsfm_flowers2_fifo_batch_verify/database.db \
  --num-threads 4 \
  --batch-size 1000 \
  > output/rustsfm_flowers2_fifo_batch_verify/colmap_verifier_trace.json
```

## Notes

Homebrew COLMAP 3.13's SQLite backend still prepares legacy pose-prior
statements against `pose_priors.image_id`. RustSFM's database opener now keeps a
nullable compatibility column for that name, so opening a copied DB once through
RustSFM is enough before running the official `colmap geometric_verifier` binary
or this probe. This compatibility is meant to unblock verifier diagnostics on
DBs without pose priors; it is not full old-schema pose-prior interop.

Repeated official `colmap geometric_verifier` runs on the same `flowers2` raw
matches are not bit-stable with `TwoViewGeometry.random_seed=-1` and multiple
threads. On the current fixture, four 4-thread runs produced two-view mismatch
counts of `2/3/4/3` against the stored reference DB, with boundary flips on
`0004-0020`, `0002-0018`, `0006-0022`, and `0001-0017`. Treat the stored
reference DB as one verifier schedule realization unless replaying a recorded
worker/PRNG schedule.

RustSFM can replay a recorded verifier schedule through:

```bash
RUSTSFM_COLMAP_FIFO_VERIFIER=1 \
RUSTSFM_COLMAP_SHARED_RANSAC_STREAM=1 \
RUSTSFM_COLMAP_FIFO_VERIFIER_REPLAY_TRACE=output/rustsfm_flowers2_fifo_batch_verify/colmap_verifier_trace_run1.json \
target/debug/rustsfm match-features \
  --database output/rustsfm_flowers2_fifo_replay_run1/database.db \
  --use-existing-matches \
  --random-seed=-1 \
  --essential-threshold-px 4.0 \
  --essential-iterations 10000 \
  --existing-match-batch-size 1000 \
  --output-json output/rustsfm_flowers2_fifo_replay_run1/match_report.json
```

The replay path assigns pairs to the recorded `worker_id` in recorded
`dequeue_order`, so each Rust worker consumes its thread-local RANSAC stream in
the same order as the trace. It also emits `verifier_trace.mode =
"colmap_fifo_shared_ransac_stream_replay"` for downstream comparison.

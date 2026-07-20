# RustSFM Sparse Maintenance Performance Benchmark

## Reproducibility

- Sparse-maintenance commit: `e830fcb903ed5a1965dc8cdf1a9e5574fb1e7b63`
- Full-registration acceptance fix: `167d521` (`fix(rustsfm): trust refined pnp over pair
  rotation metadata`)
- Branch: `feat/rustsfm-wgpu`
- Host: Apple M5 Max, 64 GiB RAM, arm64
- OS: macOS 26.5.1 (25F80), Darwin 25.5.0
- Image input: `/Users/tfjiang/Projects/RustScan/test_data/flowers2/images`
- Input images: 960
- Database: `/Users/tfjiang/Projects/RustScan/test_data/flowers2/rustsfm_full_work/database.db`
- Database size: approximately 1.0 GB
- Random seed: 0
- CPU threads: 1
- Output root: `/tmp/rustsfm-sparse-maintenance-20260720`

The benchmark uses the existing feature and match database. It does not sample images during the
final acceptance run and does not enable CPU parallelism.

## Baseline

- All-960 run: terminated at 19,509.85 seconds with no sparse export.
- Sparse maintenance profile: filter 62%, complete 30%, merge 8%.
- Observation slots: 6,612,183.
- Largest earlier model: 900 registered images and 420,294 points.
- Previous 200-image result: 362.31 seconds, 200 registered images, and 91,894 points.

## Static Verification

- Observation manager: 19 passed, 0 failed.
- Incremental triangulator: 21 passed, 0 failed.
- Non-fixture track filtering: 8 passed, 0 failed.
- Dirty-frontier filtering: 1 passed, 0 failed.
- Full non-stale library suite after the acceptance fix: 590 passed, 0 failed, 19 fixture tests
  filtered.
- `cargo check -p rustsfm`: passed.
- `cargo check -p rustgs`: passed.
- `cargo build -p rustsfm --release`: passed.
- `cargo fmt --all -- --check`: passed.
- `git diff --check`: passed.

The excluded `real_colmap_sparse` fixture family no longer represents its documented 24-image,
256-point model. The exact track-filter failure was reproduced unchanged at pre-optimization
commit `c761490`, where the fixture removed 6 observations while the stale assertion expected 2.

## 96-Image Preflight

Command:

```bash
/usr/bin/time -p target/release/rustsfm reconstruct \
  --input /Users/tfjiang/Projects/RustScan/test_data/flowers2/images \
  --output /tmp/rustsfm-sparse-maintenance-20260720/optimized96 \
  --database /Users/tfjiang/Projects/RustScan/test_data/flowers2/rustsfm_full_work/database.db \
  --max-images 96 \
  --random-seed 0 \
  --threads 1 \
  --log-level debug \
  --summary-json /tmp/rustsfm-sparse-maintenance-20260720/optimized96/summary.json
```

Results:

- Wall time: 98.30 seconds (`elapsed_ms=98272.138166`).
- Input/registered images: 96/96.
- Sparse points: 42,887.
- Pair geometries: 452.
- Complete: 214 calls, 1,002,321 frontier points, 7,221 completed observations,
  984.64 ms.
- Merge: 214 calls, 1,002,322 frontier points, 22,695 merged observations, 706.45 ms.
- Filter: 13 full calls, 194 subset calls, 810,752 points, 6,178,786 observations,
  2,779 removed observations, 755.00 ms.
- Deletion: 1,962 points, 1,960 moved tail points, 4,246 rewritten track observations,
  158.87 ms.
- Dirty frontier: peak 4,481 points, 207 cycles, 499,560 points consumed. The peak was 10.45%
  of the final point count.
- Finite-pose result: 96 image headers and zero non-finite pose values.
- Finite-point result: 42,887 points and zero non-finite XYZ values.
- COLMAP export: non-empty `sparse/0` with cameras, images, points, rigs, and frames.

The plan's `--max-frames 1` RustGS command completed one Metal iteration but intentionally reduced
the loader report to one pose. The acceptance probe therefore omitted that truncation so the
loader itself validated every registered image:

```bash
target/release/rustgs train \
  --input /tmp/rustsfm-sparse-maintenance-20260720/optimized96 \
  --image-root /Users/tfjiang/Projects/RustScan/test_data/flowers2/images \
  --output /tmp/rustsfm-sparse-maintenance-20260720/rustgs96-all.ply \
  --iterations 1 \
  --max-initial-gaussians 1000
```

RustGS resolved 96 poses with zero missing images, loaded all 42,887 initialization points, ran
one wgpu/Metal iteration on the Apple M5 Max, trained 1,000 Gaussians, wrote a non-empty PLY, and
reported `gate_status=Passed`.

## 200-Image Benchmark

Command:

```bash
/usr/bin/time -p target/release/rustsfm reconstruct \
  --input /Users/tfjiang/Projects/RustScan/test_data/flowers2/images \
  --output /tmp/rustsfm-sparse-maintenance-20260720/optimized200 \
  --database /Users/tfjiang/Projects/RustScan/test_data/flowers2/rustsfm_full_work/database.db \
  --max-images 200 \
  --random-seed 0 \
  --threads 1 \
  --log-level debug \
  --summary-json /tmp/rustsfm-sparse-maintenance-20260720/optimized200/summary.json
```

Results:

- Wall time: 268.49 seconds (`elapsed_ms=268455.234083`).
- Input/registered images: 200/200.
- Sparse points: 91,910, or 16 more than the previous 91,894-point result.
- Pair geometries: 993.
- Complete: 432 calls, 2,243,151 frontier points, 14,897 completed observations,
  2,703.83 ms.
- Merge: 432 calls, 2,243,157 frontier points, 55,470 merged observations, 1,860.33 ms.
- Filter: 18 full calls, 406 subset calls, 1,920,294 points, 15,505,166 observations,
  6,359 removed observations, 2,030.28 ms.
- Complete, merge, and filter together: 6,594.44 ms, or 32.97 ms per registered image and
  2.46% of wall time.
- Deletion: 4,589 points, 4,584 moved tail points, 10,106 rewritten track observations,
  383.84 ms.
- Dirty frontier: peak 4,970 points, 424 cycles, 1,124,253 points consumed. The peak was 5.41%
  of the final point count.
- Finite-pose result: 200 image headers and zero non-finite pose values.
- Finite-point result: 91,910 points and zero non-finite XYZ values.
- Performance delta: 93.82 seconds faster than the previous 362.31-second result, a 25.90%
  wall-time reduction, with no registration or point-coverage regression.

RustGS command:

```bash
target/release/rustgs train \
  --input /tmp/rustsfm-sparse-maintenance-20260720/optimized200 \
  --image-root /Users/tfjiang/Projects/RustScan/test_data/flowers2/images \
  --output /tmp/rustsfm-sparse-maintenance-20260720/rustgs200.ply \
  --iterations 1 \
  --max-initial-gaussians 1000
```

RustGS resolved 200 poses with zero missing images, loaded all 91,910 initialization points, ran
one wgpu/Metal iteration, trained 1,000 Gaussians, wrote a non-empty PLY, and reported
`gate_status=Passed`.

## 960-Image Acceptance

The first optimized run at `e830fcb` completed in 2,167.88 seconds but retained two overlapping
models whose largest model registered only 900 images. The two models contained 900 and 889
images, with an 842-image intersection and a 947-image union. The missing images had strong
verified database connectivity, and a fixed 900-image seed still failed every remaining PnP
attempt despite 0.899-0.996 PnP inlier ratios and 0.83-1.43 pixel mean reprojection errors.

The rejection came from a non-COLMAP post-PnP gate that averaged all pair-rotation metadata and
rejected refined absolute poses above 20 degrees. The affected images had diagnostic pair-rotation
errors from 16.56 to 44.38 degrees even though their absolute poses had high support. Commit
`167d521` keeps this value for logging and ranking but no longer hard-rejects structure-based PnP;
the structure-less consistency guard remains unchanged.

The final benchmark is a fresh, single-command reconstruction from the database, without a sparse
seed and without `--max-images`:

```bash
/usr/bin/time -p target/release/rustsfm reconstruct \
  --input /Users/tfjiang/Projects/RustScan/test_data/flowers2/images \
  --output /tmp/rustsfm-sparse-maintenance-20260720/optimized960-fixed \
  --database /Users/tfjiang/Projects/RustScan/test_data/flowers2/rustsfm_full_work/database.db \
  --random-seed 0 \
  --threads 1 \
  --log-level debug \
  --summary-json /tmp/rustsfm-sparse-maintenance-20260720/optimized960-fixed/summary.json
```

Results:

- Wall time: 1,024.23 seconds (`elapsed_ms=1024111.203709`).
- Input/registered images: 960/960 in one model.
- Sparse points: 432,201.
- Pair geometries: 4,820.
- Complete: 1,996 calls, 9,567,635 frontier points, 56,894 completed observations,
  12,939.32 ms.
- Merge: 1,996 calls, 9,567,650 frontier points, 359,627 merged observations, 8,152.97 ms.
- Filter: 26 full calls, 1,958 subset calls, 8,146,053 points, 67,168,860 observations,
  26,201 removed observations, 10,428.13 ms.
- Complete, merge, and filter together: 31,520.42 ms, or 32.83 ms per registered image and
  3.08% of wall time.
- Deletion: 22,464 points, 22,427 moved tail points, 47,956 rewritten track observations,
  1,727.95 ms.
- Dirty frontier: peak 4,970 points, 1,984 cycles, 5,153,973 points consumed. The peak was 1.15%
  of the final point count.
- Incremental registration: 958 structure-based attempts, zero structure-less attempts,
  20,542.09 ms pose solve/refinement, 15,246.86 ms triangulation.
- Finite-pose result: 960 image headers and zero non-finite pose values.
- Finite-point result: 432,201 points and zero non-finite XYZ values.
- COLMAP export: exactly one non-empty `sparse/0` model with cameras, images, points, rigs, and
  frames.
- Baseline delta: 18,485.62 seconds faster than the interrupted 19,509.85-second run, a 94.75%
  wall-time reduction and a 19.05x speedup. Unlike the baseline, this run exported a complete
  model.

RustGS command:

```bash
target/release/rustgs train \
  --input /tmp/rustsfm-sparse-maintenance-20260720/optimized960-fixed \
  --image-root /Users/tfjiang/Projects/RustScan/test_data/flowers2/images \
  --output /tmp/rustsfm-sparse-maintenance-20260720/rustgs960-fixed.ply \
  --iterations 1 \
  --max-initial-gaussians 1000
```

RustGS resolved all 960 poses with zero missing images, loaded all 432,201 initialization points,
ran one wgpu/Metal iteration on the Apple M5 Max, trained 1,000 Gaussians, wrote a non-empty PLY,
and reported `gate_status=Passed`.

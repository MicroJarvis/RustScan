# RustGS Benchmark Datasets

**Updated:** 2026-04-30

This document tracks the next dataset pack for improving RustGS quality beyond
the existing TUM `freiburg1_xyz` loop.

## Current Loader Reality

RustGS training needs initialization points. The lowest-friction inputs are:

- COLMAP scene roots with `images/` and `sparse/0/{cameras,images,points3D}.{txt,bin}`.
- `TrainingDataset` JSON files with non-empty `initial_points`.
- Nerfstudio-style directories with `transforms.json` plus an initialization
  point cloud such as `sparse_pc.ply` or `point_cloud.ply`.

Raw TUM RGB-D directories are useful source data, but the training CLI still
requires them to be converted to COLMAP-style input first.

## Priority Pack

| Priority | Scene | Input | What It Stresses |
|---:|---|---|---|
| 1 | `tum_freiburg1_xyz_colmap` | local COLMAP conversion | continuity with the current PSNR loop |
| 2 | Mip-NeRF 360 `room` | COLMAP | indoor PSNR and SH color |
| 3 | Mip-NeRF 360 `garden` | COLMAP | outdoor/unbounded scene scale |
| 4 | Tanks and Temples `truck` | COLMAP | large object geometry and topology growth |
| 5 | Deep Blending `playroom` | COLMAP | indoor lighting, occlusion, view-dependent color |
| 6 | NeRF Synthetic / Nerfstudio `lego` | `transforms.json` + point cloud | projection and coordinate sanity |

## Prepare The Local Pack

```sh
python3 scripts/prepare_rustgs_benchmark_pack.py
```

This writes:

- `artifacts/runs/rustgs_benchmark_pack/manifest.json`
- `artifacts/runs/rustgs_benchmark_pack/README.md`
- `artifacts/runs/rustgs_benchmark_pack/tum_freiburg1_xyz_colmap/` when local TUM data is present

For a smaller local TUM smoke pack:

```sh
python3 scripts/prepare_rustgs_benchmark_pack.py \
  --max-tum-frames 180 \
  --tum-frame-stride 10 \
  --tum-point-step 128
```

## Run A Quick Benchmark

```sh
cargo run -p rustgs --release --bin rustgs -- train \
  --input artifacts/runs/rustgs_benchmark_pack/tum_freiburg1_xyz_colmap \
  --output artifacts/runs/rustgs_benchmark_pack/runs/tum_freiburg1_xyz_colmap_500.ply \
  --iterations 500 \
  --render-scale 1.0 \
  --eval-after-train \
  --eval-render-scale 0.5 \
  --eval-frame-stride 10 \
  --eval-json
```

Use the same command shape for external COLMAP-format scenes after replacing
`--input`.

## External Sources

- Mip-NeRF 360: <https://jonbarron.info/mipnerf360/>
- Tanks and Temples: <https://www.tanksandtemples.org/download/>
- TUM RGB-D: <https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download>
- Graphdeco 3DGS data notes: <https://github.com/graphdeco-inria/gaussian-splatting>
- Nerfstudio dataset formats: <https://docs.nerf.studio/quickstart/existing_dataset.html>

## Suggested Acceptance Metrics

Track each scene with:

- Final PSNR from RustGS `--eval-after-train`.
- Initial and final Gaussian counts.
- Densify/prune event counts from LiteGS telemetry.
- Worst-frame PSNR list from the evaluation summary.
- Wall-clock training time.

The short-run default should stay `500` iterations for tuning. Longer `30k`
runs should be reserved for candidate defaults that already improve at short
budget.

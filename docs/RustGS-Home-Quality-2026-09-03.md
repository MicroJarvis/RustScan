# RustGS Home Quality Smoke 2026-09-03

## Scope

This is a local quality smoke test for the current RustGS wgpu training and
GPU evaluation path. It is not a replacement for the TUM/LiteGS acceptance
record: the COLMAP input below was generated locally under `output/` and is
not a versioned fixture.

## Local Input

- Sparse model: `output/profile_home/colmap60/0`
- Images: `test_data/home/images`
- Frames: first 12 frames selected by `--max-frames 12`
- Initialization: 26,518 sparse COLMAP points
- Training/evaluation render scale: 0.25
- Evaluation device: GPU
- Frame shuffle seed: 0

## Commands

```sh
# Baseline
./target/release/rustgs train \
  --input output/profile_home/colmap60/0 \
  --image-root test_data/home/images \
  --output output/profile_home/rustgs_home_baseline_1500.ply \
  --iterations 1500 \
  --max-frames 12 \
  --render-scale 0.25 \
  --frame-shuffle-seed 0 \
  --eval-after-train \
  --eval-render-scale 0.25 \
  --eval-max-frames 12 \
  --eval-frame-stride 1 \
  --eval-device gpu \
  --eval-json

# Freeze topology after epoch 80
./target/release/rustgs train \
  --input output/profile_home/colmap60/0 \
  --image-root test_data/home/images \
  --output output/profile_home/rustgs_home_freeze80_1500.ply \
  --iterations 1500 \
  --max-frames 12 \
  --render-scale 0.25 \
  --frame-shuffle-seed 0 \
  --litegs-topology-freeze-after-epoch 80 \
  --eval-after-train \
  --eval-render-scale 0.25 \
  --eval-max-frames 12 \
  --eval-frame-stride 1 \
  --eval-device gpu \
  --eval-json
```

## Results

| Metric | Baseline | Freeze80 | Delta |
|---|---:|---:|---:|
| Training time | 44.73s | 45.10s | +0.37s |
| Final splats | 26,943 | 26,899 | -44 |
| Final loss | 0.118859 | 0.119736 | +0.000877 |
| PSNR mean | 20.4425 dB | 20.4748 dB | +0.0323 dB |
| PSNR min | 19.7924 dB | 19.6736 dB | -0.1188 dB |
| PSNR max | 21.0202 dB | 21.1160 dB | +0.0958 dB |

At 500 iterations, training ended near epoch 42, so an epoch-80 topology
freeze did not activate and produced no meaningful comparison. At 1,500
iterations the baseline still performed topology work at epoch 93
(growth=1,714, clone=1,693, split=21, prune=381); freeze80 correctly prevented
that late event.

## Decision

Freeze80 preserves mean quality on this local smoke dataset, but does not
improve runtime at this scale. Do not make it the default schedule from this
measurement. Re-evaluate on the external TUM fixture, where the historical
LiteGS experiment recorded measurable late-stage topology churn and a runtime
benefit.

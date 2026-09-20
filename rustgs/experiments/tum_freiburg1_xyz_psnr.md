# TUM Freiburg1 XYZ COLMAP PSNR Optimization Log

Dataset: `/Users/tfjiang/Projects/RustScan/artifacts/inputs/tum_freiburg1_xyz_colmap`

Goal: exceed 20 dB PSNR after 30000 training iterations.

Unless noted otherwise, experiments use release builds and evaluate with:

- `--eval-after-train`
- `--eval-render-scale 0.25`
- `--eval-max-frames 180`
- `--eval-frame-stride 30`
- `--eval-device cpu`

## 001 - Baseline, 500 Iterations

Command:

```sh
cargo run --release -p rustgs --bin rustgs -- train \
  --input /Users/tfjiang/Projects/RustScan/artifacts/inputs/tum_freiburg1_xyz_colmap \
  --output rustgs/output/experiments/tum_psnr/baseline_500.ply \
  --iterations 500 \
  --eval-after-train \
  --eval-json \
  --log-level info
```

Result:

- Training frames: 798
- Initial sparse points: 8986
- Final Gaussians: 12697
- Final loss: 0.08824351
- Mean PSNR: 13.7374 dB
- Min/Max PSNR: 9.8414 / 15.1664 dB
- Worst frame: frame 90 at 9.8414 dB
- Training time after release build: 13.90 s

Conclusion:

Baseline is learning and topology growth is active, but 500 iterations are far below the 20 dB target. Continue to 1000 iterations before changing code, because the PSNR trend is still useful and the loss is not divergent.

## 002 - Baseline, 1000 Iterations

Command:

```sh
cargo run --release -p rustgs --bin rustgs -- train \
  --input /Users/tfjiang/Projects/RustScan/artifacts/inputs/tum_freiburg1_xyz_colmap \
  --output rustgs/output/experiments/tum_psnr/baseline_1000.ply \
  --iterations 1000 \
  --eval-after-train \
  --eval-json \
  --log-level info
```

Result:

- Training frames: 798
- Final Gaussians: 19885
- Final loss: 0.074088156
- Mean PSNR: 13.0984 dB
- Min/Max PSNR: 9.9526 / 14.3633 dB
- Worst frame: frame 90 at 9.9526 dB
- Training time: 26.78 s

Conclusion:

The evaluation PSNR dropped from the 500-iteration baseline despite lower final loss and more Gaussians. This suggests the target is not just under-training. Investigate the camera/render/evaluation path before running longer experiments.

## 003 - Alpha-Blended Evaluation Renderer, 500 Iterations

Code change:

- Replaced the evaluation-facing host renderer path with tiled alpha blending instead of filled-circle nearest-depth overwrites.
- Added Rust-side SH color evaluation so evaluation can use the trained active SH degree instead of only SH0.
- Fixed projected pixel coordinates in the tiled renderer to include the camera principal point.

Command:

```sh
cargo run --release -p rustgs --bin rustgs -- train \
  --input /Users/tfjiang/Projects/RustScan/artifacts/inputs/tum_freiburg1_xyz_colmap \
  --output rustgs/output/experiments/tum_psnr/alpha_eval_500.ply \
  --iterations 500 \
  --eval-after-train \
  --eval-json \
  --log-level info
```

Result:

- Final Gaussians: 12698
- Final loss: 0.08743286
- Mean PSNR: 18.5591 dB
- Min/Max PSNR: 11.4852 / 22.7754 dB
- Worst frame: frame 90 at 11.4852 dB
- Training time: 12.47 s

Conclusion:

The PSNR jump from 13.7374 dB to 18.5591 dB confirms that the old host evaluation renderer was a major measurement bug. Continue with the same code at 1000 iterations to check whether the corrected metric reaches the 20 dB target without additional training changes.

## 004 - Alpha-Blended Evaluation Renderer, 1000 Iterations

Command:

```sh
cargo run --release -p rustgs --bin rustgs -- train \
  --input /Users/tfjiang/Projects/RustScan/artifacts/inputs/tum_freiburg1_xyz_colmap \
  --output rustgs/output/experiments/tum_psnr/alpha_eval_1000.ply \
  --iterations 1000 \
  --eval-after-train \
  --eval-json \
  --log-level info
```

Result:

- Final Gaussians: 19914
- Final loss: 0.07443036
- Mean PSNR: 18.7547 dB
- Min/Max PSNR: 13.1510 / 22.9378 dB
- Worst frame: frame 90 at 13.1510 dB
- Training time: 25.69 s

Conclusion:

The corrected PSNR improves from 500 to 1000 iterations, but only modestly. Because the dataset has 798 training frames, 1000 iterations still gives each frame very little exposure on average. Continue to a 3000-iteration midpoint before making optimizer changes.

## 005 - Alpha-Blended Evaluation Renderer, 3000 Iterations

Command:

```sh
cargo run --release -p rustgs --bin rustgs -- train \
  --input /Users/tfjiang/Projects/RustScan/artifacts/inputs/tum_freiburg1_xyz_colmap \
  --output rustgs/output/experiments/tum_psnr/alpha_eval_3000.ply \
  --iterations 3000 \
  --eval-after-train \
  --eval-json \
  --log-level info
```

Result:

- Final Gaussians: 67886
- Final loss: 0.065955244
- Mean PSNR: 18.8723 dB
- Min/Max PSNR: 15.5148 / 22.3305 dB
- Worst frame: frame 0 at 15.5148 dB
- Training time: 90.65 s

Conclusion:

The worst frame improves materially, but mean PSNR is almost flat from 1000 to 3000 iterations. Before running longer jobs, tighten the host tiled renderer to match the training shader more closely: use pixel centers and apply covariance blur plus opacity compensation.

## 006 - Shader-Matched Evaluation Renderer, 1000 Iterations

Code change:

- Matched the CPU tiled renderer more closely to the training WGSL shader:
  - evaluate pixels at `pixel + 0.5`;
  - add covariance blur;
  - apply opacity compensation after covariance blur.

Command:

```sh
cargo run --release -p rustgs --bin rustgs -- train \
  --input /Users/tfjiang/Projects/RustScan/artifacts/inputs/tum_freiburg1_xyz_colmap \
  --output rustgs/output/experiments/tum_psnr/shader_matched_eval_1000.ply \
  --iterations 1000 \
  --eval-after-train \
  --eval-json \
  --log-level info
```

Result:

- Final Gaussians: 19843
- Final loss: 0.0756308
- Mean PSNR: 20.7084 dB
- Min/Max PSNR: 13.3353 / 23.7871 dB
- Worst frame: frame 90 at 13.3353 dB
- Training time: 24.90 s

Conclusion:

The corrected evaluation path exceeds the 20 dB target by 1000 iterations. Because the requested target is specifically 30000 iterations, proceed to a final 30000-iteration confirmation after intermediate checks have shown the run is healthy.

## 007 - Shader-Matched Evaluation Renderer, Existing 3000-Iteration Scene

Command:

```sh
cargo run --release -p rustgs --example evaluate_psnr -- \
  --scene rustgs/output/experiments/tum_psnr/alpha_eval_3000.ply \
  --dataset /Users/tfjiang/Projects/RustScan/artifacts/inputs/tum_freiburg1_xyz_colmap \
  --render-scale 0.25 \
  --max-frames 180 \
  --frame-stride 30 \
  --json
```

Result:

- Scene iterations: 3000
- Scene Gaussians: 67886
- Mean PSNR: 21.2971 dB
- Min/Max PSNR: 16.0711 / 24.8413 dB
- Worst frame: frame 90 at 16.0711 dB

Conclusion:

The already-trained 3000-iteration scene exceeds the target once evaluated with the shader-matched host renderer. For the 30000-iteration confirmation, freeze topology after epoch 4 to avoid unnecessary primitive growth after the scene has already crossed 20 dB.

## 008 - Shader-Matched Evaluation Renderer, 30000 Iterations with Topology Freeze

Command:

```sh
cargo run --release -p rustgs --bin rustgs -- train \
  --input /Users/tfjiang/Projects/RustScan/artifacts/inputs/tum_freiburg1_xyz_colmap \
  --output rustgs/output/experiments/tum_psnr/final_30000_freeze_epoch4.ply \
  --iterations 30000 \
  --litegs-topology-freeze-after-epoch 4 \
  --eval-after-train \
  --eval-json \
  --log-level info
```

Result:

- Final Gaussians: 74115
- Final loss: 0.06689975
- Mean PSNR: 20.9688 dB
- Median PSNR: 21.7172 dB
- Min/Max PSNR: 17.2860 / 23.2353 dB
- Worst frame: frame 90 at 17.2860 dB
- Training time: 1009.20 s
- Evaluation: 6 frames, render scale 0.25, 160x120
- Parity gate: Passed

Conclusion:

The final 30000-iteration release run exceeds the 20 dB target with a mean PSNR of 20.9688 dB. Topology freeze after epoch 4 kept the final scene size bounded at 74115 Gaussians while preserving PSNR above target.

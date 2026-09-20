# RustGS TUM Profile Comparison 2026-04-06

## Scope

本次记录只回答一个当前最直接的问题：

- 在当前 `main` 训练路径下，`LiteGS` 后期 topology churn 是否值得继续保留？
- 如果在 late-stage 直接冻结 topology，能否在几乎不伤害质量的前提下减少无效训练开销？

本次对照不讨论 `Brush` 迁移，也不讨论 `gsplat-mlx-main` 对齐。

## Experiment Setup

### Dataset

- Input: `artifacts/inputs/tum/rgbd_dataset_freiburg1_xyz`
- Training subset:
  - `max_frames=180`
  - `frame_stride=30`
  - effective training frames: `6`

### Training Profile

- profile: `litegs-mac-v1`
- iterations: `720`
- render scale: `0.5`
- max initial gaussians: `100000`

### Compared Runs

1. `baseline`
   - no topology freeze
2. `freeze80`
   - `--litegs-topology-freeze-after-epoch 80`

### Important Runtime Caveats

1. 当前数据集没有 sparse-point initialization。
   - RustGS 回退到了 frame-based initialization。

2. Metal memory guard 将初始化高斯数量从 `54985` 压到了 `552`。
   - 因此这组结果反映的是“当前真实可跑配置”下的比较，不是无约束 LiteGS 上限。

3. 本次 `720` 实验最初暴露了一个 CLI post-train eval 问题。
   - 当时 `eval_frame_stride` 同时作用在“数据集加载”和“评估帧选择”两层。
   - 对于 `180/30`，训练命令最初只评到了 `1` 帧。
   - 因此本文质量对照先以 `examples/evaluate_psnr` 的 `6` 帧结果为准。
   - 当前分支已修复此问题：评估数据集加载只保留 `max_frames` 前缀裁剪，`frame_stride` 只在 `evaluate_scene()` 内应用一次。

## Commands

### Training

```bash
target/release/rustgs train \
  --input /Users/tfjiang/Projects/RustScan/artifacts/inputs/tum/rgbd_dataset_freiburg1_xyz \
  --output /tmp/rustgs-tum-freeze-study/tum_litegs_720_baseline_release.ply \
  --training-profile litegs-mac-v1 \
  --iterations 720 \
  --max-initial-gaussians 100000 \
  --max-frames 180 \
  --frame-stride 30 \
  --metal-render-scale 0.5 \
  --log-level info \
  --eval-after-train \
  --eval-render-scale 0.25 \
  --eval-max-frames 180 \
  --eval-frame-stride 30 \
  --eval-worst-frames 5 \
  --eval-device metal \
  --eval-json
```

```bash
target/release/rustgs train \
  --input /Users/tfjiang/Projects/RustScan/artifacts/inputs/tum/rgbd_dataset_freiburg1_xyz \
  --output /tmp/rustgs-tum-freeze-study/tum_litegs_720_freeze80_release.ply \
  --training-profile litegs-mac-v1 \
  --iterations 720 \
  --max-initial-gaussians 100000 \
  --max-frames 180 \
  --frame-stride 30 \
  --metal-render-scale 0.5 \
  --log-level info \
  --litegs-topology-freeze-after-epoch 80 \
  --eval-after-train \
  --eval-render-scale 0.25 \
  --eval-max-frames 180 \
  --eval-frame-stride 30 \
  --eval-worst-frames 5 \
  --eval-device metal \
  --eval-json
```

### Comparable Evaluation Gate

```bash
cargo run --manifest-path rustscan-gs/Cargo.toml --example evaluate_psnr -- \
  --scene /tmp/rustgs-tum-freeze-study/tum_litegs_720_baseline_release.ply \
  --dataset /Users/tfjiang/Projects/RustScan/artifacts/inputs/tum/rgbd_dataset_freiburg1_xyz \
  --render-scale 0.25 \
  --max-frames 180 \
  --frame-stride 30 \
  --device cpu \
  --json
```

对 `freeze80` 组同样执行一遍。

## Results

### Topology / Runtime

| Metric | baseline | freeze80 | delta |
|---|---:|---:|---:|
| training time | `31.05s` | `27.22s` | `-3.83s` |
| final gaussians | `223` | `260` | `+37` |
| prune events | `9` | `8` | `-1` |
| prune removed | `329` | `292` | `-37` |
| late-stage prune events | `1` | `0` | `-1` |
| late-stage opacity reset events | `2` | `0` | `-2` |
| topology freeze epoch | `null` | `80` | n/a |

### Quality: 6-frame Comparable Eval

| Metric | baseline | freeze80 | delta |
|---|---:|---:|---:|
| PSNR mean | `3.9983 dB` | `3.9899 dB` | `-0.0084 dB` |
| PSNR median | `3.9012 dB` | `3.8885 dB` | `-0.0127 dB` |
| PSNR min | `3.6295 dB` | `3.6580 dB` | `+0.0286 dB` |
| PSNR max | `4.5447 dB` | `4.5512 dB` | `+0.0065 dB` |
| PSNR std | `0.2937` | `0.2907` | `-0.0030` |

### Efficiency

- baseline `PSNR/time`: `0.1288 dB/s`
- freeze80 `PSNR/time`: `0.1466 dB/s`
- relative gain: `+13.8%`

## Observations

1. 当前 late-stage churn 真实存在。
   - `baseline` 在 `late_stage_start_epoch=80` 之后仍发生了：
   - `1` 次 late-stage prune
   - `2` 次 late-stage opacity reset

2. `freeze80` 达到了预期行为。
   - `epoch >= 80` 后没有再发生 topology update。
   - `last_prune_epoch` 从 `85` 提前到 `75`
   - `last_opacity_reset_epoch` 从 `90` 提前到 `70`

3. 冻结后训练更快，而且几乎没有质量损失。
   - 训练时间减少 `12.3%`
   - `PSNR mean` 只下降 `0.0084 dB`
   - 最差帧反而改善了 `0.0286 dB`

4. 当前分支里更大的问题其实不是 freeze 本身，而是初始化约束。
   - initialization 被 memory guard 压到 `552` 个高斯
   - 这一点对绝对质量上限的影响，比 freeze/no-freeze 的差异更大

## Current Decision

基于这组 `720-iter` 实测，当前代码状态下应优先接受下面的结论：

1. `LiteGS` 后期 topology freeze 是值得保留的能力。
2. `360` 之后继续让 topology 自由 churn，没有看到足够收益。
3. 当前分支已经修复 post-train eval 的双重 stride；后续训练对照应直接复用训练命令输出的多帧摘要。
4. 在修复 memory-guard 约束或初始化策略前，不应把这组数字外推为“最终 LiteGS 上限”。

## Next Steps

1. 用已修复的 post-train eval 重新跑一次 `720` 或直接跑 `1200`，把多帧摘要直接写回 parity report。
2. 再跑一次 `1200` iterations 的 `no-freeze vs freeze80`，确认结论是否继续成立。
3. 如果 `1200` 仍然成立，把 `topology_freeze_after_epoch` 收敛为默认 schedule，而不是实验开关。
4. 在此之后，再进入 scene-scale-aware normalization。

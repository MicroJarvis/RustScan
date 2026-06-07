# RustGS Refactor Guardrails

Related design: `../RustGS/docs/plans/2026-04-09-rustgs-soa-splat-architecture-proposal.md`  
Date: 2026-04-10

## Purpose

这份文档记录当前 RustGS 必须保持稳定的 public surface，以及在继续清理表示层和训练边界时必须保持可验证的最小回归基线。

它回答两个问题：

1. 现在什么是 RustGS 的真实 public contract？
2. 继续清理 legacy/compatibility 层时，最低限度要跑哪些验证？

## Verified Snapshot

Last verified: 2026-04-10

- `cargo check -p rustgs`
- `cargo test -p rustgs`
- active docs 已与当前 crate surface 对齐
- 已删除的 legacy public names 只会出现在显式的 removed-surface 说明里

说明：

- 真正的 Metal 训练与 benchmark 仍然需要 Metal 可用。
- 在当前机器的 sandbox 环境里，Metal-backed command 可能报 unavailable；非 sandbox 环境可正常执行。

## Public Compatibility Surface

### Library Entry Points

当前保留的 RustGS public contract：

- `rustgs::load_colmap_training_dataset_with_source`
- `rustgs::load_colmap_training_dataset`
- `rustgs::train_splats`
- `rustgs::evaluate_splats`
- `rustgs::runtime_from_splats`
- `rustgs::save_splats_ply`
- `rustgs::load_splats_ply`
- `rustgs::metal_available`
- `rustgs::run_metal_training_benchmark`
- `rustgs::TrainingCheckpoint`

### CLI Surface

当前用户侧 CLI contract 是 `rustgs train`，包括：

- TUM / COLMAP / `TrainingDataset` JSON 输入解析
- `TrainingProfile` 选择
- chunked training flags
- post-train evaluation flags
- LiteGS parity sidecar 输出

### Artifact Surface

当前 refactor 必须保持的 artifact 边界：

- PLY 导出与导入走 `HostSplats`
- checkpoint 持久化 `TrainingCheckpoint { splats: HostSplats, ... }`
- post-train evaluation 产出 `SplatEvaluationSummary`
- LiteGS 训练仍然写 parity sidecar

## Explicitly Removed Surface

以下名称已经从当前 public architecture 中删除，不应重新引回：

- `train_from_slam`
- `train_from_path`
- `train_scene`
- `evaluate_scene`
- `save_scene_ply`
- `load_scene_ply`
- `SlamOutput`-centric 训练入口
- `TrainableGaussians`
- `TrainableColorRepresentation`
- `training_pipeline.rs`
- `legacy/` module

## Current Training Flow Contract

当前顶层训练 contract 是：

1. CLI 或 library entry 解析出 `TrainingDataset`
2. 初始化与 checkpoint/export 边界使用 `HostSplats`
3. 训练 step loop 的 canonical runtime state 是 `diff::Splats`
4. trainer 在边界处把 runtime snapshot 回 `HostSplats`
5. 调用方导出 splat PLY、可选 parity sidecar 和 post-train evaluation summary

允许继续变化的是内部模块边界；不允许变化的是这条数据流的基本 ownership 形状。

## Minimum Regression Baseline

### 1. Crate Build and Full Test Suite

```bash
cargo check -p rustgs
cargo test -p rustgs
```

这是本轮清理之后的最低回归基线。任何进一步的 legacy/compatibility 清理都必须至少保持这两条命令通过。

### 2. TUM Smoke Integration

```bash
cargo test --manifest-path RustGS/Cargo.toml --features gpu --test tum_training -- --nocapture
```

覆盖内容：

- TUM dataset discovery
- evaluation-frame selection
- path-based training smoke
- post-train evaluation summary

### 3. Benchmark Baseline

```bash
cargo run --manifest-path RustGS/Cargo.toml --features gpu --example training_benchmark -- --profile litegs-mac-v1 --json
```

覆盖内容：

- forward / loss / backward / optimizer / step timing
- smoke training timing

### 4. CLI Smoke Baseline

```bash
cargo run --manifest-path RustGS/Cargo.toml --features gpu --bin rustgs -- train \
  --input test_data/tum \
  --output /tmp/rustgs-guardrail-smoke.ply \
  --training-profile litegs-mac-v1 \
  --iterations 1 \
  --max-frames 90 \
  --frame-stride 30 \
  --eval-after-train \
  --eval-render-scale 0.25 \
  --eval-max-frames 90 \
  --eval-frame-stride 30
```

期望输出：

- splat PLY
- evaluation summary
- LiteGS parity sidecar

## Internal Details Allowed To Change

只要上面的 public surface 和回归基线不坏，下面这些仍然可以继续重构：

- `metal_trainer.rs` 内部布局
- `metal_runtime.rs` 及其子模块分工
- `data_loading.rs` / `frame_loader.rs` / `init_map.rs`
- `topology.rs` / `density_controller.rs` 的策略边界
- optimizer state 布局
- eval / renderer 对 AoS 兼容层的进一步清理
- `benchmark_rejects_invalid_spec_before_requesting_metal`

## Refactor Rules

During the refactor:

- do not remove or rename a public entry point without an explicit migration note
- do not change scene export or evaluation summary shape silently
- do not merge or delete topology helpers until the guardrail tests above still pass
- do not claim a benchmark baseline exists unless the benchmark example compiles and runs
- do not treat parity reporting as optional for LiteGS profile validation

## Notes

- `metal_available()` is intentionally public because smoke tests and examples need a stable capability probe.
- `run_metal_training_benchmark()` is intentionally public because the benchmark example is part of the Epic 1 baseline.
- The existence of this document does not replace tests; it identifies which tests are the current refactor gate.

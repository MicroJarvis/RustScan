# RustScan Architecture

**Updated:** 2026-04-10

## Overview

RustScan 是一个多 crate 的 3D 重建工作区，但当前主线已经明确收敛到 RustGS 的纯 3DGS 训练架构。RustGS 不再把 SLAM 输出、scene/map ownership、或 legacy compatibility API 当作核心设计前提。

## Workspace Crates

- `rustscan-types`: 跨 crate 共享的数据结构。
- `RustSLAM`: 视觉 SLAM、稀疏地图、回环与数据摄取。
- `RustGS`: 3D Gaussian Splatting 训练、评估、parity 与 chunked training。
- `RustMesh`: 网格处理与 OpenMesh 对齐算法。
- `RustViewer`: 结果检查与可视化。
- `RustFF`: 前馈式推理实验工具。

## Cross-Crate Flow

1. 外部数据源或 `RustSLAM` 提供图像、位姿和可选稀疏点。
2. `RustGS` 将 COLMAP sparse reconstruction 解析为 `TrainingDataset`。
3. `RustGS` 初始化并训练 splats，导出 splat PLY、checkpoint 与评估摘要。
4. `RustViewer` 或其他工具消费导出的 splat/checkpoint 产物。
5. `RustMesh` 只在需要网格后处理时介入，不参与 RustGS 核心训练状态设计。

## Current RustGS Training Architecture

### Public Entry Surface

当前 RustGS 保留的训练主入口是 splat-first 的：

- `rustgs::load_colmap_training_dataset_with_source`
- `rustgs::load_colmap_training_dataset`
- `rustgs::train_splats`
- `rustgs::evaluate_splats`
- `rustgs::runtime_from_splats`
- `rustgs::save_splats_ply`
- `rustgs::load_splats_ply`
- `rustgs::metal_available`
- `rustgs::run_metal_training_benchmark`
- CLI `rustgs train`

已经删除的 legacy public surface 不再属于架构契约：

- `train_from_slam`
- `train_from_path`
- `train_scene`
- `evaluate_scene`
- `save_scene_ply`
- `load_scene_ply`
- `SlamOutput`-centric flow

### Canonical State Roles

RustGS 当前的 splat 表示是分层但单向的：

- `TrainingDataset`: 输入训练样本。
- `training::HostSplats`: host 侧 SoA 边界类型，用于初始化、checkpoint、PLY 导入导出。
- `diff::Splats`: device/runtime 侧可微训练状态，是 step loop 的 canonical owner。
- `training::SplatView`: host 侧只读借用视图。

`render::Gaussian` 仍然存在，但它只是 CPU renderer / 测试 / 局部兼容路径的 AoS 适配类型，不再是 RustGS 核心 ownership 模型的一部分。

### Data and Initialization

- `data_loading.rs`: `TrainingDataset` 到训练载荷的转换。
- `frame_loader.rs`: 帧解码与缓存。
- `init_map.rs`: 稀疏点或帧驱动初始化。
- `chunk_planner.rs`: chunk planning 与 per-chunk 数据集物化。
- `splats.rs`: `HostSplats`、`SplatView` 以及 host/device 边界能力。
- `splat_interop.rs`: `HostSplats <-> render::Gaussian` 适配，仅保留必要兼容桥接。

### Execution Planning

`RustGS/src/training/mod.rs` 已经缩成训练装配层，活跃逻辑分散到更窄的模块：

- `config.rs`: `TrainingConfig`、`TrainingProfile`、LiteGS 配置。
- `orchestrator.rs`: `train_splats()` 路由。
- `execution_plan.rs`: standard / chunked route 选择。
- `chunk_training.rs`: sequential chunk 执行。
- `export.rs`: chunk artifact / report 持久化。

### Step Execution and Runtime

Metal 训练路径已经拆成显式子模块：

- `metal_forward.rs`
- `metal_loss.rs`
- `metal_backward.rs`
- `metal_optimizer.rs`
- `metal_trainer.rs`

Metal runtime 也已经从单体模块拆解为：

- `metal_kernels.rs`
- `metal_pipelines.rs`
- `metal_resources.rs`
- `metal_dispatch.rs`
- `metal_projection.rs`
- `metal_raster.rs`
- `metal_runtime.rs`

### Topology and Evaluation

- `topology.rs`: densify / prune / opacity reset 调度与 mutation。
- `density_controller.rs`: LiteGS reference/parity adapter。
- `parity_harness.rs`: LiteGS parity gate 与对照报告。
- `eval.rs`: PSNR、evaluation summary、post-train 评估。

### Removed Legacy Structure

下列结构已经不再存在于当前源码主路径：

- `RustGS/src/legacy/*`
- `RustGS/src/training/training_pipeline.rs`
- `RustGS/src/io/dataset_loader.rs`
- `RustGS/src/io/scene_io/scene_import.rs`
- `RustGS/src/io/scene_io/scene_export.rs`

## Current Architectural Constraints

- RustGS 仍然是 Metal-first 训练后端，没有引入多后端抽象。
- 用户侧默认训练 profile 已经收口到 `LiteGsMacV1` 语义；legacy 只体现在 removed-surface 说明与显式拒绝旧 flag 的测试上，不再体现在活跃实现里。
- 训练核心已经是 SoA，但评估/CPU renderer 周边仍有少量 `Gaussian` AoS 适配层。
- 质量侧工作仍未结束，TUM PSNR、scene-scale-aware normalization、parity gate 仍是后续重点。

## Canonical Companion Docs

- [current-project-status.md](current-project-status.md)
- [plans/2026-04-06-rustgs-refactor-guardrails.md](plans/2026-04-06-rustgs-refactor-guardrails.md)
- [../RustGS/docs/plans/2026-04-09-rustgs-soa-splat-architecture-proposal.md](../RustGS/docs/plans/2026-04-09-rustgs-soa-splat-architecture-proposal.md)
- [plans/2026-04-05-litegs-parity-roadmap-refresh.md](plans/2026-04-05-litegs-parity-roadmap-refresh.md)
- [RustGS-TUM-Profile-Comparison-2026-04-06.md](RustGS-TUM-Profile-Comparison-2026-04-06.md)

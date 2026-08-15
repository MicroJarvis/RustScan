# RustScan Architecture

**Updated:** 2026-08-15

## Overview

RustScan 是一个多 crate 的 3D 重建工作区。RustSFM、RustViewer 和 RustGS 是当前活跃的重建工作流；RustGS 的纯 3DGS 训练架构仍不把 SLAM 输出、scene/map ownership 或 legacy compatibility API 当作核心设计前提。

## Workspace Crates

- `rustscan-types`: 跨 crate 共享的数据结构。
- `RustSLAM`: 视觉 SLAM、稀疏地图、回环与数据摄取。
- `RustGS`: 3D Gaussian Splatting 训练、评估、parity 与 chunked training。
- `RustMesh`: 网格处理与 OpenMesh 对齐算法。
- `RustViewer`: 结果检查与可视化。
- `RustFF`: 前馈式推理实验工具。
- `RustSFM`: COLMAP-style 特征、匹配、两视图验证、增量 SfM 与序列注册。

## Cross-Crate Flow

1. 外部图像、视频或 `RustSLAM` 提供图像、位姿和可选稀疏点。
2. `RustSFM` 可从图像生成 COLMAP-compatible sparse reconstruction；`RustViewer` 负责其项目级编排。
3. `RustGS` 将 COLMAP sparse reconstruction 解析为 `TrainingDataset`，训练 splats 并导出 PLY、checkpoint 与评估摘要。
4. `RustViewer` 或其他工具消费重建与训练产物。
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
- `rustgs::gpu_available`
- `rustgs::TrainingCheckpoint` and `rustgs::SharedWgpuContext`
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
- `HostSplats`: host 侧 SoA 边界类型，用于初始化、checkpoint、PLY 导入导出。
- `training::engine::DeviceSplats`: GPU step loop 中的可微内部状态；不构成 public API。
- `TrainingCheckpoint`: 可恢复训练的持久化边界，包含 `HostSplats`、optimizer 和 topology 状态。
- `SplatView`: `HostSplats` 的 host 侧只读借用视图。

`render::Gaussian` 仍然存在，但它只是 CPU renderer / 测试 / 局部兼容路径的 AoS 适配类型，不再是 RustGS 核心 ownership 模型的一部分。

### Data and Initialization

- `training/data/frame_loader.rs`: 帧解码、缓存和预取。
- `training/data/frame_targets.rs`: 训练目标准备。
- `training/data/init_map.rs`: 从稀疏点或帧初始化 `HostSplats`。
- `training/engine/splats.rs`: `HostSplats` 与内部 `DeviceSplats` 的显式上传/读回边界。
- `core`: `HostSplats`、`SplatView`、相机和其他训练中立的数据类型。

### Execution Planning

`RustGS/src/training/mod.rs` 负责 public re-export 与训练入口；实际执行由
`training/engine` 装配：

- `config.rs`: `TrainingConfig`、`TrainingBackend::Wgpu` 和 LiteGS 配置。
- `events.rs`: 训练进度、暂停、取消、snapshot 与 checkpoint 事件边界。
- `checkpoint.rs`: 版本化 checkpoint 和训练 identity 校验。
- `engine/runtime.rs`: `train_splats()` 的运行时编排、frame 顺序和 resume 流程。
- `engine/trainer.rs`: 训练 step、snapshot、topology 累积和 telemetry。
- `engine/optimizer.rs` 与 `engine/loss.rs`: Adam 状态及损失计算。

### Step Execution and Runtime

GPU 渲染与梯度路径按前向/反向阶段拆分：

- `forward/`: projection、visibility compaction、tile mapping、sorting 和 rasterization。
- `backward/`: rasterization/projection 的反向传播与 Burn autodiff 接入。
- `gpu_primitives/`: radix sort 与 prefix sum。
- `engine/backend.rs`: Burn/CubeCL wgpu backend 类型与 device 绑定。

### Topology and Evaluation

- `topology/`: densify、prune、opacity reset 的调度、选择、snapshot 和 mutation。
- `evaluation/core.rs`: PSNR、evaluation summary、GPU renderer 和共享 wgpu context。
- `evaluation/parity.rs`: LiteGS fixture、threshold、comparison 与 gate。
- `reporting/`: training metrics、parity telemetry 与最后一次训练 telemetry。

### Removed Legacy Structure

下列结构已经不再存在于当前源码主路径：

- `RustGS/src/legacy/*`
- `RustGS/src/training/training_pipeline.rs`
- `RustGS/src/io/dataset_loader.rs`
- `RustGS/src/io/scene_io/scene_import.rs`
- `RustGS/src/io/scene_io/scene_export.rs`

## Current Architectural Constraints

- RustGS 的唯一训练后端是 Burn/CubeCL 上的 wgpu；macOS 通过 wgpu 选择 Metal adapter，而不是维护一套独立的 Metal runtime。
- `TrainingBackend` 只保留 `Wgpu`，用于兼容既有配置构造方式。
- 旧 JSON checkpoint 仍通过显式的 `legacy` namespace 与 deprecated alias 提供读取兼容；可恢复 checkpoint 使用版本化的 `TrainingCheckpoint`。
- 训练核心已经是 SoA，但评估/CPU renderer 周边仍有少量 `Gaussian` AoS 适配层。
- 质量侧工作仍未结束，TUM PSNR、scene-scale-aware normalization、parity gate 仍是后续重点。

## Companion Docs

- [current-project-status.md](current-project-status.md)
- [../RustGS/README.md](../RustGS/README.md)
- [../RustSFM/README.md](../RustSFM/README.md)
- [../RustSFM/PARITY_ROADMAP.md](../RustSFM/PARITY_ROADMAP.md)

The dated RustGS design and benchmark documents listed in `docs/index.md` are
historical evidence. They do not define the current module layout or public API.

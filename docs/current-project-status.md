# RustScan Current Project Status

**Updated:** 2026-08-31
**Branch:** `main`

## Overall

当前 `main` 同时维护 RustSFM 稀疏重建、RustViewer 工作流和 RustGS 训练架构。RustGS 的 splat-first 收口仍然有效，但它不再是描述整个 workspace 活跃状态的唯一主线。

## Verified Snapshot

本轮 RustSFM 硬化后的本地验证结果是：

- `cargo test -p rustsfm --lib`: `713 passed; 0 failed; 19 ignored`
- `cargo test -p rustsfm --lib --no-default-features`: `586 passed; 0 failed; 19 ignored`
- `cargo test -p rustsfm --test sequence_registration --no-default-features`: `62 passed; 0 failed`
- `cargo test -p rustsfm --lib -- --ignored`（本地 provision 夹具后）: `19 passed; 0 failed`

被忽略的 `real_colmap_sparse_*` 测试需要工作区外部的 `test_data/flowers2_colmap` 夹具；该夹具不在 Git 或 submodule 中，必须显式用 `--ignored` 执行。2026-08-31 通过 `scripts/provision_flowers2_colmap_fixture.sh`（SHA-256 固定内容）重新 provision 后实测为 `19 passed; 0 failed`。此前两个失败项（scheduled global BA 不触发、先验位置 BA 未对齐）已按 COLMAP `CheckRunGlobalRefinement`/`PosePriorBundleAdjuster` 语义修复（见 PARITY_ROADMAP 的 BA orchestration 修复记录）。

## Current Progress

- RustSFM 是 workspace 的主动维护 crate，提供 COLMAP-style 特征、匹配、两视图验证、增量 mapper、序列注册和文本导出。
- RustSFM 硬化已合并：keyframe mapper 输入固定为私有快照，共享输出被序列化，GPU PnP-focal 管线失败或不支持的路由（generalized rig/structureless）会返回既有 CPU fallback 并记录遥测，BA 重投影残差对负深度连续求值（`img_from_cam_unchecked`，COLMAP 语义，2026-09-02），scheduled BA 只限制中间精化为最多两轮/每轮 15 次迭代而保留 initial/final 完整质量预算（60 图固定 seed 基准快 43%，点数 +6.1%、观测 +1.2%、平均误差下降），macOS GPU context 测试也已串行化。
- RustViewer 维护导入媒体、运行 RustSFM、消费 COLMAP 产物并衔接 RustGS 训练的桌面工作流；macOS 新项目默认使用并行 CPU VLFeat SIFT（60 图实测 23.3s / 638,895 特征，对比当前 wgpu SIFT 59.6s / 424,554 特征），既有 manifest 保留用户已保存的 GPU 选择。顺序匹配保留 GPU cross-check：60 图实测 130.2s（CPU 264.5s）；禁用 cross-check 虽降到 89.1s，但 mapper 注册从 60/60 降到 56/60。
- RustGS 的公开训练路径仍收口到 splat-first API；其细节和质量路线见架构文档及 RustGS 专项文档。

## Active Gaps

- `flowers2_colmap` parity fixture 的来源、版本和 CI 获取方式尚未固化；默认测试不会再假设它存在。
- RustSFM 的 COLMAP 数值 parity、RustGS 的 LiteGS parity/TUM PSNR，以及 RustViewer 的端到端真实媒体验证仍需要各自的专门验收。
- RustSLAM 的 dependency-minimal library suite 在 2026-08-15 实测为 `244 passed; 1 failed`；失败项为 `tracker::vo::tests::test_initialize_keeps_relocalized_pose_in_global_frame`。
- RustFF 的默认 library suite 在 2026-08-15 为 `2 passed; 0 failed`，但可选 `onnx-ort` feature 仍使用旧 ORT API，当前不能编译。

## Next Priorities

1. 固化并可复现地提供 RustSFM `flowers2_colmap` parity fixture，作为独立 opt-in CI 验收。
2. 维护 RustSFM default/minimal-feature 测试与 macOS GPU context 串行化的 CI 覆盖。
3. 继续 RustGS parity/TUM 质量闭环，并在完成后更新其专项状态。

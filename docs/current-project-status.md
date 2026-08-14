# RustScan Current Project Status

**Updated:** 2026-08-14
**Branch:** `main`

## Overall

当前 `main` 同时维护 RustSFM 稀疏重建、RustViewer 工作流和 RustGS 训练架构。RustGS 的 splat-first 收口仍然有效，但它不再是描述整个 workspace 活跃状态的唯一主线。

## Verified Snapshot

本轮 RustSFM 硬化后的本地验证结果是：

- `cargo test -p rustsfm --lib`: `708 passed; 0 failed; 19 ignored`
- `cargo test -p rustsfm --lib --no-default-features`: `581 passed; 0 failed; 19 ignored`
- `cargo test -p rustsfm --test sequence_registration --no-default-features`: `62 passed; 0 failed`

被忽略的 `real_colmap_sparse_*` 测试需要工作区外部的 `test_data/flowers2_colmap` 夹具；该夹具不在 Git 或 submodule 中，必须显式用 `--ignored` 执行。

## Current Progress

- RustSFM 是 workspace 的主动维护 crate，提供 COLMAP-style 特征、匹配、两视图验证、增量 mapper、序列注册和文本导出。
- 当前 RustSFM 硬化将 keyframe mapper 输入固定为私有快照，序列化共享输出，并将 GPU PnP-focal 管线失败返回给现有 CPU fallback。
- RustViewer 维护导入媒体、运行 RustSFM、消费 COLMAP 产物并衔接 RustGS 训练的桌面工作流。
- RustGS 的公开训练路径仍收口到 splat-first API；其细节和质量路线见架构文档及 RustGS 专项文档。

## Active Gaps

- `flowers2_colmap` parity fixture 的来源、版本和 CI 获取方式尚未固化；默认测试不会再假设它存在。
- RustSFM 的 COLMAP 数值 parity、RustGS 的 LiteGS parity/TUM PSNR，以及 RustViewer 的端到端真实媒体验证仍需要各自的专门验收。

## Next Priorities

1. 固化并可复现地提供 RustSFM `flowers2_colmap` parity fixture，作为独立 opt-in CI 验收。
2. 提交并发布当前 RustSFM 硬化与 CI 覆盖。
3. 继续 RustGS parity/TUM 质量闭环，并在完成后更新其专项状态。

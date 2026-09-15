# RustScan Current Project Status

**Updated:** 2026-09-15
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

- `flowers2_colmap` 运行时树仍需 `./scripts/provision_flowers2_colmap_fixture.sh` 生成；参考文本已提交在 `test_data/fixtures/flowers2_colmap_ref_text_20260630`，opt-in CI job `rustsfm-flowers2-parity-opt-in`（`workflow_dispatch` 或 PR label `flowers2-parity`）可验收 ignored suite。默认 push/PR 仍不假设夹具存在。
- RustSFM 的 COLMAP 数值 parity、RustGS 的 LiteGS parity/TUM PSNR，以及 RustViewer 的端到端真实媒体验证仍需要各自的专门验收。
- RustSLAM 的 dependency-minimal library suite 在 2026-09-03 实测为 `245 passed; 0 failed`。此前 `tracker::vo::tests::test_initialize_keeps_relocalized_pose_in_global_frame` 的 fixture 仅有 10 个点，其中 7 个可三角化，误触发生产的 8 点退化保护；fixture 已扩展至 16 个空间分布点，生产门槛保持不变。
- RustFF 的默认 library suite 在 2026-09-03 为 `2 passed; 0 failed`，`onnx-ort` feature 已迁移至 pinned ORT 2.0 RC API 并为 `3 passed; 0 failed`；Spann3R decoder ONNX execution 仍未实现，故该 feature 不是端到端推理后端。`onnx-candle` 仍仅暴露未接线依赖。

## flowers2 960 帧基线（2026-09-13 settlement / 2026-09-15 retest）

| 阶段 | settlement | retest（`8b0b63b`） | 质量 |
|---|---:|---:|---|
| extract（CPU SIFT） | 332.3 s | — | 960 images / 6,612,183 keypoints |
| match | 2146.9 s | **1668.4 s** | 14,045 pairs / 13,894 verified / 10,019,172 matches（逐项一致） |
| reconstruct（frozen matching.db） | 598.9 s | 698.8 s（单轮，未判定是否回归） | 960/960 / 437,185→437,192 points / 1 model |

retest matching 分账（`output/flowers2_960_retest_20260915/matching.json`）：描述子匹配 GPU 等待 643 s；geometry scorer 读回等待 E 165 / F 181 / H 240 = 586 s（46 万次同步 × ~1.3 ms 固定延迟，与字节数无关，R12/S1 已证）；essential 候选生成（CPU f64 五点法）196 s；F/H 候选生成 + CPU refinement 127 s。reconstruct 内 27 次 global BA ≈ 331 s。

GPU f32 五点法实验（分支 `gpu-five-point-f32`，Q1–Q5b / P1–P3）已于 2026-09-15 冻结：生产五点法保持 CPU f64；Apple GPU 无 shader f64，df64 不值；按生产粒度 GPU 比 CPU8 慢约 5×，进生产前提是跨 pair 聚合流水线。见该分支 `docs/gpu-five-point-pipeline-TODO.md` 冻结决定。

## Next Priorities

按顺序逐项执行，每项完成后 review 全方案再进入下一项；需要时重跑 960 帧全流程更新真实数字：

1. **geometry scorer 同步点削减**（586 s）：设备端 argmax 去除独立 `mask_calls`、决策窗口扩大；单变量、逐位输出门（`tools/pair_output_hash.py`）、3×3 交错；48 帧 smoke 不够，按 96→192 升级验证（essential 每 pair 成本在 960 帧超线性）。
2. **pair 级粗粒度 CPU 候选生成 / GPU 打分流水线**：pair N+1 的 CPU 生成与 pair N 的 GPU 打分重叠，线程内部串行；禁止细粒度 rayon hand-off（R13 已证毒化 `device.poll`）。
3. **reconstruct**：核查 960 相机时 Ceres 线性求解器（48 帧为 Eigen DenseSchur；COLMAP >50 图切 SPARSE_SCHUR）；同 `matching.db` 三轮复现 599→699 s 是噪声还是回归。
4. **`post_bogus_cameras` / BA 质量债**独立复现与修复（与性能实验分目录，不混轮）。
5. **描述子匹配 GPU kernel**（643 s，算力瓶颈）：先做 kernel 级画像；或 pair 选择策略（需独立质量门）。
6. 继续 RustGS parity/TUM 质量闭环；维护 RustSFM CI 覆盖，需要时触发 flowers2 opt-in job。

# RustScan Current Project Status

**Updated:** 2026-09-18
**Branch:** `main`

## Overall

当前 `main` 同时维护 RustSFM 稀疏重建、RustViewer 工作流和 RustGS 训练架构。RustGS 的 splat-first 收口仍然有效，但它不再是描述整个 workspace 活跃状态的唯一主线。

## Verified Snapshot

2026-09-17 macOS arm64 / rustc 1.97.0 本地实测。`--no-default-features` 只保证编译和非 GPU codec 测试；需要 adapter 的测试必须显式启用 `gpu-wgpu,vlfeat-sift`。较早的 784 通过记录见 `output/p0_baseline_20260917/REPORT.md`，不能覆盖随后的初始化改动。下列数字是 R1/R2 修复后的复验，任务 4 仍未验收。2026-09-17 的 seed 0 / 1 线程对照见 `docs/RustSFM-TODO-技术方案-2026-09-17.md` 第 12.6 节：首个 accepted pair 与 COLMAP 3.13.0 都是 `frame_0002.jpg -> frame_0018.jpg`，注册顺序和模型切分未对齐。

- `cargo test -p rustsfm --lib --features gpu-wgpu,vlfeat-sift -- --test-threads=1`: `792 passed; 0 failed; 19 ignored`（测试耗时 64.23 s）
- `cargo test -p rustsfm --test sequence_registration --features gpu-wgpu,vlfeat-sift -- --test-threads=1`: `72 passed; 0 failed; 0 ignored`（58.41 s）
- `cargo test -p rustsfm --lib --features gpu-wgpu,vlfeat-sift -- --ignored --test-threads=1`（已有 `test_data/flowers2_colmap`）: `19 passed; 0 failed`（4.55 s）
- `cargo test -p rustsfm --lib --no-default-features -- --test-threads=8`: `603 passed; 0 failed; 19 ignored`
- `cargo test -p rustsfm --test sequence_registration --no-default-features -- --test-threads=8`: `55 passed; 0 failed`

被忽略的 `real_colmap_sparse_*` 测试需要工作区外部的 `test_data/flowers2_colmap` 夹具；该夹具不在 Git 或 submodule 中，必须显式用 `--ignored` 执行。2026-09-17 用 `scripts/provision_flowers2_colmap_fixture.sh` 重新 provision 后，SHA-256 与脚本钉死值一致，ignored suite 为 `19 passed; 0 failed`。这 19 个测试加载的 COLMAP 参考模型是 24/24 注册、1 个模型、6449 点、34234 观测、平均重投影误差 0.638769 px；这不是一次新的 RustSFM 全量重建。2026-08-31 的 scheduled-BA / pose-prior 修复记录仍见 PARITY_ROADMAP。

## Current Progress

- RustSFM 是 workspace 的主动维护 crate，提供 COLMAP-style 特征、匹配、两视图验证、增量 mapper、序列注册和文本导出。
- RustSFM 硬化已合并：keyframe mapper 输入固定为私有快照，共享输出被序列化，GPU PnP-focal 管线失败或不支持的路由（generalized rig/structureless）会返回既有 CPU fallback 并记录遥测，BA 重投影残差对负深度连续求值（`img_from_cam_unchecked`，COLMAP 语义，2026-09-02），scheduled BA 只限制中间精化为最多两轮/每轮 15 次迭代而保留 initial/final 完整质量预算（60 图固定 seed 基准快 43%，点数 +6.1%、观测 +1.2%、平均误差下降），macOS GPU context 测试也已串行化。
- RustViewer 维护导入媒体、运行 RustSFM、消费 COLMAP 产物并衔接 RustGS 训练的桌面工作流；macOS 新项目默认使用并行 CPU VLFeat SIFT（60 图实测 23.3s / 638,895 特征，对比当前 wgpu SIFT 59.6s / 424,554 特征），既有 manifest 保留用户已保存的 GPU 选择。顺序匹配保留 GPU cross-check：60 图实测 130.2s（CPU 264.5s）；禁用 cross-check 虽降到 89.1s，但 mapper 注册从 60/60 降到 56/60。
- RustGS 的公开训练路径仍收口到 splat-first API；R01～R12（mask detach、voxel init、Home/flowers2 验收、瓶颈排名、Abs 对照）证据见 `docs/reviews/2026-09-18-rustgs-training-review-and-home-evidence.md`。残留仅 TUM 全视图阶梯（TODO `R09b`）。

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

1. **geometry scorer 同步点削减（R14，960 已确认）**：matching **1668→1108 s**（−33.6%），geometry 998→455 s，scorer wait 586→167 s，`mask_calls=0`，pairs digest 与 settlement **逐位相同** `a4e8e8ec…2295c9`。不再做设备端 argmax、不扩大 64-trial 决策窗口。报告 `output/flowers2_960_r14_20260915/REPORT.md`。
2. **CPU/GPU 重叠（R16 保留）**：matching 1108→1094 s。R18 去掉 overlap 后 first48 15.57→19.46 s（拒绝）。R19 把 next-pair first-batch 塞进 essential scorer wait 后 first48 15.57→18.73 s（拒绝）。见 `output/pair_pipeline_r18_20260916/REPORT.md`、`output/pair_pipeline_r19_20260916/REPORT.md`。
3. **reconstruct（R17 matching.db 复测）**：606.3 s / 960/960 / 437,186 points。仍在 settlement 599–retest 699 带内，未大幅度超过主 worktree。27 次 global BA：DenseSchur×11 → SparseSchur×16，solve 316 s。无 post_bogus / camera_reset。
4. **`post_bogus_cameras` / BA 质量债**：post-BA 只回滚 bogus camera、保留 pose/point。24 帧 image-only 当前二进制：24/24、4239 points、`camera_reset=`×23、final BA 已提交、无 skip。报告 `output/post_bogus_repro_20260916/REPORT.md`。
5. **描述子匹配 GPU kernel（R17）+ 五点法去堆分配（R20）**：matching **1668→715 s**（−57%），digest 仍 `a4e8e8ec…2295c9`。R20 essential cand-gen 215→200 s。报告 `output/pair_pipeline_r17_20260916/REPORT.md`、`output/pair_pipeline_r20_20260916/REPORT.md`。
6. 补齐 RustGS TUM 全视图阶梯（`R09b`）；维护 RustSFM CI 覆盖，需要时触发 flowers2 opt-in job。

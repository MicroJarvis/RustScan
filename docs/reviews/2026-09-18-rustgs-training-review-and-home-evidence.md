# RustGS 2026-09-18 审核与 Home 实验记录

**审核版本：** `938fa8d`；对照版本：`09f9254`。

本文件保存原 TODO 的实验记录，不作为待办清单。当前执行顺序见 [RustGS TODO](../rustgs-TODO-训练效果与效率优化-2026-09-17.md)。

## 审核结论与记录限制

- 发现 4 项问题：非采样步 overflow 丢失、scan 输出缓存覆盖存活结果、保留累计可见性导致不可见窗口无法推进、Threshold 在非增密剪枝步骤关闭 opacity 判定。
- scan 生命周期问题已在 GPU 上复现：第二次同长度 scan 将第一次结果从 `[1,3,6]` 改成 `[10,30,60]`。
- 2026-09-18 审核运行结果：无 GPU 特性库测试 17/17、GPU 库测试 76/76、显式启用的 GPU integration 1/1 通过；scan 补充复现失败；`cargo fmt --package rustgs --check` 未通过。
- 下方为当时实验记录。其“通过”仅指已记录的 Home 短跑指标，不代表本次提交已通过正确性或跨场景验收。
- overflow 检测存在漏报，所以“没有报错”不能证明所有步骤均未溢出；prune 数量变化也不能单独归因于预期的可见性改进。
- 提交已经改变默认路径的实现行为。“默认配置仍不改”不能解释为默认训练行为未变；尚缺的是变更后的质量验收。
- peak RSS 是进程常驻内存，不能代替 GPU 峰值显存。TUM 全视图、外部场景和独立留出视图验证尚未完成。

## R01 修复证据（2026-09-18）

- 代码：`StickyForwardOverflow` + `accumulate_sticky_forward_overflow`（`reporting/metrics.rs`）；`train_step` 在 forward 后、loss/Adam/topology 前消费 sticky；`checkpoint()` 拒绝 sticky 状态下的成功导出；`ForwardCapacityExceeded` 增加 `first_iteration`。
- 语义：恰好满容量（`requested == capacity`）不失败；首次异常信息保留；后续健康步不能清掉 sticky。
- 验证：
  - `cargo test -p rustgs --lib --no-default-features reporting::metrics` → 6 passed（含非采样溢出策略、连续溢出、exact-full）
  - `cargo test -p rustgs --lib --features gpu-wgpu sticky_overflow_rejects` → checkpoint 拒绝通过
- 说明：健康步仍探测 1×u32 overflow 标志以在更新前硬失败；loss / 完整 intersection 计数仍按原 cadence 延迟读回。未做自动扩容重试。

## R02 修复证据（2026-09-18）

- 问题：`scratch_output` 按 len/device/dtype 复用同一 output handle；第二次同长度 scan 覆盖第一次仍持有的结果。
- 修复：`inclusive_scan` 每次分配独立 `output` 与 `block_sums`（去掉返回值池化），避免跨调用别名和递归帧互相覆盖。
- 验证：`cargo test -p rustgs --lib --features gpu-wgpu prefix_sum`
  - `consecutive_same_length_scans_keep_independent_results`：先扫 `[1,2,3]` 再扫 `[10,20,30]`，后读前者仍为 `[1,3,6]`
  - `scan_results_survive_length_changes_and_reuse`：短→长→短，旧结果不变
  - 原有 CPU reference / dispatch-count 用例仍通过

## R03 修复证据（2026-09-18）

- 增密继续使用累计 `visible_observations` 做归一化分母。
- 剪枝 `visible_count` / 不可见窗口改用与上一 topology 步的差分（`visibility_window_baseline`）；`SkipNoEligibleCandidates` 保留增密累计时仍推进 baseline。
- checkpoint 恢复后 baseline 对齐已恢复的累计可见性；mutation remap 同步 baseline。
- 验证：`visibility_window_delta_ignores_retained_cumulative_history`、`retained_accumulators_still_advance_invisible_windows`

## R04 修复证据（2026-09-18）

- Threshold：`opacity_prune_enabled = schedule.prune`（仅剪枝步可清低 opacity）。
- Weight / VisibilityWeight：opacity 清理仍跟 `schedule.densify`；prune-only 仍走 history-invisible / visibility-ratio。
- 三种模式分支拆开，避免共享 match 臂互相改写。
- 验证：`threshold_prune_only_step_removes_low_opacity_keeps_visible_high_opacity`、`weight_opacity_cleanup_stays_densify_gated_on_prune_only_steps`

## R05 修复证据（2026-09-18）

- `cargo fmt --package rustgs` 已整理并通过 `--check`。
- Adam remap 补强：
  - `optimizer_remap_keeps_surviving_moments_and_step`：三组参数 × moment1/moment2，`step=7`，新行为零，`origins=[Some(1),None,Some(0)]`
  - `optimizer_remap_checkpoint_restore_preserves_next_step_update`：remap → checkpoint → restore → 下一步后 `step=8`，参数与 twin 一致，新行 moments 非零
- 门禁（日志：`artifacts/runs/rustgs-optimization/2026-09-18/r05-gates/`）：

| 命令 | 结果 |
|---|---|
| `cargo fmt --package rustgs --check` | pass |
| `cargo test -p rustgs --lib --no-default-features --no-fail-fast` | **22 passed** / 0 failed / 0 ignored |
| `cargo test -p rustgs --lib --features gpu-wgpu --no-fail-fast` | **90 passed** / 0 failed / 0 ignored |
| `cargo test -p rustgs --test integration_test --features gpu-wgpu -- --ignored --test-threads=1` | **1 passed**（显式跑 ignored GPU adapter 用例） |
| `cargo test -p rustgs --test checkpoint_resume --features gpu-wgpu` | **49 passed** / 0 failed / 0 ignored（无 ignore 项） |

- 说明：integration / checkpoint_resume 需 `--features gpu-wgpu` 才能编译/覆盖真实训练路径；未用无 GPU 特性测试代替 GPU 验证。

## R06 修复证据（2026-09-18）

- 新增 `OptimizationReport`（`reporting/optimization_report.rs`）：environment / command / train / topology / memory / evaluation；不可测字段序列化为 `null`。
- Trainer 采样闭环：每步 `Instant` loop duration → p50/p95（`loop_timing_kind=cpu_submit_instant`）；loss/count readback 计数；按 splat 规模估计 sort/scan dispatch 与 workspace bytes；topology snapshot/plan/apply `Instant` 与 accumulator readback bytes（有 topology 步时才有值）。
- CLI：`--optimization-report`（`--eval-json` 时默认写到输出旁 `.optimization.json`）；`compare-optimization-reports` 拒绝 fingerprint / seed / scale / eval frames / resolution 不一致。
- 验证：
  - `cargo test -p rustgs --lib reporting::optimization_report --no-default-features` → 4 passed
  - Home 50-step smoke：`artifacts/runs/rustgs-optimization/2026-09-18/r06-smoke/`
    - `gpu_completion_seconds` / `peak_device_bytes` / `driver` = `null`
    - `loop_duration_p50_ms≈22.9`，`loss_readback_count=4`，`count_readback_count=12`，12 帧 evaluation
    - self-compare → compatible；改 `eval_frame_ids` → rejected

## R07 修复证据（2026-09-18）

- 新增 `backward/gradient_check.rs`：单 Gaussian projected-splat 标量目标 × centered FD；扰动未归一化 quaternion（与前向一致）。
- Shader 修正：
  - `compensate_cov2d_vjp`：把 `filter_comp` 对 raw covariance 的导数并入 `v_cov2d`
  - `sh_to_color_viewdir_vjp` + `normalize3_vjp`：SH viewdir → position；`project_bwd` 增加 `sh_coeffs` 读绑定
- 门禁：`cargo test -p rustgs --lib gradient_check --features gpu -- --test-threads=1` → **6 passed**
  - cases：`baseline_degree0`、`off_center_fx_fy`、`small_covariance`、`anisotropic_rotated`、`degree1_viewdir`
  - 常规相对容差 `<= 8e-2` 或绝对 `<= 2e-3`；弱方向 / 未归一化 quat FD 另有条件放宽并记入结果表
- 说明：强信号轴（position xy、opacity、主要 log-scale、SH DC）与 FD 对齐；光学轴 depth-scale / 小 quat 分量仍受 float FD 条件数限制，不作为放宽默认拓扑策略的依据。

## 历史实验原始记录

### 2026-09-18 Home COLMAP 500-step A/B（替代缺失的 TUM pack）

本机无 `artifacts/runs/rustgs_benchmark_pack/tum_freiburg1_xyz_colmap`，改用已有 Home 稀疏模型做 Task 1 短跑 filter。

- 数据集：`artifacts/runs/profile_home/colmap60/0` + `artifacts/inputs/home/images`
- 共同参数：`--iterations 500 --max-frames 12 --render-scale 0.25 --frame-shuffle-seed 0 --eval-after-train --eval-render-scale 0.25 --eval-max-frames 12 --eval-frame-stride 1 --eval-device gpu --eval-json --eval-worst-frames 10`
- binary：`09f9254` baseline vs `938fa8d` candidate（显式路径，避免误用仓库旧 `target/release/rustgs`）
- GPU：Apple M5 Max（wgpu）

| | baseline `09f9254` | candidate `938fa8d` | Δ |
|---|---|---|---|
| `time -l` real | 16.66 s | 12.22 s | −26.6% |
| parity `training_ms` | 14008 | 9936 | −29.1% |
| final PSNR | 18.977 dB | 19.088 dB | +0.11 dB |
| final loss | 0.145938 | 0.141879 | 更好 |
| Gaussian init→final | 26518→26649 | 26518→26649 | 相同 |
| peak RSS | 1.47 GB | 1.48 GB | ≈ |
| gate / nan / oom | Passed | Passed | OK |

- 产物：`artifacts/runs/rustgs-optimization/2026-09-18/home-500-{baseline,candidate}/`
- 无 `ForwardCapacityExceeded` / 非有限 loss；loss 日志仅在 100 步倍数出现（与延迟读回一致）
- 判定：短跑 filter **通过**（更快且 PSNR 不掉）。历史 Home 1500 smoke mean PSNR 约 20.44 dB，不可与 500-step 直接比质量。
- 早先一次「candidate」单跑曾误用 Jul 20 旧 binary，已移至 `home-500-candidate-ambiguous-old-bin/`，不作数。

Home-1500 同口径 A/B 见下一节。

### 2026-09-18 Home COLMAP 1500-step A/B

同输入 / 同 seed / 同 12 帧 / scale 0.25；binary 同上。

| | baseline `09f9254` | candidate `938fa8d` | Δ |
|---|---|---|---|
| `time -l` real | 45.19 s | 32.95 s | −27.1% |
| parity `training_ms` | 43070 | 30985 | −28.1% |
| ≈steps/s（step 100→1500） | 35.0 | 48.3 | +38% |
| final PSNR | 20.423 dB | 20.706 dB | +0.28 dB |
| worst-frame PSNR（日志 10 帧） | 19.762 dB | 20.023 dB | +0.26 dB |
| 逐帧 Δ 最小 | — | — | +0.18 dB（全部为正） |
| final loss | 0.119651 | 0.118935 | 略好 |
| Gaussian init→final | 26518→26925 | 26518→27359 | +434 最终 |
| densify_added / prune_removed | 834 / 427 | 854 / 13 | prune 大幅减少（Weight 可见性语义） |
| peak RSS | 1.52 GB | 1.49 GB | ≈ |
| gate / nan / oom | Passed | Passed | OK |

- 产物：`artifacts/runs/rustgs-optimization/2026-09-18/home-1500-{baseline,candidate}/`
- 停止条件（最差帧下降 >0.2 dB）：**未触发**
- 判定：Home smoke **通过**（更快且质量更好）。topology prune 路径相对 baseline 行为变化明显，需在外部场景再确认；默认配置仍不改（缺 TUM 全视图 + 外部场景）。

## R08 修复证据（2026-09-18）

- `DynamicMaskGradient::{StopGradient,Coupled}`；默认 `StopGradient`；mask 在 `mean(loss*w)/mean(w)` 前 `.detach()`。
- Host 复现：`coupled_dynamic_mask_can_reward_larger_residuals`；detach 固定权重 FD：`detached_dynamic_mask_keeps_non_negative_residual_derivative`。
- 验证：`cargo test -p rustgs --lib loss::tests --features gpu` → 3 passed。未启用 mask（默认阈值关闭）路径不变。

## R11 修复证据（2026-09-18）

- `select_initial_points`：预算足够时保持输入顺序；截断时 voxel 每格取首点，再按原序填满预算；`init_voxel_cell_size<=0` 时按包围盒自动 cell。
- 验证：`cargo test -p rustgs --lib data::init_map --features gpu` → 3 passed（含聚集边界保留与确定性）。

## R09 验收证据（2026-09-18，工作树 atop `938fa8d`）

- 二进制：`artifacts/runs/rustgs-optimization/2026-09-18/bin/rustgs-r09-current`；对照 `rustgs-baseline-09f9254`。
- Home 同口径（12 帧，scale 0.25，seed 0）：

| | baseline `09f9254` 500 | current 500 | current 1500 | baseline 1500（历史） |
|---|---|---|---|---|
| train elapsed | 14.61 s | 13.34 s | 40.02 s | 45.19 s wall |
| PSNR mean | 18.978 | 19.084 | 20.693 | 20.423 |
| PSNR min（worst） | 18.272 | 18.450 | 20.088 | 19.762 |
| Gaussian final | 26649 | 26649 | 27370 | 26925 |
| gate | Passed | Passed | Passed | Passed |

- 最差帧相对 `09f9254`：**上升**（500：+0.18 dB；1500：+0.33 dB），未触发拒绝门禁。
- 外部 flowers2-24v（`artifacts/runs/flowers2_colmap_ref_text_20260630` + `artifacts/inputs/flowers2/images`，1500 step）：mean 26.283 / min 24.240 dB；6449→6555 Gaussians；topology densify/prune 有事件；gate Passed。产物：`r09-flowers2-24v-1500/`。
- **TUM：** 本地 RGB 缺失；现有 pack symlink 失效；下载 403/401/超时 → **未跑**。残留见 TODO `R09b`。不据此改默认训练配置为“已全场景验收”。

## R10 瓶颈排名（2026-09-18）

依据 `r09-home-500/home.optimization.json`（CPU Instant loop；GPU completion / VRAM = null）：

| 排名 | 来源 | 证据 | 结论 |
|---|---|---|---|
| 1 | 稳态 raster/sort 循环 | `loop_duration_p50_ms≈23.1`，`sort_dispatch_count_p50=50`，`sort_workspace_bytes≈66 MiB` | 占训练时间主体；缺 GPU timer，**不**据此融合 SSIM/loss |
| 2 | 延迟读回 | `count_readback_count=78`，`loss_readback_count=26` / 500 step | 已按 cadence；再压缩需先证 stall |
| 3 | topology snapshot/apply | 500 step 内 1 次；`snapshot+plan+apply≈16.5 ms`，readback ≈1.0 MiB | 非当前短跑主因；勿上完整 GPU-native topology |
| 4 | 帧缓存 / decode | 本报告无 host/device stall 字段 | 未测出前不改 cache 策略 |

- **候选实施：** 无超过重复运行波动且可归因的单项瓶颈 → **本轮不改代码**（符合“未测出不批量加融合”）。

## R12 增密策略验证（2026-09-18）

- Accumulator 语义（`project_backwards.wgsl` / `topology/mod.rs`）：`Abs`=`abs_refine_weight_max`；`AbsPixel`=`pixel_coverage * abs`；`AbsPixelDepth`=`AbsPixel * depth_scale(γ=0.37, scene_extent)`。
- Home-500（同 seed/budget）：

| mode | PSNR mean | PSNR min | final G | train s |
|---|---|---|---|---|
| baseline | 19.085 | 18.455 | 26649 | 13.57 |
| abs | 19.075 | 18.372 | 26648 | 13.64 |
| abs-pixel | 19.086 | 18.463 | 26649 | 13.86 |
| abs-pixel-depth | 19.119 | 18.535 | 26648 | 14.00 |

- flowers2-24v-500：baseline mean/min 23.141 / 20.623；abs-pixel-depth 23.142 / 20.623（无实质差）。
- **判定：** 无跨场景稳定收益 → **保留默认 `Baseline`**。


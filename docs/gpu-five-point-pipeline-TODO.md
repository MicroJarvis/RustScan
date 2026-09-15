# GPU f32 五点法 Pipeline 优化 TODO

更新日期：2026-09-15（第十一次更新）  
开发分支：`gpu-five-point-f32`  
工作区：`/Users/tfjiang/Projects/RustScan/.worktrees/gpu-five-point-f32`  
源码基线 commit：`4748208` / docs `ac15d4b`（Q1–Q4 + P1–P3 检查点；其上叠测量 harness 小修）

## 目标与边界

同时推进两条线，但每轮只动其中一条的一个变量：

- **性能线 P1–P5**：减少资源重建、主机分配、CPU/GPU 往返和流水线空闲。
- **质量线 Q1–Q5**：定位并改善 f32 丢解、模型差异和 mask 长尾误差。

约束：

- 保持 wgpu / GPU f32 实验，不添加 CPU runtime fallback。CPU f64 仅用于离线 reference，以及现有生产路径中尚未替换的步骤。
- 不自动修改默认 matching、RANSAC、SIFT 或 BA 策略。
- 不把离线 candidate-generation 吞吐优势解释为生产加速或等质量替代。
- 每轮只改一个主要变量；资源/布局优化与数值算法改动分开验证。
- 不覆盖已有工作；提交、push 和生产接入分别执行，不由 TODO 自动授权。
- 保留历史报告和原始结果，不用新结果覆盖旧基线。

> 编号说明：本文 P1 指主机分配优化，已完成。历史上的"P1 nullspace/LU 布局实验"见历史报告，不是本文 P1。

## 当前状态一览

| 项 | 状态 | 结论 |
|---|---|---|
| B0 固定基线与验收协议 | 已完成 | 输入 digest、GPU/CPU 签名、交错计时协议均固定 |
| P1.1 消除每-trial 临时 Vec | 已完成，保留 | decode −52%/−72%，签名不变 |
| P1.2 buffer 零初始化交由设备 | 已完成，保留 | prepare −89%，端到端 −21%（65,024），签名不变 |
| roots 成本归因（测量） | 已完成 | 主要成本是 SIMD lockstep 分歧，上限 1.84–1.89× |
| Q1 失败分类（测量） | 已完成 | `unconverged` 计数 93% 是复根过滤，不是收敛失败 |
| **Q1 诊断修复** | **已完成，保留** | 三条路径独立状态码；模型签名不变；与零-root 推断逐项一致 |
| 丢解根因（测量） | 已完成 | 零空间闸门阈值作用在 σ² 上；不是 f32 精度；丢的是好样本 |
| **Q2 零空间 σ 级闸门** | **已完成，保留** | AAᵀ 上 σ₅/σ₁ > 1e-5；救援带 99.1% 保留；QR 作对照 |
| **P2 持久化 session** | **已完成，保留** | 复用 buffer/staging；`params.count`；签名=Q2；batch512 ≈−1.4% |
| **P3 提交与读回边界** | **已完成，保留 A+B** | A: merge → 1.344 s（−24% vs P2）；B: 双缓冲 → 0.740 s（−45% vs A） |
| **Q3 recovery 丢解归因（测量）** | **已完成** | 93.5% 缺口在 basis 之后；2e-5 闸门无关；系数误差是判别变量 |
| **Q4 algebra 系数精度** | **已完成，保留** | FMA 补偿；丢解 cohort p90 ↓84–103×；GPU 模型 +4,917；签名→Q4 |
| **Q5 前置测量** | **已完成** | B 误差 loss/no-loss p90 ≈30×；建议 Q5a LU/B |
| **Q5a LU 迭代精化** | **已完成，回退** | B p90 4.45e-4→4.74e-4（变差）；签名保持 Q4 |
| **Q5a-B elim/back-sub FMA** | **已完成，回退** | B p90 4.45e-4→4.36e-4（~2%，不实质）；签名保持 Q4 |
| **Q5b recovery null3** | **已完成，保留** | RF 18,063→17,475；模型 +949；签名→Q5b |
| 难度感知 roots / P4 / P5 | 之后 | 见各节；质量线建议混精 LU/B 或 P4 |

## 固定输入与签名

```text
DB: /Users/tfjiang/Projects/RustScan/output/flowers2_960_settlement_20260913/matching.db
127 pairs × 512 trials, seed=1 → 65,024 trials, 231 张不同图像
Input digest:              af07459d637b5b8c2b20c76d3ef103f058505fe0e8307578ee7e8750f3e230c5
Model signature (Q1):      030f43068d9e8f78961142915162f20bef4ecb909d49d247a4f766a2ab5412df
Model signature (Q2):      472124c8f1a9c6299d659e2f835dc11c682ff8a593757f0b145f73454f51d6ea   (历史)
Model signature (Q4):      26e825cf97002be4f7fc36b77ce4fed682445af37746504253444e0103c5e716   (历史)
Model signature (Q5b):     fb5be154bb0fa742ab1b83e5c9e62269722678cab463bfbe3827457b7424e8f0   (现行)
Diagnostic signature v1:   1f6f3f8d88a17e4e1c42abfc85aad44a66c4b864091805594e086ca80bf9b339   (历史，Q1 前)
Diagnostic signature v2:   3cd8091d6033f2fdb0b598c27483e67c4dd3ee1af1232ff4f1f5ee1da86117cb   (Q1 后)
Diagnostic signature Q2:   3d04bb3edaef183049e07d6c3841e296f963da3b279bcbd99d878e1dc090c050   (历史)
Diagnostic signature Q4:   d9e911ac6b6e2be18f7e3b8b8615417efba6257f955ce2bbf21f943a1c7f358a   (历史)
Diagnostic signature Q5b:  1412c6d5d2af9bb9ed23c8d55238af22deb2b259cfa16fa43a99fb21486b9a1a   (现行)
CPU f64 signature:         9e6764b41602d0005710187d2558d046f1a843c5f873b9542dd7b4f6c29c563a
```

资源/布局实验校验 **diagnostic** 签名（现行 **Q5b**）。诊断-only 改动另校验 **model** 签名须不变。数值算法实验允许 diagnostic 与 model 都变，但须重新确立质量接受条件。P2/P3 资源复用必须保持现行质量签名。

所有 harness 启动时先校验 digest；漂移即失败，不产出可比数字。

构建注意：本 worktree 的 `third_party/PoseLib` 是空的 submodule 目录，编译需
`POSELIB_ROOT=<主 checkout>/third_party/PoseLib`；所有构建显式 `--target-dir target`
（环境里 `CARGO_TARGET_DIR` 指向临时缓存）。poselib bridge 不在 GPU 五点路径上。

**计时原则：** GPU pass 时间与 CPU wait 重叠，不能相加；不同模式/运行的中位数不是严格可相减的时间账单。完整诊断接口与紧凑候选接口必须分别比较。

## 当前事实：性能

- P1 之后新起点：65,024 约 80.3 ms、32,768 约 86.6 ms、512（127 次调用）约 1640 ms。
- CPU8 同轮对照（P1 后，首次通过 stable-win 门）：

| Batch | CPU8 中位数 | GPU 中位数 | CPU/GPU | GPU 获胜轮次 | stable_win |
|---:|---:|---:|---:|---:|---|
| 32,768 | 121.922 ms | 94.375 ms | 1.292 | 8/8 | true |
| 65,024 | 121.647 ms | 89.304 ms | 1.362 | 8/8 | true |

  该轮 CPU8 比上一轮（111.052 / 111.735 ms）慢约 9%，CPU 代码与签名一致，原因未定位；用上一轮更有利的 CPU 样本做敏感性检查结论仍成立。**该胜出仅为离线 candidate generation 吞吐，输出与 CPU f64 不等价**（`equal_quality: false`），不是生产加速。
- **batch 512 的瓶颈**是每次调用的 wait（约 1.37–1.43 s 合计）与 readback（约 0.20 s），不在主机准备与解码。这是 P2/P3 的依据，且不受下面的分歧结论影响。
- **roots 成本归因**（65,024 单次调用，仅重排输入顺序、不改算术，三种顺序逐 trial 结果逐位相同）：`roots` 33.7–36.2 ms 中有 24–27 ms 是 SIMD lockstep 分歧浪费。按难度排序后 `roots` 9.3–9.5 ms（3.62–3.82×），九 kernel 合计 54.1–57.1 → 29.3–30.6 ms（1.84–1.89×）。随机打乱作对照与原顺序持平；`recover` 同轮内三种顺序差 ≤0.07 ms 作负对照。机制：8.7% 的 trial 触到 384 迭代上限，却污染 2,032 个 SIMD-32 组中的 93.0%（迭代数中位 17、p90 74、p99 384）。排序是 oracle（需上一轮迭代数），只界定上限，不是可实施策略。
- Q2 改了 nullspace 算术（多一次 5×5 Jacobi），**尚未**重跑 CPU8 同口径计时。
- **P2 session（保留）**：127×512 稳态中位 1.775 s vs one-shot 1.801 s（约 −1.4%）。创建 session ≈ 19 µs，可忽略。
- **P3（保留 A+B）**：merge → 1.344 s；双缓冲 → 0.740 s（Q2 当时）。**Q4 shader 下复测**（`experiments/p3-double-buffer-q4shader-20260915.json`）：双缓冲中位 **0.674 s**，同 run merge 1.306 s；Q4 签名门通过。

## 当前事实：质量

- **Q2 后全量计数（历史）**：CPU 288,264；GPU 243,967；空结果 2,942；CPU 有解/GPU 空 2,899；`RankDeficient` 621。
- **Q4 后全量计数（历史）**：CPU 288,264；GPU **248,884**（+4,917）；空结果 **2,172**；CPU 有解/GPU 空 **2,127**；`RankDeficient` 仍 621。同 basis 下游缺口 41,404→36,487。
- **Q5b 后全量计数（现行）**：CPU 288,264；GPU **249,833**（+949 vs Q4）；CPU 有解/GPU 空 **1,883**；Matched RecoveryFailed 18,063→**17,475**（tol 1e-3）。
- **Q2 闸门**：对缩放后的 A 做 AAᵀ（5×5）Jacobi 得 σ，接受条件 `σ₅/σ₁ > 1e-5`；基仍由 AᵀA 特征向量给出。禁止只下调旧的 `1e-6·trace(AᵀA)` 常数；改的是作用对象（σ 而非 σ²）与估计路径（AAᵀ 而非 AᵀA 噪声底）。
- **救援带 `[1e-5,1e-2)`**：40,448 / 40,832 保留（99.1%）；其中出模型的样本 Sampson 相对 pair 最优 p50=0.592、均值 0.589。`[1e-4,1e-2)` 几乎全保留；`[1e-5,1e-4)` 仅 57.5%。
- **真退化残留**：f64 `σ₅/σ₁ ≤ 1e-7` 仍有 88/197 被保留并出模型——f32 Gram 高估近零 σ₅；再拧常数无效（1e-6 与 1e-5 只差 2 trial）。CPU 也无秩检验。更严拒绝留给后续 QR/SVD，不作为本轮回滚理由。
- **`NotConverged` 已拆分（Q1，保留）**：见历史。
- QR of Aᵀ 本轮作对照：A*N≈0 通过，但合成 algebra fixture 破坏 f64 五点行列式恒等式（多项式误差 ~1e-2），未采用。
- **Q3 归因（2026-09-15，纯测量）**：缺口 44,297 = basis/闸门 2,893 + 下游 41,404。下游按根分类（tol 1e-3/1e-2）：MissingNoSlot 37,345/21,467、RecoveryFailed 9,862/14,445、NotDone 5,458/8,793、实轴+polish 闸门仅 21/142。系数误差（同 basis f64 对照）：丢解 cohort p90=0.084，无丢解 p90=2.9e-5。详见 [Q3 报告](gpu-five-point-q3-recovery-attribution-20260915.md)。
- **Q4 系数精度（保留）**：同一 B 上多项式系数 FMA 补偿；固定丢解 cohort p90 0.084→0.001（84×）；MissingNoSlot 37,345→10,094；质量 trial 5,701 优 / 4,200 劣 / 55,123 平。剩余误差主要在 LU/B（same-basis 总误差 p90≈1.2e-4 vs same-B 4.4e-8）。根匹配后 RecoveryFailed/NotDone 上升为重新归类，非 recovery 变差证据。详见 [Q4 报告](gpu-five-point-q4-coefficients-verified-20260915.md)。
- **Q5 前置（2026-09-15）**：`B_gpu_vs_f64_same_basis` 丢解/无丢解 p90 ≈ **30×**；InvalidEssential 11,907 / NullVectorFailure 9,530 slots。Q5a / Q5a-B 均已回退；详见 [Q5](gpu-five-point-q5-premeasure-20260915.md) / [Q5a](gpu-five-point-q5a-lu-refinement-reverted-20260915.md) / [Q5a-B](gpu-five-point-q5a-b-fma-elim-reverted-20260915.md)。
- **Q5a-B（2026-09-15，回退）**：消去/回代 FMA；固定 loss B p90 4.447e-4→4.361e-4（~2%）；模型 −2；质量近似持平略负。
- **Q5b recovery（2026-09-15，保留）**：离线 good-B matched RF **6,759**（勿停）；`null3` FMA Gram + min‖Bv‖ + 逆迭代；RF −588；模型 +949；质量 net +569；签名→Q5b。详见 [Q5b 报告](gpu-five-point-q5b-recovery-retained-20260915.md)。

## 历史报告（相对本文件）

- [大 batch 实验](gpu-five-point-batch-sweep-20260914.md)
- [设备容量与真实样本覆盖](gpu-five-point-capacity-sweep-20260914.md)
- [阶段计时与丢解归因](gpu-five-point-profile-attribution-20260914.md)
- [CPU4/CPU8 对照，nullspace/LU 优化前](gpu-five-point-cpu-threads-comparison-20260914.md)
- [nullspace/LU 布局实验及最终 CPU8 对照](gpu-five-point-p1-layout-20260914.md)
- [P1 主机分配与 buffer 初始化，首次 CPU8 stable win](gpu-five-point-p1-host-20260914.md)
- [roots 成本归因与 Q1 失败分类（纯测量）](gpu-five-point-roots-attribution-20260914.md)
- [CPU 有解/GPU 无解根因与质量权衡（纯测量）](gpu-five-point-lost-solutions-20260914.md)
- [Q1 诊断修复：拆分 NotConverged（保留）](gpu-five-point-q1-diagnostics-20260914.md)
- [Q2 零空间 σ 级闸门（保留）](gpu-five-point-q2-nullspace-20260914.md)
- [P2 持久化 session（保留）](gpu-five-point-p2-session-20260914.md)
- [P3 提交与读回边界（保留 A+B）](gpu-five-point-p3-submit-20260914.md)
- [Q3 recovery 丢解归因（纯测量）](gpu-five-point-q3-recovery-attribution-20260915.md)
- [Q4 algebra 系数精度（保留）](gpu-five-point-q4-coefficients-verified-20260915.md)
- [Q5 前置测量（纯离线）](gpu-five-point-q5-premeasure-20260915.md)
- [Q5a LU 迭代精化（回退）](gpu-five-point-q5a-lu-refinement-reverted-20260915.md)
- [Q5a-B elim/back-sub FMA（回退）](gpu-five-point-q5a-b-fma-elim-reverted-20260915.md)
- [Q5b recovery null3（保留）](gpu-five-point-q5b-recovery-retained-20260915.md)

测量 harness：`five_point_gpu_p1_host.rs`、`five_point_gpu_roots_attribution.rs`、`five_point_gpu_lost_solutions.rs`、`five_point_gpu_q1_diagnostics.rs`、`five_point_gpu_q2_nullspace.rs`、`five_point_gpu_p2_session.rs`、`five_point_gpu_p3_submit.rs`、`five_point_gpu_p3_double.rs`、`five_point_gpu_recovery_attribution.rs`、`five_point_gpu_q4_coefficients.rs`、`five_point_gpu_q5b_recovery_measure.rs`；汇总脚本 `summarize_five_point_q4.py`、`summarize_five_point_q5_premeasure.py`；原始结果在 `experiments/`。

P1 轮的保留产物（baseline/candidate 二进制、`baseline-uncommitted.diff`、`P1.2-buffer-field-inventory.md`）在 `output/five-point-p1host-20260914-ThcNbcf6/`。下文提到这两个文件均指该目录。

---

## 已完成

### B0 — 固定基线与验收协议【完成】

详见 [P1 报告](gpu-five-point-p1-host-20260914.md)。要点：基线 commit `3ab9c09`，上一轮未提交差异保存为 `baseline-uncommitted.diff`；baseline/candidate 二进制均保留；13 个进程逐位一致，含前缀 1/31/32/33/63/65/65,023 与尺寸交替；每进程每 batch 3 预热 + 5 测量、交错顺序；主机分项与九 pass 计时分开采集；CPU8 使用未修改 harness、8 线程专用池。

### P1 — 主机分配与初始化优化【完成，两项保留】

报告：[gpu-five-point-p1-host-20260914.md](gpu-five-point-p1-host-20260914.md)。

- **P1.1** 直接构造固定长度 slot 数组：decode 65,024 下 15.571→7.505 ms（−51.8%），512 下 −72.1%；端到端 65,024 −8.9%、32,768 −10.9%；分配次数 195,498→426。新增无 GPU 依赖的 decode 回归测试（含五种非法编码与错误优先级）。
- **P1.2** `create_buffer` + `mapped_at_creation: false`，零初始化交由 wgpu 设备端（公开文档保证 "Every `Buffer` starts with all bytes zeroed."）：prepare 20.737→2.365 ms（−88.6%），端到端 −21.0%，主机零字节 161.0→0.0 MiB；峰值驻留未变；每 buffer 字段清单见 `P1.2-buffer-field-inventory.md`。
- 合计端到端相对原基线：65,024 −28.3%、32,768 −25.8%、512 −3.6%（重叠，不宣称）。签名逐位不变。

### 三轮测量 + Q1 诊断修复【完成】

见"当前事实"两节与对应报告。测量轮新增只读 harness；Q1 改动见下。

### Q1 — 诊断修复【完成，保留】

报告：[gpu-five-point-q1-diagnostics-20260914.md](gpu-five-point-q1-diagnostics-20260914.md)。

- [x] 三条路径独立状态码：`RealAxisRejected`(3.0)、`RootNotDone`(7.0)、`PolishRejected`(8.0)。阈值与拒绝逻辑未改。
- [x] 主机侧枚举与 decode；保留 header `unconverged` 合计，新增三分项（由 slot 状态派生，须与合计一致）。
- [x] 模型签名 `030f4306…12df` 与修复前逐位相同。
- [x] 诊断签名版本化为 v2：`3cd8091d…17cb`；v1 保留为历史。
- [x] 与零-root 推断交叉验证：276,906 / 130 / 20,786 三项全中。新信息：上限类全部是 `RootNotDone`。
- [x] 1,938 个求根阶段丢解按新状态码精确拆分（见"当前事实：质量"）。
- [x] 回归 fixtures：`experiments/q1-fixtures-20260914.json`（18 个零空间分层样本 + 5 个求根阶段标签样本）。

**解除的耦合：** 分类不再依赖"未写入的 root 仍为零"；P2 复用 buffer 后诊断不会静默失效。

### Q2 — 零空间 σ 级闸门【完成，保留】

报告：[gpu-five-point-q2-nullspace-20260914.md](gpu-five-point-q2-nullspace-20260914.md)。

- [x] 先写明秩判据：对缩放后 A 做 AAᵀ（5×5）Jacobi 得 σ，接受 `σ₅/σ₁ > 1e-5`；基仍来自 AᵀA。
- [x] QR of Aᵀ 作对照（代数 fixture 破，未采用）；未单纯下调旧 `1e-6·trace` 常数。
- [x] 导出 `min_diagonal_ratio = σ₅/σ₁`；验收用分桶表 + Sampson，不是丢解总数。
- [x] 模型/诊断签名更新为 Q2 基线；CPU f64 签名不变。
- [x] 救援带 99.1% 保留；真退化残留与 `[1e-5,1e-4)` 部分保留已写入报告。
- [ ] CPU8 同口径计时尚未重跑（若引用吞吐需补）。

### P2 — 持久化 GPU 求解 session【完成，保留】

报告：[gpu-five-point-p2-session-20260914.md](gpu-five-point-p2-session-20260914.md)。

- [x] `FivePointSession`：预分配/增长复用 input、中间、输出与 staging。
- [x] `params.count` 与容量分离；`nullspace`/`algebra`/`roots` 不再用 `arrayLength` 作 N。
- [x] 按 P1.2 清单对 diagnostics/basis/algebra/out 做 `clear_buffer`。
- [x] 同时只允许一批在途；小→大→小、成功→失败→成功、空输入、非整 workgroup 回归。
- [x] 签名与 Q2 基线一致；分开报告 create / cold / steady / oneshot（含 batch 512）。

### P3 — 提交与读回边界【完成，保留 A+B】

报告：[gpu-five-point-p3-submit-20260914.md](gpu-five-point-p3-submit-20260914.md)。

- [x] Candidate A：compute + staging copy 合并为单次 submit/wait；127×512 1.775→1.344 s（−24%）；签名=Q2。
- [x] Candidate B：有界双缓冲（最多 2 in-flight）；0.740 s（−45% vs A）；签名=Q2；opt-in API。
- [x] 以 wall time 为停止条件，不以 submit 次数单独宣称收益。
- [x] `submit_count` / `wait_count` 计数与 session 回归（含 double-buffer bits 门）。

### Q4 — algebra 阶段 f32 系数精度【完成，保留】

报告：[gpu-five-point-q4-coefficients-verified-20260915.md](gpu-five-point-q4-coefficients-verified-20260915.md)。

- [x] 唯一变量：生成的行列式多项式系数 FMA 补偿（`generate_five_point_wgsl.py` → `five_point_generated.wgsl`）；不动 LU、roots、recovery、闸门。
- [x] 固定丢解 cohort 系数误差 p90 ↓84–103×；MissingNoSlot 37,345→10,094；GPU 模型 +4,917。
- [x] 上游 basis/A/solve/B 与基线逐位相同；仅 coeff11 变化。
- [x] 质量表重立（Sampson）；签名切换为 Q4 model/diagnostic。
- [x] 性能中性（~1%，不作加速结论）；Metal FMA 不重结合依赖已实测记录。

---

## 待做（按执行顺序）

### Q5a — LU / B 矩阵 f32 精度【已完成，回退】

报告：[Q5a 迭代精化回退](gpu-five-point-q5a-lu-refinement-reverted-20260915.md)。

- [x] 唯一变量：10×10 LU 一步迭代精化（第二遍 GE+solve 于残差 `b−Ax₀`）。
      不动 roots、recovery、闸门、多项式 FMA。
- [x] 验收：Q4 harness `--candidate` + Q4 baseline trials 固定 loss cohort。
- [x] **回退**：固定 loss cohort `B_gpu_vs_f64_same_basis` p90 **4.447e-4 → 4.740e-4**
      （约 7% 变差）；GPU 模型 −240；MissingNoSlot +6,768；质量 net worse。
      签名保持 Q4；候选 run 见 `experiments/q5a-candidate-tol1e{3,2}-20260915.*`。
- [x] 性能：batch 512 中位 1.492→1.573 s（+5.5%）；65,024 ~1×。

**下一轮：** 勿再试纯 f32 同路径 LU 修补。优先二选一（仍单变量）：
(1) ~~**Q5b recovery**~~（已保留）；(2) 设备混精 / f64 残差若 API 可行。
P4 可在现行 **Q5b** 质量门下单独开性能轮。

### Q5a-B — elim/back-sub FMA【已完成，回退】

报告：[Q5a-B FMA 回退](gpu-five-point-q5a-b-fma-elim-reverted-20260915.md)。

- [x] 唯一变量：消去与回代 FMA 累加（B 组装本身已是精确 `x−y`）。
- [x] **回退**：固定 loss cohort B p90 **4.447e-4 → 4.361e-4**（~2%，不实质）；
      模型 −2；质量 3751/3755/57518；签名保持 Q4。
- [x] 产物：`experiments/q5a-b-fma-tol1e3-20260915.*`。

### Q5b — recovery 3×3 null3【已完成，保留】

报告：[Q5b recovery 保留](gpu-five-point-q5b-recovery-retained-20260915.md)。

- [x] 离线：matched RF 按 NullVector/InvalidEssential × good-B/bad-B 交叉表；
      good-B RF **6,759** → 继续实施（勿停）。
- [x] 唯一变量：`null3` FMA Gram + min‖Bv‖ 选向量 + 2 步阻尼逆迭代；
      不放宽 accept 阈值；不动 algebra/roots/闸门。
- [x] 验收：RF 18,063→17,475（−588）；模型 +949；Sampson net +569；
      签名→Q5b；focused tests 11/11。
- [x] 活跃 harness 质量门切 Q5b。

### Q5 前置测量【完成】

报告：[gpu-five-point-q5-premeasure-20260915.md](gpu-five-point-q5-premeasure-20260915.md)。

- [x] 拆分 InvalidEssential / NullVectorFailure 绝对 slot 计数。
- [x] B 误差 vs loss / recovery-fail 判别（~30×）。
- [x] 建议 Q5a（LU/B），Q5b（recovery）排队。

### 第 5 项：难度感知 roots【推后】

依赖：P2（session 结构决定能否在设备端重排/压缩）。已有上限数字：大 batch 下 `roots` 3.62–3.82×，九 kernel 合计 1.84–1.89×。不作用于 batch 512。

- [ ] 设计如何在求解前预测难度，或改为收敛 lane 提前退出/工作重分配的 kernel 结构；oracle 排序本身不可实施。
- [ ] 任何方案先用 `roots_attribution` harness 的"逐 trial 结果逐位相同"门验证，再看时间。

### 第 6 项：recovery 阶段丢解【并入 Q3/Q5】

已被 Q3 归因与 Q5 前置测量取代；勿再按旧清单单独开轮。
### 第 7 项：P4 — 拆分 GPU 求解与诊断读回，连接 scorer

依赖：P2/P3 的设备资源生命周期已明确。仅实验集成，不代表生产准入。

```text
输入 → GPU candidate generation → 设备端模型与 trial/slot 映射
                               ├─ 完整诊断读回
                               └─ GPU scorer → 决策摘要/选中模型
```

- [ ] 提供设备端结果句柄，明确 buffer 生命周期、有效数量和状态。
- [ ] 保留现有完整诊断读回作为适配层。
- [ ] 先做单 pair、同一个 context 的 solver→scorer 连接。
- [ ] 优先固定 slot + 有效标记，保持顺序；是否 compact 由实测决定。
- [ ] 不再强制 E 从 GPU 下载、转为 CPU 对象后再上传评分。
- [ ] 仅读回决策摘要，以及 CPU 后续决策/refinement 真正需要的模型。
- [ ] 对照"相同 GPU 模型读回再上传评分"和"设备端直接评分"，确认结果一致。
- [ ] 分别报告完整诊断接口与设备端连接接口的时间和数据量。

**验收：** trial/slot/模型数量与顺序不变；scorer 使用正确观测、阈值和模型类型。不得把非最小 LO/refinement 的输入当作多个 minimal 五点样本处理。不直接将不等价 solver 接入默认 RANSAC。

### 第 8 项：P5 — 生产跨 pair 有界流水线

依赖：P4；生产接入还必须通过质量门。先做设计与实验，不盲目扩大单 pair 试算。

```text
各 pair 当前需要的 trial → 有界聚合 → GPU 求解与分段评分 → 按 pair/trial 返回摘要 → 各 pair 推进原有 RANSAC
```

- [ ] 建立每 pair 独立 sampler、best、动态 trial 上限、decision frontier 和 refinement 状态。
- [ ] 维护 `(pair, trial, slot)` 映射，不能拆坏同一 trial 的候选组。
- [ ] 扩展 scorer 的观测 offset/count、阈值、模型类型与 pair 关联。
- [ ] 单独处理 shared RNG 模式，验证采样流消耗顺序。
- [ ] 设计 CPU refinement 与 GPU 工作衔接，不改变非最小 refit 语义。
- [ ] 设计 Taskflow admission、DB 提交、进度、取消、错误传播和 drain 边界。
- [ ] 限制在途任务/内存与背压；Taskflow grant 不是 RSS 上限。
- [ ] 统计提前终止后丢弃的预计算工作、mask 预取浪费，不能只统计吞吐。
- [ ] 先验证实验路径；质量与语义通过后运行 first48，再评估完整 960 张 matching/重建。

**验收：** 保留 sampling、候选消费顺序、决策窗口和 early termination 语义；pair 输出与重建质量满足事先确定的门禁。离线能聚合 65,024 trials 不等于生产能持续提供相同规模的有效工作。

---

## 约束检查项（不是待实施功能）

- 不再优先单纯扩大 batch。
- 不一次重写/融合所有 shader。
- 不放宽阈值让失败计数更好看（具体到零空间：不单纯下调 `1e-6`）。
- 不自动改变生产 CPU 线程数和 RANSAC 策略。
- 质量门前不以完整 960 张运行宣称生产加速。
- CPU f64 不是退化性的 ground truth；GPU/CPU 对照单列真退化样本。

## 每轮交付模板

- 本轮假说、唯一主要变量、改动文件和精确源码基线。
- focused tests、实际 GPU 验证、格式与 diff 检查；生成代码改动运行生成一致性检查。
- 输入 fingerprint、完整结果签名（或模型签名 + 诊断签名版本）或算法质量对照。
- baseline/candidate 逐轮计时、kernel/主机分项、端到端、内存范围。
- 同口径 CPU8 结果；需要时补 CPU4。
- 明确 pass / fail / unavailable / blocked，保留失败结果。
- 给出保留或回退决定，报告未解决质量风险。
- 更新本文进度和关联报告链接。

## 轮次历史

| 轮 | 类型 | 改动 | 结论 |
|---|---|---|---|
| B0 + P1.1 + P1.2 | 实施 | `five_point_f32_complete.rs` 及测试 | 两项保留；端到端 65,024 −28.3%；签名不变；CPU8 首次 stable win（仅离线吞吐） |
| roots 归因 / Q1 分类 | 测量 | 新增 `five_point_gpu_roots_attribution.rs` | 分歧上限 1.84–1.89×；`unconverged` 93% 是复根过滤；83.8% 丢解在零空间 |
| 丢解根因 / 质量 | 测量 | 新增 `five_point_gpu_lost_solutions.rs` | 根因是 σ² 级阈值而非 f32 精度（H2 = 0）；被拒样本 96% 温和病态且质量不低于保留样本；pair 级最终模型几乎不受影响，代价是 15.4% 有效样本 |
| Q1 诊断修复 | 实施 | `five_point_recovery.wgsl` + 主机枚举/decode | **保留**；模型签名不变；诊断签名 → v2；与零-root 推断三项全中；上限类 20,786 全部是 `RootNotDone` |
| Q2 零空间闸门 | 实施 | nullspace AAᵀ σ 门 | **保留**；模型 209,600→243,967；救援带 99.1% |
| P2 session | 实施 | `five_point_f32_session.rs` | **保留**；127×512 ≈−1.4% vs oneshot |
| P3 A+B | 实施 | merge submit + 双缓冲 | **保留**；1.775→1.344→**0.740** s；签名=Q2（当时） |
| P3 复审 | 修复 | 双缓冲中途扩容 drain | 消除静默错误结果路径；签名与耗时不变 |
| Q3 recovery 归因 | 测量 | 新增 `five_point_gpu_recovery_attribution.rs` + f64 参考重构 | 系数误差是判别变量；闸门无关；见 Q4 依据 |
| Q4 系数精度 | 实施+验证 | FMA 补偿行列式系数 | **保留**；模型 +4,917；签名→Q4；残余在 LU/B |
| Q4 处置 | 文档+门 | 签名切 Q4；P3 复测 0.674 s | 活跃 harness 已切门 |
| Q5 前置 | 离线测量 | B 误差 vs loss ~30× | **建议 Q5a LU/B** |
| Q5a 迭代精化 | 实施+验证 | 一步 LU 残差修正 | **回退**；B p90 变差；见 Q5a 报告 |
| Q5a-B FMA elim | 实施+验证 | 消去/回代 FMA | **回退**；B p90 ~2% 不足；见 Q5a-B 报告 |
| Q5b recovery | 离线+实施 | null3 FMA/选向量/逆迭代 | **保留**；RF −588；签名→Q5b |

## 下一轮明确范围与停止条件

**Q5b 已保留；Q5a/Q5a-B 均已回退。** 下一轮勿重复纯 f32 LU 修补或再拧
recovery 阈值；选 **混精残差（LU/B）** 或在 **Q5b** 门下开 **P4**。仍单变量。
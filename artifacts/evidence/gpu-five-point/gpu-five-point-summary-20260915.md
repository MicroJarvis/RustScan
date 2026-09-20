# GPU f32 五点法实验最终摘要

**状态：已冻结（2026-09-15）**

这份摘要替代本目录之前的逐轮实验报告。它只保留已经验证、仍然有效的
实现结论；原始 JSON、日志、shader 快照和运行目录继续作为可复核证据保留
在 `artifacts/evidence/` 与 `artifacts/runs/`。

## 最终决策

- 生产五点法继续使用 CPU `f64`。GPU `f32` 没有接入默认 RANSAC，也没有替代
  CPU 求解器。
- 实验线冻结在 Q5b。后续不再开展 Q5c、df64、混合精度 LU/B，也不使用这条
  f32 solver 继续推进 P4/P5。
- Apple GPU/Metal 没有 shader `f64`；按生产粒度每 pair 512 trials 测得约
  5.3 ms，而 CPU8 约 1 ms，GPU f32 不能形成生产加速。后续性能重点转向
  geometry scorer 同步、pair 级流水线、重建/BA 和描述子 kernel。

## 保留的有效改动

| 阶段 | 保留内容 | 已验证结论 |
| --- | --- | --- |
| P1 layout | nullspace 与 algebra 使用按 trial 的 `global_invocation_id`，保持公式、阈值、buffer 和 pass 顺序不变 | 实际 GPU 测试通过；端到端单轮降低约 13.7%–21.3%；CPU8 对照仍为 GPU 0/8 胜出 |
| P1.1 | 解码时去掉每 trial 的临时 slot `Vec`，使用固定数组和原有错误优先级 | decode 阶段降低约 52%–72%，签名不变 |
| P1.2 | 输出 buffer 的零初始化交给设备，保留显式计数器和未使用 slot 语义 | prepare 降低约 88.6%，65,024 trials 端到端降低约 21%，签名不变 |
| Q1 | 将 `NotConverged` 拆为 real-axis reject、root-not-done、polish reject 三类状态 | 诊断计数与旧推断逐项一致，模型签名 bit-identical；消除 session 复用后的诊断歧义 |
| Q2 | 用 `AAᵀ` 的奇异值比 `σ₅/σ₁ > 1e-5` 做零空间门控 | rescue band 保留 99.1%；未放宽阈值伪造成功，残余真退化风险仍被记录 |
| P2 | `FivePointSession` 复用 GPU buffer/staging，显式传入 `params.count`，每轮清理依赖零值的区域 | 小→大→小、失败→成功和部分 workgroup 回归通过；稳态 wall time 约改善 1.4% |
| P3-A | 合并 compute 与 staging copy，单次 solve 只提交/等待一次 | 127×512 中位约 1.344 s，相比 P2 降低约 24% |
| P3-B | 在 A 之上使用最多两个 in-flight slot 的有界双缓冲；扩容前先 drain | Q4 门控复测约 0.674 s；签名通过；保持 opt-in，不改变单 slot API |
| Q4 | 对生成的代数多项式系数进行 FMA 补偿，其他阶段不变 | 固定丢解 cohort 的系数误差 p90 下降约 84–103×；GPU 模型数增加 4,917；残余主要在 LU/B |
| Q5b | recovery 的 3×3 null-vector 使用 FMA Gram、按 `‖Bv‖` 选向量和两次阻尼逆迭代 | 相比 Q4 模型数 +949、RecoveryFailed −588，Sampson 质量净增 569；最终质量签名切换为 Q5b |

## 只保留为结论、没有保留为实现的测量

- roots 阶段的主要性能浪费来自 SIMD lockstep 分歧；按难度排序可以得到理论上限，
  但需要未来信息，不能直接作为生产策略。
- GPU 丢解的主要早期原因是零空间条件门，而不是简单的 `f32` 算术误差；Q3
  进一步确认真正可行动的判别变量是系数精度，因此采用 Q4。
- Q5 前置测量显示 LU/B 误差与丢解高度相关，但只作为后续判断依据，没有把离线
  误差分析误当作生产质量证明。
- 大 batch 和设备容量实验证明 shader 可以完成目标容量和 dispatch，但不代表按
  pair 的生产吞吐优于 CPU。

## 已回退或明确不采用的方案

- Q5a：10×10 LU 一步迭代精化使固定丢解 cohort 的 B 误差 p90 从约
  `4.45e-4` 变为 `4.74e-4`，质量变差，已回退。
- Q5a-B：消去/回代 FMA 只带来约 2% 的 B 误差变化，端到端模型和质量没有实质
  改善，已回退。
- 不放宽 `essential_residual`、`|z|` 或零空间阈值来追求更好看的计数。
- 不把离线 candidate-generation 吞吐、非等价 CPU/GPU 输出或单机 GPU 计时宣称为
  生产加速。

## 验收依据

- 输入：Apple M5 Max，127 pairs × 512 trials = 65,024；输入 digest 为
  `af07459d637b5b8c2b20c76d3ef103f058505fe0e8307578ee7e8750f3e230c5`。
- Q5b 当前 model signature：
  `fb5be154bb0fa742ab1b83e5c9e62269722678cab463bfbe3827457b7424e8f0`。
- Q5b 当前 diagnostic signature：
  `1412c6d5d2af9bb9ed23c8d55238af22deb2b259cfa16fa43a99fb21486b9a1a`。
- CPU `f64` reference signature 保持不变：
  `9e6764b41602d0005710187d2558d046f1a843c5f873b9542dd7b4f6c29c563a`。
- 运行过实际 GPU stage tests、session/double-buffer 回归、生成 shader 一致性检查、
  `cargo fmt` 和相关 crate 测试；原始结果仍可从 `artifacts/evidence/` 和
  `artifacts/runs/` 中按报告中的输入摘要追溯。

新 GPU 五点法工作必须创建新的 task 或 change package，并重新声明单变量、质量门、
签名和生产接入条件；不得把本摘要重新当作待办清单。

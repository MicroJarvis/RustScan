# RustSFM 性能优化计划

更新时间：2026-09-13

本文档记录 RustSFM 后续性能工作的执行顺序、测量边界和验收标准。目标是先用数据定位瓶颈，再做一次只改变一个变量的优化；不把 Taskflow、匹配算法、BA 质量和 Viewer 延迟问题混在同一轮实验中。

## 当前状态

最新进展：已实现默认关闭的 `RUSTSFM_PROFILE_CERES=1` stderr 原生 summary 画像，完成 flowers2 前48/960帧 off/on 各三次对照。六次均48/48、427 pairs、17144 points，数据库哈希不变。Global BA 原生时间中线性求解中位数3138.333ms、Jacobian与残差1101.546ms、总计5000.713ms（嵌套计时不可重复相加）；实际使用 Eigen DenseSchur。未改求解策略。详见 [Ceres 内部画像报告](output/ceres_profile_flowers2_20260907/REPORT.md)。随后三次只读调用栈采样定位到 Schur 消元/组装：调用线程 Eliminate/SolveImpl 样本比约98%，约化稠密求解样本很少，消元栈内存在 mutex 等待；不能将线程样本数当作精确耗时或 CPU 占比。见 [DenseSchur 采样报告](output/dense_schur_sample_flowers2_20260907/REPORT.md)。已完成独立控制 BA 请求线程数的1/2/4各三次对照：Mapper grant保持4，原有小问题串行门保留；重建中位数分别11.076/10.755/10.022秒。1线程线性求解变快，但Jacobian变慢，整体无收益；临时源码和默认二进制已恢复，不修改默认策略。见 [线程对照报告](output/ceres_threads_flowers2_20260907/REPORT.md)。后续[接口核查](output/ceres_threads_flowers2_20260907/INTERFACE_AUDIT.md)确认当前Ceres公开Solver接口不能独立控制评估/消元线程；继续该方向需单独立项可回退的原生依赖实验，不能只加Rust配置。当前线程调优分支到此关闭。

真实共享 Runtime 验证已完成：新增 `examples/taskflow_reconstruction.rs`，一个进程/Runtime预算8CPU，每项目请求4CPU，flowers2等规模2/4/8项目及2大+6小混合项目各三轮，共66次重建全部通过，最终资源归零。12帧项目均12/12、65pairs、3735points，48帧项目均48/48、427pairs、17144points。等规模8项目小任务queue P95中位数5.525秒，混合场景14.085秒（小样本批次统计，不是线上尾延迟估计）；存在整段重建持有额度导致的短任务等待，未证明持续到达下无饥饿。详见[共享Runtime真实重建报告](output/shared_workflows_flowers2_20260907/REPORT.md)。后续 C2/C3、C4 已完成；C5 在独立 runner 退出/信号/reap 回归通过后，经新授权于 _02 完成2烟测和6轮正式RSS对照，15 workflows/120逐图完整质量一致、最终账本归零、8个child均真实wait4回收。512/1024MiB预算 makespan中位数5.007502/5.050906s、OS峰值3787145216/3799597056 bytes，未证明并发加速。见 [C5 _02 完成报告](output/c5_flowers2_20260908_02/REPORT.md)；[旧 _01 停止报告](output/c5_flowers2_20260907_01/REPORT.md)及未知exit/RSS原样保留。不拆 DAG/改优先级，不进入 C6。

## 执行任务表（按顺序）

| 组别 | 任务 | 状态 | 当前验收/下一步 |
|---|---|---|---|
| A | A1 固定输入、配置、版本和 hash | ✅ 已完成 | 继续保留现有 provenance 与独立 output |
| A | A2 image-only flowers2 前 48 帧三次完整画像 | ✅ 已完成（有 profiling 异常说明） | 48/48、428 pairs、1 model；matching/search 仍为主要 pair 热点；报告见 `output/imageonly_flowers2_a2_20260907_REPORT.md` |
| A | A3 database-first flowers2 前 48 帧三次画像 | ✅ 已完成 | 仅作为数据库复用基线，不与 image-only 直接宣称优劣 |
| A | A4 960 帧扩展前资源/时间预算检查 | ⏳ 待做 | 先完成 B1、B2，再决定是否扩展 |
| B | B1 匹配完整输出 regression fixture | ✅ 已完成（真实子集 + 人工边界 + 独立 oracle） | `sift::tests::uint8_matching_fixture_preserves_complete_output_semantics`；固定完整输出 BLAKE3 `3e9d58d11f4f059fbd8c068a03bccacd15cd87b84af93cc56480199715e102d7` |
| B | B2 搜索内核 SIMD/cache 行为分析 | ✅ 已完成 | 当前整数路径约 348.27 ms，中位数；旧浮点参考约 1654.72 ms；汇编显示两者均有 NEON，但生产路径为整数扩展/累加；报告见 `output/sift_search_b2_20260907_REPORT.md` |
| B | B3 选择一个精确搜索候选 | ✅ 本候选关闭：不采用 | 64 行单查询分块未证明收益，临时代码已移除 |
| B | B4 搜索微基准 | ✅ 本候选完成 | 6 轮：baseline 350.528 ms / blocked 350.205 ms；不足以保留 |
| B | B5 image-only 三次端到端验证 | ⏸ 本候选不执行 | 微基准无稳定收益，不运行昂贵端到端实验 |
| C | C1 共享 Runtime 多 workflow 基线 | ✅ 已完成 | 已证明准入/释放/排队可观测，不证明比固定槽位更快 |
| C | C2 确定性晚到小任务实验 | ✅ 已完成（3 次） | 2×48 帧真实 Started 屏障 + CPU8/pending2 快照后提交 12 帧，pending3 确认排队；小任务 queue/service/end-to-end 中位数 10.102/1.633/11.692s；质量计数一致、最终资源全零；证据 `output/c2c3_flowers2_20260907_01/REPORT.md`（屏障位于 reconstruction 内首次 local BA Started，不是函数入口） |
| C | C3 排队取消响应实验 | ✅ 已完成（3 次；仅准入取消） | 已知排队时取消，cancel-to-return 中位数 4.994ms；typed Cancelled、grant/service=0、无事件/模型，未进入 reconstruction body；取消返回且 pending 恢复2后才释放大任务，全部大任务完成、最终 CPU/memory/pending/IO/GPU/exclusive 全零；同 C2 报告。不证明运行中 BA/native 工作中断或真实 GPU 负载清理；该轮当时止于 C2/C3，后续授权 C4 已另行完成 |
| C | C4 无准入/固定槽位/Taskflow 对照 | ✅ 已完成（用户批准继续；三模式各 3 次） | 2×48+6×12 DB复用、固定10ms提交序列、轮换交错顺序；independent（每项目独立Runtime，并非不用Taskflow库）/外部FIFO fixed2+独立Runtime/shared CPU8 eachjob4；makespan中位数14.259/19.527/19.592s，小任务端到端P95中位数5.107/19.467/19.532s，RSS中位数828784640/639877120/633454592 bytes。fixed2外部等待计入总queue；72次质量计数一致，51个最终Runtime快照全零。shared未证明快于fixed2；independent允许更高总并发，非等物理CPU效率结论。取消未对照，C2/C3仍仅适用于shared；详见 `output/c4_flowers2_20260907_01/REPORT.md`。该轮当时止于C4；后续授权C5已另行实施并按安全门停止，不改优先级/DAG |
| C | C5 特征提取并发 RSS 压力 | ✅ 有界对照验收完成（_02：2烟测+6正式） | 前8图、默认CPU SIFT8192、每job4线程、共享CPU8、stage估算512MiB不变。512/1024MiB预算各3轮：makespan中位数5.007502/5.050906s，OS峰值3787145216/3799597056 bytes，账本峰值536870912/1073741824；parallel约+0.87%且三对方向不一致，不证明加速。15 workflows/120逐图完整bits/bytes hash一致、43093 keypoints/job，最终资源全零；8次退出ESRCH均获得真实wait4 exit0/RSS，无其他监控/回收失败，所有峰值<8GiB。旧_01失败冻结，历史EPERM仍未解释；RSS不是账本预算，SIGKILL等无watchdog保证。见 `output/c5_flowers2_20260908_02/REPORT.md`；不改生产默认/DAG/优先级，止于C5，不开展C6 |
| C | C6 阶段粒度调整 | ⏳ 暂不开始 | 只有 C 数据支持才改 DAG |
| D | D1 SIFT NEON | ✅ 已完成 | 已通过逐位 oracle、回归和 Flowers24 交错对照 |
| D | D2 Ceres profiling | ✅ 已完成 | 已定位 DenseSchur/Schur elimination 热点 |
| D | D3 Ceres 线程 1/2/4 | ✅ 已完成 | 4 线程整体最佳，关闭线程调优分支 |
| D | D4 Ceres native fork 评估 | ⏳ 独立立项 | 不自动开始，不改变当前默认依赖 |
| D | D5 SIFT 后续卷积/缓存优化 | ⏳ 暂不开始 | 只有匹配优化后仍为主要瓶颈才进入 |
| E | E1/E2/E3 96/192/960 帧扩展 | ⏳ 待做 | 按 48 → 96 → 192 → 960 逐级预算检查 |

**执行规则：**64 行候选已关闭，不再调 block size。B1 的真实 descriptor 固定输出及独立预期值已补齐，C2/C3、C4 已完成。C5 旧_01按监控失败停止且冻结保留；OS退出/信号/reap独立回归后已获新授权，在_02按smoke1→smoke2→serial-r1→parallel-r1→parallel-r2→serial-r2→serial-r3→parallel-r3逐条门禁全部通过，最终-O analyzer仅一次通过，C5有界对照验收完成但未证明加速。现止于C5，不自动重试/扩展、不改变默认或进入C6；任何后续诊断须另行授权。此前把人工 fixture 通过标为 B1 全部完成过早，本轮已补齐缺口；B2 已有微基准和历史汇编证据，但未测 cache miss，不能宣称缓存归因已证实。

### Review 后修复进展（2026-09-10 更新）

用户已授权推进 review Top3；以下更新取代上方 C5 当轮“止于 C5、后续另行授权”的执行停止点，但不改变历史实验结论。

| 任务 | 当前状态 | 验收与剩余范围 |
|---|---|---|
| R1 VLFeat 指数表并发初始化 | ✅ 修复并验证 | POSIX/Windows once，仅同步初始化；修改前 TSan 复现 expn_tab 写竞争，修改后 normal/count/TSan 各 10 个新进程×8线程×12次提取逐位一致，SIFT focused 18 tests 通过。Windows 分支未编译验证；见 `output/expn_once_20260908_01/REPORT.md` |
| R2a admission 执行池与 worker 上下文 | ✅ 合成回归通过 | 每次 admission 独占惰性池，弱引用 TLS 覆盖各 worker，嵌套借用父额度；本池 worker 的 native bridge 至多 CPU1，fixed minimum>1 拒绝，coordinator 保留原额度。default/no-default execution 各23 tests通过；不保证 detached spawn 或任意外部线程传播 |
| R2b 特征内存规划 | ✅ 组合入口接线并通过 focused gates | 显式绑定的标准 CPU default/selected DB、mapper image-only、sequence keyframe/remaining/adaptive 均按尺寸规划；per-image 窗口统一最大估算 floor 避免异构任务与顺序 commit 等待环。mapper 在 admission 后、首个 decode 前复核尺寸/encoded length；动态预算不足仍等待/可取消，超 ceiling/父 grant 拒绝。默认无绑定、custom/nonstandard 保留旧估算契约，无自动工作集保障。默认 mapper 9/9、no-default 2/2；sequence memory/taskflow default 4/6/2、no-default 3/4；memory review 4/4。见各 `output/*memory*` 报告 |
| R3 BA 失败状态闭环 | ✅ focused 回归通过 | outcome 区分未尝试/失败/提交及后处理；失败不推进成功 watermark，dirty 保留 final refinement 资格；未改变 solver、迭代预算或强制 full-budget final。global_ba_ 22 passed、6 ignored，schedule/caps筛选通过（有重叠）；见 `output/ba3_review_20260908T092053Z/REPORT.md` |
| R4 Taskflow DAG artifact liveness 准入检查 | ✅ 完成并修复集成回归 | 提交前将节点直接依赖的 output lease 与当前 working/output 共同检查；producer 900 + direct consumer working 200 在 1000 ceiling 下返回 `Unschedulable`。更早祖先不重复累计，跨节点存活必须由中间 output 显式声明；新增三节点 release 回归并恢复 RustSFM ordered extraction window。runtime 28/28、feature window 4/4 通过；仍不是精确 RSS/allocator 上限 |
| R5 queue timing decomposition | ✅ 完成并验证 | `queue_ms` 保留 workflow 创建到 task 开始的总等待，新增 `dependency_wait_ms` 与 `resource_wait_ms`；高层 stage/sequence report 和 runtime `TaskReport` 均接线，pre-admission 未获 grant 的失败/取消报告 timing 为 0；runtime 27、execution 23、BA 7 focused tests 通过。此修复只改善正确性与观测，不构成加速证据 |
| R6 mapper DB cache 生命周期边界 | ✅ 完成并验证 | `prepare_mapper_features` 只在 admission 前解析路径和 SIFT 计划，不再预加载 `DatabaseCache`；已有 DB 在 reconstruction stage 获得 grant 后加载，等待期间 DB 消失则 fail closed。新增损坏 DB + queued cancellation 回归，mapper composition focused 9 tests 通过。 |
| R7 sequence initial reconstruction 生命周期 | ✅ 完成并验证（2026-09-12） | admission 前只校验固定 DB/sparse 路径并对全部可能 target 做保守的 missing-feature plan；完整 DB/sparse 校验、`Reconstruction` 单次加载和 partial-keyframe registration plan 构建均移入 stage grant。queued cancellation 严格在模型加载前返回 typed cancellation；等待期间 sparse 消失在恢复预算后 fail closed；sequence integration 69/69 通过。model growth 仍由 caller other-work floor 估算覆盖，grant 不是 RSS 上限；未据此宣称加速 |
| R8 GPU-only matching 迁移门 | ✅ focused 与 CI 配置完成（2026-09-12） | 产品约束为 SIFT matching 仅使用 wgpu，不提供 CPU fallback。修复 `gpu-wgpu` 条件编译；旧 FIFO verifier 控制不再生成 trace；no-default 仅作 lib compile check，GPU integration 显式启用 `gpu-wgpu,vlfeat-sift`，Linux CI 安装 Mesa Vulkan 并串行执行 GPU 测试。matching 30/30、GPU integration 81 passed/1 ignored、默认 lib 760 passed/19 ignored；fixture reference 提交前仍须确认纳入 Git |
| R9 packed GPU SIFT distance | ✅ 交错验证完成，候选保留（2026-09-13） | CPU packing 为每个 descriptor 附加精确 `Σx²`，WGSL 使用 packed `dot4U8Packed` 计算 `‖q‖²+‖t‖²-2q·t`。flowers2 first48 baseline/candidate 各3轮，wall 中位数 63.80→41.64s（-34.7%），descriptor matching 39.82→17.04s（-57.2%），RSS 基本不变；6轮完整 pairs SHA-256 均为 `613a73f…97216`。focused、matching 30/30、完整 lib 760 passed/19 ignored。下一步只画像并优化 GPU geometry，不改 matching policy |
| R10 geometry compute+copy 单提交 | ⛔ smoke 无收益，已回退（2026-09-13） | scorer 将 compute 与 readback copy command buffers 合为一次 queue submission；focused 3/3、pairs hash 不变，但 smoke wall 43.86s、geometry 25.80s，劣于相邻 packed-only 的约41.3/23.73s，E/F/H readback wait 几乎不变。源码已恢复，不做多轮、不重试 |
| R11 bounded GPU geometry masks | ✅ 交错验证完成，候选保留（2026-09-13） | 每个64-trial decision window 按原 ordinal 即时预取最多4个当前 contender mask，CPU 仍逐候选重新比较/refine/update frontier。3×3 wall 中位数42.96→40.20s（-6.4%），geometry 25.63→22.87s（-10.8%），readback wait 15.31→13.19s；完整 pairs hash 一致。整窗口预取因严重过算已拒绝；batch failure 回到当前 candidate 的标量 GPU mask 并关闭本窗口 batching |
| R16 pair-level descriptor/geometry overlap | ✅ first48+960 质量门通过（2026-09-16） | 960 matching 1108→1094 s，digest `a4e8e8ec…2295c9`。描述子 kernel wait 未变。见 `output/pair_pipeline_r16_20260916/REPORT.md` |
| R17 tiled GPU SIFT matcher | ✅ first48+960 质量门通过（2026-09-16） | 960 matching 1668→722 s（−57%），desc 669→196 s，digest `a4e8e8ec…2295c9`。见 `output/pair_pipeline_r17_20260916/REPORT.md` |
| R12 geometry scorer session buffer 复用 | ⚠️ 目标指标改善但端到端不可证明（2026-09-13） | scorer session 复用 model/summary/mask/params buffer 与 bind group，按容量翻倍增长。逐位输出一致，focused 与完整 lib 763 passed/19 ignored 通过。scorer buffer_prepare 0.568→0.397s（-30%）、submit 0.152→0.091s（-40%），但 readback wait 13.13→13.20s 不变、wall 40.74→40.95s，收益被同步延迟完全掩盖。**本轮真正产出是瓶颈定位见 R12 小节，不作加速宣称** |

| R8 GPU-only matching 测试/CI 迁移 | ✅ focused 门完成（2026-09-12） | 不保留 CPU matching fallback。`no-default-features` 改为最小编译门；matching/sequence/adaptive 集成门显式启用 `gpu-wgpu,vlfeat-sift`，Linux CI 安装 Mesa Vulkan。GPU session 测试按 feature gate 编译，旧 FIFO verifier 控制不再生成 trace 且不得改变 GPU 结果。no-default release check 通过；GPU matching 30/30、GPU integration 81 passed/1 ignored。完整默认 lib 门及真实 flowers2 尚待后续验证 |

- 内存/执行最终集成门：default/no-default memory 18/15、feature_extraction 31/27、execution 23/23，task_control 10；新增组合入口 focused 门：mapper 9/2、sequence memory 4/3、sequence integration memory 6/4、taskflow sequence 2/0、memory review 4/4。测试集合有重叠，不累加为独立总数。见 `output/feature_memory_gate_20260908T142841Z/FINAL_AUDIT.md`、`output/composition_memory_mapper_20260909/REPORT.md`、`output/sequence_memory_gates_20260909T090715Z/REPORT.md` 与 `output/memory_review_20260909T091946Z/REPORT.md`。
- runtime `queue_time`/高层 `queue_ms` 现在表示从 workflow 创建到 task/stage 开始的总等待；`dependency_wait_time`/`dependency_wait_ms` 表示依赖完成前等待，`resource_wait_time`/`resource_wait_ms` 表示 ready 后等待资源。不要再把 `queue_ms` 单独解释为纯 resource wait；没有 runtime grant 的 pre-admission 失败/取消报告 timing 为 0。
- 本轮只完成 fail-closed correctness、observability 和 mapper cache lifecycle 修复，未运行新的真实 flowers2 大实验，不能据此宣称 Taskflow 加速；历史 C4/C5 结论不变。
- 独立只读复核确认异构窗口等待环、默认入口兼容性、transient budget 误拒绝、worker native 超额、mapper 排队后输入变化、adaptive 的 PnP-only GPU 误申请、取消错误优先级及 keyframe 静态校验顺序问题均已关闭。窗口 floor 依赖当前 scheduler 的 ready 顺序，并可能减少异构输入的并行度。
- 估算是分配规划而非 RSS/allocator 上限，数据相关候选余量不是已证明的最坏上界。本轮未运行大图、真实重建或新的 C5 对照；历史性能结果不能用作新执行池的加速证据。
- 下一步：以 [`output/b1_r7_baseline_flowers2_20260910/REPORT.md`](../output/b1_r7_baseline_flowers2_20260910/REPORT.md) 为 R7 后 48 帧参考基线。R9 已按新测瓶颈证据完成；下一单变量方向是 GPU geometry 画像与候选筛选。暂不拆 DAG、提高默认预算或扩至960帧。

### R9：packed GPU SIFT distance（2026-09-13）

**状态：候选保留，尚未提交。** 实验目录：`output/gpu_only_profile_flowers2_20260913/`。

- 输入固定为 `artifacts/inputs/flowers2/images` 前48帧、8192 features、sequential matching、`random_seed=1`；每轮均从 SHA-256 `af18e5b2…ea2abd` 的 frozen DB 创建新副本，不复用已写数据库。
- 单一实现变量：CPU 将每个128维 `u8` descriptor 打包为32个 little-endian `u32` 加一个 `Σx²` norm；WGSL 每个 word 只调用一次 `dot4U8Packed`，以 `norm(query) + norm(target) - 2 * dot(query,target)` 得到平方距离。最大值 `128 * 255² = 8,323,200 < 2²⁴`，转为 `f32` 精确；未改 ratio、distance、cross-check、pair policy、geometry 或 Taskflow。
- 运行顺序包含初始 `baseline-1/candidate-1`，随后交错 `baseline-2 → candidate-2 → candidate-3 → baseline-3`。baseline wall 为 63.80/63.72/64.45s，candidate 为 54.42/41.64/41.05s；全部 candidate 均快于全部 baseline。
- 三轮中位数：wall `63.80 → 41.64s`（约 -34.7%）；CLI matching `62.940 → 40.922s`（约 -35.0%）；GPU descriptor matching `39.820 → 17.042s`（约 -57.2%）；descriptor readback wait `38.919 → 16.497s`（约 -57.6%）。readback calls/bytes 保持 922 / 157,757,216，说明该 wait 含 kernel 完成等待，不能解释成纯复制收益。
- GPU geometry 中位数 `22.895 → 23.733s`，接近但略高；`candidate-1` 的 32.354s 是本组明显高值，后两轮均约23.73s。不能据此宣称 geometry 改善；R9 收益来自 descriptor kernel。
- RSS 中位数 `241,811,456 → 241,991,680 bytes`，约 +0.07%，没有有意义的工作集增加。descriptor storage 理论上由32增至33 words（+3.125%），但生命周期保持在单次匹配调用内。
- 质量：6轮均为461 matched/verified pairs、259,206 matches；规范化完整 `pairs` SHA-256 均为 `613a73f3838832d706f1574792d0ece141fb89ce0afcac91ca6fa3e4dbb97216`，逐对象完全一致。
- 门禁：`cargo fmt --all -- --check`、`git diff --check`、GPU matcher focused、`feature_matching_db::tests` 30/30、完整默认 lib 760 passed/19 ignored 全部通过；均为 release/offline、Cargo `-j1`、测试线程1并限制三项 native thread 环境。
- **下一热点：GPU geometry。** 先拆分 essential/fundamental/homography 的 candidate generation、scorer score/mask、readback wait 与 CPU refinement；只选择一个有证据的最窄候选。不要仅凭 readback calls 多就直接合并，也不在同一轮改变 RANSAC/质量策略。

### R10：geometry compute+copy 单提交候选关闭（2026-09-13）

- 画像依据：packed-only 三轮中，homography scorer 中位数为4,037 score calls、3,321 mask calls、7,358 readbacks、约9.354s readback wait；fundamental 为2,500 readbacks/3.179s，essential 为2,167/2.758s。
- 单一候选：保持现有 compute/copy command buffers 和同步 map/decode，只将两次 `queue.submit` 合并为一次有序 submission；不改 shader、RANSAC sampling/frontier、candidate ordinal、threshold、tie-breaking 或 CPU refinement。
- scorer support/mask/Sampson focused 3/3 通过；flowers2 smoke 仍为461 pairs、259,206 matches，完整 pairs SHA-256 仍为 `613a73f…97216`。
- packed-only 相邻两轮 wall 41.64/41.05s、geometry 23.733/23.724s；候选 smoke wall 43.86s、geometry 25.801s。E/F/H readback wait 为2.768/3.184/9.368s，与 packed-only 的约2.757/3.178/9.352s 基本相同，没有机制级收益证据。
- 按 smoke 停止条件回退 `gpu/context.rs` 与 `gpu/scorer.rs`，不运行多轮或完整门禁。后续不要重试单纯合并 submission；若继续 geometry，应先设计保持64-trial decision 顺序的 bounded batched-mask 实验，并单独证明内存上界和逐位输出。

### R11：bounded GPU geometry masks（2026-09-13）

**状态：候选保留，尚未提交。** 实验目录：`output/gpu_only_profile_flowers2_20260913/geometry-mask4-interleaved/`。

- 首个整64-trial窗口预取候选被 smoke 拒绝：essential/fundamental/homography mask 计算膨胀到129,441/71,658/30,902，geometry 28.176s、wall 46.28s。该策略未保留。
- 最终候选只在当前 candidate 确实需要 mask 时，从当前位置按 ordinal 选择最多4个相对当前 best 的 contender，一次二维 GPU dispatch/readback；CPU 仍按原顺序重新判断、构建 support、refine、更新 best 与 dynamic frontier。下一批只能在前一批最早4个 contender 已消费或失效后启动，因此 host 保留上界为4个 observation masks。
- scalar synthetic scorer 保留原错误/顺序门；新增 batched synthetic scorer 对照，最终 model、support bits、完整 mask、sampling 和 refinement 顺序一致。真实 WGSL 回归验证两个可区分模型的一次2D dispatch、model-major mask 顺序、一次 readback。
- flowers2 first48 使用同一 frozen DB。三轮 baseline wall 41.05/43.39/42.96s，三轮 candidate 40.73/40.16/40.20s，candidate 全部更快；中位数 wall `42.96 → 40.20s`（约-6.4%）、matching `42.814 → 39.998s`（约-6.6%）、geometry `25.628 → 22.872s`（约-10.8%）、geometry scorer readback wait `15.306 → 13.185s`（约-13.9%）。descriptor 约17.0s不变。
- geometry scorer readbacks 中位数12,025→10,352（-13.9%），readback bytes 51,838,616→63,637,340（+22.8%）；以有限额外GPU计算/传输换取更少同步点。RSS 中位数246,005,760→243,810,304 bytes，无增长信号，但不是一般输入的RSS上限证明。
- 6轮均为461 matched/verified pairs、259,206 matches，规范化完整 pairs SHA-256 均为 `613a73f3838832d706f1574792d0ece141fb89ce0afcac91ca6fa3e4dbb97216`。
- 错误与内存边界：转换、buffer size和dispatch limit在batch前验证；batch失败或返回数量异常时，丢弃预取结果并从当前candidate走原标量GPU mask，随后关闭本decision window batching，使错误按当前candidate路径暴露。每个已越过candidate的预取mask立即释放；新增失败注入对照确认最终best、sampling、refinement和mask消费与标量路径一致。这里的“标量”指单模型GPU dispatch/readback，不是CPU geometry fallback。

### R12：geometry scorer session buffer 复用与同步延迟瓶颈定位（2026-09-13）

**状态：代码保留（无回归），但不构成加速证据；本轮主要价值是证伪与瓶颈定位。** 实验目录：`output/gpu_only_profile_flowers2_20260913/geometry-scratch-reuse-smoke/`。

- 单一候选：`WgpuModelScoringSession` 在 session 生命周期内复用 model/summary/mask/params buffer 与 bind group，容量不足时翻倍增长并受设备 storage 限制裁剪；未改 WGSL、candidate ordinal、summary 严格排序、local refinement、dynamic frontier、batch 上限或任何策略参数。
- 复用安全性依据：scoring shader 用 `params.model_count`/`params.observation_count` 而非 `arrayLength` 约束全部访问，且每次 dispatch 完整写满随后读回的区间，readback 只拷贝请求的元素范围；因此超额分配的 buffer 既不越界也不会读到上一次 dispatch 的残留值。新增 `wgpu_model_scorer_reused_buffers_match_isolated_sessions`，用降序再升序的批量大小（4/1/2/1/4）交替 score 与 mask，对照每批独立 session，两种 residual kind 下 support、完整 mask 与 readback bytes 全部相同。
- 门禁：`cargo fmt --all -- --check`、`git diff --check`、GPU scorer focused 7/7、RANSAC focused 41/41、完整默认 lib **763 passed / 19 ignored** 全部通过。
- 质量：flowers2 first48 仍为 461 matched/verified pairs、259,206 matches。用新增 [pair_output_hash.py](tools/pair_output_hash.py) 对 `matches` 与 `two_view_geometries` 全部行（含原始 blob、float 按位）规范化后，R11 已提交 smoke 与本候选 smoke 的 digest 同为 `fbe4acf540b62c1866f3ccacc0d549b24c039ec1ef0a6ae192fb44a26d5c1a32`，即逐位一致。该脚本的规范化与历史 `613a73f…97216` 不是同一算法，两者不可互比。
- 目标指标确实下降：scorer `buffer_prepare` 合计 `0.568 → 0.397s`（-30%），`submit` 合计 `0.152 → 0.091s`（-40%），每次 score/mask 的 4 次 buffer 创建加 1 次 bind group 创建被完全消除（约 10,352 次调用 × 5）。score/mask/readback 计数逐项不变。
- 但端到端不可见：`readback wait` 13.132 → 13.199s、geometry 22.356 → 22.920s、wall 40.74 → 40.95s、RSS 248,676,352 → 244,547,584 bytes。分配开销只占 wall 约 1.4%，完全被同步等待掩盖，单轮差异在既有轮间噪声（约 ±1s）之内。因此**不跑 3×3 交错，不宣称加速**。
- **瓶颈定位（本轮关键结论）**：per-readback 等待在三个 model family 间高度一致——essential `2.431s / 1896 = 1282µs`、fundamental `2.673s / 2100 = 1273µs`、homography `8.095s / 6356 = 1274µs`。三者传输量（20.9 / 16.6 / 26.1 MB）与每次计算量（平均每次 score 的模型数 2190 / 1231 / 476）差异巨大，per-call 延迟却只差 0.7%。这说明 geometry readback wait 由**每个同步点约 1.275ms 的固定往返延迟**主导，与字节数和 kernel 工作量基本无关：`10,352 × 1.275ms ≈ 13.2s`，占 geometry 的约 58%、wall 的约 32%。
- 由此排除的方向：继续压缩 buffer 分配、staging 复用或纯传输量优化都不可能有实质收益（R10 合并 submission 的失败与此一致）。**唯一有效杠杆是减少同步点数量**，即减少 `score_calls + mask_calls`（当前 5,970 + 4,382 = 10,352），而不是让每次同步更便宜。
- 限制：1.275ms 是本机 Metal/wgpu 后端在此工作负载下的观测值，不是设备无关常数，也未分离驱动唤醒、`map_async` 回调与 GPU 实际执行；不可外推到 960 帧或其他后端。

### S1：flowers2 960 帧全流程结算（2026-09-13）

**状态：已完成。** 报告：`output/flowers2_960_settlement_20260913/REPORT.md`；每阶段单轮，是占比普查而非验证过的 delta。

| 阶段 | wall | RSS | 输出 |
|---|---:|---:|---|
| extract | 332.3s | 3.95GB | 960 images / 6,612,183 keypoints |
| match | 2150.9s | 4.85GB | 14,045 pairs / 13,894 verified / 10,019,172 matches |
| reconstruct | 599.0s | 9.33GB | 960/960 registered / 437,185 points / 1 model |
| 合计 | 3082.2s (51.4 min) | | |

- 阶段占比：match 69.8%、reconstruct 19.4%、extract 10.8%。
- 瓶颈普查（占全流程）：geometry GPU 同步等待 684.6s/22.2%、descriptor readback 等待 639.6s/20.8%、**essential candidate generation（CPU 五点法）590.6s/19.2%**、CPU SIFT extract 332.3s/10.8%、global BA solve 330.8s/10.7%。前三项 62.2%，累计归类 93.5%。
- **固定同步延迟结论在 44 倍规模下完全复现**：per-readback 等待 essential 1490µs、fundamental 1489µs、homography 1486µs，离散度 0.3%；同步点从 10,352 增到 460,025，per-call 成本只涨 17%。
- **essential candidate generation 每 pair 超线性恶化**：12.2 → 42.0 ms/pair（3.4×），score calls/pair 1.74 → 5.92（3.4×），而 models/score call 不变（2190 → 2187）。因此不是 batching 问题，而是低 inlier-ratio pair 把动态 RANSAC 推向迭代上限。181.9M 个 essential 模型 / 590.6s = 3.25µs每模型，对比 fundamental 0.52µs、homography 0.64µs。
- **matching 是严格串行的 CPU↔GPU 交替阻塞**：user/real = 0.38（18 核机器）；CPU 忙 687.5s（candidate gen）+ 阻塞等 GPU 1324.2s = pair_compute 的 93.8%。CPU 生成模型时 GPU 空闲，GPU 打分时 CPU 空闲。
- reconstruct 内部：BA 占 404.7s（solve 346.7 / setup 27.4 / postprocess 30.6）；**27 次 global BA = 330.8s（平均 12.3s/次，占 BA 时间 81.7%）**，958 次 local BA 仅 73.9s（77ms/次）。
- 限制：单轮无交错基线；1.49ms 为本机 Metal/wgpu 观测值；Taskflow 授予 512MiB 是准入预算，不约束 9.33GB RSS；extract 使用 CPU SIFT 以与 B1 可比。

### R13：candidate generation 有界池并行候选关闭（2026-09-13）

**状态：ⴛ smoke 严重退化，已完整回退。** 实验目录：`output/gpu_only_profile_flowers2_20260913/parallel-candgen-smoke/`。

- 单一候选：在 `run_gpu_ransac_batches` 内将采样与模型求解分离——采样仍严格逐 trial 顶序，仅对无状态的最小解求解在 `execution::parallel` 有界池（matching admission = 4 threads）上用 indexed `par_iter` 并行，按 ordinal 收集。三个 family 的 `generate_models` 均只捕获不可变引用，确为纯函数，因此语义不变。
- 质量确实不变：461 pairs、259,206 matches，规范化 pairs digest 仍为 `fbe4acf5…1a32`，**逐位一致**；RANSAC focused 42/42 通过。
- 但端到端严重退化：matching `39.96 → 64.11s`、wall `40.95 → 118.70s`（+190%）、user CPU `10.22 → 31.93s`（+212%）。
- 关键证据：**意图收益几乎不存在，而无关的 GPU 路径被毒化**。candidate generation 仅 `7.465 → 6.893s`（-0.57s）；而 descriptor matching 的 readback 调用数和字节数完全不变（922 次），其 readback wait 却 `16.326 → 33.500s`（+105%），geometry readback wait `13.199 → 16.197s`（+23%）。我未触碰这些代码路径。
- 机制解释：rayon worker 在每次 install 后自旋再休眠，数千次 install 使 4 个 worker 持续热转，抢占了延迟敏感的 `device.poll(Wait)` 路径的 CPU，直接抬高 GPU 同步等待。user CPU 三倍与此一致。wall 减 matching 的缺口从 0.99s 涨到 54.59s，提示额外的线程 churn。
- 48 帧 smoke 对此候选本身也不是有效代理：该规模下 candidate generation 仅 7.5s，而 960 帧为 687.5s。但自旋竞争机制与规模无关且在 960 帧下会更差，故不升级到 960 帧重试。
- 保留产出：新增 `gpu_ransac_one_physical_batch_scores_every_trial_in_order`，固定“一个 physical batch 必须将全部 trial ordinal 按序无重复无缺口交给 scorer”这一契约，供任何后续改动参考。
- **后续结论：不要在这个热路径上引入细粒度线程池 hand-off。** 若要同时吃掉 CPU 空等和 GPU 空等，必须是粗粒度的 pair 级流水线（pair N+1 的 CPU 生成与 pair N 的 GPU 打分重叠），且每个线程内部保持串行。

### R14：GPU geometry 同步点削减——融合 score+mask 与几何增长 batch（2026-09-15）

**状态：两项候选均保留。** 报告：`output/geometry_scored_masks_20260915/REPORT.md`；对照 `main` `400e0cb`（= `8b0b63b` 代码）。

- 依据 R12/S1：geometry 读回等待由每同步点约 1.3–1.5 ms 固定延迟主导，唯一杠杆是减少 `score_calls + mask_calls`。
- **R14a 融合 score+mask（单变量）**：新 WGSL 入口 `score_models_with_masks` 在打分 dispatch 内同时写出每个模型的 packed inlier bitmask（每 32 观测一个 u32），summary 与 mask 一次 readback（`read_two_buffers_profiled`）。per-lane 观测顺序与树形归约与 `score_models` 完全一致，summary 逐位不变；mask 位 = `is_inlier(model_residual())`，与 `write_mask` 同一谓词。决策循环优先取已到主机的 mask，独立 `mask_calls` → 0；批量/标量 mask 路径保留为 fallback。容量规划计入 mask buffer。
- **R14b 几何增长物理 batch（单变量）**：`GpuRansacBatchPolicy` 增加 `max_score_trials`；物理 score batch 按 512→1024→2048→4096 增长，决策窗口保持 64，`dynamic_max_trials` 仍在 `gpu_ransac_batch_end` 处截断；shared COLMAP RNG stream 模式保持 64/64（不得越过决策前沿采样）。输出与固定 batch 逐位一致（已有 frontier 独立性测试 + 新增 grown-vs-fixed 测试）。
- flowers2 first48 交错三轮中位数：wall **43.42 → 34.36 → 26.06 s**（−40.0%），geometry **25.48 → 16.95 → 9.06 s**（−64.4%），同步点 **10,352 → 5,970 → 2,789**（E/F/H 1896/2100/6356 → 559/607/1623），readback wait 15.62 → 9.06 → 4.25 s，候选生成 6.08 → 3.25 s，descriptor ≈17 s 不变，RSS 不变。每个 candidate 轮均快于每个 baseline 轮。
- 质量：12 轮全部 461 pairs / 259,206 matches，`pair_output_hash.py` digest 均为 `fbe4acf5…1a32`（与 R11/R12 相同，逐位一致）。
- cap 探测：8192/16384 同步点与 4096 完全相同（本数据集无 run 超过 7,680 trials），保留 4096（最小满收益 cap、最小投机浪费）。
- 门禁：fmt/diff-check、clippy 警告集 ⊆ 基线（少 1）、GPU scorer + RANSAC focused 40/40、完整默认 lib **774 passed / 19 ignored**；新增真实 GPU 测试覆盖 1…713 观测数的字边界与多轮 64-lane pass。
- 代价：readback 字节 E/F/H 每 run 20.9/16.6/26.1 → 97.5/61.9/108.3 MB，每同步延迟 1.3 → 1.5 ms；传输不是瓶颈，净收益为正。
- **Task 1 复盘（960 已确认）**：matching 1668→1108 s，geometry wait 586→167 s，pairs digest 与 settlement 逐位相同。设备端 argmax 不再需要；64-trial 决策窗口不可扩大。
- **R15 预取已回退**：同一 RANSAC 内预取下一批会改 sequential sampler，first48 82/461 two-view 行变化。overlapping readback API 保留；后续 pair 级重叠不得共用飞行中的 sampler。
- **R16 pair 级重叠（first48 质量门已过）**：pair N+1 描述子 GPU wait 期间串行生成 pair N 的 E/F/H 首个 physical batch（独立 sampler，wait 内不用 Rayon）。first48 digest 仍为 `fbe4acf5…1a32`；matching 24.97 s（R14b 中位数 25.70 s，单轮）。960 待测。

### R22：guided densify u8+Rayon（2026-09-16）

**状态：first48 门通过；960 guided match+recon 进行中。** 报告：`output/pair_pipeline_r22_20260916/REPORT.md`。

- 单一变量：`match_sift_guided_with_options` 在 DB/GPU 路径使用已有 `descriptors_u8`，避免每次极线候选把 float 描述子重新量化；forward/reverse 用 indexed Rayon；u8 路径在平方 L2 空间做 ratio/distance；保留 R21 的 F 下快速 densify（无二次 GPU RANSAC）。
- first48：matching **16.998 s**（ungided 15.545 / 旧 guided 49.148），matches **488246**（与旧 guided 相同），points **40194**（ungided 17143，约 2.34×），recon **28.9 s**。
- 预估 960 matching ≈ 782 s（相对 main retest 1668 s 约 0.47×），质量靠 densify 拉开；recon 墙钟待 960 实测（更密 track 可能抬高 BA）。
- R21 final BA early-exit（changed≤0.0015）在相同 R20 matching.db 上：584 s / 437212 points，相对 main 仍非大幅度双胜。

### R23：dense BA policy（2026-09-16）

**状态：完成；960 guided reconstruct 1007s/852k。** 报告：`output/pair_pipeline_r23_20260916/REPORT.md`。final gate 0.005、large scheduled 1 round、size=large 自动 ignore-redundant。global BA 760→175s；recon 仍高于 main。

### R24：capped guided densify（2026-09-16）

**状态：first48 门通过；960 进行中。** 报告：`output/pair_pipeline_r24_20260916/REPORT.md`。每对 densify 上限 1280；跳过 large scheduled prepare retriangulate；final gate 0.008。first48：16.0s / 405939 matches / 34535 points / recon 20.8s。

### R29–R30：stored-pose reuse and registration filter（2026-09-16）

**状态：matching 与点数已大幅超过 main；reconstruct wall 仍只略快于 settlement，目标未完成。**

- 前几轮 960 `pairs_ms≈110s` 不是 lookup miss。新 binary 计数：`no_geometry=175`，`stored_accepted=10174`，`stored_rejected=3545`，拒绝原因全是 stored mean error > 8px，随后 10000-iter five-point。把这类 fallback 限到 256 iter 后 `timing_pairs_ms=6.1s`。
- R29 960（R24 matching.db，seed 1，threads 4）：recon **562.1s** / 960/960 / **594264** points。相对 settlement 599s 为 0.94×，相对 retest 699s 为 0.80×，点数 1.36×。scheduled round-2 已消失；final 一轮。
- R30 把每帧全量 observation 扫描改成 observation manager 的 `num_visible_points3d`：`filter_frames_ms=0.44s`。总时间只到 **556.8s** / 595677 points。incremental 分段：`local_ba_ms=210s`，`next_image_ms=42s`，`triangulation_ms=37s`，`snapshot_ms=9s`。`sparse_maintenance`：complete 31s、merge 37s、filter 51s、delete 10s，frontier 约 11.5M。下一刀应减 local BA 后的 complete/merge/filter，不能靠丢掉 guided 点数换速度。

### B1：R7 后 flowers2 first48 对照基线（2026-09-10）

**状态：已完成（新目录；非相对 2026-09-07 的加速宣称）。** 报告：[REPORT.md](../output/b1_r7_baseline_flowers2_20260910/REPORT.md)；可复算 [metrics.json](../output/b1_r7_baseline_flowers2_20260910/metrics.json)。

- 输入：`artifacts/inputs/flowers2/images` 前 48 帧 symlink；8192 features；matching/reconstruction `random_seed=1`；threads=4；三项 BLAS/OpenMP=1。
- 准备一次：259,829 keypoints；461 matched/verified pairs；259,206 matches；DB SHA-256 `af18e5b2…ea2abd`；三份 frozen 副本重建后哈希不变。
- **database-first ×3**：均为 48/48、427 pairs、17,144 points、1 model；median process real **9.73 s**、RSS **323,698,688**、`elapsed_ms` **9723.65**。stage：`queue_ms`/`resource_wait_ms` median ≈ **0.007**，`dependency_wait_ms` **0**，grant 4/512MiB，`cancelled_or_failed=false`。
- **image-only ×3**（`--local-matching`）：均为 48/48、428 pairs、1 model；points 16079–16113（median **16082**）；median process real **70.64 s**、RSS **4,118,626,304**。queue 分解同为近零 bookkeeping。
- 与文档中的 2026-09-07 数量级一致（dbfirst ~9.98 s / 17144 pts；image-only 单次 ~65.68 s / 16090 pts）；差异视为噪声，**不作 R7 收益**。本路径是 mapper `reconstruct`，未直接锻炼 sequence R7 reload。
- 限制：单工作流无竞争，不能证明多项目饥饿/资源账本；summary 无旧 `timing_*` 分段字段；不与跨路径总耗时比优劣。

### B1：真实匹配输出回归门完成（2026-09-07）

- 新增 `src/feature/sift_regression_tests.rs`，仅通过 `sift.rs` cfg-gated 测试模块接入，不改生产算法或公开 API。
- 固定 flowers（不是 flowers2）历史 corpus 中 128/127 行真实描述子，共 32,640 bytes；保留原始 bytes 与选取索引，普通测试 `include_bytes!` 加载，不依赖 ignored output、图像提取或网络。来源和复现见 [fixture README](src/feature/fixtures/sift_b1/README.md)。这是回归子集，不是性能代表性样本。
- 独立 scalar float distance + 全排序 oracle 校验双向最近邻/次近邻索引与距离 bits；独立匹配筛选校验完整输出。覆盖 1/4 线程、indexed/brute、plain/profiled；人工明确预期覆盖空/单项、零与非零 ties、严格 ratio 边界、包含等号的距离边界、reverse ratio、mutuality、稳定排序和真正生效的截断（含切断 ties）。
- 真实子集输出：67 forward / 66 mutual，各自截断到 7；完整四场景生产输出在逐字段 oracle 通过后固定 BLAKE3 `fbb55fac40c5e1a1d2504548b5246f331a7f4936c46735551440d32579ac6e52`。原人工 hash 保留。
- 串行 release/offline focused 验证通过：新增 `b1_` 3 tests、原 index tests、原人工 hash；no-default index tests 和 lib check 通过。无默认特性下不把 uint8 matcher 预期用于 Lowe BBF。所有 Cargo 使用三项 native thread 环境为 1、240 秒 timeout。
- 只读审查未发现 oracle/wiring 正确性问题；发现 Python assert 可被 `-O` 禁用，已改显式异常校验，普通和 `python3 -O` 字节复现检查均通过，无 `--write`。
- 限制：不覆盖完整 corpus、GPU/guided/Lowe BBF、float descriptor fallback、execution-scoped pool routing 或重建精度；本轮不跑新 benchmark/reconstruction，不提出加速结论。

### B3/B4：64 行单查询分块候选关闭（2026-09-07）

- 六轮交错双向搜索，当前顺序扫描中位数 **350.5284795 ms**，候选 **350.2050835 ms**，差异约 **0.092%**，范围重叠，未证明稳定收益。原始数据：[benchmark JSON](output/sift_search_bench_b3_20260907.json)。
- 每轮候选的全部最近邻/次近邻索引和距离 bits 与独立 float reference 相同。1718 个默认 matches 来自未切换候选的生产 matcher，不能当作候选完整 matcher/hash 验证。
- 此候选只是每个 query 内把原 target 遍历分为 64 行，未改变访问顺序、未引入跨 query 复用；结果不否定真正的跨查询 tiling，也未证明 memory/cache bottleneck。
- 已移除生产模块中的候选方法/常量和 example 临时入口，保留原整数距离实现、原 index tests 和人工匹配 fixture。删除前源码归档：[rejected candidate](output/rejected_b3_blocked64_20260907_01/)。没有 reset 或覆盖历史实验。
- 清理后串行 `--release --offline -j 1` 验证：`sift_index::tests` **3 passed**；人工完整输出 fixture **1 passed**；`cargo check --example sift_search_bench` **通过**。三项原生线程环境均为 1，每命令 timeout 240 秒，无超时，仅编译器 warnings。
- 不运行 B5，不新增候选，不改变匹配阈值、BA、Taskflow DAG 或 Viewer。A2 的约 23 秒 search outlier 和模型点数波动仍未完成根因诊断，不能直接认定为调度问题或数值噪声。

### 已完成

1. RustSFM 高层流程已接入 Taskflow，支持阶段准入、CPU 额度、有界 Rayon、GPU 独占门和 sequence 两节点 DAG。
2. 已完成一次真实 Flowers 24 帧单工作流画像。
3. 已定位并优化 CPU SIFT 精确距离计算：
   - 128 维 `u8` 描述子平方距离使用 `u32` 整数累加，最后转换为 `f32`。
   - 最大距离 `128 * 255^2 = 8,323,200`，小于 `2^24`，转换精确。
   - 最近邻、次近邻、tie-breaking、ratio、cross-check 和默认匹配输出保持不变。
4. 固定真实描述子基准：双向搜索中位数约 `1654.0 ms -> 349.2 ms`，下降约 `78.9%`。
5. 完整 24 帧 CPU image-only Mapper：
   - 配对阶段中位数约 `59.231 s -> 14.136 s`，下降约 `76.1%`。
   - Mapper 总计时中位数约 `81.909 s -> 36.328 s`，下降约 `55.7%`。
   - 两组均为 24/24 注册、186 个有效配对，pair-quality 统计一致。
6. 最终 Global BA 的 `post_bogus_cameras` 回退仍存在。它是独立的质量问题，本轮没有把它和性能优化混在一起。
7. 已完成高层 Taskflow stage report（P1）：
   - `SfmTaskflow::run_with_reports` 和 `sequence_with_reports` 暴露 requested/granted CPU、memory、queue/service/total 时间及失败状态。
   - `ReconstructionSummary.stage_reports` 和 `SfmTaskContext::stage_reports()` 提供调用层读取入口。
   - 无 task context 的高层 reconstruction 路径也会记录顶层 `reconstruction` stage；sequence 会按 stage 记录。
8. 已完成 SIFT 内部阶段 profiling（P2）：
   - `RUSTSFM_PROFILE_FEATURES=1` 默认关闭，并区分图像准备、输入转换、backend kernel、结果转换、RGB/颜色/wide descriptor 和每图总耗时。
   - Flowers 24 帧真实运行确认：24/24 注册、186 个有效配对，SIFT backend kernel 是主要耗时。
   - VLFeat 标准 SIFT 子阶段进一步拆分：平均每图输入转换 `0.381 ms`、尺度空间 `2553.106 ms`、检测 `60.291 ms`、方向 `158.590 ms`、descriptor `105.664 ms`、输出整理 `0.208 ms`；子阶段和与 `sift_kernel_ms` 的最大残差约 `0.45 ms`。
9. P2 已评估优化候选，但尚未保留 SIFT kernel 加速。当前数据不支持优先优化复制或图像读取；每轮只选择一个实现点，并用同一输入做质量和速度对照。
10. 已评估一个 VLFeat backend 微优化候选：融合 descriptor 量化和 UBC 维度重排以移除临时拷贝。三次 24 帧对照未改善 feature wall span 或总耗时，且出现一次 22/24 注册；候选已回退，当前主代码不包含该改动。

详细证据：

- [`rustsfm/output/pair_profile_flowers24/report.md`](output/pair_profile_flowers24/report.md)
- [`rustsfm/output/sift_distance_kernel/`](output/sift_distance_kernel/)
- [`rustsfm/output/taskflow_feature_profile_check_fixed.json`](output/taskflow_feature_profile_check_fixed.json)
- [`rustsfm/output/taskflow_vlfeat_stage_profile_check.json`](output/taskflow_vlfeat_stage_profile_check.json)
- [`rustsfm/output/taskflow_feature_profile_baseline_seed1_run1.json`](output/taskflow_feature_profile_baseline_seed1_run1.json)
- [`rustsfm/output/taskflow_feature_profile_baseline_seed1_run2.json`](output/taskflow_feature_profile_baseline_seed1_run2.json)
- [`rustsfm/output/taskflow_feature_profile_baseline_seed1_run3.json`](output/taskflow_feature_profile_baseline_seed1_run3.json)
- [`rustsfm/output/taskflow_feature_profile_quantized_seed1_run1.json`](output/taskflow_feature_profile_quantized_seed1_run1.json)
- [`rustsfm/output/taskflow_feature_profile_quantized_seed1_run2.json`](output/taskflow_feature_profile_quantized_seed1_run2.json)
- [`rustsfm/output/taskflow_feature_profile_quantized_seed1_run3.json`](output/taskflow_feature_profile_quantized_seed1_run3.json)
- [`rustsfm/src/feature/sift_index.rs`](src/feature/sift_index.rs)
- [`rustsfm/examples/sift_search_bench.rs`](examples/sift_search_bench.rs)

## 总原则

- 所有 Rust 编译、测试和性能运行使用 `--release`。
- 启动进程前设置原生线程限制：

  ```sh
  VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
  ```

- 每轮实验只改变一个主要变量。
- 保留原始 JSON、日志和输出目录，不覆盖历史结果。
- 同时检查速度、注册率、有效配对数、几何质量、BA 状态和资源归零。
- 不把并行任务累计墙钟 span 当成 CPU 时间；不把单次异常样本当成稳定结论。
- 未做多轮数据前，不宣称普遍加速。

## 已完成阶段：拆解剩余的图像/特征准备时间

距离搜索优化后，24 帧运行中提取/准备阶段约 17～18 秒，已经是最大耗时段，约占优化后总时间的一半。本阶段已完成测量，未立即修改算法。

### 必须增加的计时点

在 `rustsfm/src/sfm/mapper/image_features.rs` 和必要的图像加载路径中，增加默认关闭的阶段计时：

1. 灰度图解码/读取；
2. SIFT 提取内核；
3. RGB 图像再次读取；
4. 颜色采样；
5. wide descriptor 构建；
6. 每张图总耗时；
7. 24 张图的累计墙钟 span、P50/P95、最慢 10 张图。

建议沿用 `RUSTSFM_PROFILE_PAIRS=1` 的思路，使用独立开关，例如：

```sh
RUSTSFM_PROFILE_FEATURES=1
```

不要在 Rayon worker 内逐条打印。所有记录在并行阶段完成后由调用线程统一序列化到 `--summary-json`，避免日志锁和 I/O 改变测量结果。

### 验收标准

- 默认不开启 profiling 时，生产输出和性能路径不改变。
- profiling 开启时能区分上述 6 类时间。
- 使用相同 24 帧连续运行至少 3 次。
- 输出 24/24 注册、186 有效配对、pair-quality 和 descriptor 数量。
- 根据结果决定主要优化方向：
  - 灰度/RGB 读取占主导：检查重复解码、图片格式和缓存边界；
  - SIFT 占主导：检查特征数量、VLFeat 调用、每图并行粒度和内存复制；
  - wide descriptor 占主导：检查是否所有 Mapper 路径都需要它，以及是否可延迟构建；
  - 颜色采样占主导：检查是否可以延迟到真正需要导出颜色时。

## 第二阶段：只优化特征准备的首要热点

得到第一阶段数据后，只选择一个热点处理。候选方向如下。

### A. 减少重复图像读取

当前 SIFT 路径会读取灰度图，随后再次读取 RGB 图，并在后续路径中可能再次取得灰度数据。只有在计时确认读取占比高时，才考虑：

- 在单张图工作项内复用已加载像素数据；
- 明确 RGB/灰度内存的生命周期；
- 避免为了减少读取而让所有图像长期驻留内存。

验收时必须同时观察峰值 RSS，不能只看耗时。

### B. 优化 wide descriptor 构建

只有当 wide descriptor 明显占用时间时才处理。需要先确认：

- 哪些匹配/注册路径确实使用 wide descriptor；
- 是否能按需构建，而不是所有 Mapper 输入都无条件构建；
- 延迟构建是否会把时间移动到交互路径或单张图注册路径。

### C. 优化 SIFT 提取

只有在 SIFT 内核占主导时处理。优先级：

1. 检查每图任务是否已经使用 grant 大小的 Rayon pool；
2. 检查图像解码、格式转换和 descriptor 复制是否包住了真正的 kernel；
3. 再考虑减少复制或后端级优化；
4. 不先降低 `max_features` 或改变检测阈值制造速度差异。

### 尺度空间路径核查与优化前回归基线（2026-09-06）

标准 SIFT 的实际路径是 `vl_sift_process_first_octave/next_octave -> _vl_sift_smooth -> vl_imconvcol_vf`，不是 `vl_imsmooth_f`。`_vl_sift_smooth` 已使用 `VlSiftFilt::temp` 复用图像临时缓冲区，并缓存最近一个 sigma 的 Gaussian filter。`scalespace.c -> vl_imsmooth_f` 每次分配图像缓冲区的现象属于 covdet 路径，不能据此解释或优化当前标准 SIFT 的 Flowers 热点。

arm64 构建禁用 SSE2/AVX，优化前 `vl_imconvcol_vf` 使用标量路径。2026-09-06 实现并拒绝了一个跨独立像素的卷积 SIMD 候选（下述 shifted-support 边界语义不一致），当时移除全部候选生产代码。2026-09-07 经用户批准，仅实现一个修正三段边界语义的 NEON 候选，通过正确性和当前 Flowers24 对照后保留，详见后续报告；不为当前标准 SIFT 新增 workspace API。

已添加 `sift::tests::vlfeat_profiling_preserves_feature_bits`：固定 `193x157` 非对称灰度纹理，对标准、L2/upright、covdet/DSP 三种配置分别比较重复的非计时及计时提取结果。比较数量、所有 keypoint/COLMAP 字段、float descriptor 的 `to_bits()` 和完整 uint8 descriptor。以下为当前 macOS arm64 release 构建实际输出的优化前 BLAKE3（显式小端字段序列化，不含 struct padding）：

| 配置 | 特征数 | BLAKE3 |
|---|---:|---|
| standard | 582 | `6b5675e5f9b52de5094772730a717dcb25440bb1f3ee7131055c77187ea1fe70` |
| standard_l2_upright | 177 | `9f57e514d5ee246fc43ed86610c58f52a3f964fbea290ecd4878302ff8f03651` |
| covdet_dsp | 608 | `9ee1e70d22c5b6770a44f048c051c7bf7c7f53312a665460ba8bf71ed07a42f7` |

该测试现在在 macOS arm64 上直接断言上表的固定数量和哈希，同时保留同一构建内 profiling 等价性和重复确定性。候选前使用 Apple clang 21.0.0 (`clang-2100.1.1.101`)、rustc 1.97.0 (`2d8144b78`) release 重新验证全部哈希；候选不能自更新此基线。不假设其他平台/工具链逐位相同：其他平台仅运行重复/profiling 门，工具链升级导致固定哈希失败时必须独立重新验证，不能用候选输出覆盖。

### Top3 实施与停止报告（2026-09-06）

**保留的改动**

- `src/native/vlfeat_sift.c`：恢复 covdet 成功路径的 `free(patch_xy)`，每次非空 covdet 提取原本漏掉 `2 * 31 * 31 * sizeof(float) = 7,688` 字节。审查确认 patch 分配失败和 `vl_sift_new` 失败分支已有释放；更早的错误尚未分配 patch，输出分配失败发生在 patch 释放之后，patch extraction 的 `continue` 最终经过统一释放。
- `src/feature/sift.rs`：固定优化前 hash/count 门（上表），保留逐字段序列化和重复/计时对照。
- `build.rs`：对解析后的完整 VLFeat 源目录发出 `rerun-if-changed`，覆盖所有使用的 C、头文件和递归模板，以及外部 `VLFEAT_ROOT`。有意允许无关 VLFeat 文件触发保守重编译。实际 `touch third_party/native/vlfeat/float.h` 后 Cargo 报告源目录 Dirty、重新执行 native build，固定哈希仍通过；见 [`output/imconv_header_rebuild_20260906.log`](output/imconv_header_rebuild_20260906.log)。头文件内容未改动。
- `tools/vlfeat_imconv_scalar.c`：从候选前实现冻结的独立标量 oracle，单独编译，不从当前生产内核运行时生成。
- `tools/vlfeat_imconv_test.c`：14,976 个逐位案例，覆盖宽度 1–193、所有四 lane 尾数、高度 1–157、宽于图像和非对称/负向偏移 filter、step 1/2/3、zero/continuity padding、transpose、src/dst stride padding、非对齐读取和输出 guard。数据为固定 PRNG 有限浮点，不声称覆盖所有 NaN/Inf 或任意编译器浮点模式。
- `tools/vlfeat_covdet_alloc_test.c` 和 `tools/test_vlfeat_native.py`：不改生产 API 的 bridge allocation instrumentation。计时开/关分别验证成功、8 个 bridge malloc 失败点及 `vl_sift_new` 失败后 live allocations 为零。临时副本删除此 free 的负向对照准确报 `live=1`。不注入 VLFeat 内部每个分配点。

**唯一 SIMD 候选：拒绝，不继续调参**

候选对 arm64 float 每次处理四个独立列，降序累加 filter taps，使用 `vfmaq_n_f32`，保留 double 和标量尾部；转置结果逐 lane 写回。Apple clang `-O3 -DNDEBUG` 的原始卷积汇编确认为 scalar `fmadd`，没有换用非融合乘加或重排 taps。候选用按行 clamp 实现 continuity padding，但这与原始实现的全部有效 support 行为并不相同：

```text
FAIL w=4 h=1 support=[-7,-2] step=1 transpose=0 pad=1 stride_pad=0
index=0 scalar=00000000 actual=3fc35278
```

原始标量先将 `v=0`，仅在 top/interior chunk 中更新它；上述所有 taps 均在图像后方，故 bottom chunk 仍累加零。候选直接 clamp 到最后一行，结果非零。即使标准 SIFT 的中心 Gaussian 不触发此案例，也不满足本轮通用 `vl_imconvcol_vf` 精确保持门。没有顺便修改上游语义、增加 fallback 或试第二候选。实验 patch 留存于 [`output/imconv_neon_rejected_20260906.patch`](output/imconv_neon_rejected_20260906.patch)，生产 `third_party/native/vlfeat/imopv.c` 与任务开始时完全一致（`git diff --exit-code` 通过）。

依照批准的 first-semantic-failure 停止条件，本轮**未运行 Flowers24 baseline/candidate 三轮对照**，未测 candidate feature wall/scale-space/total/RSS、24/24、186 pairs 或 pair-quality，不提出性能改善结论。既有 Flowers 数据不当成本候选结果。未重试 descriptor 量化/重排融合，未改 max_features、阈值、matching、BA、Taskflow DAG、Viewer。

**验证与复现**

所有下列 Cargo 命令串行执行，前缀均为 `VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`，每条 timeout 180–240 秒；没有超时：

```sh
cargo test -p rustsfm --release --offline --lib sift::tests::vlfeat_profiling_preserves_feature_bits -- --nocapture
cargo build -p rustsfm --release --offline --bin rustsfm
cargo test -p rustsfm --release --offline --lib sift::tests:: -- --nocapture
cargo test -p rustsfm --release --offline --lib -- --skip gpu::
cargo check -p rustsfm --release --offline --no-default-features --all-targets
# Header-only dirty/rebuild proof; full log linked above:
touch third_party/native/vlfeat/float.h
cargo test -p rustsfm --release --offline --lib sift::tests::vlfeat_profiling_preserves_feature_bits -vv -- --nocapture
```

结果：固定哈希 1/1（候选前与移除后/header rebuild 均通过）；focused filter 29/29（filter 同时选中部分 `gpu::sift` 测试）；广域 lib 656 passed / 19 ignored / 59 filtered；无默认特性 all-targets check 与 release binary build 通过。存在原有 Ceres unused-parameter 和 Rust unused/dead-code warnings，没有扩大范围修复。

```sh
python3 rustsfm/tools/test_vlfeat_native.py
python3 rustsfm/tools/test_vlfeat_native.py --sanitize
rustfmt --check --edition 2021 rustsfm/build.rs rustsfm/src/feature/sift.rs
git diff --check
git diff --exit-code -- third_party/native/vlfeat/imopv.c
```

Standalone runner 设置相同三个线程限制，以 `-O3 -DNDEBUG -DVL_DISABLE_AVX -DVL_DISABLE_SSE2 -DVL_DISABLE_OPENMP` 编译原生源，每个子进程 timeout 90 秒，整轮 180 秒。普通 release 与 AddressSanitizer 均通过 14,976 cases、allocation gates 和负向对照；rustfmt（直接运行以避免不适用的 Cargo release/offline fmt 参数）、diff check 通过。

**限制：UBSan 非通过。** 最初同时开启 ASan/UBSan 时，冻结 scalar 的 `dst += 1 * 1 - dheight * dst_stride`（`tools/vlfeat_imconv_scalar.c:84`，对应原始 `imopv.c:208`）被报 unsigned-offset pointer overflow。未修改 upstream/reference 来消警；runner 保留 `--ubsan` 可复现此既有问题。ASan 单独运行通过不代表 UBSan 或所有 VLFeat 内部分配已验证。

### 修正 NEON 候选：通过并保留（2026-09-07）

**范围与实现**

- 本次仅改 `third_party/native/vlfeat/imopv.c`、`tools/vlfeat_imconv_test.c` 和本计划；实验输出使用独立 `output/neon_corrected_*` 名称。未改已有 `free(patch_xy)`、固定 feature hashes、build tracking、冻结 scalar oracle、算法/阈值/max_features/matching/BA/Taskflow/Viewer；没有 commit、branch 或 reset。
- 唯一修正候选每次计算四个独立列，逐像素按原始 descending tap 顺序 `vfmaq_n_f32` 累加。严格复现 top/interior/bottom 三段状态：bottom continuity 使用最后的 `v`，若前两段均未加载，仍使用初始零；不是 clamp 到最后一行。`4x1 [-7,-2]` 的原始失败已通过。不改上游边界行为。
- 只在 `__aarch64__` float 且 `vl_get_simd_enabled()` 时启用；double、其他架构、scalar tails 与原始 scalar loop 不变。`vl_set_simd_enabled(0)` 保留原始路径且已测。transpose、step、src/dst padding 和非对齐输入均经过 bitwise gate。标准 SIFT 仍走 `_vl_sift_smooth/f->temp`，无 workspace API。
- 当前 macOS arm64 / Apple clang 21.0.0 (`clang-2100.1.1.101`) / rustc 1.97.0 (`2d8144b78`)；`VLFEAT_ROOT` 未设置，使用 vendored source。`build.rs` 没有新配置要求，外部 `VLFEAT_ROOT` 仍照原逻辑编译其源，不会自动注入 vendored 优化。未做其他架构/外部源构建实测；不声称非当前浮点编译模式也逐位一致。

**先保存当前 baseline，再改生产代码**

当前工作树 release CLI build 成功后、编辑 `imopv.c` 之前，保存 baseline binary；不是借用前次实验数据。copy tool 无法解析实际存在的 ignored `target/release/rustsfm`，故仅对此使用 `cp -p` fallback，SHA-256 验证一致。随后 baseline 固定 feature hash 测试通过。候选正确性通过后构建并另存 candidate，最后重建 CLI 与 candidate hash 仍一致：

| binary（均在 `output/`） | SHA-256 |
|---|---|
| `neon_corrected_baseline_bin` | `8ebf6bda8e77499e8ba344a33dac0d56f1cd3b6cedacdf1e8c492ca8eb05b96e` |
| `neon_corrected_candidate_bin` | `56c7eb643beddf54056fa05bcbd79134cb2dcf882db5d274e8d2684b6a10bc4c` |

**正确性与构建证据**

先运行原始 14,976 cases：baseline 和唯一修正 candidate 均通过（`neon_corrected_baseline_native.log` / `neon_corrected_candidate_native14976.log`）。然后新增 negative-only supports `[-1,-1],[-2,-1],[-7,-7],[-8,-6],[-16,-16],[-17,-15],[-33,-33],[-34,-32],[-157,-157],[-158,-156],[-159,-158]`，覆盖 no-interior-tap、最后有效行及 bottom transition。最终 **35,568 cases × SIMD off/on 两种模式**，普通及 ASan 均通过，包含输出 padding guards；两种模式重置 PRNG 以使用相同数据。allocation gates（profile on/off、8 个 bridge allocation failures、sift failure）及 removed-free 负向对照仍通过。

baseline/candidate 各自固定哈希测试通过：standard **582** / `6b5675e5f9b52de5094772730a717dcb25440bb1f3ee7131055c77187ea1fe70`；L2 upright **177** / `9f57e514d5ee246fc43ed86610c58f52a3f964fbea290ecd4878302ff8f03651`；covdet DSP **608** / `9ee1e70d22c5b6770a44f048c051c7bf7c7f53312a665460ba8bf71ed07a42f7`。未更新任何预优化哈希。

所有 Cargo 命令串行、`--release --offline`，每条工具 timeout 240 秒；以下命令均从 workspace root 执行，Cargo 前缀统一为 `VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`：

```sh
cargo build -p rustsfm --release --offline --bin rustsfm
cp -p target/release/rustsfm rustsfm/output/neon_corrected_baseline_bin
shasum -a 256 target/release/rustsfm rustsfm/output/neon_corrected_baseline_bin
cargo test -p rustsfm --release --offline --lib sift::tests::vlfeat_profiling_preserves_feature_bits -- --nocapture
# Above precedes production edits; after native correctness, repeat fixed-hash test and build:
cargo test -p rustsfm --release --offline --lib sift::tests::vlfeat_profiling_preserves_feature_bits -- --nocapture
cargo build -p rustsfm --release --offline --bin rustsfm
cp -p target/release/rustsfm rustsfm/output/neon_corrected_candidate_bin
shasum -a 256 rustsfm/output/neon_corrected_baseline_bin rustsfm/output/neon_corrected_candidate_bin
# After six Flowers runs:
cargo test -p rustsfm --release --offline --lib sift::tests:: -- --nocapture
cargo test -p rustsfm --release --offline --lib -- --skip gpu::
cargo check -p rustsfm --release --offline --no-default-features --all-targets
cargo build -p rustsfm --release --offline --bin rustsfm
shasum -a 256 target/release/rustsfm rustsfm/output/neon_corrected_candidate_bin
```

结果：focused **29 passed**；lib **656 passed / 19 ignored / 59 filtered**；no-default all-targets check、final CLI build 通过。保留原有 Ceres/Rust warnings，没有修无关问题。日志分别为 `output/neon_corrected_{baseline_build,baseline_hash,candidate_hash,candidate_build,focused,lib,no_default,final_build}.log`。

```sh
python3 rustsfm/tools/test_vlfeat_native.py
python3 rustsfm/tools/test_vlfeat_native.py --sanitize
python3 rustsfm/tools/test_vlfeat_native.py --ubsan
clang -O3 -DNDEBUG -DVL_DISABLE_AVX -DVL_DISABLE_SSE2 -DVL_DISABLE_OPENMP -I third_party/native/vlfeat -S rustsfm/tools/vlfeat_imconv_scalar.c -o rustsfm/output/neon_corrected_scalar.s
clang -O3 -DNDEBUG -DVL_DISABLE_AVX -DVL_DISABLE_SSE2 -DVL_DISABLE_OPENMP -I third_party/native/vlfeat -S third_party/native/vlfeat/imopv.c -o rustsfm/output/neon_corrected_candidate.s
grep -n -E 'fmadd|fmla' rustsfm/output/neon_corrected_scalar.s rustsfm/output/neon_corrected_candidate.s
rustfmt --check --edition 2021 rustsfm/build.rs rustsfm/src/feature/sift.rs
git diff --check
```

Native runner 每子进程 timeout 90 秒，工具总 timeout 180–240 秒；没有 timeout。最终 native / ASan 证据为 `output/neon_corrected_final_native.log` / `output/neon_corrected_final_asan.log`。汇编验证原始 scalar `fmadd`、四 lane candidate `fmla.4s`、scalar tail `fmadd`，没有跨 tap reduction 或非融合替代。

**明确非通过：UBSan** 命令 exit 1，仍在冻结 scalar `vlfeat_imconv_scalar.c:84:11` 报 `addition of unsigned offset ... overflowed`（`output/neon_corrected_ubsan.log`）。未 mask、未改 oracle/upstream 消警；ASan 单独通过不等于 UBSan 通过。有限 PRNG 测试不穷举 NaN/Inf、极端维度、原始实现本身未定义的 supports 或所有编译器 FP 模式。

**Flowers24：实际交错 B1,C1,B2,C2,B3,C3，各三轮**

每次独立进程，严格串行且测量期间未并行编译/测试；以下完整命令仅将 binary/tag 对应替换为表中六行，无其他参数变化，每条 timeout 240 秒。stdout/stderr 合并保存为相同 tag `.log`；summary 和 reconstruction 使用独立路径，不复用旧输出。`/usr/bin/time -l` 可用，RSS 为 bytes。

```sh
RUSTSFM_PROFILE_FEATURES=1 RUSTSFM_PROFILE_PAIRS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/time -l rustsfm/output/neon_corrected_baseline_bin reconstruct --input artifacts/inputs/flowers/images --output rustsfm/output/neon_corrected_b1 --max-images 24 --max-features 8192 --local-matching --random-seed 1 --summary-json rustsfm/output/neon_corrected_b1.json > rustsfm/output/neon_corrected_b1.log 2>&1
RUSTSFM_PROFILE_FEATURES=1 RUSTSFM_PROFILE_PAIRS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/time -l rustsfm/output/neon_corrected_candidate_bin reconstruct --input artifacts/inputs/flowers/images --output rustsfm/output/neon_corrected_c1 --max-images 24 --max-features 8192 --local-matching --random-seed 1 --summary-json rustsfm/output/neon_corrected_c1.json > rustsfm/output/neon_corrected_c1.log 2>&1
RUSTSFM_PROFILE_FEATURES=1 RUSTSFM_PROFILE_PAIRS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/time -l rustsfm/output/neon_corrected_baseline_bin reconstruct --input artifacts/inputs/flowers/images --output rustsfm/output/neon_corrected_b2 --max-images 24 --max-features 8192 --local-matching --random-seed 1 --summary-json rustsfm/output/neon_corrected_b2.json > rustsfm/output/neon_corrected_b2.log 2>&1
RUSTSFM_PROFILE_FEATURES=1 RUSTSFM_PROFILE_PAIRS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/time -l rustsfm/output/neon_corrected_candidate_bin reconstruct --input artifacts/inputs/flowers/images --output rustsfm/output/neon_corrected_c2 --max-images 24 --max-features 8192 --local-matching --random-seed 1 --summary-json rustsfm/output/neon_corrected_c2.json > rustsfm/output/neon_corrected_c2.log 2>&1
RUSTSFM_PROFILE_FEATURES=1 RUSTSFM_PROFILE_PAIRS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/time -l rustsfm/output/neon_corrected_baseline_bin reconstruct --input artifacts/inputs/flowers/images --output rustsfm/output/neon_corrected_b3 --max-images 24 --max-features 8192 --local-matching --random-seed 1 --summary-json rustsfm/output/neon_corrected_b3.json > rustsfm/output/neon_corrected_b3.log 2>&1
RUSTSFM_PROFILE_FEATURES=1 RUSTSFM_PROFILE_PAIRS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/time -l rustsfm/output/neon_corrected_candidate_bin reconstruct --input artifacts/inputs/flowers/images --output rustsfm/output/neon_corrected_c3 --max-images 24 --max-features 8192 --local-matching --random-seed 1 --summary-json rustsfm/output/neon_corrected_c3.json > rustsfm/output/neon_corrected_c3.log 2>&1
```

| Run | Scale-space sum (s) | Feature wall (s) | Summary total (s) | time real (s) | Peak RSS (bytes) | Registered / pairs | Points |
|---|---:|---:|---:|---:|---:|---|---:|
| B1 | 61.213550 | 17.617419 | 33.114613 | 34.47 | 3951591424 | 24/24 / 186 | 3455 |
| C1 | 19.103076 | 7.172077 | 23.190095 | 23.78 | 3924557824 | 24/24 / 186 | 5047 |
| B2 | 61.215890 | 17.648984 | 33.922314 | 33.95 | 3932553216 | 24/24 / 186 | 3674 |
| C2 | 19.023216 | 7.210774 | 23.397394 | 23.42 | 3933962240 | 24/24 / 186 | 3187 |
| B3 | 61.809207 | 17.951190 | 34.520989 | 34.55 | 3928752128 | 24/24 / 186 | 3288 |
| C3 | 19.047049 | 7.249219 | 23.643500 | 23.67 | 3932258304 | 24/24 / 186 | 3465 |

Median：scale-space sum **61.215890 → 19.047049 s (-68.89%)**；feature wall **17.648984 → 7.210774 s (-59.14%)**；summary total **33.922314 → 23.397394 s (-31.03%)**；process real **34.47 → 23.67 s (-31.33%)**；RSS **3932553216 → 3932258304 bytes (-0.0075%)**。Scale-space 是 24 张图各自 backend elapsed 的和，**不是 wall time**。feature wall 来自 `feature_profile.wall_clock_span_ms`，total 来自顶层 `elapsed_ms`。三组逐对均有明显改善，候选与 baseline 的时间范围完全不重叠，改善远大于本次观察到的波动；不声称三次运行构成跨机器统计结论。

质量：六轮均为 **24 images / 24 registered / 186 retained pairs / 1 model**，189 attempted pairs。逐 pair 比较顶层全部非 timing 字段（移除 `*_ms` 和 `sift` 子对象；另断言所有 `sift.uint8_backend` 均为 true）完全一致：image IDs/names、每侧 descriptor count、matches、inliers、acceptance。每轮 **118006 features/descriptors、90256 matches、82656 inliers**（后二者为全部 attempted pairs 之和）。规范化 JSON（`json.dumps(q, sort_keys=True)`）SHA-256 均为 `5d24f237e062bb9c2a656f2d1e567903d0c13806cbef2446ef13570c60fd38d8`。汇总见 [`output/neon_corrected_metrics.json`](output/neon_corrected_metrics.json)，原始六份 `.json`/`.log` 可独立复算。

**实际验收：保留唯一修正候选。** 所要求的 bitwise fixture/native、registration、pairs、稳定 pair-quality、重复性能和 RSS 门通过。Flowers 没有导出每个完整 descriptor 的 bitwise hash；完整字段/descriptor 逐位证据来自固定 SIFT regression fixture，Flowers 证据为 counts 和全部 pair-quality 字段。最终 points 在 baseline 也有波动，candidate 范围更宽；仅凭三轮不能归因或声称最终 BA/model 逐位相同，没有为此修改 downstream，也不宣称最终几何精度提升。未做真实 DB/Viewer/GPU 性能实验。

## 第三阶段：真实数据库路径对照

当前 24 帧主要测量 `--local-matching` image-only 路径。完成特征准备优化后，需要单独测量数据库复用路径，避免把提取和匹配重复计算混入比较。

至少建立两种场景：

1. image-only：从图片提取特征并构建配对；
2. database-first：已有数据库特征、matches 和 two-view geometries，Mapper 只加载和使用已有结果。

比较：

- 数据库打开和 cache 加载；
- 特征准备；
- 配对几何处理；
- 增量重建；
- 总耗时和峰值 RSS。

两种路径不能直接用总耗时互相宣称优劣，因为工作内容不同。

### Flowers2 database-first 固定子集基线（2026-09-07）

**状态：真实数据库复用基线已验证（48/960 固定子集），不是全量验收。** 报告与限制见 [`output/dbfirst_flowers2_20260907/REPORT.md`](output/dbfirst_flowers2_20260907/REPORT.md)，可复算数据见 [`metrics.json`](output/dbfirst_flowers2_20260907/metrics.json)。本轮无生产代码修改。

- 输入严格使用 `/Users/tfjiang/Projects/RustScan/artifacts/inputs/flowers2/images`：960 张 1536×2048 JPEG，共 468,689,310 bytes。运行前固定选择排序前 48 张（`frame_0001.jpg`–`frame_0048.jpg`），不复用 flowers24/186 pairs 预期。为维持每 workflow 240 s 上限，未执行全量；48 张提取实测 15.08 s，线性规划估算全量提取约 302 s（非实测）。
- 专用 `mapper` CLI 无图像数量限制，因此使用同一 `run_reconstruction` 实现的 `reconstruct --database ... --max-images 48`；没有 `--local-matching`。输入目录不替换、不拷贝到其他输入位置。8192 features、默认 matching/BA 阈值不变，matching/reconstruction seed=1；三项 BLAS/OpenMP 环境变量均为 1。
- 仅在独立输出目录新建 metadata DB，CLI 提取和 sequential matching/verification 各一次：259,829 features、461 matches rows / 259,206 correspondences、461 two-view rows / 237,836 inliers，全部有 stored poses。准备 process real 分别 **15.14 / 39.60 s**，不计入复用重建。
- 已关闭且无 WAL sidecar 的 populated DB 独立复制三份，按相同参数**串行**运行，输出模型各自独立。三轮均 **48/48 registered、427 retained pairs、1 model、17,144 points**；DB whole-file SHA-256、全部表 schema/content hashes、counts 和完整性检查前后完全一致。代码分支与启用 profiling 的日志共同证明无重新提取/descriptor matching；Mapper 使用已有 verified correspondences 和 stored geometries，不读取 descriptors 做匹配。
- 三轮 median：DB/cache/frame preparation **14.17 ms**（原日志名 `timing_extract_ms`，不是提取）、pair processing **119.35 ms**、incremental **9665.16 ms**、summary total **9974.722500 ms**、process real **9.98 s**、RSS **325,582,848 bytes**。数据库 open/cache 无独立计时；不声明 cold-cache。stage report 仅有 reconstruction，4/4 threads、无失败；pending/resource/GPU cleanup 归零计数不暴露，不能伪称通过。
- 一次相同 48 图/参数 image-only 对照：prep/pairs/incremental **14.44375 / 39.73896 / 11.30360 s**，process real **65.68 s**，RSS **4,130,078,720 bytes**，48/48、428 pairs、16,090 points。与 DB reuse 的总工作量及 pair/model 输出不同，不计算直接 speedup 或宣称几何等价。
- 下一热点有依据地转向 **incremental（约总耗时 97%）/global BA**：三轮 logged global BA elapsed sum median **5328.63 ms**。每轮 14 条 global BA 均为 DenseSchur / MaxIterations，无 BA fallback 日志；这不是 global BA 已收敛或历史 fallback 问题已修复。最终模型 hash 不同，点数及 pair-quality summary 稳定，不宣称 BA 逐位确定性。
- **限制：stored-pose 被拒后代码允许 pose re-estimation，但无运行时计数；本轮没有证明“零几何重估”。** 若要求这一严格门，需另行批准最小 counter/trace 或严格模式；不为验证任务修改生产代码。全量 960、独立 cache 计时、资源归零和外部几何精度对照仍未验证。本轮到此停止，不扩展 SIMD/BA 优化。

### Flowers2 incremental / Global BA 只读归因（2026-09-07）

**状态：已有三轮证据足以定位下一主热点，本轮停止；无生产代码/BA 策略修改，无新重建 workload。** 详细报告：[`output/ba_diagnosis_flowers2_20260907/REPORT.md`](output/ba_diagnosis_flowers2_20260907/REPORT.md)；逐调用表：[`BA_CALLS.md`](output/ba_diagnosis_flowers2_20260907/BA_CALLS.md)；可复算 [`metrics.json`](output/ba_diagnosis_flowers2_20260907/metrics.json) / [`analyze.py`](output/ba_diagnosis_flowers2_20260907/analyze.py)。旧基线目录不修改。

- 同一 first48/960、8192 features、seed1、三份独立 frozen DB 的原始 JSON debug logs：global BA 14 calls/run = **1 initial + 8 scheduled invocations（其中 5 次两轮）**；触发注册帧数 **3,4,6,9,14,21,32,48**。7 次可由 frame ratio=1.5 解释；3→4 的 point-growth 具体 operands 未记录，不能伪称有精确 predicate counter。
- global adapter elapsed median **5328.63 ms** 中 setup **316.51 ms**、`problem.solve` wrapper **5000.26 ms**、postprocess **11.08 ms**；solve 占 incremental **51.75%**、global elapsed **93.93%**。wrapper 包含 native solve、参数拷贝/问题销毁，不等同 DenseSchur 线性分解耗时。32/48 帧四个 calls 共 **3413.37 ms（63.92% global）**，5 个 second rounds 共 **2430.55 ms（45.61%）**；两组重叠，不能相加。
- 已有 scheduled cap=15 iterations、最多2轮保持不变。所有 **42/42 global** 及 **138/138 可见 local last-round** objective 均下降；global MaxIterations 不等于 divergence/fallback。`iterations=a/b` 是 successful/attempted steps，不是当前/上限。二轮触发源是 changed-observation ratio>0.0005，不是 NoConvergence。最后48帧 scheduled mark 使独立 final BA check 跳过，因此不能假定后面还有 full-budget quality closure，也不建议据低 marginal cost drop 直接跳过二轮。
- local 46 lines/run 实际覆盖 **68 successful refinements**，22 个前轮报告被覆盖；**779.84 ms** 仅为最后一轮 BA elapsed sum。incremental 减去可见 global/local BA 剩余 **3556.69 ms** 不是纯 overhead。PnP solve/refine **243.80 ms**；registration triangulation **205.08 ms** 与 complete/merge timers 重叠，delete 与 merge/filter 重叠，不作重复计时加总。颜色、初始化、snapshot 等独立耗时缺失。
- 质量仍为 **48/48、427 pairs、17144 points**；exported mean point error 三轮约 **0.813808 px**。新发现两处原有 local observations/residuals 小差异（frame0043、frame0010），所以不能宣称所有 local solver inputs 或模型逐位相同。global size/steps 一致、无 global skipped/post-bogus 行，不代表历史 post_bogus 质量问题已解决。
- **下一唯一具体干预建议（未实施）**：仅在现有 Ceres adapter 中 default-off 保存已暴露的 `SolverSummary::full_report/message`，重点分解32/48帧 solve 的 residual/Jacobian evaluation 与 linear solve；复用现有 binding，不扩 public API，不改 solver inputs/options/schedule。拿到该证据再选实现优化。本轮已有证据足以完成主路径归因，不扩展成全量 instrumentation 或新 full960 benchmark。
- 原始 binary 和11项 source/config provenance hashes 匹配；新任务开始/结束比较原始 artifacts、source/images hashes 与 staged/unstaged production diff fingerprints，并只读 immutable 检查四份 DB integrity/schema/hash 不变。无 Cargo/build/新 off-on parity（本轮无生产改动）；若后续实施 profiling，须按原约束补 focused tests、off/on 输入不变证明及至少三份 fresh frozen DB 串行重建。

## 第四阶段：真实多 workflow 排队分析

只有单工作流分段稳定后，才测 Taskflow 的排队行为。目标是回答“多个项目同时运行时，谁在等待什么”，而不是再次证明单工作流速度。

### 测量内容

对 2、4、8 个真实小项目并发运行，记录每个 workflow：

- 阶段提交时间；
- 阶段获得 grant 的时间；
- 阶段实际开始时间；
- 阶段完成时间；
- CPU grant、memory grant、GPU exclusive 状态；
- `queue_ms`；
- 阶段服务时间；
- 整体完成时间；
- 小任务 P50/P95/P99；
- runtime 的 pending_tasks、CPU 和 memory 快照。

建议先选 6～12 帧的小项目，避免单个真实重建过长导致实验成本过高。每个 workflow 使用相同输入规模和相同配置，再增加一组大小不均的 workload，观察大任务是否饿死小任务。

### 调度观测（已完成）

高层 `SfmTaskflow::run/sequence` 已通过 `SfmStageReport` 暴露：

```text
stage_name
requested_threads
granted_threads
requested_memory
granted_memory
queue_ms
dependency_wait_ms
resource_wait_ms
service_ms
total_ms
cancelled_or_failed
```

runtime `TaskReport`/`Event::Started` 同样提供 `queue_time`、`dependency_wait_time` 和 `resource_wait_time`。这些字段不改变同步借用 callback 的线程归属，也不在 Taskflow worker 内同步提交同一个 Runtime。没有实际 grant 的 pre-admission 失败/取消不伪造 resource wait，timing 字段保持为 0。

### 排队分析的验收标准

- 能区分排队时间和真正计算时间；
- 能区分 Taskflow admission queue 和 OS run queue；
- 能看到阶段持有 CPU 额度但内部实际低利用率的情况；
- 能发现 GPU 独占键造成的等待；
- 能发现大阶段阻塞短任务的 P95/P99 影响；
- 不在没有数据时修改优先级或拆分 DAG。

## 第五阶段：GPU 与 Viewer 专项

这部分不和 CPU 搜索优化混测。

### GPU

单独测量：

- GPU 阶段 admission queue；
- CPU 准备时间；
- GPU submit 时间；
- readback/wait 时间；
- GPU stage cleanup 时间；
- CPU/GPU 同时存在时的互相等待。

当前 GPU 调度仍是保守的 `rustsfm/default-gpu` exclusive key，不是多 GPU placement、VRAM 预算或自动 CPU/GPU 路由。报告必须保持这个边界。

### Viewer

以用户可感知的交互指标为目标，而不是只看后台吞吐：

- pause/cancel 到安全边界的响应时间；
- 阶段排队期间的响应时间；
- 单帧注册 P50/P95；
- final Global BA 对后续帧或 UI 的阻塞时间；
- 事件发送和渲染线程是否出现积压。

当前取消是协作式的，Ceres/GPU 原生工作不能保证立即中断；测试结果必须明确这一点。

## 独立质量问题：Global BA 回退

`post_bogus_cameras` 不属于当前性能优化的直接结果。2026-09-16 独立调查（`output/post_bogus_repro_20260916/REPORT.md`）：

- 历史触发面是 **image-only `--local-matching` 每图一个 PINHOLE**（flowers 24，`neon_corrected_*.json`），不是 flowers2 960 的共享相机 DB 路径。960 recon 无 post_bogus 行。
- Mapper 在 Ceres 成功后若任一注册相机 `HasBogusParams`，会回滚 **整次** BA（pose+point+camera）。COLMAP IncrementalMapper 的 global BA 不会这样整包丢弃。
- PINHOLE 不精化 extra / principal（默认），预期失败模式是 **focal ratio** 离开 `[0.1, 10]`。skip 日志现带 `audit=[idx:model fx=… reasons=focal(…)]`。
- `reconstruct --ImageReader.single_camera` 已接线（默认 0，保持历史每图相机）。当前二进制 24 帧 image-only 复现须等 GPU matching 空闲，再与 `--ImageReader.single_camera 1` 对照。
- **已改提交语义（待 24 帧确认）：** post-BA bogus 只把那些 camera 恢复到 BA 前快照，**保留 pose/point**；成功行带 `camera_reset=[…]`。pre-BA bogus 仍整次跳过。unit test：`restore_bogus_cameras_keeps_other_ba_state`。

性能实验可以暂时以 pair metadata、pair-quality、注册率和资源清理作为回归门；质量修复应有自己的数值验收。

## 每轮实验的固定命令模板

构建：

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  cargo build -p rustsfm --release --bin rustsfm --offline
```

真实 24 帧画像：

```sh
RUSTSFM_PROFILE_PAIRS=1 \
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  target/release/rustsfm reconstruct \
  --input artifacts/inputs/flowers/images \
  --output rustsfm/output/<unique-run-dir> \
  --max-images 24 \
  --max-features 8192 \
  --local-matching \
  --summary-json rustsfm/output/<unique-summary>.json
```

回归：

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  cargo test -p rustsfm --release --lib --offline -- --skip gpu::

VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  cargo check -p rustsfm --release --no-default-features --all-targets --offline
```

## 停止条件

满足以下条件后停止继续扩大本轮范围：

- 已确认一个主要耗时来源；
- 只改一个实现点；
- 固定输入基准和真实流水线都测过；
- 结果语义回归通过；
- 没有新的未解释质量退化；
- 资源额度、pending tasks 和 GPU cleanup 归零；
- 结论和限制已写入对应报告。

不要在同一轮同时做以下事情：降低特征数量、改变匹配阈值、切换 ANN、拆 DAG、修改 BA 策略和调整 Viewer 事件节流。那样无法判断收益来自哪里，也无法证明重建质量没有被牺牲。

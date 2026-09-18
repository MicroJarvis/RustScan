# RustGS 训练效果与效率优化 TODO

**制定日期：** 2026-09-17  
**范围：** RustGS wgpu 训练、拓扑管理、渲染反向传播、训练数据和评估闭环  
**当前基线：** `RustGS/experiments/brush_vs_rustgs_tum_freiburg1_xyz.md` 中的 stabilized 30k TUM 运行；平均 PSNR 约 `21.10 dB`，Brush 对照约 `21.29 dB`。  
**当前测试基线：** `cargo test -p rustgs --lib --no-default-features --no-fail-fast`，16 passed；GPU integration benchmark 尚未在本轮复测。

## 使用规则

- 每个任务完成前必须保留基线命令、候选命令、训练时间、最终 Gaussian 数、平均/最差 PSNR 和显存峰值。
- 性能优化必须同时通过质量 gate；不能以平均 PSNR 上升掩盖最差帧、动态区域或边缘质量下降。
- 所有 GPU 优化先在小型确定性 fixture 上验证，再在 TUM 全视图和至少一个外部场景上复测。
- 文档中的“待验证”表示代码证据支持的假设，不表示已经测得收益。

## P0：先修训练流水线的同步瓶颈

### [ ] P0-01 去除每步前向的 GPU/CPU 计数同步

**现状：** `render_forward_with_active_sh` 在排序和 tile mapping 前读取 `num_visible`、`num_intersections`，每个 iteration 都发生一次 GPU 到 CPU 的同步。[`RustGS/src/training/forward/mod.rs:224`](/Users/tfjiang/Projects/RustScan/RustGS/src/training/forward/mod.rs:224)

**方案：**

- 第一阶段使用固定容量的 visible/intersection buffer，并在 GPU 内做 compaction。
- 第二阶段使用 GPU counter + indirect dispatch，避免 CPU 参与动态尺寸决策。
- 超过容量时只触发一次可观测的容量扩展，不在正常路径中读回计数。

**验收：**

- 训练 loop 中不再对 visible/intersection count 调用 `into_data_async`。
- 500/3000 iteration 的 steps/s 提升，且 GPU 时间线上不再出现每步固定 host gap。
- 输出图、Gaussian 数和 PSNR 与基线误差在 `0.05 dB` 内。

### [ ] P0-02 延迟 loss scalar 读回

**现状：** `train_step` 在 backward 前读取 loss scalar，阻塞 GPU 队列。[`RustGS/src/training/engine/trainer.rs:536`](/Users/tfjiang/Projects/RustScan/RustGS/src/training/engine/trainer.rs:536)

**方案：**

- backward 直接消费 device loss。
- 仅在每 `N` 步、checkpoint、训练结束或 GPU-side finite flag 触发时读取 scalar。
- 保留非有限 loss 的终止语义，不能为了吞吐取消数值安全检查。

**验收：**

- 普通 iteration 不发生 loss scalar readback。
- 非有限 loss 仍能在下一个检查点终止并产生 `TrainingRunFailed`。
- 同一 seed 下最终 loss/PSNR 与基线一致。

## P1：修拓扑质量和状态连续性

### [ ] P1-01 修复默认 Weight pruning 的真实可见性语义

**现状：** 默认 `Weight` 模式下，trainer 不使用实际 visibility，shader 对每个 splat 累加 `1.0`，使历史不可见 pruning 基本不可达。[`RustGS/src/training/engine/trainer.rs:616`](/Users/tfjiang/Projects/RustScan/RustGS/src/training/engine/trainer.rs:616)、[`RustGS/src/training/shaders/accumulate_topology_stats.wgsl:79`](/Users/tfjiang/Projects/RustScan/RustGS/src/training/shaders/accumulate_topology_stats.wgsl:79)

**方案：** 明确三种 prune mode 的定义；`Weight` 至少使用真实 visibility 或明确改成 opacity/contribution weight 语义。补充不可见 splat 的端到端 topology 测试。

**验收：**

- 人工构造长期不可见、低 opacity Gaussian，默认配置能够按规则进入候选。
- pruning 不删除持续有贡献的 splat。
- TUM 动态帧最差 PSNR 不下降超过 `0.2 dB`，Gaussian 数和 raster intersection 数下降或保持稳定。

### [ ] P1-02 保留 surviving Gaussian 的 Adam moments

**现状：** topology mutation 或 opacity reset 后调用 `optimizer.reset()`，清空所有 moment 和 step。[`RustGS/src/training/engine/trainer.rs:1045`](/Users/tfjiang/Projects/RustScan/RustGS/src/training/engine/trainer.rs:1045)、[`RustGS/src/training/engine/optimizer.rs:336`](/Users/tfjiang/Projects/RustScan/RustGS/src/training/engine/optimizer.rs:336)

**方案：** 根据 `TopologyMutationPlan::origins()` remap surviving rows 的一阶/二阶矩；新 Gaussian 只初始化新增行；删除行丢弃状态。仅在 tensor layout 或 SH storage 发生实质变化时整体 rebuild。

**验收：**

- checkpoint round-trip 后，mutation 前后 surviving row 的 optimizer state 可追踪。
- topology step 后学习率 bias correction 不重新从 step 0 开始。
- TUM 10k/30k 运行的动态帧和边缘清晰度不劣于当前 stabilized baseline。

### [ ] P1-03 明确无候选 topology step 的统计窗口策略

**现状：** 即使没有候选，scheduled topology step 仍会清空全部 accumulator；`SkipNoEligibleCandidates` 的 disposition 没有被 trainer 消费。[`RustGS/src/training/engine/trainer.rs:1048`](/Users/tfjiang/Projects/RustScan/RustGS/src/training/engine/trainer.rs:1048)

**方案：** 选择并记录一种策略：无候选时保留统计并延长窗口，或明确开始新窗口；不能由实现细节隐式决定。

**验收：**

- 添加 schedule/topology 单测，覆盖“无候选、只有 reset、有候选 mutation”三种情况。
- telemetry 能区分 scheduled step、actual mutation 和 accumulator reset。

### [ ] P1-04 将 topology snapshot/apply 从全量 CPU 往返改为增量路径

**现状：** topology step 会读回全部 splat 和 9～10 组 N 长度统计，再 CPU 重建并完整上传。[`RustGS/src/training/topology/bridge.rs:39`](/Users/tfjiang/Projects/RustScan/RustGS/src/training/topology/bridge.rs:39)、[`RustGS/src/training/topology/apply.rs:20`](/Users/tfjiang/Projects/RustScan/RustGS/src/training/topology/apply.rs:20)

**方案：**

- 短期：只读回 candidate mask/index，host mirror 保留 splat 数据。
- 中期：GPU 上完成 score、select、compact 和 state remap。
- topology telemetry 单独采集 snapshot、plan、upload 三段耗时。

**验收：**

- topology step 的 GPU/CPU 往返字节数按 splat 数量增长，而不是完整状态数量增长。
- 30k 训练中 topology pause 时间下降至少 30%。
- topology 前后场景输出与当前实现一致。

## P1：替换高 dispatch 成本的 GPU primitive

### [ ] P1-05 用真正的 GPU radix sort 替换 bitonic sort

**现状：** `radix_sort.rs` 实际使用 bitonic compare-exchange，按 `(k,j)` 阶段多次全数组 dispatch，并分配五个大 buffer。[`RustGS/src/training/gpu_primitives/radix_sort.rs:99`](/Users/tfjiang/Projects/RustScan/RustGS/src/training/gpu_primitives/radix_sort.rs:99)、[`RustGS/src/training/gpu_primitives/radix_sort.rs:172`](/Users/tfjiang/Projects/RustScan/RustGS/src/training/gpu_primitives/radix_sort.rs:172)

**方案：** 实现 block radix pass + histogram/prefix + scatter；depth sort 和 tile sort 共享 workspace，避免每次重新分配。

**验收：**

- 排序结果与 CPU reference 对每个 key/value 配对一致。
- 记录 dispatch 数、临时显存、sort 时间；目标是 dispatch 数显著低于 `O(log^2 N)`。
- 训练 PSNR、可见 splat 数和 tile intersection 数与基线一致。

### [ ] P1-06 将全局 prefix sum 改为 hierarchical scan

**现状：** 当前 prefix sum 对每个 offset 做一次全数组 kernel，使用两个 N-sized scratch buffer。[`RustGS/src/training/gpu_primitives/prefix_sum.rs:74`](/Users/tfjiang/Projects/RustScan/RustGS/src/training/gpu_primitives/prefix_sum.rs:74)、[`RustGS/src/training/gpu_primitives/prefix_sum.rs:110`](/Users/tfjiang/Projects/RustScan/RustGS/src/training/gpu_primitives/prefix_sum.rs:110)

**方案：** block-local scan、block sums scan、uniform add；复用 workspace 并支持空输入/非二次幂长度。

**验收：**

- 对随机长度、全零、最大值和非二次幂输入与 CPU reference 一致。
- tile mapping 的 scan 时间和 dispatch 数显著下降。

## P2：降低数据和 loss 的内存/调度成本

### [ ] P2-01 精简 frame cache，改为按字节预算

当前 `DecodedFrame` 同时保存 `color_u8`、resize 后 `target_rgb` 和 depth，而训练主循环只消费 target tensor。[`RustGS/src/training/data/frame_loader.rs:17`](/Users/tfjiang/Projects/RustScan/RustGS/src/training/data/frame_loader.rs:17)

- 区分训练路径和 depth/init 路径，避免缓存无用 representation。
- 用总字节数而不是 frame 数限制 host/device cache。
- 评估 pinned staging 或直接使用预处理的低分辨率 target。

### [ ] P2-02 优化 resize 和 target upload

当前 resize 是逐目标像素遍历源区域，并先生成完整 source `f32` buffer。[`RustGS/src/training/data/frame_targets.rs:12`](/Users/tfjiang/Projects/RustScan/RustGS/src/training/data/frame_targets.rs:12)

- 使用 image crate 的 SIMD resize 或一次性预处理缓存。
- 对固定 render scale 的数据集生成离线 target pack。
- 统计 decode、resize、upload 各自耗时和 cache hit rate。

### [ ] P2-03 融合 SSIM/gradient loss

默认 loss 包含多次 grouped convolution、slice、subtract 和中间 tensor。[`RustGS/src/training/engine/loss.rs:38`](/Users/tfjiang/Projects/RustScan/RustGS/src/training/engine/loss.rs:38)

- 先用 profiler 确认 SSIM、gradient、L1 的实际占比。
- 将 SSIM window 或 gradient residual 融合到单个 WGSL kernel。
- 训练路径保留可切换的 reference loss，确保数值 parity。

### [ ] P2-04 优化 rasterizer 的 tile 工作分配

当前固定 16x16 tile，每个 pixel 遍历 tile 内所有 intersection，并在 batch 间同步。[`RustGS/src/training/shaders/rasterize.wgsl:56`](/Users/tfjiang/Projects/RustScan/RustGS/src/training/shaders/rasterize.wgsl:56)

- 统计 tile intersection histogram、early termination 比例和空 tile 比例。
- 评估自适应 tile size、空 tile indirect dispatch 和 tile-level culling。
- 保持前向/反向 tile ordering 一致，避免先优化 forward 后破坏 VJP。

## P2：补齐效果质量能力

### [ ] P2-05 改进 sparse point 初始化覆盖

当前初始化取 `initial_points` 的前 N 个点，没有空间采样或 visibility 分层。[`RustGS/src/training/data/init_map.rs:17`](/Users/tfjiang/Projects/RustScan/RustGS/src/training/data/init_map.rs:17)

- 实现 voxel/grid sampling。
- 保留长 track、高可见性和场景边界点的代表性。
- 对相同点数比较空间覆盖、初始投影 coverage 和早期 PSNR。

### [ ] P2-06 支持背景和曝光策略

当前训练背景固定为黑色。[`RustGS/src/training/engine/trainer.rs:491`](/Users/tfjiang/Projects/RustScan/RustGS/src/training/engine/trainer.rs:491)

- 支持 black/white/random/dataset-estimated background。
- 保证训练和评估使用同一背景定义。
- 在白墙、室外和动态遮挡场景上比较边缘污染和最差帧。

### [ ] P2-07 增加手写反向 shader 的 finite-difference parity

覆盖 mean、scale、quaternion、opacity、SH、perspective projection、covariance inverse 和 clipping boundary；重点测试非中心 principal point、`fx != fy`、near plane、小 covariance、低 alpha。

**验收：** analytic gradient 与 centered finite difference 在合理容差内一致，所有退化输入保持 finite；失败时阻止 GPU backend 质量 gate。

## 实施记录

### 2026-09-17 代码落地，质量 gate 未跑

P0-01 / P0-02 / P1-01 的代码已写入，**验收框保持未勾选**。

- P0-01：训练路径 `CountPolicy::Bounded` 不再 `into_data_async` 读 visible/intersection count；eval/viewport 仍走 `CountPolicy::Exact`。计数留在 device，经 `write_dispatch` + `client.flush()` 写 indirect dispatch。排序用固定 8-pass 4-bit device radix，避免按 hard bound 跑 bitonic。容量上限 `8_388_608`，overflow 只在 loss 采样步读回并尝试扩容。
- P0-02：`train_step` 先 `loss.backward()`，host scalar 仅在 iteration 1、每 20 步、日志步、checkpoint、pause 或最后一步读取。非有限 loss 在 device 上累加，下一次采样读回后以 `TrainingError::TrainingFailed` 终止（runtime 转为 `TrainingRunFailed`）。
- P1-01：默认 `Weight` 改为累加 rasterizer visibility bit，并在 densify 步也允许 history-invisible 进入候选。持续可见且 opacity 高于阈值的 splat 不因该规则被删。这是默认 prune 语义变更，未做 TUM / Home / 外部场景复测，不能当作已接受的默认配置。
- P1-02：topology / opacity reset 不再 `optimizer.reset()`。surviving row 按 `origins()` gather moment，新行置零，Adam step 保持。只有 moment 内层 layout 对不上时才清零该参数组。
- P1-03：无候选且无 opacity reset 时保留 accumulator，并记 `skipped_no_eligible_candidates`。有 mutation 或 reset 时清空并记 `accumulator_resets`。`scheduled_steps` 两者都计数。
- P1-05：已知长度排序不再走 bitonic。depth/tile sort 共用 8-pass radix，ping-pong workspace 按容量复用。`radix_sort.wgsl` 已删除。dispatch 数在 4k–8M 上低于 bitonic（单测）。GPU fixture 与稳定 unsigned CPU 排序一致，含非二次幂、重复 key、有符号 bit、以及 count < capacity 的尾填充。真实训练的 sort 时间、PSNR、可见 splat 数未测。
- P1-06：prefix sum 改为 block-local scan + block-sum scan + uniform add。空输入和长度 1 不 dispatch。非二次幂、全零、`i32::MAX` 回绕与 CPU 一致（GPU 单测）。1e6 长度约 4 次 dispatch，低于旧 Hillis-Steele。tile mapping scan 时间未测。

### 2026-09-18 按详细实施计划收紧 Task 1 验收接口

对照 `docs/superpowers/plans/2026-09-17-rustgs-training-quality-efficiency.md`：

- 新增 `TrainingError::ForwardCapacityExceeded { logical_intersections, capacity }`；overflow 采样后硬失败，不再静默扩容。
- `write_dispatch` 额外写出 `requested_intersections`；遥测 `ForwardCapacityTelemetry` 进入 training telemetry。
- `should_read_loss(iteration, total, cadence, checkpoint_due, paused)`；`TrainingIterationMetrics` 增加 `loop_duration` / `loss_readback`。
- Task 2 fixture 扩展到长度 `0,1,17,255,256,257,4093`；Task 3 Adam remap 覆盖 `origins=[Some(1),None,Some(0)]` 且 `step=7`。
- 本机无标准 TUM COLMAP 路径，500-step baseline/candidate 未跑。验收框仍不勾。

### 2026-09-18 Home COLMAP 500-step candidate（替代缺失的 TUM pack）

本机无 `output/rustgs_benchmark_pack/tum_freiburg1_xyz_colmap`，改用已有 Home 稀疏模型做 Task 1 短跑 filter。

- 数据集：`output/profile_home/colmap60/0` + `test_data/home/images`
- 命令：`target/release/rustgs train --input output/profile_home/colmap60/0 --image-root test_data/home/images --output output/rustgs-optimization/2026-09-18/home-500-candidate/home-500.ply --iterations 500 --max-frames 12 --render-scale 0.25 --frame-shuffle-seed 0 --eval-after-train --eval-render-scale 0.25 --eval-max-frames 12 --eval-frame-stride 1 --eval-device gpu --eval-json --eval-worst-frames 10`
- wall-clock（`time -l` real）：17.11s；CLI training elapsed：15.04s；eval：0.66s
- 初始 / 最终 Gaussian：26518 / 26649
- final loss：0.145818
- PSNR mean / min / max：18.9807 / 18.2698 / 19.6772 dB（12 帧，scale 0.25）
- peak RSS：约 1.49 GB；parity gate：Passed
- 无 `ForwardCapacityExceeded` / 非有限 loss；loss 日志仅在 100 步倍数出现（与延迟读回一致）
- 说明：这是当前 candidate 单跑，不是与旧 binary 的 A/B。历史 Home 1500 smoke（同输入、同 12 帧、scale 0.25）mean PSNR 约 20.44 dB，不可与 500-step 直接比质量，只能作规模参考。

未记录：baseline/candidate 命令、训练时间、Gaussian 数、mean/worst PSNR、峰值显存。因此这些项都还不是完成状态。

## 实施顺序

1. P0-01、P0-02：先消除每步同步，建立真实 steps/s 基线。
2. P1-01、P1-02、P1-03：修 topology 语义和 optimizer continuity，避免性能优化建立在错误训练状态上。
3. P1-05、P1-06：替换 sort/scan，并复用 workspace。
4. P1-04：将 topology snapshot/apply 改为增量或 GPU-native。
5. P2-01～P2-04：处理数据、loss 和 rasterizer 的次级成本。
6. P2-05～P2-07：提升初始化、背景适配和反向梯度可信度。

## 每项任务的统一验收记录

每个 TODO 完成后追加一段实验记录，至少包含：

- 数据集、帧数、render scale、seed、iterations、GPU/driver；
- wall-clock、training loop time、steps/s、GPU 峰值显存；
- 初始/最终 Gaussian 数、intersection 数、topology event 数；
- mean/median/worst PSNR、worst frame id、edge/sharpness 指标；
- 与当前 baseline 的质量差、性能差和是否接受；
- 失败实验及停止原因。

## 停止条件

- 任何优化让最差帧下降超过 `0.2 dB`，或产生可见的全局 fog/ghosting，先回滚候选并保留报告。
- 单场景、低分辨率、短 iteration 的收益不能直接改默认配置。
- 默认配置变更前，至少通过 TUM 全视图、Home smoke 和一个外部场景的同口径评估。
- GPU primitive 替换必须同时通过单元测试、确定性小 fixture 和真实 adapter integration test。

# RustGS 训练 Pipeline 修复与验证实施计划

> **供自动化开发工具执行：** 建议使用 `superpowers:subagent-driven-development` 或 `superpowers:executing-plans`，严格按任务顺序实施。每个任务都要完成失败测试、最小实现、回归验证和独立提交。

**目标：** 修复 `3686466` 之后仍存在的训练状态污染、逐步 GPU→CPU 同步和断点恢复不连续问题，并建立真实评价 GPU 效率和独立视角质量的验收闭环。

**架构：** 训练器持有一个贯穿 forward、loss、topology 和 optimizer 的 device-resident sticky status。GPU kernel 在状态异常时禁止所有持久状态写入，host 只在 loss cadence、topology、checkpoint、暂停/取消和训练结束等安全点读取。正确性闭环后，再引入 scan workspace 复用、真实 profiling、固定 train/eval split 和跨场景实验。

**技术栈：** Rust 2021、Burn、burn-wgpu、CubeCL/WGSL、wgpu、serde JSON、现有 RustGS CLI 与 evaluation pipeline。

**代码基线：** `3686466 fix(rustgs): close training correctness review and R08–R12 gates`

**审查依据：** [2026-09-18 RustGS 审核记录](../../reviews/2026-09-18-rustgs-training-review-and-home-evidence.md)

## 全局约束

- 异常 step 不得修改 Gaussian 参数、Adam moment、Adam step、topology accumulator、visibility history 或 topology host vectors。
- overflow 与 non-finite 状态必须 sticky，首次异常信息不得被后续健康 step 清掉。
- 正常训练 step 不允许为判断 overflow 或 finite 而做 host readback。
- 所有 host topology mutation、checkpoint、暂停、取消和成功结束之前必须读取并验证 status。
- 不可测指标写 `null` 并记录原因；不能把 CPU submit `Instant` 标成 GPU 执行时间。
- checkpoint 升级必须能读取 v1；v1 恢复要显式记录兼容迁移。
- baseline/candidate 固定 revision、数据 fingerprint、训练帧、评估帧、seed、分辨率、迭代数、loss、topology 和 SH schedule。
- Home/flowers2 只做 smoke；质量结论必须包含独立 holdout 与 TUM 或同等级场景。
- 每个 Task 单独提交，不混入当前 RustSFM/RustViewer 未提交修改。

## 阶段依赖

```text
1 device 状态屏障 → 2 checkpoint/topology 连续性 → 3 bounded parity/scan 复用
→ 4 真实性能测量 → 5 holdout/gradient 质量 → 6 跨场景实验与收口
```

阶段 1、2 通过前，不接受新的默认性能算法。

---

# 阶段 1：统一 Device 状态屏障

## Task 1.1：新增共享 device training status

**文件：**

- Create: `RustGS/src/training/engine/device_status.rs`
- Create: `RustGS/src/training/shaders/update_training_status.wgsl`
- Modify: `RustGS/src/training/engine/mod.rs`
- Modify: `RustGS/src/training/engine/trainer.rs`
- Modify: `RustGS/src/training/reporting/telemetry.rs`
- Test: `RustGS/src/training/engine/device_status.rs`

**接口：**

```rust
pub(crate) const STATUS_FORWARD_OVERFLOW: u32 = 1 << 0;
pub(crate) const STATUS_NON_FINITE_LOSS: u32 = 1 << 1;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct TrainingStatusSnapshot {
    pub flags: u32,
    pub first_invalid_iteration: u32,
    pub requested_intersections: u32,
    pub intersection_capacity: u32,
    pub committed_optimizer_steps: u32,
}

pub(crate) struct DeviceTrainingStatus<B: Backend> { /* one u32[8] buffer */ }
impl<B: Backend> DeviceTrainingStatus<B> {
    pub(crate) fn new(device: &B::Device, restored_step: u32) -> Self;
    pub(crate) async fn read(&self) -> Result<TrainingStatusSnapshot, TrainingError>;
}
```

word 0 为 sticky flags；word 1 为首次异常 iteration（初始 `u32::MAX`）；word 2/3 为首次 overflow requested/capacity；word 4 为 committed Adam step；word 5 为当前 mutation gate；word 6/7 保留。状态 buffer 必须是连续资源，避免 Rust/WGSL 对齐差异。

- [x] 写无异常、overflow、non-finite、双 flag、首次 iteration 保留五个 host 解码测试。
- [x] 运行 `cargo test -p rustgs --lib --features gpu-wgpu engine::device_status -- --test-threads=1`，先确认测试失败。
- [x] 实现状态初始化、解码和错误转换；non-finite 与 capacity 错误必须能区分。
- [x] 把 trainer 的 `non_finite_loss_steps` 与 host-only overflow 状态替换为共享 status。
- [x] 运行 CPU/GPU library tests，提交 `refactor(rustgs): centralize device training status`。

## Task 1.2：Forward overflow 只写 device sticky flag

**文件：** `forward/dispatch.rs`、`forward/mod.rs`、`shaders/write_dispatch.wgsl`、`backward/autodiff.rs`、`engine/trainer.rs`；测试同名模块。

把 `WriteDispatchBackend::write_forward_dispatch` 增加 `iteration` 和 status buffer 参数。shader 仍将 logical intersections clamp 到 capacity，但在 `requested > capacity` 时 `atomicOr(flags, STATUS_FORWARD_OVERFLOW)`、`atomicMin(first_iteration, iteration)`，并只在首次异常写 requested/capacity。

- [x] 覆盖 `requested == capacity`、`capacity + 1`、连续 overflow 三个边界测试。
- [x] training forward 传入共享 status；evaluation exact path 使用独立 dummy status。
- [x] 删除 `note_sticky_forward_overflow()` 中每 step 的 `into_scalar_async()`。
- [x] telemetry 分开记录 `status_readbacks`、`capacity_telemetry_readbacks`、`loss_value_readbacks`、`checkpoint_tensor_readbacks`。
- [x] 运行 forward/GPU integration tests，提交 `perf(rustgs): keep overflow detection on device`。

## Task 1.3：Non-finite loss 在 GPU 上先标记

**文件：** 新建 `shaders/mark_non_finite_loss.wgsl`；修改 `engine/loss.rs`、`engine/trainer.rs`；测试 `loss.rs`。

增加 `LossStatusBackend::mark_non_finite_loss(loss, status, iteration)`。命令顺序必须是 `loss → mark status → backward → mutation gate → topology/Adam`。finite、NaN、+Inf、-Inf 都要覆盖；采样 loss scalar readback 仍保留用于报错数值，但不再承担安全职责。

- [x] 写四个 GPU fixture 并确认旧实现无法提前阻断。
- [x] 实现 kernel/backend trait 和首次 iteration 记录。
- [x] 删除 `non_finite_loss_steps` 的累加路径。
- [x] 运行 loss/trainer tests，提交 `fix(rustgs): mark non-finite loss before state mutation`。

## Task 1.4：Topology 与 Adam 消费同一个 mutation gate

**文件：** `shaders/accumulate_topology_stats.wgsl`、`engine/topology_accum.rs`、`shaders/adam_update.wgsl`、`engine/optimizer.rs`、`engine/trainer.rs`；测试 optimizer/trainer。

新增 `prepare_optimizer_step.wgsl`：读取 status flags，健康时将 word 5 设为 1 并把 word 4 加一；异常时设为 0 且不加 step。Adam 三组参数 kernel 都读取 word 5/4；gate=0 立即 return，不写 param/moment，gate=1 使用同一 committed step 做 bias correction。无 scaling 的路径也必须走 gated kernel。Topology accumulator 在每个写入前检查 flags。

- [x] 用非零 param/moment、step=7 写异常 gate 测试，验证参数、moment、step 全不变。
- [x] 用非零哨兵 accumulator 写 topology gate 测试。
- [x] 增加真实 `train_step` 故障注入：第 2 step、`read_loss=false` 注入 NaN，checkpoint 前后比较全部 state。
- [x] 增加小 capacity overflow 故障注入，断言同样不变。
- [x] 运行 GPU library、integration、checkpoint tests，提交 `fix(rustgs): gate optimizer and topology mutations on device`。

## Task 1.5：集中 host 安全点

**文件：** `engine/trainer.rs`、`engine/runtime.rs`、`reporting/metrics.rs`、`reporting/optimization_report.rs`。

新增：

```rust
enum StatusReadbackReason { LossCadence, TopologyBoundary, Checkpoint, Pause, Cancel, TrainingEnd }
async fn ensure_device_status_healthy(&mut self, reason: StatusReadbackReason)
    -> Result<TrainingStatusSnapshot, TrainingError>;
```

普通 step 不读；loss cadence、topology snapshot 前、checkpoint、pause/cancel、training end 才读。删除分散的 status readback，并在 report 写出各 reason 计数。

- [x] 覆盖 checkpoint 不在 loss cadence 的测试。
- [x] 100-step 正常训练断言 readback 数量等于安全点数量而不是 step 数。
- [x] 阶段门禁：`cargo fmt --package rustgs --check`、CPU/GPU library、integration ignored、checkpoint_resume、`git diff --check`。
- [x] 提交 `fix(rustgs): validate device status at training safety points`。

**阶段 1 验收：** 正常 step 零 status/overflow/finite readback；异常 step 零参数、moment、step、topology mutation；首次异常信息准确；checkpoint 和成功结束都不能越过异常。

---

# 阶段 2：Checkpoint 与 Topology 连续性

## Task 2.1：Checkpoint v2 保存 visibility baselines

**文件：** `training/checkpoint.rs`、`engine/topology_accum.rs`、`engine/trainer.rs`、`engine/runtime.rs`；测试 checkpoint_resume。

将 `TRAINING_CHECKPOINT_VERSION` 升为 2，`TopologyCheckpoint` 增加：

```rust
pub visibility_window_baseline: Vec<f32>;
pub actual_visibility_window_baseline: Vec<f32>;
```

v2 要求长度等于 splat count、值 finite 且非负；v1 用旧累计 visibility 构造 baseline，并返回 `CheckpointMigration::V1BaselineReset`，日志和 report 必须记录迁移；不能用 `serde(default)` 静默接受损坏的空数组。

- [x] 增加 v2 round-trip、长度/NaN 校验和固定 v1 JSON fixture。
- [x] restore 直接使用 checkpoint baseline，禁止再由累计 tensor 覆盖。
- [x] 运行 checkpoint tests，提交 `fix(rustgs): preserve topology visibility windows in checkpoints`。

## Task 2.2：Topology window 中途 resume parity

在 `RustGS/tests/checkpoint_resume.rs` 增加 topology interval=4、iteration 2 checkpoint 的双路径测试：A 直训到 8，B 从 checkpoint 训到 8。比较 topology event fingerprint（iteration、disposition、origins、增密/剪枝数、两个 visibility delta、invisible windows）、HostSplats、三组 Adam state、committed step 和 accumulator，浮点误差 `1e-6`。

- [x] 先让旧 baseline reset 使测试失败，再实现后使其通过。
- [x] 覆盖 checkpoint 不在 topology 边界和 checkpoint 不在 loss cadence 两种位置。
- [x] 提交 `test(rustgs): prove mid-window topology resume parity`。

## Task 2.3：Densify/prune/opacity reset 后 Adam continuity

**文件：** `engine/optimizer.rs`、`topology/apply.rs` 或实际 apply 文件、`tests/checkpoint_resume.rs`。

用可识别 row 值覆盖：非连续大量 prune、多个 `None` densify、混合 origins、opacity reset、全部候选 prune 的保护逻辑。survivor 的 param/moment/age/baseline 必须同源；new row moment 为 0；opacity reset 时只清对应 opacity moment，transform/SH moment 保留；step 不回退。增加 apply→checkpoint→restore→next-step parity，提交 `fix(rustgs): preserve optimizer continuity across topology changes`。

- [x] opacity reset 只清 opacity moment；混合 prune/densify remap 保留 survivor、新行为零；step 不回退。
- [x] 提交 `fix(rustgs): preserve optimizer continuity across topology changes`。

**阶段 2 验收：** v2 严格验证、v1 显式迁移；mid-window resume 与 uninterrupted 事件相同；大规模 mutation 后所有 tensor/vector shape 和来源一致。

---

# 阶段 3：Bounded Forward Parity 与 Scan Workspace

## Task 3.1：Exact/bounded 完整 parity

新建 `RustGS/tests/bounded_forward_parity.rs`，覆盖 zero visible、zero intersections、1/255/256/257 intersections、恰好满容量、capacity 少 1、partial tile、duplicate depth key、low alpha、near-plane。容量足够时比较 logical/requested count、排序 key/value、tile offsets、RGB/depth/visibility 和三组 gradient；整数完全相等，浮点默认 `1e-5`，只有能解释归约顺序时才局部放宽到 `1e-4`。capacity 少 1 必须复用阶段 1 的状态不变断言。

- [x] 先写 fixture 并确认没有 parity 入口。
- [x] 实现 exact/bounded 双跑 helper 和测试所需最小入口暴露。
- [x] 修复差异，不得用整体放宽容差绕过。
- [x] 提交 `test(rustgs): cover bounded forward pipeline parity`。

## Task 3.2：PrefixSum workspace 复用

**文件：** `gpu_primitives/prefix_sum.rs`、`forward/tile_mapping.rs`、`forward/mod.rs`、`engine/trainer.rs`。

实现：

```rust
struct PrefixSumWorkspace<B: Backend> { capacity: usize, levels: Vec<PrefixSumLevel<B>> }
struct PrefixSumLevel<B: Backend> { block_sums: Tensor<B,1,Int>, block_prefix: Tensor<B,1,Int> }
fn reserve(&mut self, capacity: usize, device: &B::Device);
fn inclusive_scan_into(&mut self, input: Tensor<B,1,Int>, len: usize,
    output: Tensor<B,1,Int>) -> Result<Tensor<B,1,Int>, String>;
```

公共 `inclusive_scan()` 继续返回独立 allocation；训练路径使用 caller-owned output。每个递归层独立 buffer，workspace 生命周期覆盖当前 step backward，下一 step 才复用。

- [x] 保留连续同长度、短→长→短旧结果测试。
- [x] 增加 capacity 不增长、capacity 增长、fresh allocation count 和 output 生命周期测试。
- [x] telemetry 记录实际 reserved bytes、growth count、per-step fresh allocations。
- [x] 运行 prefix/radix/parity tests，提交 `perf(rustgs): reuse owned prefix scan workspaces`。

**阶段 3 验收：** exact/bounded parity、overflow hard-stop、scan 无 alias、稳态递归 scratch 不再逐 step 分配。

---

# 阶段 4：真实性能测量

## Task 4.1：补齐实验身份

修改 `OptimizationCommand` 增加 `train_frame_ids`、`effective_max_frames`、`loss_config_fingerprint`、`topology_config_fingerprint`、`sh_schedule_fingerprint`、`training_config_fingerprint`。`effective_max_frames` 必须是真正进入 loader 的 pose 数量。comparator 对这些字段以及已有 dataset/seed/scale/eval IDs/resolution 全部做 mismatch reject。

- [ ] 每个字段写 mismatch test。
- [ ] 用 canonical JSON 生成 fingerprint，排除路径和日志级别。
- [ ] report 使用实际 train/eval selection。
- [ ] 提交 `fix(rustgs): reject incomparable training reports`。

## Task 4.2：GPU timing 和 allocation evidence

新建 `training/reporting/gpu_profiler.rs`。报告至少包含 `supported`、`unsupported_reason`、GPU step p50/p95、sample count、adapter/backend/driver、workspace current/peak bytes、growth count、fresh-step allocations、runtime peak device bytes（拿不到则 null+reason）。有 timestamp query 才使用 GPU completion；否则保留清晰命名的 `cpu_submit_instant`。

- [ ] 写 capability 与 null/reason 序列化测试。
- [ ] 接入 adapter metadata、workspace tracker 和可选 timestamp。
- [ ] 100-step smoke 验证计数单调、p95≥p50、unsupported 字段自洽。
- [ ] 提交 `feat(rustgs): report gpu timing and workspace allocation evidence`。

## Task 4.3：先测量再决定 topology/cache/fused loss

记录 frame wait、decode、resize、upload、forward、backward、optimizer、topology snapshot/plan/apply/upload/remap。只有同一瓶颈在至少两个场景重复出现才新建后续优化任务；建议门槛为 topology pause <5%、frame wait p95 <总 step p95 的10%、loss <GPU step的10%时不改对应模块。

- [ ] 添加 percentile tests 和无行为变化的 timing points。
- [ ] 生成 Home 500/1500 报告，提交 `perf(rustgs): measure topology and input stalls`。

**阶段 4 验收：** comparator 可拒绝不可比报告；CPU/GPU timing 分离；readback、workspace、topology pause、frame stall 可从 JSON 判断；不可测指标带原因。

---

# 阶段 5：独立视角与梯度质量

## Task 5.1：显式 train/in-view/holdout split

新建 `training/evaluation/split.rs`，定义 `EvaluationSplitKind::{InView,Holdout}` 和带 dataset fingerprint、train IDs、in-view IDs、holdout IDs 的 `FrameSplitManifest`。CLI 增加 `--frame-split-manifest`、`--eval-split in-view|holdout`。

- [ ] 校验重复、train/holdout 交叉、未知 ID、fingerprint mismatch。
- [ ] 使用 `ScenePose.frame_id`，不能依赖 vector index。
- [ ] report 保存 split kind 和完整 ID；comparator 比较它们。
- [ ] 未提供 holdout 却要求 holdout 时直接报错。
- [ ] 提交 `feat(rustgs): add reproducible holdout evaluation splits`。

## Task 5.2：Gradient boundary suite

在 `backward/gradient_check.rs` 增加 near-plane 内外、clip 边界、low alpha 阈值两侧、covariance blur boundary、SH degree 2/3、off-axis camera、depth-sensitive scale。平滑强信号使用 `relative<=3e-2 || absolute<=1e-3`；弱信号只在双方均 `<5e-3` 通过；不连续阈值两侧至少 `4*epsilon`，不得在不连续点做 centered FD。Quaternion 主门禁改用切空间 perturbation，旧未归一化分量检查保留为诊断。

- [ ] 写 case builder 和分层 tolerance。
- [ ] 输出每参数 analytic/numeric/error 表。
- [ ] 只修对应数学链路，不放宽全局容差。
- [ ] 提交 `test(rustgs): harden gradient checks at render boundaries`。

## Task 5.3：显式 SH schedule

在 `TrainingConfig` 增加 `ShScheduleConfig { initial_degree, increment_every, max_degree }`，默认保持当前每 1000 iteration 升一级。 `active_sh_degree_at()` 改为纯函数；checkpoint identity、active degree validation 和 optimization fingerprint 都包含完整 schedule。

- [ ] 测试 iteration 0/1/999/1000/1001 和 resume 边界。
- [ ] validate `increment_every>=1`、degree≤3。
- [ ] 提交 `feat(rustgs): make sh training schedule reproducible`。

**阶段 5 验收：** train/holdout 不重叠，report 明确 split；gradient boundary suite 通过；SH schedule 在 config/checkpoint/report 一致。

---

# 阶段 6：跨场景实验与收口

## Task 6.1：冻结实验包

目录：`output/rustgs-optimization/2026-09-18-remediation/{bin,manifests,home,flowers2,tum}`。每次保存 command、revision、binary sha256、split manifest、optimization JSON、evaluation JSON、stdout、PLY。一次构建 release binary，baseline/candidate 不在实验中途重建。

## Task 6.2：执行阶梯

Home 500 → flowers2 500 → TUM 500 → TUM 3k → 10k → 30k。正式阶段 baseline/candidate 各至少 3 次，首轮 warmup 不计入统计，报告 mean/stddev/p50/p95。

拒绝条件：任意 overflow/non-finite/invalid dispatch/checkpoint failure；resume fingerprint 不一致；普通 step status readback；workspace 持续增长；holdout mean 下降超过 0.10 dB；worst frame 下降超过 0.20 dB；出现新 fog/ghosting/floaters。

接受条件：先通过 correctness/quality，再证明同 GPU/driver 下稳定 steps/s 提升，p95 不恶化，并能从报告归因到同步、workspace 或 kernel 时间。

## Task 6.3：更新 TODO 与最终门禁

修改 TODO、审核记录和 `docs/index.md`。只在代码和实验门禁通过后删除事项；记录 revision、命令、报告路径和数字；修复审核文档 EOF 空行。

```bash
cargo fmt --package rustgs --check
cargo test -p rustgs --lib --no-default-features --no-fail-fast
cargo test -p rustgs --lib --features gpu-wgpu --no-fail-fast -- --test-threads=1
cargo test -p rustgs --test integration_test --features gpu-wgpu -- --ignored --test-threads=1
cargo test -p rustgs --test checkpoint_resume --features gpu-wgpu -- --test-threads=1
git diff --check
```

## 完成定义

只有同时满足以下条件才算完成：device status 覆盖 overflow、non-finite、optimizer、topology、checkpoint 和成功结束；v2 checkpoint 与 mid-window resume parity 通过；bounded/exact parity 和 fault injection 通过；scan 无 alias 且稳态复用有实测；report 能拒绝不可比实验并区分 CPU/GPU；holdout split 固化；gradient boundary suite 通过；Home、flowers2、TUM 阶梯达到门禁；TODO 只保留真实未完成事项；`git diff --check` 通过。

## 开发工具执行规则

1. 每次只执行一个 Task，先写失败测试再实现。
2. 发现接口冲突时，优先保证“异常 step 零持久 mutation”和“正常 step 零 status readback”。
3. 不得用放宽全局容差、减少测试覆盖或删除 telemetry 通过门禁。
4. 阶段 1/2 失败时停止性能阶段。
5. 没有 profiling 证据，不实现 GPU-native topology、fused loss 或大规模 cache 重构。
6. 提交前检查 `git status --short`，不得带入 RustSFM/RustViewer 修改。


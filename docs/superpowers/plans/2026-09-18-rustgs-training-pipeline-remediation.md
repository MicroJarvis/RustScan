# RustGS 训练 Pipeline 修复与验证实施计划

> 供自动化开发工具执行：使用 superpowers:subagent-driven-development 或 superpowers:executing-plans，按优先级逐项实施。每个任务先写失败测试，再做最小实现，独立验证后提交。

**目标：** 在已完成的训练状态屏障、checkpoint 连续性和 exact/bounded parity 基础上，补齐异常 step 的硬停止、GPU 资源复用证据、真实性能测量、独立视角质量验证和跨场景实验闭环。

**架构：** 训练器继续使用贯穿 forward、loss、backward、topology 和 optimizer 的 device-resident sticky status。异常状态必须在 device 上阻止后续计算和持久写入；host 只在安全点读回状态。scan workspace、profiling、split manifest 和实验 comparator 都必须有明确的所有权、版本和可追溯字段。

**技术栈：** Rust 2021、Burn、burn-wgpu、CubeCL/WGSL、wgpu、serde JSON、现有 RustGS CLI 与 evaluation pipeline。

**代码基线：** device status、checkpoint v2、resume parity、bounded parity、scan workspace、experiment identity 的已有提交视为历史基线；本计划只列当前仍真实需要完成的工作。

**审查依据：** docs/reviews/2026-09-18-rustgs-training-review-and-home-evidence.md 及最近提交的 pipeline review。

## 当前状态与结论

阶段 1、阶段 2，以及阶段 3 的 exact/bounded parity 已实现并通过对应 GPU 测试；阶段 4.1 的实验身份 comparator 已实现。整个 remediation 计划尚未完成，最近 review 仍发现：

- **P0/P1 forward hard-stop 缺口：** RustGS/src/training/engine/trainer.rs:1379 的 ensure_forward_capacity_before_update() 只检查 forward 之后的 host 状态镜像。GPU overflow 发生后，read_loss=false 的 step 仍可能进入 loss/backward，随后才在安全点读回；Adam/topology gate 能阻止持久 mutation，但不能满足异常 step 立即停止，也浪费 backward 计算。
- **P1 scan 所有权缺口：** RustGS/src/training/gpu_primitives/prefix_sum.rs:371 的训练路径每次仍调用 empty_tensor 创建输出。workspace 只复用递归 scratch，step_fresh_allocations 没有统计该输出分配；prefix_sum.rs:291 与 trainer.rs:720 的 TLS raw pointer 绑定跨越 await，在多线程异步 runtime 下不安全。
- **P1 阶段 4.2–6 不完整：** 尚无完整 training/reporting/gpu_profiler.rs、GPU completion timing、adapter/driver metadata、runtime device memory、unsupported reason、可复现实验 split manifest、完整 gradient boundary suite、显式 SH schedule 和 TUM 阶梯包。
- **P2 格式门禁失败：** cargo fmt --package rustgs --check 当前仍报告 RustGS/src/lib.rs、RustGS/src/training/engine/trainer.rs、RustGS/src/training/forward/parity.rs、RustGS/src/training/forward/sorting.rs 差异。

已验证结果：GPU bounded parity 通过；GPU checkpoint resume 通过（53 tests）；GPU library 通过（130 tests）；ignored GPU integration 通过；CPU library 通过（29 tests）；git diff --check 通过。上述结果不能替代 fmt、profiling、holdout 和跨场景门禁。

## 已完成且冻结的基线

- device sticky status 已覆盖 forward overflow、non-finite loss、首次异常 iteration、requested/capacity 和 committed optimizer step；host 只在 loss cadence、topology、checkpoint、暂停/取消和训练结束等安全点读取。
- overflow/non-finite 状态已在 GPU 标记；Adam、topology accumulator 和 topology host mutation 均受同一 mutation gate 保护，异常时不写参数、moment、step 或 accumulator。
- checkpoint v2 保存 visibility baselines；v1 恢复显式执行迁移并记录原因；mid-window topology resume 与 uninterrupted training 做过事件和状态 parity。
- exact/bounded forward parity 已覆盖边界 intersection、容量和稳定 equal-depth 排序；实验 identity comparator 已能拒绝 revision、dataset、seed、帧选择、分辨率、迭代数、loss/topology/SH 配置不一致的报告。

## 全局约束

- 异常 step 不得修改 Gaussian 参数、Adam moment、Adam step、topology accumulator、visibility history 或 topology host vectors。
- overflow 与 non-finite 必须 sticky；首次异常信息不得被健康 step 清除。
- 正常 step 不得为判断 overflow 或 finite 而做 host readback。
- 所有 host topology mutation、checkpoint、暂停、取消和成功结束之前必须读取并验证 status。
- 不可测指标写 null 并记录结构化原因；CPU submit 时间不能伪装成 GPU 执行时间。
- 只修改 RustGS 相关文件和本计划；不得带入 RustSFM/RustViewer 未提交修改。

## 优先级与依赖

1. **P0：异常 forward 的 device-side hard-stop。** 依赖已完成的 status buffer；完成前不得宣称训练错误路径闭环。
2. **P1：scan workspace/output 所有权与分配证据。** 依赖 P0 的 step 生命周期约束；完成前不做 fused loss 或大规模 topology cache。
3. **P1：GPU profiler、pipeline timing、split manifest。** 可与 scan 并行，但必须使用稳定的 step/safety-point 语义。
4. **P2：gradient boundary suite、显式 SH schedule。** 依赖 split/config identity 定义。
5. **P2：TUM/跨场景实验包、格式修复和最终收口。** 依赖前四项全部通过。

---

# P0：Forward hard-stop

## Task P0.1：把 forward overflow 变成同一 step 的 device gate

**文件：** 修改 RustGS/src/training/forward/dispatch.rs、forward/mod.rs、forward/shaders/write_dispatch.wgsl、engine/trainer.rs、engine/loss.rs、backward/autodiff.rs；测试 RustGS/tests/bounded_forward_parity.rs、RustGS/tests/checkpoint_resume.rs 及 engine GPU fixtures。

**接口与不变量：**

    pub(crate) struct ForwardDispatchResult<B: Backend> {
        pub logical_intersections: Tensor<B, 1, Int>,
        pub mutation_gate: Tensor<B, 1, Int>,
    }

    pub(crate) async fn dispatch_forward(
        num_visible: Tensor<B, 1, Int>,
        num_intersections: Tensor<B, 1, Int>,
        intersection_capacity: usize,
        status: &DeviceTrainingStatus<B>,
        iteration: u32,
    ) -> Result<ForwardDispatchResult<B>, TrainingError>;

shader 在写 dispatch 前用 atomicOr/atomicMin 记录 overflow，随后写入 device scalar gate。gate=0 时 loss kernel、backward launch、topology accumulation 和 Adam kernels 直接 return，且不得读取或写入持久状态。不能依赖 stale host mirror，也不能把“稍后安全点读回”当作当前 step 的停止条件。

- [x] 写 capacity 少 1、连续 overflow、read_loss=false 三个失败 fixture；断言 backward dispatch 不发生，所有参数/moment/step/accumulator 与 step 前 bitwise 相同。
- [x] 在 forward dispatch 后增加 device gate producer；把 gate 作为 loss/backward/topology/Adam 的显式输入，删除当前 step 对 host mirror 的分支判断。
- [x] 保留 host safety-point readback，错误中同时携带首次 iteration、requested 和 capacity。
- [x] 为 evaluation exact path 传入独立 dummy status，确保训练 gate 不污染评估。
- [x] 运行 GPU library、bounded parity、integration ignored 和 checkpoint resume；检查正常 step 的 status readback 仍为零。

**验收标准：** overflow step 不提交 loss/backward/optimizer/topology 命令；异常后下一步不能清除 sticky flag；capacity 恢复前训练直接返回结构化错误；正常 step 仍保持零逐步 host readback。

**P0.1 落地说明（2026-09-19）：** `write_dispatch` 在 overflow 时立刻 `atomicStore(status[5], 0)`；`mark_non_finite_loss` 同样清 gate；`rasterize_backwards` / `project_backwards` 读取 sticky flags 后 early-return；`RenderCheckpoint` 携带训练 status（评估路径用独立 dummy）；删除 `ensure_forward_capacity_before_update` 中途 host mirror 分支。连续 overflow / read_loss=false / capacity−1 fixture 与 GPU library + bounded parity 已通过。

## Task P0.2：修复安全点错误传播与状态镜像

**文件：** RustGS/src/training/engine/trainer.rs、engine/runtime.rs、reporting/telemetry.rs。

- [ ] 为 StatusReadbackReason 增加 ForwardAbort，仅用于 gate 已判定异常后的诊断读回，不在健康 step 调用。
- [ ] 将失败 step 的命令计数、backward skipped、optimizer skipped、topology skipped 写入 telemetry；区分 GPU gate skip 和 host safety-point abort。
- [ ] 增加训练 API 测试：forward overflow、loss NaN、取消和 checkpoint 错误都返回唯一错误类别，且不会被后续 healthy step 覆盖。
- [ ] 验证错误路径不会把部分写入的 transient tensor 当作可恢复 checkpoint 状态。

**验收标准：** 每种失败都能从 error、status snapshot 和 telemetry 三处定位；错误发生后不会继续完成一个逻辑 iteration。

---

# P1：Scan workspace 与分配证据

## Task P1.1：取消 TLS raw pointer，改为显式 workspace 所有权

**文件：** RustGS/src/training/gpu_primitives/prefix_sum.rs、forward/tile_mapping.rs、forward/mod.rs、engine/trainer.rs。

    pub(crate) struct PrefixSumWorkspace<B: Backend> {
        capacity: usize,
        levels: Vec<PrefixSumLevel<B>>,
        output: Option<Tensor<B, 1, Int>>,
    }

    impl<B: Backend> PrefixSumWorkspace<B> {
        pub(crate) fn reserve(&mut self, capacity: usize, device: &B::Device);
        pub(crate) fn inclusive_scan_into(
            &mut self, input: Tensor<B, 1, Int>, len: usize,
        ) -> Result<Tensor<B, 1, Int>, String>;
    }

workspace 必须由 trainer/forward context 持有，并通过 &mut PrefixSumWorkspace 传入；不得再通过 thread-local raw pointer 绑定。inclusive_scan_into 返回 workspace-owned output，调用方在 backward 完成前不能触发下一次 scan 或 reserve。

- [ ] 先添加多线程 tokio/wgpu fixture，证明旧 TLS 绑定可被不同 task 交错覆盖；测试当前实现失败。
- [ ] 删除 TLS pointer 和 bind/unbind API，改为显式 &mut 传递；把 workspace 生命周期覆盖 forward 到 backward 的最后一个 consumer。
- [ ] 明确 input/output alias 规则：输入不能与 output 共享 storage；reserve 增长时不得使当前 step 仍在使用的 tensor 失效。
- [ ] 对训练路径使用 inclusive_scan_into；公共 convenience inclusive_scan 保持独立 allocation 语义。
- [ ] 运行 prefix/radix/parity 和单线程、多线程 GPU tests。

**验收标准：** 无 unsafe TLS pointer；并发测试无数据竞争；scan output 在 backward 完成前稳定；短→长→短容量复用结果正确。

## Task P1.2：消除每次 scan 的 output allocation 并建立计数

**文件：** RustGS/src/training/gpu_primitives/prefix_sum.rs、training/reporting/telemetry.rs、engine/trainer.rs。

- [ ] 为每个递归 level 和最终 output 预留 capacity；只在 len > capacity 时增长，并记录 old/new bytes、growth count。
- [ ] 把 output allocation 纳入 step_fresh_allocations，区分 workspace growth allocation 与稳态 allocation；不能只统计 scratch。
- [ ] 在连续 100 step 的固定 shape smoke 中断言：第一次允许增长，之后 output/scratch fresh allocation 为 0；capacity 变化时增长次数单调且 bytes 可复算。
- [ ] 加入 alias/shape 检查，防止调用方传入错误 len 导致越界或静默截断。

**验收标准：** JSON 中能看到 workspace current/peak bytes、growth count、per-step fresh allocations；稳态 allocation 证据为零；parity 不回退。

---

# P1：GPU Profiling、Pipeline Timing 与实验可比性

## Task P1.3：实现 GPU profiler 数据模型

**文件：** 新建 RustGS/src/training/reporting/gpu_profiler.rs；修改 training/reporting/mod.rs、telemetry.rs、optimization_report.rs、engine/runtime.rs；测试 gpu_profiler.rs 单元测试和 report JSON fixtures。

    #[derive(Serialize, Deserialize)]
    pub struct GpuProfilerReport {
        pub supported: bool,
        pub unsupported_reason: Option<String>,
        pub backend: String,
        pub adapter: Option<String>,
        pub driver: Option<String>,
        pub gpu_step_p50_ms: Option<f64>,
        pub gpu_step_p95_ms: Option<f64>,
        pub sample_count: u64,
        pub workspace_current_bytes: u64,
        pub workspace_peak_bytes: u64,
        pub workspace_growth_count: u64,
        pub fresh_step_allocations: u64,
        pub runtime_peak_device_bytes: Option<u64>,
        pub runtime_peak_device_bytes_reason: Option<String>,
    }

- [ ] 写 capability matrix、unsupported/null/reason、自洽性和 percentile 单元测试；要求 p95>=p50，unsupported 时 GPU percentile 为 null。
- [ ] 接入 wgpu timestamp query；只有 query resolve/readback 完成后才计算 GPU completion interval；无 query 时保留明确命名的 cpu_submit_instant，不得映射到 GPU 字段。
- [ ] 接入 adapter/backend/driver metadata、workspace tracker 和 runtime device memory；拿不到 runtime peak 时写 null+reason。
- [ ] 在 100-step smoke 中验证 sample_count、p50/p95、growth 和 fresh allocation 计数单调。

**验收标准：** report 能区分 CPU submit 与 GPU completion；unsupported 字段自洽；所有不可测指标有机器可读原因。

## Task P1.4：补齐完整 pipeline timing

**文件：** training/reporting/telemetry.rs、engine/trainer.rs、io/loader.rs、optimization_report.rs。

- [ ] 为 frame wait、decode、resize、upload、forward、backward、optimizer、topology snapshot/plan/apply/upload/remap 建立统一 span 名称和 sample counter。
- [ ] 使用同一 warmup/drop policy 计算 p50/p95；报告总 step 与子阶段不能重复累计。
- [ ] 在 Home 500/1500 生成基线报告，确认 topology pause <5%、frame wait p95 <总 step p95 的 10%、loss < GPU step 的 10% 等门槛；未达门槛才创建后续优化任务。
- [ ] 禁止仅凭 CPU wall clock 宣称 GPU kernel 加速。

**验收标准：** JSON 足以归因同步、workspace、输入 stall、topology 或 kernel 时间；同一 revision 的重复运行字段可比较。

## Task P1.5：固定 train/in-view/holdout split manifest

**文件：** 新建 RustGS/src/training/evaluation/split.rs；修改 RustGS/src/bin/rustgs/train_command.rs、training/evaluation/mod.rs、optimization_report.rs、comparator；测试 split manifest unit/integration tests。

    pub enum EvaluationSplitKind { InView, Holdout }
    pub struct FrameSplitManifest {
        pub dataset_fingerprint: String,
        pub train_ids: Vec<String>,
        pub in_view_ids: Vec<String>,
        pub holdout_ids: Vec<String>,
    }

- [ ] CLI 增加 --frame-split-manifest 与 --eval-split in-view|holdout；使用 ScenePose.frame_id，不能依赖 vector index。
- [ ] 校验重复 ID、train/holdout 交叉、未知 ID、dataset fingerprint mismatch；要求 holdout 时缺少 manifest 直接报错。
- [ ] report 保存 split kind、完整 ID 和 manifest fingerprint；comparator 比较这些字段。
- [ ] 用固定 fixture 覆盖空 split、单帧 split、重复和未知 ID 错误。

**验收标准：** 同一 manifest 在重复运行中产生相同 eval selection；holdout 与 train 不重叠；报告明确说明 in-view/holdout。

---

# P2：质量边界、配置可复现与实验收口

## Task P2.1：Gradient boundary suite

**文件：** RustGS/src/training/backward/gradient_check.rs、RustGS/tests/gradient_boundary.rs。

- [ ] 覆盖 near-plane 内外、clip 边界、low alpha 阈值两侧、covariance blur boundary、SH degree 2/3、off-axis camera、depth-sensitive scale。
- [ ] 平滑强信号使用 relative <= 3e-2 或 absolute <= 1e-3；弱信号仅在双方均 <5e-3 时通过；不连续点两侧至少 4*epsilon，不得在不连续点做 centered FD。
- [ ] Quaternion 主门禁使用切空间 perturbation；旧未归一化分量检查保留为诊断。输出每参数 analytic/numeric/error 表。
- [ ] 只修对应数学链路，不放宽全局容差；CPU 与 GPU fixture 都要能定位具体边界。

**验收标准：** boundary suite 全通过；失败信息包含 case、参数、analytic、numeric、relative/absolute error。

## Task P2.2：显式 SH schedule 与 identity

**文件：** RustGS/src/training/config.rs、engine/trainer.rs、training/checkpoint.rs、comparator。

    pub struct ShScheduleConfig {
        pub initial_degree: u8,
        pub increment_every: u32,
        pub max_degree: u8,
    }

    pub fn active_sh_degree_at(cfg: ShScheduleConfig, iteration: u32) -> u8;

- [ ] 保持当前默认语义：每 1000 iteration 升一级；测试 0/1/999/1000/1001 和 resume 边界。
- [ ] validate increment_every >= 1、degree <= 3、initial <= max；非法配置在 CLI 解析阶段失败。
- [ ] checkpoint 保存 schedule；optimization fingerprint 使用 canonical JSON；恢复时拒绝 schedule mismatch。

**验收标准：** 同一 config 在 uninterrupted/resume 下 active degree 完全一致；报告可追溯完整 schedule。

## Task P2.3：冻结跨场景实验包并执行阶梯

**目录与产物：** output/rustgs-optimization/2026-09-18-remediation/{bin,manifests,home,flowers2,tum}；每次保存 command、git revision、binary sha256、split manifest、optimization JSON、evaluation JSON、stdout/stderr、PLY 和环境元数据。

- [ ] 一次构建 release binary；baseline/candidate 实验期间禁止重建。
- [ ] 执行 Home 500 → flowers2 500 → TUM 500 → TUM 3k → 10k → 30k；正式 baseline/candidate 各至少 3 次，首轮 warmup 不计入统计，报告 mean/stddev/p50/p95。
- [ ] 拒绝条件：overflow/non-finite/invalid dispatch/checkpoint failure；resume fingerprint mismatch；普通 step status readback；workspace 持续增长；holdout mean 下降超过 0.10 dB；worst frame 下降超过 0.20 dB；出现新 fog/ghosting/floaters。
- [ ] 接受条件：先通过 correctness/quality，再在同 GPU/driver 下证明稳定 steps/s 提升且 p95 不恶化，并能从 JSON 归因到同步、workspace 或 kernel 时间。

**验收标准：** 每个场景都能由 manifest、report 和 binary hash 完整复现；质量和效率门槛均有数字证据。

## Task P2.4：修复格式门禁并更新文档

**文件：** RustGS/src/lib.rs、RustGS/src/training/engine/trainer.rs、RustGS/src/training/forward/parity.rs、RustGS/src/training/forward/sorting.rs，以及本计划和对应 review 记录。

- [ ] 运行 cargo fmt --package rustgs，只接受 rustfmt 产生的格式变更，不顺手改行为。
- [ ] 用 git diff --check 检查空白；确认 git status 不含 RustSFM/RustViewer 变更的 staged/committed 混入。
- [ ] 将本计划中的 checkbox 仅在对应代码、测试和 artifact 已验收后勾选；删除过时事项，不保留“已完成但无证据”的任务。

**验收标准：** fmt check、CPU/GPU library、ignored integration、checkpoint resume、bounded parity、gradient boundary、split/comparator 和实验报告全部通过。

---

## 最终验证命令

在 P0–P2 任务完成后执行：

    cargo fmt --package rustgs --check
    cargo test -p rustgs --lib --no-default-features --no-fail-fast
    cargo test -p rustgs --lib --features gpu-wgpu --no-fail-fast -- --test-threads=1
    cargo test -p rustgs --test integration_test --features gpu-wgpu -- --ignored --test-threads=1
    cargo test -p rustgs --test checkpoint_resume --features gpu-wgpu -- --test-threads=1
    git diff --check

此外必须运行 bounded parity、gradient boundary、split manifest/comparator 单测，并在最终实验包中保存命令输出、revision、binary hash、报告路径和关键数字。

## 完成定义

只有同时满足以下条件才算完成：forward overflow 和 non-finite 在 device 上阻止当前 step 后续计算；异常 step 零持久 mutation；checkpoint v2 与 mid-window resume parity 通过；scan 无 TLS alias 且稳态 output/scratch allocation 有实测为零；GPU profiler 能区分 completion/submit 并记录 unsupported reason；split manifest 固化 holdout；gradient boundary 和 SH schedule 可复现；Home、flowers2、TUM 阶梯达到 correctness、quality、efficiency 门禁；fmt 和 diff check 通过；本文只保留真实未完成事项。

## 执行规则

1. 每次只执行一个 Task，先写失败测试，再实现最小改动，独立运行该 Task 的验证命令。
2. P0 未通过前停止性能算法；没有 profiling 证据，不实现 GPU-native topology、fused loss 或大规模 cache 重构。
3. 不得用放宽全局容差、减少测试覆盖、删除 telemetry 或跳过 holdout 来通过门禁。
4. 每个 Task 单独提交，提交前检查 git status --short；不得带入当前 RustSFM/RustViewer 未提交修改。
5. 发现接口冲突时，优先保证“异常 step 零持久 mutation”和“正常 step 零 status readback”。

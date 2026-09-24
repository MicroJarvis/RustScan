# RustGS 分阶段 Cursor 开发任务卡

> 这份文档用于把 RustGS 训练 pipeline remediation 拆成可逐张交给 Cursor 的小任务。
> Cursor 每次只执行一张卡；完成后必须停止，由维护者完成代码 review、独立验证和任务状态更新，确认通过后再发下一张卡。

## 使用规则

1. 开始任何一张卡前，先阅读根目录 `AGENTS.md`、`docs/agent/rust-style.md`、`docs/agent/task-protocol.md`，以及本 change package 下的 `design.md` 和 `tasks.md`。
2. Cursor 必须先执行 `git status --short`，记录已有 dirty files。不得回滚、覆盖或格式化无关改动，不得使用 `git reset --hard`、`git clean` 或 `git checkout --`。
3. 每张卡只修改卡片列出的文件。发现必须扩大范围时先停止并报告，不要自行扩大任务。
4. 每张卡都必须先添加回归测试，再实现代码，再运行验证命令。
5. GPU 测试必须在真实 wgpu 路径上执行；不能只用 host mock、只读 telemetry 或只检查返回计数来替代 GPU 行为验证。
6. Cursor 完成一张卡后不得继续下一张卡，必须输出：修改文件、测试命令和结果、未解决限制、建议下一张卡，然后停止。
7. 维护者审核重点是：正确性、GPU buffer 边界、异步生命周期、异常 step 的状态一致性、公共 API 兼容性、测试是否真的覆盖触发条件。审核未通过时只修当前卡，不进入下一卡。

## 顺序和依赖

| 顺序 | 任务卡 | 目标 | 依赖 |
| --- | --- | --- | --- |
| 1 | C1 | bounded map 越界与 WGSL 条件访问 | 现有 bounded forward/status 基线 |
| 2 | C2 | `Committed/Aborted` 训练 step 语义 | C1 的 device gate |
| 3 | C3 | prefix scan output 所有权、复用和分配证据 | C1、C2 |
| 4 | C4 | GPU completion timing 和 pipeline telemetry | C2、C3 的稳定 step 语义 |
| 5 | C5 | stable frame identity、canonical selection、split manifest | 与 C1-C4 可并行，建议先完成 C1-C3 |
| 6 | C6 | quaternion、sRGB、depth 契约 | C5 的 selection/report identity |
| 7 | C7 | rasterizer backward finite difference 和 SH schedule | C6 的数学契约 |
| 8 | C8 | 跨场景实验、comparator 和最终收口 | C1-C7 全部通过 |

每张卡的完成不等于整个 remediation 完成。只有 C8 的实验和全部验证通过后，才能在 `tasks.md` 中关闭整个计划。

---

## C1：修复 bounded forward 的 GPU 越界和不安全 WGSL 条件

### 给 Cursor 的任务提示

```text
只执行 C1，不要处理其他任务。请在 /Users/tfjiang/Projects/RustScan 修复 RustGS bounded forward 的真实 GPU 越界风险。

先阅读：AGENTS.md、docs/agent/rust-style.md、docs/agent/task-protocol.md、当前 change package 的 design.md/tasks.md。先运行 git status --short，保留所有已有用户改动。只允许修改以下范围：

- rustscan-gs/src/training/forward/tile_mapping.rs
- rustscan-gs/src/training/forward/mod.rs
- rustscan-gs/src/training/forward/dispatch.rs
- rustscan-gs/src/training/shaders/map_gaussian_to_intersects.wgsl
- rustscan-gs/src/training/shaders/write_dispatch.wgsl
- rustscan-gs/src/training/shaders/scan_block.wgsl
- rustscan-gs/src/training/shaders/radix_scatter.wgsl
- 与本回归测试直接相关的 rustscan-gs/tests/bounded_forward_parity.rs 或同目录 GPU test

问题背景：bounded 路径按 intersection_capacity 分配 tile_id_from_isect 和 compact_gid_from_isect，但 map shader 仍用真实 prefix sum 的 cum_tiles_hit 计算 isect_id，并直接写数组。真实 requested_intersections > capacity 时可能越界。另有 WGSL select 被误当作短路条件：map 的 cum_tiles_hit[compact_gid - 1u]、scan 的 input_values[idx]、radix 的 keys[idx] 都可能在尾部 lane 或首 lane 产生非法访问。

请按以下顺序执行：

1. 先写真实 GPU 回归测试：强制 capacity=1，构造至少两个会产生 intersection 的 visible Gaussian，在 wgpu validation 开启的路径执行。测试必须证明没有 validation/device error，overflow 被 sticky status 记录，后续 sort/raster 不使用越界输出，且调用方能得到 ForwardCapacityExceeded。不要只检查 host requested/capacity 数字。
2. 让 map shader 获得 output capacity（或等价的 device gate），每次写 tile_id_from_isect/compact_gid_from_isect 前显式检查 isect_id < capacity。检查必须发生在实际数组写入之前。
3. 保证 overflow 后下游 mapping、sort、tile offset、raster 的写入不会继续污染持久训练状态；保持现有 device-resident status gate 语义，不能增加普通 step 的 host readback。
4. 把 map、scan、radix 中依赖 select 的边界访问改成显式 if 分支，确保未命中的分支不会构造非法索引。覆盖输入长度 255、256、257，至少包含 scan 和 radix 的尾部 lane。
5. 保持 evaluation exact path 和正常 bounded path 的现有输出契约，不要借机重构 sorting 或 optimizer。

必须运行：

- cargo fmt --all -- --check
- cargo test -p rustscan-gs --lib --features gpu-wgpu
- cargo test -p rustscan-gs --test bounded_forward_parity --features gpu-wgpu
- 你新增的 overflow/boundary GPU 测试
- git diff --check

交付时只报告本卡：修改文件、每条命令的结果、使用的 GPU/backend、如果 validation 不可用的准确原因、以及仍存在的风险。不要开始 C2。
```

### 审核门

- capacity=1 的测试确实执行了 map shader，而不是提前在 host 返回错误。
- map shader 的 capacity 检查覆盖所有实际写入路径。
- `select` 已替换为显式分支，255/256/257 测试真实通过。
- overflow 时没有下游 buffer 污染，正常 exact/bounded parity 没有回归。
- 维护者独立运行 C1 命令并检查 diff 后，才能进入 C2。

---

## C2：修复 `read_loss=false` 的 committed/aborted step 语义

### 给 Cursor 的任务提示

```text
只执行 C2，不要处理其他任务。前提是 C1 已经由维护者审核通过。目标是修复 RustGS 在 read_loss=false 时把异常 step 误记为完成的问题。

先阅读 AGENTS.md、docs/agent/rust-style.md、docs/agent/task-protocol.md、change package 的 design.md/tasks.md，并运行 git status --short。只允许修改：

- rustscan-gs/src/training/engine/trainer.rs
- 训练错误/报告类型所在的 rustscan-gs/src/training/engine 或 reporting 文件（仅在引入状态类型确实需要时）
- rustscan-gs/tests/checkpoint_resume.rs
- 与 outer-loop 回归直接相关的 rustscan-gs 测试文件

当前问题：train_step(..., read_loss=false) 返回 Ok(None)，但 train_with_frame_loader 随后无条件 record_loop_step、record_completed_step，并可能触发 progress、snapshot、checkpoint。overflow/non-finite step 可能没有提交 optimizer mutation，却被记录为 completed_iterations。

请实现明确的 step 结果，例如 StepDisposition::Committed { loss: Option<f32> } 和 StepDisposition::Aborted { reason: ... }，名称可按现有错误体系调整，但不能继续用 Option 同时表达“没有读 loss”和“step 没提交”。

要求：

1. 只有确认 device mutation gate 允许提交、optimizer step 已完成、且没有 sticky overflow/non-finite 的 Committed step 才能更新 completed_iterations、final loss、progress、snapshot、checkpoint、topology report。
2. read_loss=false 只禁止 loss scalar readback，不能跳过异常状态判定。
3. overflow/non-finite 的 aborted step 必须保持 Gaussian 参数、Adam moment、Adam step、topology accumulator、visibility history 和 report completed_iterations 不变。
4. 保留现有安全点 status readback；不要为每个正常 step 增加 host readback。
5. 增加 outer-loop GPU 测试：overflow + read_loss=false、non-finite + read_loss=false、连续 overflow、正常 committed step、取消/暂停和 checkpoint 边界。测试必须检查 report 和 checkpoint 中的 completed_iterations，而不只是 train_step 的返回值。
6. 若已有 API 对外暴露 Option<f32>，使用最小兼容改造，并在边界处明确转换，不要无理由破坏公共 API。

必须运行：

- cargo fmt --all -- --check
- cargo test -p rustscan-gs --lib --features gpu-wgpu
- cargo test -p rustscan-gs --test checkpoint_resume --features gpu-wgpu
- 相关新增 outer-loop GPU 测试
- git diff --check

完成后停止，不要开始 C3。
```

### 审核门

- 测试失败时能证明 optimizer/topology 没有写入，而不是仅凭错误返回判断。
- `completed_iterations`、checkpoint、progress 和 snapshot 都只由 committed step 更新。
- read_loss=false 的正常 step 仍能继续训练，loss cadence 没有被破坏。
- checkpoint resume 与现有 53 个测试不回归。

---

## C3：完成 prefix scan output 复用和分配计数

### 给 Cursor 的任务提示

```text
只执行 C3，不要处理 profiler、split 或数学问题。前提是 C1、C2 已审核通过。

只允许修改：

- rustscan-gs/src/training/gpu_primitives/prefix_sum.rs
- rustscan-gs/src/training/forward/tile_mapping.rs（仅为显式 workspace 生命周期所需）
- rustscan-gs/src/training/forward/mod.rs（仅为显式 workspace 生命周期所需）
- rustscan-gs/src/training/engine/trainer.rs（仅接 telemetry）
- rustscan-gs/src/training/reporting/telemetry.rs
- prefix/radix/workspace 相关测试

目标：训练路径不能每次 scan 都重新分配最终 output；workspace 必须明确拥有 output 和递归 levels，并能证明稳态没有 fresh allocation。不得重新引入 thread_local、raw pointer、Arc<Mutex> 或跨 await 的悬空引用。

要求：

1. PrefixSumWorkspace 为每个递归 level 和最终 output 保存可复用 capacity；只有 len 超过 capacity 时增长。
2. 明确 input/output 不得共享 storage；reserve 或增长不能使当前 step 尚未消费的 tensor 失效。
3. inclusive_scan_into 返回的 output 在 forward 到 backward 最后一个 consumer 完成前保持有效；下一次 scan 不能覆盖仍在使用的结果。
4. step_fresh_allocations 必须包含 output allocation，并区分 workspace growth、scratch growth、output growth 和稳态分配。
5. 增加固定 shape 连续 100 step 测试：首次允许增长，之后 fresh allocation=0；短->长->短结果正确；growth count 单调；reported bytes 能由 capacity 复算。
6. 保留无 workspace convenience API 的独立 allocation 语义，避免 evaluation 路径行为改变。

必须运行：

- cargo fmt --all -- --check
- cargo test -p rustscan-gs --lib --features gpu-wgpu
- prefix/radix/workspace 相关测试
- cargo test -p rustscan-gs --test bounded_forward_parity --features gpu-wgpu
- git diff --check

输出分配不能以“无法测量”代替；如果某个 GPU backend 无法提供字节数，必须在 telemetry 中写 null 和结构化原因。完成后停止，不要开始 C4。
```

### 审核门

- 没有 TLS/raw pointer，且 output 生命周期经过真实 forward/backward 调用链。
- 100 step 测试测的是 GPU workspace，不是只测 Rust counter。
- output reuse 没有 alias 覆盖；短长短序列结果与 fresh path 一致。
- telemetry 中的 allocation 数字和实际 buffer growth 对得上。

---

## C4：实现 GPU completion profiler 和统一 pipeline timing

### 给 Cursor 的任务提示

```text
只执行 C4，不要修改训练数学或 split 逻辑。前提是 C2、C3 已审核通过。

允许修改/新增：

- rustscan-gs/src/training/reporting/gpu_profiler.rs（新文件）
- rustscan-gs/src/training/reporting/mod.rs
- rustscan-gs/src/training/reporting/telemetry.rs
- rustscan-gs/src/training/reporting/optimization_report.rs
- rustscan-gs/src/training/engine/runtime.rs
- rustscan-gs/src/training/engine/trainer.rs
- rustscan-gs/src/bin/rustgs/train_command.rs
- profiler/report JSON 测试 fixture

目标：报告必须区分 CPU submit/wall time 和真实 GPU completion time。不能把 into_scalar_async 包含的同步时间标成 GPU kernel 时间。

实现要求：

1. 定义可序列化 GpuProfilerReport，至少包含 supported、unsupported_reason、backend、adapter、driver、GPU step p50/p95、sample_count、workspace current/peak bytes、growth count、fresh step allocations、runtime peak device bytes 及不可用原因。
2. 如果 wgpu timestamp query 可用，按 query resolve/readback 完成后的 GPU timestamp 计算 completion interval；不得用 host Instant 伪装 GPU 时间。
3. timestamp query 不可用时，GPU percentile 必须为 null，unsupported_reason 必须是结构化稳定字符串；CPU 时间使用明确字段名。
4. 真实采集 adapter/backend/driver metadata；拿不到就 null+reason，删除固定伪造的 adapter_name=None/driver=None，除非底层确实返回 unavailable。
5. 为 frame wait、decode、resize、upload、forward、backward、optimizer、topology snapshot/plan/apply/upload/remap 定义统一 span 名称、sample count 和 warmup/drop policy。
6. p50/p95 计算要有单元测试，验证 p95>=p50、sample_count 一致、unsupported 字段自洽。
7. 100-step smoke 验证 sample count、workspace growth 和 allocation 统计不会回退或重复累计。

必须运行：

- cargo fmt --all -- --check
- cargo test -p rustscan-gs --lib --features gpu-wgpu
- profiler/report JSON 单元测试
- 真实 GPU 100-step smoke（若设备不支持 timestamp，保存 unsupported reason）
- git diff --check

完成后停止，不要开始 C5。
```

### 审核门

- report 能从字段上区分 CPU 和 GPU 时间。
- timestamp 不可用时没有伪造的 GPU p50/p95。
- adapter、driver、backend 信息来自运行时，而非硬编码。
- span 不重复累计，父级总时间与子级时间关系有明确语义。

---

## C5：统一 stable frame identity、canonical selection 和 split manifest

### 给 Cursor 的任务提示

```text
只执行 C5，不要修改 rasterizer 或 optimizer。先阅读 AGENTS.md、rust-style.md、task-protocol.md 和现有 evaluation/report 代码。允许修改：

- rustscan-gs/src/io/colmap_dataset.rs
- rustscan-gs/src/bin/rustgs/train_command.rs
- rustscan-gs/src/training/evaluation/core.rs
- rustscan-gs/src/training/evaluation/mod.rs
- rustscan-gs/src/training/evaluation/split.rs（新文件）
- rustscan-gs/src/training/reporting/optimization_report.rs
- comparator 相关 Rust 文件
- split/selection/report 测试

目标：同一 COLMAP image 在 max_frames、stride、缺失图片变化后仍拥有同一个 stable frame identity；train/eval/crop/report 必须使用同一个 canonical selection；holdout 不能和 train 重叠。

要求：

1. `ScenePose.frame_id` 使用 `image.image_id as u64`，不要使用过滤后重新 enumerate 的 frame_idx。需要 vector index 时单独表达 dataset index。
2. 清理 report/eval 中 frame_id 的 u64->u32 截断，或在明确兼容边界的地方拒绝超范围值。
3. 新增 `FrameSelection` 或等价结构，统一处理 stable IDs、include/exclude、max_frames、stride；selection 只能构造一次，eval、crop、report 复用同一结果。
4. 明确并测试选择顺序：manifest/include/exclude 先按 stable ID 解析，再执行 max/stride，最后生成 selection fingerprint。
5. 支持 `--frame-split-manifest` 和 `--eval-split in-view|holdout`。manifest 至少保存 dataset fingerprint、train IDs、in-view IDs、holdout IDs。
6. 启动时拒绝重复 ID、未知 ID、train/holdout 交叉、dataset fingerprint mismatch；holdout 缺 manifest 时直接报错。
7. report/checkpoint 保存 split kind、完整 stable IDs、manifest fingerprint 和 selection fingerprint。in-view 评估必须显式标记，不能假装 holdout。
8. 添加 fixture：空 split、单帧 split、重复 ID、未知 ID、fingerprint mismatch、重复运行 selection 稳定性、include 一个原始排序靠后的 frame ID。

必须运行：

- cargo fmt --all -- --check
- cargo test -p rustscan-gs --lib --features gpu-wgpu
- split/selection/comparator 相关测试
- cargo test -p rustscan-gs --test checkpoint_resume --features gpu-wgpu
- git diff --check

完成后停止，不要开始 C6。
```

### 审核门

- frame ID 来自 COLMAP image_id，缺失图像不会重编号已有帧。
- include/max/stride 的顺序由一个实现负责，不存在 eval/crop 第二套选择逻辑。
- holdout 与 train 集合按 stable ID 校验不相交。
- report 能复现实际评估帧，而不是只记录数量。

---

## C6：补齐 quaternion、sRGB resize 和 depth 契约

### 给 Cursor 的任务提示

```text
只执行 C6。前提是 C5 已由维护者审核通过。此卡只处理三个数据/数学边界，不做 rasterizer 全量 finite difference 和 SH schedule。

允许修改：

- rustscan-gs/src/training/shaders/project_visible.wgsl
- rustscan-gs/src/training/shaders/project_forward.wgsl
- rustscan-gs/src/training/shaders/project_backwards.wgsl
- rustscan-gs/src/training/evaluation/core.rs
- rustscan-gs/src/training/data/frame_loader.rs
- depth/config/checkpoint/report 中实际承载契约的 Rust 文件
- 相关 GPU/unit tests

Quaternion 要求：统一 visible、forward、backward 的小范数 epsilon guard；零/近零 quaternion 不能产生 NaN；无效 quaternion 在 forward/backward 的 skip 或梯度策略一致；新增真实 GPU 测试。有限差分使用归一化 quaternion 的切空间 perturbation。

sRGB 要求：先确认训练 target 和 evaluation target 的色彩空间约定，并在代码中明确写出。若输入是 sRGB，resize 流程必须是 sRGB->linear->resize->metric；不得继续在 /255 后的 sRGB 数值空间直接 box average。增加 checkerboard、高对比边缘、1x 和非整数缩放测试。

Depth 要求：明确 depth format、unit、scale、invalid value policy、zero semantics；raw little-endian f32 与 Luma16 的单位不能靠隐式假设。启动时校验 depth contract，checkpoint/report 保存解析后的 contract。增加 raw f32、Luma16、无效值、零值和错误 scale 测试。

必须运行：

- cargo fmt --all -- --check
- cargo test -p rustscan-gs --lib --features gpu-wgpu
- quaternion GPU tests
- sRGB/depth unit tests
- cargo test -p rustscan-gs --test bounded_forward_parity --features gpu-wgpu
- git diff --check

如果现有数据集的色彩或深度约定无法确定，先实现显式配置和错误提示，不要猜一个默认值掩盖不确定性。完成后停止，不要开始 C7。
```

### 审核门

- forward、visible、backward 使用同一 quaternion guard。
- sRGB 测试能区分 linear resize 和 gamma-space resize。
- raw depth/Luma16 的单位和 invalid policy 在 report 中可见。
- 未知或矛盾契约会失败，而不是静默产生错误尺度。

---

## C7：增加 rasterizer backward finite difference 和可配置 SH schedule

### 给 Cursor 的任务提示

```text
只执行 C7。前提是 C6 已审核通过。允许修改：

- rustscan-gs/src/training/backward/gradient_check.rs
- rustscan-gs/src/training/shaders/rasterize_backwards.wgsl（只有测试证明数学实现确实错误时才改）
- rustscan-gs/tests/gradient_boundary.rs（新文件）
- rustscan-gs/src/training/config.rs
- rustscan-gs/src/training/engine/trainer.rs
- rustscan-gs/src/training/checkpoint.rs
- comparator/report fingerprint 相关文件

第一部分是 rasterizer backward finite difference：现有 gradient_check 主要覆盖 projection VJP，需要新增端到端 rasterizer 测试，至少覆盖单 Gaussian、两 Gaussian 遮挡、background、alpha/transmittance、early termination、depth、covariance blur、SH degree 0 和高 degree。测试输出 case、parameter、analytic、numeric、absolute/relative error。连续区域使用 centered FD，不连续边界使用单侧 FD 或避开边界；不能通过全局放宽容差让测试通过。

第二部分是 SH schedule：新增可序列化配置：

```rust
struct ShScheduleConfig {
    initial_degree: u8,
    increment_every: u32,
    max_degree: u8,
}
```

默认语义必须保持每 1000 iteration 升一级。验证 iteration 0、1、999、1000、1001 和 checkpoint resume 边界；拒绝 increment_every=0、degree>3、initial>max；checkpoint 保存 schedule；canonical fingerprint 和 comparator 包含 schedule；resume 遇到 schedule mismatch 必须拒绝。

必须运行：

- cargo fmt --all -- --check
- cargo test -p rustscan-gs --lib --features gpu-wgpu
- cargo test -p rustscan-gs --test gradient_boundary --features gpu-wgpu
- cargo test -p rustscan-gs --test checkpoint_resume --features gpu-wgpu
- comparator/config/checkpoint tests
- git diff --check

完成后停止，不要开始 C8。
```

### 审核门

- finite difference 覆盖实际 rasterizer compositing，而不是复用 projection-only helper。
- 失败输出足以定位具体 Gaussian、参数和边界。
- SH 默认行为与旧 checkpoint 兼容，非默认 schedule 能被保存、恢复和比较。

---

## C8：跨场景实验、comparator 和最终收口

### 给 Cursor 的任务提示

```text
只执行 C8。只有 C1-C7 都由维护者审核通过后才能执行。此卡以实验和验证为主，除非实验暴露明确 bug，不再进行新的架构重构。

先运行 git status --short，确认没有把用户已有改动混入实验。所有产物必须写入：

artifacts/runs/rustgs-optimization/2026-09-18-remediation/

目录至少包括 bin、manifests、home、flowers2、tum。不要创建顶层 output、test_data 或 experiments。

实验步骤：

1. baseline 和 candidate 各构建一次 release binary；记录 git revision、binary sha256、Cargo features、adapter/backend/driver、command 和环境信息；实验期间禁止重建 binary。
2. 按 Home 500、flowers2 500、TUM 500、TUM 3k、TUM 10k、TUM 30k 执行固定 manifest 的训练和评估。baseline/candidate 正式结果各至少 3 次，首轮 warmup 不计入统计。
3. 保存 split manifest、optimization JSON、evaluation JSON、stdout/stderr、PLY 和 timing/allocation evidence；汇总 mean/stddev/p50/p95。
4. comparator 必须拒绝 binary revision、dataset fingerprint、split、seed、分辨率、iteration、loss/topology/SH schedule 或 GPU environment 不一致的报告；不可测字段必须标明原因。
5. 任一 overflow、non-finite、invalid dispatch、checkpoint failure、resume fingerprint mismatch、**超出 C2 安全点契约的逐步 disposition status readback**、workspace 持续增长、holdout mean 下降超过 0.10 dB、worst frame 下降超过 0.20 dB 或新增 fog/ghosting/floaters，都判定 candidate 失败。baseline/candidate 各自绑定自己的 binary revision/hash；不可把“binary revision 必须相同”当成跨版本 A/B 的硬门槛。
6. 先证明 correctness 和 quality，再在同一 GPU/driver 上比较 steps/s 和 p95；不能用不同环境的数字宣称性能提升。
7. 如果 TUM 数据不可用，保留 blocked reason、数据路径和尝试过的命令，不得伪造结果；完成可运行的 Home/flowers2 和全部代码验证。

必须运行并保存结果：

- cargo fmt --all -- --check
- cargo check --workspace --all-targets
- cargo test -p rustscan-gs --all-targets --features gpu-wgpu
- cargo test -p rustscan-gs --test bounded_forward_parity --features gpu-wgpu
- cargo test -p rustscan-gs --test checkpoint_resume --features gpu-wgpu
- cargo test -p rustscan-gs --test gradient_boundary --features gpu-wgpu
- cargo clippy -p rustscan-gs --all-targets --features gpu-wgpu -- -D warnings
- git diff --check
- git status --short

完成后停止并提交最终 handoff，不要自行修改 tasks.md 的已完成状态；由维护者审核实验产物和最终 diff 后更新状态。
```

### 审核门

- baseline/candidate 使用固定 binary 和同一 GPU/driver。
- 每次结果都能由 manifest、report、binary hash 和 command 复现。
- comparator 没有把不同 split 或不同环境误判为可比较。
- 所有失败门槛都有数字证据，TUM 缺失时有明确阻断记录。

---

## 维护者每张卡的审核流程

Cursor 停止后，维护者按以下顺序审核，不通过时只回发当前卡的修订要求：

1. 查看 `git diff --stat`、`git diff --check` 和完整 diff，确认没有越界文件。
2. 对照本卡“审核门”逐条检查；重点看测试是否真正触发 GPU/异步/异常路径。
3. 独立重新运行本卡命令，不把 Cursor 的文字报告当作测试证据。
4. 运行本卡要求之外的最小回归测试，确认已有 RustGS baseline 没有回退。
5. 检查 handoff 是否写清 changed files、base/final commit、命令结果、环境前提、限制和下一步。
6. 审核通过后，在当前 change package 的 `tasks.md` 或 `verification.md` 记录证据，再把下一张卡的完整提示发送给 Cursor。

## 任务卡完成状态

Fix branch base: `701d051`. C1–C4 code is on `fix/rs-2026-002-c1-c4-review` pending maintainer review; C5–C8 not started.

- [ ] C1 bounded map 边界和 WGSL 显式条件 *(implemented on fix branch; maintainer review)*
- [ ] C2 committed/aborted step 语义 *(fa970ed; maintainer review)*
- [ ] C3 scan output 复用和分配证据 *(implemented on fix branch; maintainer review)*
- [ ] C4 GPU profiler 和 pipeline timing *(ae55d66 + measurement fix; maintainer review)*
- [ ] C5 stable frame identity、selection、split manifest
- [ ] C6 quaternion、sRGB、depth 契约
- [ ] C7 rasterizer finite difference、SH schedule
- [ ] C8 跨场景实验和最终收口

状态只能由维护者在审核和独立验证完成后更新。Cursor 不得自行勾选，也不得因为单元测试通过就宣称整个 RustGS remediation 完成。

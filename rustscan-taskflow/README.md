# rustscan-taskflow

独立的、进程内资源感知 CPU/GPU DAG 调度 crate。多个 workflow 共用一个 `Runtime`，而不是各自创建不受约束的执行器。当前提供可运行的调度核心、异构执行示例和测试；RustSFM 已提供显式启用的 CPU 特征提取/数据库写入和 Ceres BA 接入（见 `RustSFM/README.md`）。**尚未接管 RustViewer 或整个 SfM pipeline，也不代表已经提升了 SfM 性能**。

## 快速使用

```rust
use rustscan_taskflow::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = Runtime::new(RuntimeConfig {
        budget: Budget {
            cpu_threads: 4,
            memory_bytes: 512 * 1024 * 1024,
            io_slots: 2,
        },
        ..RuntimeConfig::default()
    })?;
    let mut graph = TaskGraph::new();
    let mut request = ResourceRequest::cpu(CpuRequest::fixed(1));
    request.output_memory_bytes = 3 * 8;
    let input = graph.task("load", vec![TaskVariant::cpu("cpu", request, |_| {
        Ok(vec![1u64, 2, 3])
    })])?;
    let input_id = input.id();
    let mut request = ResourceRequest::cpu(CpuRequest::scalable(1, 2, 4));
    request.output_memory_bytes = 8;
    let sum = graph.task("sum", vec![TaskVariant::cpu("cpu", request, move |ctx| {
        use rayon::prelude::*;
        let values = ctx.input(&input)?;
        ctx.parallel(|| values.par_iter().copied().sum::<u64>())
    })])?;
    graph.depends_on(sum.id(), input_id)?;
    let run = runtime.submit(graph)?;
    let report = run.wait(); // GUI/UI 线程请使用非阻塞查询或其他等待线程。
    assert!(report.succeeded(), "{report:?}");
    assert_eq!(*run.output(&sum)?, 6);
    drop(sum); // 不再需要的 handle 应及时释放，归还输出内存额度。
    runtime.shutdown();
    Ok(())
}
```

`TaskGraph` 是一次性静态 DAG；提交后不能再添加节点。`depends_on(child, parent)` 声明先后顺序；仅捕获一个 handle **不会**自动建立依赖。`ctx.input()` 只允许读取显式声明的直接依赖，禁止跨图读取。循环、外来节点、非法请求以及在初始配置下无任何可行实现的任务会在执行前被拒绝。

## 执行模型

- 一个 scheduler 线程维护所有 workflow 的 ready queues 和全局资源账本；任务按 workflow 轮转准入。真正计算在 Rayon dispatch workers 或外部 GPU/backend 上执行，**不是单线程计算**。
- 任务所需的 CPU、host memory、GPU memory / in-flight、I/O slots、exclusive keys 一次性检查和获取；不能先占一部分资源再等待另一部分。
- 一个节点可提供多个返回同一类型的 `TaskVariant`，按声明顺序选择第一个当前可行的实现。例如 GPU 繁忙时立即选 CPU fallback。**这是可行性选择，不是耗时预测，也不会在执行失败后自动尝试下一种实现**。
- `CpuRequest::scalable(min, preferred, max)` 在空闲额度不足 preferred 时降到不少于 min；目前不会主动分配超过 preferred 的线程。运行期间不动态扩缩。
- `max_bypasses` 限制 CPU 大任务被小任务超越的次数；只有其他资源已满足时才为它保留 CPU 准入机会，避免挡住释放内存的消费者。非 CPU 资源没有严格无饥饿保证。
- `max_pending_tasks` 限制所有 workflow 的未完成任务总数，超出时提交返回 `Error::QueueFull`；不是整个进程内存或所有元数据的硬上限。图结束前仍保留其已完成任务报告。

## 资源与生命周期约定

| 资源 | 配置 / 请求 | 归还时机 |
| --- | --- | --- |
| CPU | `Budget::cpu_threads` / `CpuRequest` | CPU/提交闭包及其 scoped compute pool 结束 |
| Host working memory | `working_memory_bytes` | 提交闭包和异步完成均结束 |
| 输出内存 | `output_memory_bytes` | 所有对应 `TaskHandle` / `Artifact` 释放 |
| GPU | `GpuCapacity` / `GpuRequest` | working memory / in-flight 在真正完成后归还，输出内存跟随 artifact |
| I/O | `io_slots` | 提交闭包和异步完成均结束 |
| 独占应用资源 | `exclusive: Vec<String>` | 提交闭包和异步完成均结束 |

`GpuRequest::device = None` 可选择任一有额度的已配置设备；指定 ID 则只能用该设备。Apple 统一内存设备应设置 `shared_host_memory = true`，其 GPU 内存额外计入 host budget；请求中的 host bytes 应填写独立的 host 分配，避免对同一份共享分配重复申报。

内存额度是**调用方估算的协作式准入控制**，不是 allocator/显存硬隔离，不保证不会 OOM。申请应包含工作集和输出，输入由已有 artifact 的 lease 计费。对于内部共享指针、外部缓存等逃逸生命周期，调用方必须自行保证申报和存活期一致。

输出直到最后一个 handle/reader 释放才归还额度，workflow 完成不等于输出释放。及时将输入 handle 移入消费者、释放不再需要的克隆。提交时会做保守的 DAG 内存存活检查：显式依赖上的前置输出按可能继续存活计入消费者的 working/output 请求；无法满足的图返回 `Error::Unschedulable`，而不是接受后永久等待。该检查可能拒绝依赖分支共享释放协议的图；此版本仍不做全图精确内存证明、磁盘溢写或自动分批。跨 workflow 的 output lease 仍按运行时 backpressure 处理，直到外部 handle/reader 释放。

### CPU 并行

`ctx.parallel()` 按任务 grant 懒创建独立 Rayon pool，dispatch worker 等待其完成。多个任务的**已准入计算额度**不超过全局 CPU budget，但 OS 线程总数可能更多（scheduler、等待中的 dispatch workers、外部 backend、短暂线程池清理）。短小任务直接执行，避免为微任务创建 pool。

Ceres / BLAS / OpenMP 等必须由适配层显式将线程数设为 `ctx.grant().cpu_threads`。不要使用 Rayon 全局池、逃逸的 detached work，或自行启动额外计算线程绕过额度。运行在同一进程内但未接入本 runtime 的计算不受它控制。独立进程中的多个 runtime 也不共享资源账本。

### 异步完成与取消

`TaskVariant::asynchronous` 接收 `Completion<T>`：提交闭包应尽快返回，将 token 移入实际完成回调。返回后可以释放 CPU，但 GPU、I/O、独占和工作内存额度继续占用。即使先收到 completion，也要等提交闭包退出后才发布结果和触发下游。

**只有 backend/硬件确实不再访问任务资源时，才可以调用 `complete()` 或丢弃 token。** Drop token 会报告 `CompletionDropped`，不会取消硬件。提交后发生 panic 且 token 已移交外部时，仍需等待 token 结束。

`run.cancel()` 取消未开始节点，运行中任务通过 `ctx.check_cancelled()` / `ctx.cancellation()` 协作退出；不会抢占 CPU、强停 GPU，也不会提前回收它们的资源。取消后的结果不再发布。普通失败/panic 只阻止其下游，独立分支继续执行。

`Runtime::shutdown()` 和 Drop 会取消并等待全部运行任务。**不协作或永不归还 token 的 backend 可能导致无限等待；不要在任务自身或负责 GPU polling 的 UI 线程上销毁 runtime。** 应先停止提交、取消任务、继续驱动 backend 完成，然后在合适的线程关闭 runtime。

## 系统压力自适应

默认不启动采样器。启用 `system-monitor` 后：

```rust
# #[cfg(feature = "system-monitor")]
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use rustscan_taskflow::*;
use std::time::Duration;

let runtime = Runtime::new(RuntimeConfig::default())?;
let monitor = SystemMonitor::start(&runtime, Duration::from_millis(500), 256 * 1024 * 1024)?;
// 在此期间提交 workflow；退出时先停采样器，再关闭 runtime。
drop(monitor);
runtime.shutdown();
# Ok(())
# }
# #[cfg(not(feature = "system-monitor"))]
# fn main() {}
```

采样间隔至少 250ms；根据系统 CPU 使用率减去本进程 CPU 使用率估算外部竞争，结合可用内存和预留 headroom 调整预算。CPU 下降每次一步，连续四个恢复采样才上升一步，减少振荡。系统采样有延迟且是估算，不能准确感知同进程内未接管的其他计算。

也可自行向 `AdaptivePolicy::update()` 输入 `PressureSample`，再调用 `Runtime::set_budget()`。不要把包含自身有效计算的总 CPU 使用率直接当外部压力，否则会错误降速。每个 runtime 应只有一个预算控制器，不要让手工调节与多个 monitor 相互覆盖。

动态 budget 只能不超过初始配置，作用于**新任务准入**，不缩减已有任务；降低后 snapshot 中的使用量可暂时超过新预算。手动将 CPU 设为 0 可暂停新计算，恢复需重新设置。GPU 容量/in-flight 当前为静态配置；**未实现自动 GPU 利用率/显存压力采样或根据 GPU 压力动态调整配额**。

## 接入现有 wgpu

启用 `wgpu-backend`，使用 `WgpuBackend::new(id, Arc<Device>, Arc<Queue>)` 包装已有 device/queue。ID 必须匹配 `RuntimeConfig::gpus`，queue 必须来自同一 device。适配器不创建第二个 GPU device 或自己的 polling 线程。

在 asynchronous 闭包中按 grant 分配/编码，然后调用 `backend.submit(ctx, command_buffers, output, done)`。宿主必须继续 `device.poll(...)` 驱动完成回调；共享队列上的其他工作可能保守地延迟完成通知。

Queue completion **不等于** validation/device-loss 错误检查。应用需处理 wgpu 错误；复杂错误传播或 readback 可直接使用 `TaskVariant::asynchronous` 自定义适配器。真实计算、GPU readback、依赖发布示例见 `tests/wgpu.rs`。

## 观测与验证

- `Runtime::snapshot()`：预算、已占资源和未完成任务数。
- `RunHandle::try_event()`：提交、开始、CPU 提交结束、完成事件；有界队列满时丢弃观测事件，不丢 completion。
- `RunReport`：每节点状态、错误、实际 grant、`queue_time`（从 workflow 创建到开始的总等待）、`dependency_wait_time`（等待依赖完成）和 `resource_wait_time`（ready 后等待资源准入）、执行时间、丢失事件数。

在 workspace 根目录运行，全部使用 **release**：

```sh
cargo test -p rustscan-taskflow --release --offline
cargo test -p rustscan-taskflow --release --all-features --offline
cargo clippy -p rustscan-taskflow --release --all-features --all-targets --offline -- -D warnings
cargo run -p rustscan-taskflow --release --example heterogeneous --offline
cargo run -p rustscan-taskflow --release --example scheduler_bench --offline
cargo doc -p rustscan-taskflow --release --all-features --no-deps --offline
cargo fmt -p rustscan-taskflow --check
```

`heterogeneous` 用外部线程模拟 GPU 生命周期，**不是 GPU 性能测试**。`scheduler_bench` 测量空任务 DAG 的调度开销，不代表 SfM 提速。

真实 GPU 测试默认 ignored，必须显式执行（需要可访问的真实 wgpu adapter；沙箱可能屏蔽 Metal，无 adapter 会失败而非假装通过）：

```sh
cargo test -p rustscan-taskflow --release --features wgpu-backend --test wgpu --offline -- --ignored --nocapture
```

该测试实际执行 WGSL kernel、复制结果、异步 readback 并逐项验证值。`--offline` 需要依赖已缓存；首次准备环境时可去掉该参数。

## 当前边界

这是静态、进程内调度基础层，不是分布式工作流引擎。暂不包含动态图、持久化恢复、自动重试、强制抢占、跨进程资源仲裁、GPU kernel 自动生成或性能模型。当前 RustSFM 已接入 CPU 特征提取、有序写入和 Ceres BA 的线程/临时内存准入；GPU SIFT、其他未接管阶段以及底层 BLAS/OpenMP 线程仍需宿主单独协调。后续仍需在实际 pipeline 上比较吞吐、尾延迟和 UI 帧时间，不应将局部接入等同于整个进程已受控。

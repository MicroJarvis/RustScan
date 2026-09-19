# RustSFM 后续 ToDo 与技术方案

**更新日期：** 2026-09-19
**范围：** RustSFM 非 GUI、非 Python 方向
**执行原则：** 只保留未完成事项。严格按优先级执行；当前级别通过验收后再进入下一级。

## P0：验证当前代码状态

先证明当前工作树状态通过完整门禁，再继续修改 mapper 行为。

### ToDo

- [ ] 记录 `HEAD`、未提交 diff 指纹、平台、Rust 版本和启用的 features。
- [ ] 运行 no-default 编译门禁。
- [ ] 运行 GPU library、sequence 和 ignored sparse fixture 测试。
- [ ] 将同一次运行的结果同步到 `docs/current-project-status.md` 和 `RustSFM/PARITY_ROADMAP.md`，删除互相冲突的旧测试数字。

### 验收命令

```bash
cargo fmt --package rustsfm --check
cargo check -p rustsfm --release --no-default-features --all-targets
cargo test -p rustsfm --lib --features gpu-wgpu,vlfeat-sift -- --test-threads=1
cargo test -p rustsfm --test sequence_registration --features gpu-wgpu,vlfeat-sift -- --test-threads=1
./scripts/provision_flowers2_colmap_fixture.sh
cargo test -p rustsfm --lib --features gpu-wgpu,vlfeat-sift -- --ignored --test-threads=1
```

### 完成标准

- 所有命令退出码为 0。
- 报告包含通过数、失败数、ignored 数、fixture hash、feature、平台、`HEAD` 和 diff 指纹。
- 文档只保留这次运行的当前基线；旧状态移到 Git 历史，不继续累积在 ToDo 中。

## P1：关闭 24 图 mapper 注册顺序分叉

当前固定 replay 的第一个可复现分叉点位于 `frame_0017.jpg` 注册后的三角化与过滤：RustSFM 保留过多 point/observation，使模型 0 继续增长；COLMAP 则结束 3 图模型并创建第二个模型。后续 registration order 的差异应从 track/visibility 状态解决，不修改 `FindNext` 排序器或放宽 PnP/init 阈值。

### P1.1 固化可机器比较的 golden trace

- [ ] 固定 COLMAP `3.13.0`、数据库 SHA-256 `21b74e7302acf6f79364eba3ece36530cfdc24067b78eee648c3bfdc304c16f0`、seed `0`、单线程和相同 mapper 配置。
- [ ] 把 COLMAP 每个注册步骤转换成结构化 golden：model index、initial pair、registered image、points、observations、候选 visibility score、Create/Continue/Merge/Complete 数量、各过滤阶段移除数量和 BA 事件。
- [ ] RustSFM 输出相同字段，比较工具按阶段报告第一个分叉点。

COLMAP 默认日志没有逐候选 rejection reason，因此它不作为跨实现 parity 指标。RustSFM 仍需保留结构化 rejection reason，用于自身回归和失败诊断。

### P1.2 对齐 `frame_0017.jpg` 后的 track/visibility 状态

按以下顺序调查，每一步先记录输入输出数量，再决定是否修改实现：

1. 对照 `TriangulateImage(frame_0017)` 的 correspondence 集合、two-view observation 判定和三角化估计结果。
2. 对照 Create/Continue/Merge/Complete 的 track 生命周期，确认同一 feature 是否被重复建点或错误延长 track。
3. 对照负深度、重投影误差、三角化角和 two-view track 过滤；分别记录移除 observation 和删除 point 的数量。
4. 对照 local BA 后 `FilterPoints3DInImages` 的图像集合，以及 global BA 前后的过滤顺序。
5. 每次改动后重新检查 visibility pyramid score；只有输入 point/observation 已一致但排序仍不同，才检查 `FindNext` 实现。

### 必须补充的回归测试

- [ ] two-view observation 不应在不满足 COLMAP 条件时创建独立 point。
- [ ] local BA 后按本地参与图像集合过滤，而不是只处理 modified points。
- [ ] global BA 前清理负深度 observation，并保持 track/observation 双向一致。
- [ ] rollback 或过滤后，增量状态与从 reconstruction fresh rebuild 的状态一致。
- [ ] 固定 replay 能定位并断言首个分叉阶段。

### P1 完成标准

固定 replay 必须同时满足：

- 数据库 pair 为 `89`，其中 `CALIBRATED=87`、`UNCALIBRATED=2`。
- model 0 initial pair 为 image `#2/#12`，下一张为 `#16`，最终保留 3 张图。
- model 1 initial pair 为 image `#13/#10`。
- 24 张图全部注册，模型数量为 2。
- 每个模型的完整 registration order 与 golden trace 一致；不使用模糊的“预设 parity”表述。
- 每个注册步骤的 points、observations 和候选 visibility score 与 golden 一致；若浮点边界导致差异，必须在报告中给出字段级容差及依据。
- RustSFM 的全部候选失败和放宽重试路径仍保留 stage、candidate 和 rejection reason。

## P2：建立唯一的 24 图稀疏 SfM 基线

P1 完成后，用最终代码重新生成一次可引用的 24 图报告。此前生成的中间 replay 和测试计数不再作为当前结论。

### ToDo

- [ ] 运行 P0 的全部测试门禁。
- [ ] 运行固定 24 图 replay，并保存机器可读 parity 报告。
- [ ] 记录 registration order、models、points、observations、track histogram、平均重投影误差、BA 事件和总耗时。
- [ ] 更新 `docs/current-project-status.md` 与 `RustSFM/PARITY_ROADMAP.md`，两处引用同一个报告和同一组数字。

### 完成标准

- P0 门禁全部通过。
- P1 的 fixed replay 条件全部满足。
- 报告能够由其中记录的命令、输入 hash、feature、seed 和线程数复现。

## P3：执行 960 图产品验收

24 图 parity 通过后，再验证近期产品目标。测试 suite 或 24 图 fixture 不能替代 960 图完整重建。

### 固定输入与运行方式

- 输入图像：`test_data/flowers2/images` 的 960 张 `frame_*.jpg`。
- 使用冻结的 matching database；报告记录数据库路径和 SHA-256。
- release 构建，固定 seed、线程数、feature 和 mapper 配置。
- 使用 `/usr/bin/time` 或等价工具记录 wall time 与峰值 RSS；summary 记录 GPU wait 和 BA solve time。

### 质量门禁

- [ ] 注册 `960/960`。
- [ ] 只产生 1 个模型。
- [ ] 所有已注册 pose、point 和导出值有限且可被 COLMAP sparse reader 重新加载。
- [ ] points、observations、track histogram 和平均重投影误差相对冻结基线无未解释退化。
- [ ] pairs digest、verified pair 数、matches digest 与冻结 matching database 一致。
- [ ] wall time、峰值 RSS、GPU wait 和 BA solve time 已记录；超出冻结基线范围时给出原因，不仅报告最终注册数。

### 完成标准

生成一个新的 960 图验收报告，至少包含命令、`HEAD`、diff 指纹、输入 hash、配置、质量指标和资源指标。只有该报告通过后才能声称“稳定的稀疏 SfM 产品目标已达到”。

## P4：收口 sparse mapper 与 BA parity

P1–P3 完成后按以下顺序推进。

### P4.1 Track/observation 生命周期

- [ ] 对 Create/Continue/Complete/Merge/Retriangulate 记录输入 track、inlier mask、过滤原因和输出 track。
- [ ] 对齐 split gate、two-view suppression、角度/重投影/负深度过滤。
- [ ] 覆盖 registration rollback、local BA failure、bogus camera rollback 和 registered-frame filtering。
- [ ] 验证 reconstruction、observation manager、trial counters 和 registration counters 在 rollback 后一致。

### P4.2 BA 调度与数值边界

- [ ] 将 initial、scheduled、final-all BA 的触发条件、轮数、迭代上限、normalization 和 post-processing 顺序固化为状态表。
- [ ] 为每次 BA 输出 reason、变量/常量图像、point/residual 数量、solver、迭代数和过滤统计。
- [ ] 对齐 COLMAP final global refinement 的 image/point selection 与调度。
- [ ] 盘点 pose、point、keypoint 和 camera 参数的 `f32`/`f64` 边界，再决定是否迁移核心 storage。
- [ ] 只在真实 large fixture 证明现有后端是瓶颈后评估 Eigen/CHOLMOD sparse backend。

### P4.3 Rig、frame 与 pose prior

1. generalized absolute pose 的 camera reset/refinement schedule；
2. frame-level retry bookkeeping 和 deregistration counters；
3. pose prior 从数据库到 mapper/BA 的完整生命周期；
4. rig/sensor refinement 和 covariance；
5. shared camera grouping/reset 的 image-only 与 database 路径。

每项必须有真实 sparse-text 或合成 rig fixture，不能只依赖 solver 单元测试。

## P5：匹配、检索与性能

### 匹配 parity

- [ ] fixed descriptor fixture 覆盖 uint8 distance、ratio、cross-check 和 max matches。
- [ ] guided matching 覆盖端到端数据库写回。
- [ ] 所有优化使用 pairs digest、verified pair 数、matches digest 和 reconstruction registration 作为质量门禁。

### Retrieval 决策

- [ ] 明确产品是否需要大场景检索。
- [ ] 中小序列路线：保留 vocab tree，补召回率和耗时基准。
- [ ] 大场景路线：单独立项 FAISS-compatible index，以固定 descriptor corpus 验证召回率和 pair 数。

在 mapper parity 未稳定前，不实施会改变候选顺序、RANSAC 调度或重建质量的性能优化。

## P6：长期 COLMAP parity

以下事项不阻塞近期稀疏 SfM 产品目标，但在声称完整 COLMAP parity 前必须完成或明确移出范围：

- [ ] 完整 scene/sensor、reconstruction manager、clustering/pruning 和 camera semantics。
- [ ] 完整 controller、CLI 工具和 callback orchestration。
- [ ] FAISS VisualIndex、Hamming embedding 和 vote-and-verify。
- [ ] dense workspace、undistortion、PatchMatch、fusion 和 meshing。
- [ ] 完整 pose prior、rig refinement、covariance 和跨平台数值 parity。
- [ ] 废弃或替换未通过真实数据验证的 `pose_graph.rs` heuristic。

## 统一报告要求

所有后续报告必须包含：

- `HEAD` 与未提交 diff 指纹；
- 平台、Rust/COLMAP 版本和 features；
- 输入路径、fixture/database hash、seed 和线程数；
- 完整命令与退出码；
- registration、models、points、observations、track、误差和 BA 指标；
- wall time、峰值 RSS、GPU wait 和 BA solve time；
- 与 golden 的首个分叉点及字段级差异。

没有上述证据时，只能标记“已实现，待验收”，不能标记完成。

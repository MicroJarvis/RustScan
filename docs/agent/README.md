# RustScan 通用 Agent 文档索引

本目录是 RustScan 的模型和工具无关协作入口，只依赖 Markdown、YAML、Git、
shell、Rust、Cargo 和 CI，不要求 Codex、Claude Code、OpenSpec、Spec Kit 或
其他插件。

根目录的 [`AGENTS.md`](../../AGENTS.md) 只保留所有任务都必须知道的硬约束。
Agent 按任务类型读取下面的详细文档：

| 文档 | 什么时候读取 |
| --- | --- |
| [`rust-style.md`](rust-style.md) | 修改 Rust、Cargo、FFI、GPU 边界、build script、examples 或测试 |
| [`repository-layout.md`](repository-layout.md) | 新增、移动、删除、生成文件或调整目录 |
| [`task-protocol.md`](task-protocol.md) | 多 Agent、worktree、任务状态、handoff 或 review |
| [`handoff-template.md`](handoff-template.md) | 交接实现结果 |
| [`review-template.md`](review-template.md) | 记录代码审查结果 |
| [`task-template.yaml`](task-template.yaml) | 创建小任务的 task YAML |

## 任务路径

- 小型单文件、文档、明确编译错误或不改变公共 API 的修复，使用
  `docs/agent/tasks/<task-id>.yaml`。
- 跨 crate、公共 API、矩阵/向量/四元数、数值算法、GPU/CPU 边界或并行
  任务，使用 `docs/agent/changes/<task-id>/`，通常包含 proposal、design、
  tasks 和 verification。

任务必须声明 owner、base commit、branch、worktree、范围、验收条件和验证
命令；完成时记录准确结果或明确的 unavailable/blocked 原因。

# RustScan 通用 Agent 协作协议

本目录是 RustScan 的模型和工具无关协作入口。它只依赖 Markdown、YAML、Git、shell、Cargo 和 CI；不要求 Codex、Claude Code、OpenSpec、Spec Kit 或其他插件。

## 任务入口

每个任务必须有唯一 ID，并在 `docs/agent/tasks/<task-id>.yaml` 中记录 owner、base commit、branch、worktree、范围、验收条件和验证命令。一个任务只能有一个 active owner，一个 worktree 只能服务一个任务。

### 快速路径

单文件修改、文档修改、明确的编译错误和不改变公共 API 的小修复，只创建一个 task YAML。Agent 读取根 `AGENTS.md` 和该 task 文件后即可执行。

### 完整路径

跨 crate、公共 API、矩阵/向量/四元数、数值算法、GPU/CPU 边界和多 Agent 并行任务，使用 `docs/agent/changes/<task-id>/`。建议包含：

```text
proposal.md       目标、范围、约束和不做什么
design.md         设计决策、边界和兼容性
tasks.yaml        可独立执行的任务和依赖
verification.md   命令、环境、fixture 和结果
```

不要求所有小任务创建完整变更包。

## 生命周期

```text
planner → implementer → verifier → reviewer → integrator
```

- planner 定义范围、依赖、验收条件和验证矩阵；
- implementer 只修改 task scope；
- verifier 根据 task、diff 和验收条件独立运行检查；
- reviewer 检查行为、架构、范围和证据；
- integrator 解决冲突、合并分支、更新状态并归档任务。

`main` 只用于协调和集成。实现任务必须在独立 worktree 和 task branch 中进行。任务完成后，使用 [handoff-template.md](handoff-template.md) 交接，使用 [review-template.md](review-template.md) 记录审查。

## 验证要求

验证记录必须包含准确命令、commit、机器或运行环境、feature/fixture 前提和结果。纯文档任务使用轻量检查；Rust 任务至少运行相关 crate 的 fmt、check 和测试；跨 crate、数值和 GPU 任务按 task 中声明的矩阵验证。没有测试输出或明确的 unavailable/blocked 原因，不能标记完成。

## 工具适配规则

`.codex/`、`.claude/`、`.agents/skills/` 和其他工具目录不是项目事实来源。工具可以读取本协议，也可以在本地提供适配命令，但不能修改任务范围、验收条件或仓库规则。

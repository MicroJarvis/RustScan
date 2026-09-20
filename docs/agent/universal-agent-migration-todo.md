# RustScan 通用 AI Agent 协作迁移 TODO

**目标：** 将 RustScan 整理成只依赖仓库内通用规则、Markdown/YAML 任务文件、Git worktree 和标准命令的 AI Agent 开发环境，使模型或 Agent 工具可以随时替换。

**范围：** Agent 协作协议、任务记录、工具依赖、文档入口、worktree 生命周期、验证门禁和历史计划整理。

**明确决策：**

- 不再把厂商专用 Agent 工具作为仓库流程、文档或执行前提。
- 不再把 `.codex/` 作为仓库配置或任务入口。
- OpenSpec 不作为必需运行时；可以借鉴其 change/spec/design/tasks 结构，但仓库流程必须在没有 OpenSpec CLI 的情况下仍然可执行。
- `AGENTS.md` 是稳定规则入口；`docs/agent/` 是协作流程入口。
- 任何厂商专用目录只能作为个人本地适配，不能成为仓库事实来源。

## 目标结构

```text
AGENTS.md
docs/agent/
├── README.md
├── task-template.yaml
├── handoff-template.md
├── review-template.md
├── tasks/
│   └── <task-id>.yaml
├── changes/
│   └── <task-id>/
│       ├── proposal.md
│       ├── design.md
│       ├── tasks.yaml
│       └── verification.md
└── archive/
scripts/agent/
├── preflight.sh
├── check-scope.sh
└── check-policy.sh
```

小任务只需要 `docs/agent/tasks/<task-id>.yaml`。跨 crate、公共 API、数值算法、GPU/CPU 边界或多 Agent 并行任务，才创建 `docs/agent/changes/<task-id>/` 完整变更包。

## 阶段 0：建立迁移基线和保护边界

- [ ] 在独立分支或独立 worktree 中执行本迁移；不要在当前已有大量 dirty 修改的 `main` 工作树上删除工具文件。
- [ ] 记录迁移开始时的 base commit、工作树路径、分支和现有 dirty 文件清单。
- [ ] 为当前未提交文件标注 owner；迁移工作不得覆盖这些文件。
- [ ] 盘点以下目录和文件的实际来源：`.codex/`、`.claude/`、`.agents/`、`openspec/`、所有 `CLAUDE.md` 和 `AGENTS.md`。
- [ ] 将盘点结果写入迁移分支的验证记录，区分 Git 跟踪文件、个人本地文件和历史文档。

**阶段验收：** 能够说明每个待处理目录是否被 Git 跟踪、是否属于当前任务、是否允许删除；没有执行 `git reset --hard`、`git clean` 或覆盖用户 dirty 修改。

## 阶段 1：建立工具无关的规则入口

### 1.1 扩展根规则

- [ ] 在 [AGENTS.md](../../AGENTS.md) 增加“Tool-neutral Agent protocol”章节。
- [ ] 明确规则优先级：用户请求 > 根/最近的 `AGENTS.md` > 当前 task 文件 > 历史文档 > 工具本地配置。
- [ ] 明确仓库不要求 Codex、Claude Code、OpenSpec、Spec Kit 或其他厂商插件。
- [ ] 明确 Agent 不得覆盖其他任务的 dirty 修改，不得在共享 `main` 工作树直接实现任务。
- [ ] 明确每个任务必须声明 task ID、owner、base commit、branch、worktree、scope、out-of-scope、验收条件和验证命令。
- [ ] 保留现有 `nalgebra`、CPU/GPU 边界、共享 pose 类型和测试要求，不把稳定编码规则复制到多个工具文件。

### 1.2 创建通用流程文档

- [ ] 创建 `docs/agent/README.md`，说明小任务路径、完整变更路径、角色分工、worktree 规则、交接和 review 规则。
- [ ] 创建 `docs/agent/task-template.yaml`，字段至少包括 `id`、`status`、`owner`、`base_commit`、`branch`、`worktree`、`scope`、`out_of_scope`、`acceptance` 和 `verification`。
- [ ] 创建 `docs/agent/handoff-template.md`，固定记录 branch、base commit、文件、行为变化、测试命令、结果、限制和下一步。
- [ ] 创建 `docs/agent/review-template.md`，固定记录范围、正确性、兼容性、测试证据、未解决问题和结论。
- [ ] 约定每个任务一个文件，避免维护一个所有 Agent 都会写入的共享 `active.yaml`，减少并行冲突。

**阶段验收：** 一个没有安装任何 Agent 插件的模型，只读取 `AGENTS.md`、`docs/agent/README.md` 和一个 task 文件，就能知道允许修改什么、如何验证和如何交接。

## 阶段 2：移除厂商专用 Agent 和 Codex 依赖

- [ ] 删除 Git 跟踪的 `.codex/skills/` 文件；删除前先在迁移文档中记录其仍有价值的流程内容。
- [ ] 不把 `.codex/`、`.claude/` 或 `.agents/skills/` 作为仓库规范入口。
- [ ] 从当前有效文档中移除工具专用指令，改写成普通的实现、验证和交接要求。
- [x] 历史计划已完成审查；已完成或被替代的计划删除，仍然有效的 RustGS remediation 已迁移到 `docs/agent/changes/RS-2026-002-rustgs-training-pipeline-remediation/`，其历史目录已删除。
- [ ] 检查 worktree 中的旧 `CLAUDE.md`；将有效规则合并到 `AGENTS.md`，其余文件标记过期或删除，避免不同 worktree 读取不同项目结构。
- [ ] 审查 `.gitignore`：不使用项目规则掩盖误生成的厂商专用目录；个人机器缓存如需忽略，改用用户级全局 ignore。
- [ ] 不自动删除个人本地未跟踪缓存；只删除仓库跟踪的工具文件，个人目录由其 owner 单独清理。

**阶段验收：** `git ls-files` 不再返回 `.codex/` 或项目级厂商专用文件；当前有效文档不要求安装这些工具；仓库搜索结果中不再出现已移除工具名称。

## 阶段 3：决定 OpenSpec 的位置

- [ ] 保持 OpenSpec CLI 为可选工具，不在 README、CI 或 `AGENTS.md` 中要求安装。
- [ ] 不创建依赖 OpenSpec 才能读取的 active change；没有 OpenSpec CLI 时，Agent 仍能按 `docs/agent/` 文件完成任务。
- [ ] 对需要完整设计的任务，使用 `docs/agent/changes/<task-id>/proposal.md`、`design.md`、`tasks.yaml` 和 `verification.md`。
- [ ] 将 OpenSpec 的 delta spec、变更归档和验证思想转写为普通 Markdown/YAML 约定，不复制其 slash command 或技能名称。
- [ ] 如未来试用 OpenSpec 1.13+，只在独立 worktree 验证 `agents`/Universal 适配，不把生成的 `.agents/skills/` 提交为仓库必需内容。

**阶段验收：** 删除或隐藏 OpenSpec CLI 后，仍可创建任务、执行代码变更、运行验证、完成 handoff 和 review。

## 阶段 4：统一 Git branch/worktree 生命周期

- [ ] 规定 `main` 只用于协调、查看和集成，不用于多 Agent 并行编辑。
- [ ] 规定 branch 命名为 `agent/<task-id>/<short-name>` 或仓库选定的等价格式。
- [ ] 规定一个 task ID 只能有一个 active owner，一个 worktree 只能服务一个 task ID。
- [ ] 规定任务启动时记录 base commit，任务完成时记录最终 commit 和验证 commit。
- [ ] 规定 Agent 不得在其他任务 worktree 中修改文件，不得直接改动未声明 scope。
- [ ] 规定任务完成顺序为：implementer 自检 → 独立 verifier → reviewer → integrator 合并 → task 归档 → 清理 worktree。
- [ ] 盘点并标记现有 detached、长期和 prunable worktree；删除或归档动作必须由对应 owner 确认，不能在迁移中盲目清理。

**阶段验收：** 新建一个示例 task 时，能够从 task 文件追溯到唯一 branch、worktree、base commit、测试结果和最终状态。

## 阶段 5：增加标准命令门禁

- [ ] 创建 `scripts/agent/preflight.sh`，检查 Git 根目录、当前 branch、dirty 文件、Rust 工具链和 task 文件是否存在。
- [ ] 创建 `scripts/agent/check-scope.sh`，比较 `git diff --name-only` 与 task 的 `scope`/`out_of_scope`，越界时失败。
- [ ] 创建 `scripts/agent/check-policy.sh`，检查新增 CPU 代码是否引入 `glam::Mat*`、`glam::Vec*`、`glam::Quat` 或其他不允许的线性代数类型。
- [ ] 为文档任务提供轻量 Markdown/link 检查，不让纯文档任务强制运行完整 workspace 测试。
- [ ] 为单 crate、跨 crate、数值算法、GPU/FFI 和合并任务定义最小验证矩阵。
- [ ] 在 CI 中加入工具无关门禁：task ID、scope、vendor-specific 规则引用、数学类型策略和基础文档链接检查。

**阶段验收：** Agent 无需记住复杂流程，只要运行标准 shell 脚本和 Cargo 命令，就能得到明确的通过/失败原因。

## 阶段 6：整理文档真相来源

- [ ] 规定 `docs/current-project-status.md` 是当前状态唯一入口；README 只保留项目简介和链接。
- [ ] 给历史计划、实验报告和 review 文档增加 `historical`、`superseded`、`completed` 或 `active` 状态。
- [ ] 将仍然有效的 RustSFM、RustGS 和矩阵迁移任务登记到 `docs/agent/tasks/`，避免 Agent 从多个 TODO 猜测当前优先级。
- [ ] 每条验证记录包含日期、commit、机器、命令、feature、fixture 前提、结果和限制。
- [ ] 更新 `docs/index.md`，把 `docs/agent/README.md` 和当前迁移 TODO 加入维护入口。

**阶段验收：** 新 Agent 只需读取文档索引、当前状态和指定 task，不会把历史计划误当成 active 任务。

## 阶段 7：用两个不同 Agent 做兼容性演练

- [ ] 选择一个低风险文档任务和一个小型 Rust 修复任务作为演练，不触碰当前大型 dirty 改动。
- [ ] 使用两种不同的 Agent/模型，只提供同一组 `AGENTS.md`、task 文件和标准命令，不提供厂商专用技能。
- [ ] 检查两个 Agent 是否能独立理解 scope、worktree、测试和 handoff 格式。
- [ ] 检查 verifier 能否只根据 task、diff 和验证命令完成独立检查。
- [ ] 记录遗漏的字段、重复读取的文档和不必要的 token 消耗，回写到 `docs/agent/README.md`。

**最终验收：** 更换模型后，不需要迁移 `.codex`、`.claude` 或 OpenSpec 技能文件，仍可以从 active task 继续开发并完成验证。

## 推荐执行顺序

```text
阶段 0 基线保护
  → 阶段 1 通用协议
  → 阶段 2 移除厂商专用 Agent/Codex 依赖
  → 阶段 3 OpenSpec 可选化
  → 阶段 4 worktree 生命周期
  → 阶段 5 标准门禁
  → 阶段 6 文档收口
  → 阶段 7 多模型演练
```

不要在阶段 0 完成前删除工具目录，也不要在阶段 1 的通用协议完成前把现有任务迁移到新结构。每个阶段完成后都应单独 review 并提交，避免一次迁移产生无法归因的大 diff。

## 不纳入本次迁移的内容

- 不重写 Rust 业务代码。
- 不在本 TODO 中执行 `git reset`、`git clean` 或删除用户已有 worktree。
- 不强制引入 OpenSpec、Spec Kit、BMAD 或其他新的 Agent 框架。
- 不把完整项目历史压缩进 `AGENTS.md`。
- 不为每个小任务生成完整 proposal/design 文档。

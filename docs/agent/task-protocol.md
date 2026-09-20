# Tool-Neutral Agent Task Protocol

Read this file for multi-agent work, worktrees, task status, handoffs, or
changes that need a review trail. It depends only on Markdown, YAML, Git,
shell, Cargo, and CI; no vendor-specific Agent plugin is required.

## Task entry and scope

- Every non-trivial task has a unique ID and one owner.
- Record base commit, branch, worktree, in-scope paths, explicitly out-of-scope
  paths, acceptance conditions, and exact verification commands in
  `docs/agent/tasks/<task-id>.yaml` or a change package under
  `docs/agent/changes/<task-id>/`.
- A single task YAML is sufficient for a small file or documentation change.
  Cross-crate, public API, numerical, GPU/CPU-boundary, or parallel work
  should use a change package containing, as applicable:

  ```text
  proposal.md       goal, scope, constraints, and non-goals
  design.md         decisions, boundaries, and compatibility
  tasks.yaml        independently executable tasks and dependencies
  verification.md   commands, environment, fixtures, and results
  ```

- The implementer changes only the declared scope. Do not overwrite another
  task's dirty changes or silently widen acceptance conditions.

## Worktrees and lifecycle

- `main` is for coordination and integration. Implementation work uses an
  isolated task branch and worktree unless the task is a trivial documentation
  edit explicitly kept in the current checkout.
- Never use `git reset --hard`, `git clean`, force pushes, or broad deletion to
  resolve uncertainty unless the user explicitly authorizes that exact action.
- The normal lifecycle is:

  ```text
  planner → implementer → verifier → reviewer → integrator
  ```

  The planner defines scope and acceptance; the implementer changes code; the
  verifier independently runs checks; the reviewer checks behavior, design,
  scope, and evidence; the integrator resolves conflicts and updates status.

## Handoff and verification

- Use [`handoff-template.md`](handoff-template.md) for task completion and
  [`review-template.md`](review-template.md) for review results.
- A handoff must list changed files, base and final commits, exact commands,
  environment and feature/fixture prerequisites, results, limitations, and
  the next action.
- Pure documentation tasks may use lightweight checks. Rust tasks run relevant
  fmt, check, lint, and tests. Cross-crate, numerical, GPU, native, and data
  tasks run the full matrix declared by the task.
- No task is complete without test output or a precise unavailable/blocked
  reason. Do not mark an ignored test as a substitute for verification.

## Tool portability

`.codex/`, `.claude/`, `.agents/skills/`, and other local tool directories are
not project facts. Tools may read this protocol and provide local adapters, but
they cannot change task scope, acceptance conditions, or repository rules.

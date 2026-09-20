# RS-2026-001 Migration Baseline

**Captured:** 2026-09-19  
**Base commit:** `f78c3f0`  
**Source worktree:** `/Users/tfjiang/Projects/RustScan`  
**Migration worktree:** `/Users/tfjiang/.codex/worktrees/universal-agent-migration/RustScan`  
**Migration branch:** `agent/RS-2026-001/universal-agent-migration`

## Source worktree state

The source checkout was on `main`. It had no modified tracked files and two
untracked paths:

```text
docs/rustsfm-TODO-技术方案-2026-09-17.md
docs/agent/
```

Those paths are outside this migration worktree and must not be overwritten or
cleaned by migration agents.

## Tool-specific files found

- `.codex/skills/openspec-apply-change/SKILL.md` (tracked)
- `.codex/skills/openspec-archive-change/SKILL.md` (tracked)
- `.codex/skills/openspec-explore/SKILL.md` (tracked)
- `.codex/skills/openspec-propose/SKILL.md` (tracked)
- historical execution plans and design specs (tracked historical records)
- a legacy local Agent-plugin directory (ignored when present)
- `.worktrees/retry-state-sync/CLAUDE.md` and `.worktrees/rustscan-macos-app/CLAUDE.md` (worktree-local rules)

No active `openspec/` directory was present. The installed OpenSpec CLI was
`1.2.0`, but it is not a prerequisite for this migration.

## Existing worktrees requiring later owner review

The source repository also contained detached, long-lived, and prunable
worktrees. This migration records them but does not delete or alter them.

Observed at baseline:

```text
/private/tmp/rustgs-baseline-09f9254       09f9254 detached
/private/tmp/rustscan-baseline-400e0cb     400e0cb detached, prunable
.worktrees/gpu-five-point-baseline         f313dbd detached
.worktrees/gpu-five-point-f32              4a53dc6 branch gpu-five-point-f32
.worktrees/retry-state-sync                d1d1023 branch codex/retry-state-sync
.worktrees/rustscan-macos-app              930fbc7 branch feature/rustscan-macos-app
```

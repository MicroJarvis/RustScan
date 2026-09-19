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
docs/RustSFM-TODO-技术方案-2026-09-17.md
docs/agent/
```

Those paths are outside this migration worktree and must not be overwritten or
cleaned by migration agents.

## Tool-specific files found

- `.codex/skills/openspec-apply-change/SKILL.md` (tracked)
- `.codex/skills/openspec-archive-change/SKILL.md` (tracked)
- `.codex/skills/openspec-explore/SKILL.md` (tracked)
- `.codex/skills/openspec-propose/SKILL.md` (tracked)
- `docs/superpowers/plans/` and `docs/superpowers/specs/` (tracked historical records)
- `.superpowers/` (ignored local directory when present)
- `.worktrees/retry-state-sync/CLAUDE.md` and `.worktrees/rustscan-macos-app/CLAUDE.md` (worktree-local rules)

No active `openspec/` directory was present. The installed OpenSpec CLI was
`1.2.0`, but it is not a prerequisite for this migration.

## Existing worktrees requiring later owner review

The source repository also contained detached, long-lived, and prunable
worktrees. This migration records them but does not delete or alter them.

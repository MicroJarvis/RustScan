# RustScan Roadmap

**Updated:** 2026-08-15

This roadmap is intentionally high level. Verified branch facts live in
[`docs/current-project-status.md`](./docs/current-project-status.md) and the
crate-specific READMEs; this file only tracks forward work.

## Current State

- RustSFM, RustViewer, and RustGS are the current reconstruction workflow focus.
- RustMesh and RustSLAM remain independently buildable workspace crates.
- The current verified RustSFM test snapshot and known gaps are recorded in
  `docs/current-project-status.md`.

## Near-Term Priorities

### 1. RustSFM COLMAP parity

- Provision and version the external `flowers2_colmap` fixture.
- Close the remaining numerical and bundle-adjustment parity gaps against COLMAP.
- Keep default and dependency-minimal test paths reproducible in CI.

### 2. RustViewer end-to-end workflow

- Validate reconstruction, sparse export, RustGS handoff, and artifact loading on
  real image sequences.
- Keep unsupported video paths explicitly gated until they have the same artifact
  validation guarantees.

### 3. RustGS quality loop

- Continue LiteGS parity and TUM PSNR validation with dated, reproducible reports.
- Keep the splat-first public API and `HostSplats`/device ownership boundaries stable.

### 4. Documentation discipline

- Keep only one maintained status source per topic.
- Keep dated plans as historical records only when they still provide audit value.
- Avoid claims that cannot be reproduced from a documented command and fixture.

## RustMesh and RustSLAM

Use each crate README for current capabilities and commands. Do not infer current
status from the historical `rm-opt` worktree notes that were removed from the
maintained documentation set.

## Workspace Direction

- Keep the crates loosely coupled and testable in isolation.
- Prefer verification-backed claims over marketing-style completeness claims.
- Reduce documentation churn by centralizing status reporting in a few maintained entry points.

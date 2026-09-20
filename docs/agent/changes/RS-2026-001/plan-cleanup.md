# Historical Plan Cleanup

**Date:** 2026-09-19

The 35 historical execution plans were audited against
implementation commits, current source, and `docs/current-project-status.md`.
All plans except the 2026-09-18 RustGS remediation were complete, superseded,
or obsolete as an execution plan. The active remediation was migrated to
`docs/agent/changes/RS-2026-002-rustgs-training-pipeline-remediation/`; its
historical source was removed.

## Deleted completed or superseded plans

- 2026-07-15 RustSFM safety remediation (`44fec9e` and subsequent hardening)
- 2026-07-18 wgpu model scorer, PnP, RANSAC integration, and SIFT (`ed9c893`, `f832049`, `99d5ecb`)
- 2026-07-19 incremental registration hot path, macOS BA backends, and mapper performance (`d916816`, `e736215`, `e830fcb`)
- 2026-07-20 resumable RustGS checkpoints, PoseLib default, sparse maintenance, sequence registration, and RustViewer workspace/media plans (`06085bd` through `b030559`, `c8d3271`, `e830fcb`, `474c80d`, `df1f991`)
- 2026-07-22 project-store hardening (`b7abb6d` and follow-up fixes)
- 2026-07-25 media bridge, reconstruction prototype, RustSFM bridge, and workbench phase one (`1697cb5`, `df1f991`)
- 2026-07-26 Vulkan runtime, import-dialog threading, and RustSFM→RustGS orchestration (`3719b98`, `fce914f`, `6dd4871`)
- 2026-07-27 adaptive keyframes (`5f6d60d`)
- 2026-08-02 GPU PnP-f (`f832049`)
- 2026-08-03 pipeline state sync and project-open retry (`1190d90`, `b15ba1e`)
- 2026-08-08/09 observability plans (`f6e7adc`, `e4db884`)
- 2026-08-10 RANSAC 512 experiment and 2026-08-11 decision batching (`16c27ac`, `99d5ecb`); the later full-parity result superseded the early bounded-only NO MERGE note
- 2026-08-14 RustSFM review hardening (`273a40a`, `bbac4e2`)
- 2026-09-17 RustGS quality/efficiency plan, superseded by `docs/rustgs-TODO-训练效果与效率优化-2026-09-17.md`
- 2026-09-19 nalgebra review fixes (`f78c3f0`)

The deleted files are historical source material only; their implementation
evidence remains in Git history and current status/review documents.

# RustScan Documentation Index

**Updated:** 2026-08-15

此索引先列出当前作为执行、验证或状态依据的文档；带日期的设计、计划和实验记录仅在其仍有审计价值时保留，不替代当前入口。

## Canonical Status Docs

| Document | Purpose |
|---|---|
| [current-project-status.md](current-project-status.md) | 当前仓库主线状态、已验证结果与下一步优先级 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 当前 workspace 结构与 RustGS 训练架构边界 |

## Active RustSFM Docs

| Document | Purpose |
|---|---|
| [../RustSFM/README.md](../RustSFM/README.md) | RustSFM build, test, optional parity-fixture, and CLI entry points |
| [../RustSFM/PARITY_ROADMAP.md](../RustSFM/PARITY_ROADMAP.md) | COLMAP parity status and remaining numerical work |
| [../RustSFM/COLMAP_COMPAT_TODO.md](../RustSFM/COLMAP_COMPAT_TODO.md) | COLMAP compatibility backlog |

## Historical RustSFM Records

| Document | Purpose |
|---|---|
| [superpowers/plans/2026-08-14-rustsfm-review-hardening.md](superpowers/plans/2026-08-14-rustsfm-review-hardening.md) | Completed build, output, CI, and GPU error-handling hardening record |

## Current RustGS Docs

| Document | Purpose |
|---|---|
| [../RustGS/README.md](../RustGS/README.md) | build、test、CLI 与 artifact contract |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 当前 wgpu training module layout、public surface 和 ownership boundary |

The dated RustGS TUM benchmark and dataset-research records remain outside the
maintained entry set. Read them only as historical evidence and verify every
command against the current RustGS CLI.

## Retention Rule

- 当前文档必须明确命令前提、外部夹具和验证日期。
- 历史设计记录不构成当前 API、测试或 workspace 状态的权威来源。
- `docs/plans/`、`docs/reviews/`、`docs/superpowers/` 中的日期文档只有在仍有
  审计价值时保留；完成或被后续设计取代的文档不再作为入口引用。

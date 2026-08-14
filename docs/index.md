# RustScan Documentation Index

**Updated:** 2026-08-14

此索引列出当前仍作为执行、验证或状态依据的文档。历史计划可能仍保留在 `docs/superpowers/` 供追溯，但不替代下列当前入口。

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
| [superpowers/plans/2026-08-14-rustsfm-review-hardening.md](superpowers/plans/2026-08-14-rustsfm-review-hardening.md) | Current RustSFM output, build, CI, and GPU error-handling hardening |

## Active RustGS Docs

| Document | Purpose |
|---|---|
| [plans/2026-04-06-rustgs-refactor-guardrails.md](plans/2026-04-06-rustgs-refactor-guardrails.md) | 当前 public surface、回归基线与 guardrail 命令 |
| [../RustGS/docs/plans/2026-04-09-rustgs-soa-splat-architecture-proposal.md](../RustGS/docs/plans/2026-04-09-rustgs-soa-splat-architecture-proposal.md) | RustGS 当前唯一的 splat 表示设计文档与收口状态 |
| [plans/2026-04-05-litegs-parity-roadmap-refresh.md](plans/2026-04-05-litegs-parity-roadmap-refresh.md) | 当前剩余 LiteGS parity 工作与优先级 |
| [RustGS-TUM-Profile-Comparison-2026-04-06.md](RustGS-TUM-Profile-Comparison-2026-04-06.md) | 当前有效的 TUM 训练对照记录与 topology-freeze 决策依据 |
| [rustgs-benchmark-datasets.md](rustgs-benchmark-datasets.md) | RustGS 下一批公开训练/评测数据集、准备脚本与运行口径 |

## Retention Rule

- 当前文档必须明确命令前提、外部夹具和验证日期。
- 历史设计记录不构成当前 API、测试或 workspace 状态的权威来源。

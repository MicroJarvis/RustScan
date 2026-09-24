# RS-2026-002 Verification

This change was migrated from a historical RustGS remediation plan on 2026-09-19.

## C1–C4 fix branch (review remediation)

| Field | Value |
| --- | --- |
| Base SHA | `701d051` (`docs: record T1-T4 integration and worktree cleanup`) |
| Branch | `fix/rs-2026-002-c1-c4-review` |
| C2 commit | `fa970ed` — confirm commits at safety points only |
| C4 safety commit | `ae55d66` — harden profiler execution safety |
| C4 measurement commit | `fix(rustgs): correct C4 timing scope and reports` on this branch (see `git rev-parse HEAD`) |

C1 capacity guards, WGSL explicit branches, and C3 workspace ownership are preserved on this branch. Do not start C5–C8 from this worktree.

### Profiler measurement contract (C4)

- GPU timestamp samples are **forward-only** (`gpu_timing_scope: "forward"`).
- Totals use `gpu_forward_sum_ms` / `gpu_forward_sum_seconds` from accepted post-warmup samples; never `p50 × completed_iterations`.
- `gpu_completion_seconds` stays null in optimization JSON.
- Adapter/backend/driver/timestamp capability come from the training `GsDevice` / `SharedWgpuContext`, not a fresh HighPerformance probe.
- `timestamp_query_available` is capability; `gpu_timing_enabled` is config; `measurement_success` is post-warmup sample presence.
- `runtime_peak_device_bytes` is sampled `bytes_in_use` HWM (`sampled_bytes_in_use_high_water`).
- Workspace / fresh allocation fields are prefix-sum scoped (`workspace_scope: "prefix_sum"`).
- Warmup drops the first sample per series; aborted runs keep collected samples; resume starts a new collector.

### C2 / C8 status-readback note

Ordinary healthy `read_loss=false` steps do **not** perform per-step status readback under the C2 commit-confirmation contract (`design.md`). Safety-point reads (loss cadence, topology, checkpoint/pause/cancel, training end, ForwardAbort) remain. C8 reject language that said “ordinary step status readback” is clarified to mean **unexpected per-step disposition reads beyond the safety-point contract**, not the declared safety-point cadence.

Run the commands in `tasks.yaml` after each implementation checkpoint and record commit, machine, features, fixtures, results, and limitations here.

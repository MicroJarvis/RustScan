# P2 persistent five-point session (2026-09-14)

## Decision

**Retain.** `FivePointSession` preallocates and reuses input / intermediate /
output buffers and readback staging. Active trial count is passed as a uniform
(`params.count`); shaders no longer treat `arrayLength` as N. Zero-dependent
ranges (`diagnostics`, `basis`, `algebra`, `out`) are cleared with
`clear_buffer` before each solve. One solve in flight per session.

## Scope

Single variable: resource lifetime / reuse for the complete solve path. No
change to nullspace rank criterion, roots, or recovery arithmetic beyond the
count uniform needed for capacity > N.

## Design

```text
WgpuFivePointF32  — pipelines / layouts (unchanged ownership)
FivePointSession  — left/right, constraints, diagnostics, basis, algebra,
                    out, params, staging; grow-only capacity
```

- Grow policy: double until device workgroup limit, never shrink.
- Host API: `gpu.session(capacity)`, then `session.solve_essential`.
- One-shot `WgpuFivePointF32::solve_essential` still allocates per call and
  remains the reference path; it also binds `params.count` so both paths share
  shaders.

Shaders updated for explicit count: `nullspace`, `algebra`, `roots`.
Dispatch-count–bounded entry points (`main`, `pack`, `elimination`,
`polynomial`, `validate_polynomial`, `recover`) unchanged.

## Provenance

- Worktree `.worktrees/gpu-five-point-f32`.
- Device `Apple M5 Max`.
- Input digest / Q2 signatures unchanged and asserted:
  - model `472124c8…d6ea`
  - diagnostic `3d04bb3e…c050`
  - CPU f64 `9e6764b4…3a`
- Harness: `RustSFM/examples/five_point_gpu_p2_session.rs`
- Results: `experiments/p2-session-20260914.json` (first timing shape),
  `experiments/p2-session-final-20260914.json` (final).

## Acceptance

| Check | Result |
| --- | --- |
| One-shot full 65,024 signature == Q2 | pass |
| Session 127×512 signature == Q2 | pass |
| small→large→small model signature | pass |
| Unit: reuse / fail→success / partial WG | pass (`five_point_session_reuses_buffers_without_stale_state`) |

## Timing (127 × batch 512, 2 warm + 5 measure, median seconds)

| Mode | Median |
| --- | ---: |
| Session create (capacity 512) | 0.000019 |
| Session cold (create + one full pass) | 1.734 |
| Session steady (reuse) | 1.775 |
| One-shot (alloc every call) | 1.801 |

Steady vs one-shot ≈ **1.4%** wall improvement. Buffer construction is
negligible; the batch-512 cost remains wait + readback across 127 submits.
That is the P3 target, not a P2 failure: P2 removes allocation/churn and makes
reuse safe under oversized capacity.

## Residual

- Bind groups are still created per pass (not cached). Cheap relative to wait.
- Timestamp profiling on the session path allocates a query set per profiled
  call; unprofiled path does not.
- CPU8 same-caliber timing vs Q2 was not re-run this round; offline throughput
  claims should use a fresh CPU8 pass if needed.

## Next

P3 — submit / readback boundary (fewer waits, optional bounded double-buffer).

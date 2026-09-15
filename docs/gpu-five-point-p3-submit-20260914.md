# P3 submit / readback boundary (2026-09-14)

## Decision

**Retain Candidate A and Candidate B.**

| Candidate | Change | 127×512 steady median | vs prior |
| --- | --- | ---: | --- |
| P2 baseline | session reuse, 2 syncs/call | 1.775 s | — |
| **A (retain)** | merge compute + staging copy → 1 submit/wait per solve | **1.344 s** | −24% vs P2 |
| **B (retain)** | bounded double-buffer (2 slots) on top of A | **0.740 s** | −45% vs A / −58% vs P2 |

Signatures stay on the Q2 gate. Double-buffer remains **opt-in**
(`enable_double_buffer` + `solve_essential_batches`); single-slot
`solve_essential` keeps the A path.

## Scope

Single performance variable for this round: **CPU↔GPU sync boundary** on the
session path. No roots/recovery arithmetic changes; nullspace gate unchanged.

## Candidate A — merge compute + copy

Previously each unprofiled solve submitted compute, waited, then submitted the
`out → staging` copy and waited again (254 submits/waits per 127-call pass).
Folding the copy into the same command encoder yields **127 submits + 127 waits**.

Stop rule applied: wall time improved; submit-count reduction alone would not
have been enough.

- Harness: `RustSFM/examples/five_point_gpu_p3_submit.rs`
- Results: `experiments/p3-submit-merge-20260914.json`
- Counters: `submit_count` / `wait_count` / `reset_sync_counters`

## Candidate B — bounded double-buffer

Two `SessionSlot` buffer sets. Pipeline:

```text
submit(N+1)  while  finish/map/decode(N)
```

At most two in flight; `busy` backpressure; trial indices remapped in input
order. Timestamp resolve stays off the result path.

- Harness: `RustSFM/examples/five_point_gpu_p3_double.rs`
- Results: `experiments/p3-double-buffer-20260914.json`
- Unit coverage extended in `five_point_session_reuses_buffers_without_stale_state`

### Timing (same input, Apple M5 Max, 2 warm + 5 measure)

| Path | Median (s) | Samples |
| --- | ---: | --- |
| P3-A recorded baseline | 1.344 | from merge JSON |
| Merge same-run (this harness) | 1.410 | 1.407–1.414 |
| Double-buffer | **0.740** | 0.729–0.757 |

Relative: **−44.9% vs recorded P3-A**, **−47.5% vs same-run merge**.
Submits/waits still 127 per pass (overlap hides latency; count is unchanged).

## Provenance

- Input digest `af07459d…30c5`
- Model `472124c8…d6ea`, diagnostic `3d04bb3e…c050`
- Device `Apple M5 Max`
- Worktree `.worktrees/gpu-five-point-f32`

## Acceptance

| Check | Result |
| --- | --- |
| Merge path signature == Q2 | pass |
| Double-buffer path signature == Q2 | pass |
| Unit: merge 1 submit/wait per solve | pass |
| Unit: double-buffer bits == sequential | pass |
| Wall A < P2 | pass → retain A |
| Wall B < A | pass → retain B |

## Residual

- ~~Grow-during-pipeline unsafe~~ **Fixed (2026-09-15 review):** growing capacity
  mid-`solve_essential_batches` used to replace all slots (including the staging
  buffer holding the in-flight batch), silently decoding zeros. The pipeline now
  drains the in-flight batch before `ensure_capacity`, and `ensure_capacity`
  preserves submit/wait counters and the busy flag. Regression: mid-pipeline
  growth case in `five_point_session_reuses_buffers_without_stale_state`.
  Re-run after fix: double-buffer median 0.709 s, signatures Q2
  (`experiments/p3-double-buffer-growfix-20260915.json`).
- Bind groups still rebuilt per pass.
- Offline batch-512 throughput only; not production RANSAC admission.

## Next

Difficulty-aware roots (large-batch ceiling) and/or recovery-quality work;
P4 solver→scorer device handle when integrating.

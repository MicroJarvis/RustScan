# RS-2026-002 Design

The trainer keeps a device-resident sticky status across forward, loss,
backward, topology, and optimizer. A failed step gates subsequent device work;
the host reads status only at declared safety points. Workspace ownership is
explicit and must not use thread-local raw pointers. Reports distinguish GPU
completion timing from CPU submission timing and carry adapter, split, and
binary identity when available. GPU timestamp samples are forward-render
scoped; optimization JSON publishes real sample sums (`gpu_forward_sum_*`)
and leaves `gpu_completion_seconds` null rather than inventing p50×N totals.

The change is staged as P0 correctness, P1 ownership and measurement, and P2
quality/reproducibility. Existing P0 and P1.1 evidence in `tasks.md` is the
baseline; future work starts at the first unchecked item.

## Commit confirmation contract (C2 / F01)

Device word 4 (`committed_optimizer_steps`) is authoritative for whether an
optimizer mutation actually committed. The host Adam step counter may advance
optimistically and is resynced at safety points.

- **Safety-point status reads only:** loss cadence, topology boundary,
  checkpoint / pause / cancel, training end, and ForwardAbort refresh when the
  host mirror already knows a sticky anomaly. Healthy `read_loss=false` steps
  do **not** read the status buffer.
- **Unread steps are submitted, not confirmed.** `TrainStepDisposition` may be
  `SubmittedUnconfirmed` after a normal unread step. That outcome must not
  update `completed_iterations`, progress, snapshot, or checkpoint.
- **Confirmation** happens at the next safety-point read: host sets
  `completed_iterations = start_iteration + (device_committed - committed_baseline)`
  (clamped to the highest submitted iteration). Catch-up progress may emit for
  newly confirmed iterations.
- **Abort:** sticky overflow / non-finite discovered at a safety point leaves
  `completed_iterations` at the last confirmed value, does not write checkpoint
  / progress for unconfirmed attempts, and preserves Gaussian / Adam / topology
  state that the device gate already blocked.

This keeps anomaly steps from being recorded as completed without restoring
per-step unread-loss status readback.

Telemetry still exports `status_readbacks_step_disposition` so totals remain
recomputable from reason fields. Under this contract the counter stays at zero:
healthy unread steps never perform a disposition status read.

**Late pause/cancel (R01):** If pause or shutdown is requested after an unread
submit, the boundary is a new safety point: sync device status, confirm word4,
then write the checkpoint at the confirmed iteration. Do not fail with
“before commit confirmation” when the device has (or has not) committed — for
pause/shutdown, checkpoint the last confirmed state; device sticky errors still
block the checkpoint write.

**Exact snapshot cadence (R03):** `snapshot_every` iterations are confirmation
safety points. Snapshots are captured only when the just-confirmed iteration
equals the labeled cadence point, so the exported model matches the iteration
number (no coalesced historical labels on a later model).

**Config continuity fingerprint (R02):** Resume identity hashes training
continuity fields only. `iterations` is zeroed and nested `profiler` is omitted
so legacy checkpoints and profiler-only toggles remain restorable; optimizer /
loss / topology / data / raster / litegs changes still mismatch.

## Profiler contracts (C4 / R04–R08 remediations)

**R04 — disabled Instant creation:** When `profiler.enabled=false`, the
train loop, forward/backward/optimizer host walls, upload `CpuSpanTimer`, and
prefetch decode/resize paths must not create profiling `Instant`s. Recording
entry points (`record_span*`, `record_cpu_step`, `record_gpu_step_ms`) are
no-ops. Topology still keeps its own training telemetry vectors; those are not
pipeline-profiler series.

**R06 — split forward series:** GPU-sampled iterations record
`forward_gpu_sampled` (`synchronized_boundary`, includes timestamp resolve
wait). Unsampled iterations record `forward_cpu_submit` (`cpu_submit`). Do not
mix the two into one percentile series.

**R07 — resolve failures:** `resolve_device_gpu_ms` catches CubeCL resolve
panics. Measurement-only failure with a healthy device drops the sample,
increments resolve/drop counters, and continues. Device unusable →
`TrainingError::Gpu`. Work runs at most once. `measurement_success` reflects
whether any post-warmup GPU sample exists; later failures do not erase prior
accepted samples. Fault injection uses the production
`profile_device_gpu_step_with_fault` / `finish_profiled_output` path.

**R08 — validator:** Unsupported reports must set `measurement_success=false`.
Illegal combinations are rejected after JSON round-trip as well as in-memory.

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

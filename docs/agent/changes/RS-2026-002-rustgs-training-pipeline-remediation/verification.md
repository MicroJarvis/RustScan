# RS-2026-002 Verification

This change was migrated from a historical RustGS remediation plan on 2026-09-19.

## C1–C4 fix branch (review + re-review remediation)

| Field | Value |
| --- | --- |
| Base SHA | `701d051814a29ab3ee4fbefc01355112f71b5a47` |
| Branch | `fix/rs-2026-002-c1-c4-review` |
| Worktree | `.worktrees/rs-2026-002-c1-c4-review` |
| Prior reviewed commit | `f5dbf678362bdc7e2e45e16661488b29d97371dd` |
| R01–R03 commit | `1c11e938eedd8efb66aa37a0d9fbf894a9f9c09f` |
| R04–R09 commit | `311cd1775a40ae25546f3976f56b8eaaa663b1b0` |
| Branch tip (handoff) | recorded in commit message / `git rev-parse HEAD` |
| Code package tip | `3bf9b84644c81f66766d1644658afc4bd9fa02a3` |

C1 capacity guards, WGSL explicit branches, and C3 workspace ownership are preserved. Do not start C5–C8 from this worktree.

### Re-review R01–R09 (2026-09-24)

| ID | Fix summary |
| --- | --- |
| R01 | Late pause/cancel after unread submit syncs device status, confirms word4, then checkpoints |
| R02 | Continuity fingerprint zeros `iterations` and omits nested `profiler`; real training params still mismatch |
| R03 | `snapshot_every` is a confirmation safety point; label matches captured model |
| R04 | `profiler.enabled=false` gates CPU span/step/loop sample accumulation |
| R05 | DefaultDevice/BestAvailable metadata is null+reason (no HighPerformance re-pick) |
| R06 | `iteration_wall` + kinds: host_wait / worker_wall / cpu_submit / synchronized_boundary |
| R07 | Shared `classify_profile_recovery`; start/end/resolve + dropped counters; work ≤1 |
| R08 | Nonzero samples require finite p50/p95; zero samples require dual null |
| R09 | Optimization JSON exports mode, failures, pipeline_spans via `write_optimization_report` |

### Profiler / commit contracts

- GPU timestamp samples are **forward-only** (`gpu_timing_scope: "forward"`).
- Totals use real `gpu_forward_sum_*`; never `p50 × completed_iterations`.
- Ordinary healthy unread steps do **not** status-read; safety points include loss cadence, topology, checkpoint/pause/cancel, snapshot cadence, training end, ForwardAbort.
- C8 comparator: each run matches its own recorded binary revision/hash; cross-version A/B may differ by revision when each is bound correctly (`cursor-task-cards.md`).

### Implementer gate results (2026-09-24)

Environment: macOS / arm64; `POSELIB_ROOT=/Users/tfjiang/Projects/RustScan/third_party/native/PoseLib` for workspace check.

| Command | Result |
| --- | --- |
| `cargo fmt --all -- --check` | PASS |
| `cargo check --workspace --all-targets` | PASS (pre-existing viewer/sfm warnings) |
| `cargo test -p rustscan-gs --all-targets --features gpu-wgpu -- --test-threads=1` | PASS: lib 185, CLI 25, bounded 1, checkpoint 53; integration default-ignored 1 |
| `cargo test -p rustscan-gs --no-default-features --lib -- --test-threads=1` | PASS: 38 |
| `cargo test -p rustscan-gs --test integration_test --features gpu-wgpu -- --ignored --test-threads=1` | PASS: 1 |
| `cargo clippy -p rustscan-gs --all-targets --all-features -- -D warnings` | FAIL: blocked first by `rustscan-slam` (~85 diagnostics); not a RustGS gate pass |
| `cargo clippy -p rustscan-gs --all-targets --features gpu-wgpu --no-deps -- -D warnings` | FAIL: ~42 baseline RustGS diagnostics (duplicated_attributes, unused Int, gradient dead_code, chunks_exact, too_many_arguments, …). R07-introduced MutexGuard/`recover_profile_result` unused fixed in follow-up. No global allow added. |

### Remaining limitations

- CubeCL `ProfileDuration::resolve` may still panic on map failure (documented; not silent success).
- DefaultDevice adapter name/driver may be null + reason when metadata was not captured at device creation.
- Worker Instant collection may still run when profiler is off (samples are not recorded).
- Clippy `-D warnings` not green on baseline; all-features blocked by slam.
- Stopped for independent re-review; not merged to main; C5–C8 not started.

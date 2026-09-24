# RS-2026-002 Verification

This change was migrated from a historical RustGS remediation plan on 2026-09-19.

## C1–C4 fix branch (review + re-review remediations)

| Field | Value |
| --- | --- |
| Base SHA | `701d051814a29ab3ee4fbefc01355112f71b5a47` |
| Branch | `fix/rs-2026-002-c1-c4-review` |
| Worktree | `.worktrees/rs-2026-002-c1-c4-review` |
| Prior tip (CHANGES_REQUESTED) | `a0d5c7ec1f93eacae12e6e9602a7098d8972377e` |
| Round-2 code package | `6bfab3ae491a2f7de7f5310ac2f0ad4719bce9b7` |
| Branch tip (docs) | `aa7dd1fa156ae19ad28ca4ab1f6ef6e7989e219a` |

C1 capacity guards, WGSL explicit branches, and C3 workspace ownership are preserved. Do not start C5–C8 from this worktree.

### Round-2 remediations (remaining R04 / R06 / R07 / R08 + Clippy)

| ID | Fix summary |
| --- | --- |
| R07 | `resolve_device_gpu_ms` catch_unwind; healthy device → drop sample + counters; unhealthy → `TrainingError::Gpu`; production `profile_device_gpu_step_with_fault` / `finish_profiled_output` |
| R04 | `profiling_instant` / `CpuSpanTimer::start_enabled`; prefetch `measure_timing`; `record_gpu_step_ms` gated on profiler+gpu_timing |
| R06 | Split `forward_gpu_sampled` (synchronized_boundary) vs `forward_cpu_submit` (cpu_submit) |
| R08 | Unsupported rejects `measurement_success=true`; JSON ser/de consistency tests |
| Clippy | Moved `fingerprint_tests` after helpers (`items_after_test_module` fixed); no global allow |

### Implementer gate results (2026-09-24 round 2)

Logs: `artifacts/runs/rs-2026-002-c1-c4-rereview-round2/`

| Command | Result |
| --- | --- |
| `cargo fmt --all -- --check` | PASS |
| `cargo check --workspace --all-targets` | PASS (pre-existing viewer warnings) |
| `cargo test -p rustscan-gs --all-targets --features gpu-wgpu -- --test-threads=1` | PASS: lib 190, CLI 25, bounded 1, checkpoint 53; integration default-ignored 1 |
| `cargo test -p rustscan-gs --no-default-features --lib -- --test-threads=1` | PASS: 38 |
| `cargo test -p rustscan-gs --test integration_test --features gpu-wgpu -- --ignored --test-threads=1` | PASS: 1 |
| `cargo clippy -p rustscan-gs --all-targets --features gpu-wgpu --no-deps -- -D warnings` | FAIL: baseline only (~33 lib + test extras). **New** `items_after_test_module` and `ProfileFaultStage` dead_code **cleared**. No global allow. |
| `git diff --check` | PASS |

### Remaining limitations

- Clippy `-D warnings` not green on pre-existing rustscan-gs diagnostics (gradient dead_code, chunks_exact, too_many_arguments, …).
- Topology still uses Instant for training telemetry vectors when profiler is off; those are not pipeline-profiler series.
- Stopped for independent re-review; not merged to main; C5–C8 not started.

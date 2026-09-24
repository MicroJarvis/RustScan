# RS-2026-002 Verification

## C1–C4 fix branch (round 3 — R02 / R07 / R04)

| Field | Value |
| --- | --- |
| Base SHA | `701d051814a29ab3ee4fbefc01355112f71b5a47` |
| Branch | `fix/rs-2026-002-c1-c4-review` |
| Worktree | `.worktrees/rs-2026-002-c1-c4-review` |
| Prior tip (CHANGES_REQUESTED) | `1786e6b83ecebb767edf53fcd59df0256730c2e9` |
| Round-3 code package | filled at commit |

R06 / R08 / `items_after_test_module` remain closed. C5–C8 not started.

### Round-3 remediations

| ID | Fix summary |
| --- | --- |
| R02 | `ContinuityFingerprintConfig` struct-order bytes (701d051 algorithm); frozen hash `5fc39ced…`; Value-order rejected as oracle |
| R07 | Tensor write/readback health probe; removed force-bool; child-process destroyed-device test |
| R04 | `profile_step` requires `profiler.enabled`; `note_debug_profile_readback` + Debug-log regression |

### Gate results (2026-09-24 round 3)

Logs: `artifacts/runs/rs-2026-002-c1-c4-rereview-round3/`

| Command | Result |
| --- | --- |
| `cargo fmt --all -- --check` | PASS |
| `cargo check --workspace --all-targets` | PASS (pre-existing viewer warnings) |
| `cargo test -p rustscan-gs --all-targets --features gpu-wgpu -- --test-threads=1` | PASS: lib 192, CLI 25, bounded 1, checkpoint 53 |
| `cargo test -p rustscan-gs --no-default-features --lib -- --test-threads=1` | PASS: 38 |
| `cargo test -p rustscan-gs --test integration_test --features gpu-wgpu -- --ignored --test-threads=1` | PASS: 1 |
| `cargo clippy -p rustscan-gs --all-targets --features gpu-wgpu --no-deps -- -D warnings` | FAIL: baseline only (chunks_exact, too_many_arguments, gradient dead_code, …). **No new diagnostics** in changed round-3 files. |
| `git diff --check` | PASS |

### Remaining limitations

- Clippy `-D warnings` not green on pre-existing rustscan-gs diagnostics.
- Destroyed-device regression uses intentional `Device::destroy` in a child process.
- Stopped for independent re-review; not merged to main.

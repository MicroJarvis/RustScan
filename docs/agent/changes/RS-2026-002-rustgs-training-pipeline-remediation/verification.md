# RS-2026-002 Verification

## C1–C4 fix branch (round 4 — R04 diagnostic gate)

| Field | Value |
| --- | --- |
| Base SHA | `701d051814a29ab3ee4fbefc01355112f71b5a47` |
| Branch | `fix/rs-2026-002-c1-c4-review` |
| Worktree | `.worktrees/rs-2026-002-c1-c4-review` |
| Prior tip | `eb8f3f3bf866f13a92c0c6e4505c3607dec9f841` |
| Round-4 code package | `cc4323b57f8aa23bd268583a93c804c58572cf38` |

R02 / R06 / R07 / R08 remain closed. C5–C8 not started.

### Round-4 remediation

| ID | Fix summary |
| --- | --- |
| R04 | `should_log_diagnostics` requires `profiler_on`; six diagnostic `into_scalar_async` paths call `note_debug_profile_readback`; disabled+Debug test asserts total readbacks == 0 and no profile/diagnostics log lines |

### Gate results (2026-09-25 round 4)

Logs: `artifacts/runs/rs-2026-002-c1-c4-rereview-round4-r04/`

| Command | Result |
| --- | --- |
| `cargo fmt --all -- --check` | PASS |
| `cargo check --workspace --all-targets` | PASS with `POSELIB_ROOT` pointing at main-tree PoseLib (worktree `third_party/native/PoseLib` empty). Without it: FAIL `rustscan-sfm` build.rs PoseLib missing (env, not this change). Targeted `cargo check -p rustscan-gs --all-targets --features gpu-wgpu` PASS. |
| `cargo test -p rustscan-gs --lib --features gpu-wgpu -- --test-threads=1` | PASS: 192 (`profiler_disabled_with_debug_log_skips_profile_readbacks` ok) |
| `cargo test -p rustscan-gs --no-default-features --lib -- --test-threads=1` | PASS: 38 |
| `cargo test -p rustscan-gs --test integration_test --features gpu-wgpu -- --ignored --test-threads=1` | PASS: 1 |
| `cargo clippy -p rustscan-gs --all-targets --features gpu-wgpu --no-deps -- -D warnings` | FAIL: baseline only (~33). **No new diagnostics** in `trainer.rs` / `gpu_profiler.rs`. |
| `git diff --check` | PASS |

### Clippy baseline (unchanged; not introduced this round)

duplicated attribute; unused `Int` import; dead_code in `gradient_check` / radix / prefix_sum / device_status / loss / optimizer; `chunks_exact` const size; `too_many_arguments`; `clone_on_copy`; field_reassign_with_default; useless_conversion; complex type; identical if blocks; useless `vec!`.

### Remaining limitations

- Clippy `-D warnings` not green on pre-existing rustscan-gs diagnostics.
- Worktree PoseLib checkout empty; workspace check needs `POSELIB_ROOT` or vendored tree.
- Stopped for independent re-review; not merged to main.

## Prior: round 3 — R02 / R07 / R04

| Field | Value |
| --- | --- |
| Round-3 code package | `4f539d8af475936022c2a930d856a6c2765a4136` |
| Tip after docs | `eb8f3f3bf866f13a92c0c6e4505c3607dec9f841` |
| Logs | `artifacts/runs/rs-2026-002-c1-c4-rereview-round3/` |

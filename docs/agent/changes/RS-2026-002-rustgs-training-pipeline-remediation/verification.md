# RS-2026-002 Verification

## C5 frame selection (pending independent review)

| Field | Value |
| --- | --- |
| Base SHA | `e6db819e43764d536fc51a83e72395e8aa76632d` (main at worktree creation) |
| Branch | `agent/RS-2026-002/c5-frame-selection` |
| Worktree | `.worktrees/rs-2026-002-c5-frame-selection` |
| Code tip | `841c9d28af3b66354836a6d7cbebcc4bbe8fbeed` |

C5: `implemented_pending_review`. C6–C8: `not_started`. Not merged; not marked complete by implementer.

### C5 remediations

| Item | Summary |
| --- | --- |
| Stable ID | `ScenePose.frame_id = image.image_id as u64`; missing images do not renumber survivors |
| FrameSelection | include/exclude → max_frames → stride → selection fingerprint (one implementation) |
| Split manifest | `--frame-split-manifest` + `--eval-split in-view\|holdout`; startup rejects dup/unknown/overlap/fingerprint mismatch; holdout requires manifest |
| Report | `eval_frame_ids` / `train_frame_ids` / worst frames are `u64`; split kind + manifest/selection fingerprints persisted |
| Examples | `evaluate_psnr` / `rustgs_eval_suite` / `rustgs_residual_heatmap` use one `FrameSelection`; evaluate/crop reuse selected dataset with max=0 stride=1 |
| Checkpoint | Pre-C5 enumerated identity accepted with warn; otherwise reject with explicit C5 frame-identity message |
| Report IDs | `train_frame_ids` = canonical selection; `train_loader_frame_ids` = oversampled/shuffled loader order |

### Gate results (review fixes)

Logs: `artifacts/runs/rs-2026-002-c5-review-fixes/`

Environment: `POSELIB_ROOT=/Users/tfjiang/Projects/RustScan/third_party/native/PoseLib`

| Command | Result |
| --- | --- |
| `cargo fmt --all -- --check` | PASS |
| `cargo test -p rustscan-gs --lib --features gpu-wgpu -- --test-threads=1` | PASS: **207** |
| `cargo test -p rustscan-gs --test checkpoint_resume --features gpu-wgpu -- --test-threads=1` | PASS: **53** |
| `cargo check -p rustscan-gs --all-targets --features gpu-wgpu` | PASS |
| `git diff --check` | PASS |

### Known limitations

- `ColmapConfig::{max_frames,frame_stride}` still apply a max/stride-only `FrameSelection` at load end for API compatibility when callers do not use include/exclude; the CLI and migrated examples load full then select once.
- Clippy `-D warnings` not part of this C5 gate matrix.
- Pre-C5 identity acceptance does not rewrite the checkpoint on disk; re-save to persist stable-image_id hashes.

## C1–C4 merged to main (2026-09-25)

| Field | Value |
| --- | --- |
| Merge | `--no-ff` `fix/rs-2026-002-c1-c4-review` |
| Merge commit | `4fdf506e6819fd30c74ad8245e85c9f1df625220` |
| Main before | `c71f8092778e2abb1feaaf03c9d784ea25434349` |
| Included tip | `993ac093344d68096e1bb92cd3c6d620d8c03a58` |
| Logs | `artifacts/runs/rs-2026-002-c1-c4-main-merge/` |

C1–C4: `merged`. C5–C8: `not_started`. Not pushed.

Post-merge gates (`POSELIB_ROOT` = main-tree PoseLib): workspace `cargo check` PASS; `rustscan-gs` gpu-wgpu lib tests PASS **192**; `cargo fmt --all -- --check` PASS; `git diff --check` PASS.

## C1–C4 final handoff (R04 review passed)

| Field | Value |
| --- | --- |
| Base SHA | `701d051814a29ab3ee4fbefc01355112f71b5a47` |
| Branch | `fix/rs-2026-002-c1-c4-review` |
| Worktree | `.worktrees/rs-2026-002-c1-c4-review` |
| R04 code package | `cc4323b57f8aa23bd268583a93c804c58572cf38` |
| Prior verification docs | `d878109f42dd67f38bcf88a1894ad70609f1811a` |
| Handoff tip | `783eb2bac8ab0e6878e8d0c42a5b9dfb78a31b6e` |

C1–C4: `implemented_pending_merge`. C5–C8 not started. No merge, push, or worktree deletion.

### Review items closed

| ID | Status | Evidence |
| --- | --- | --- |
| R02 | closed | Historical struct-order continuity fingerprint (`701d051` algorithm); frozen fixture hash; profiler-only restore; real param changes rejected |
| R04 | closed | `profile_step` and `should_log_diagnostics` both require `profiler.enabled`; disabled+Debug path: **total Debug GPU readbacks = 0** (profile_step + 6 diagnostic `into_scalar_async`); no `WGPU train profile step` / `WGPU train diagnostics step` logs |
| R06 | closed | Split `forward_gpu_sampled` / `forward_cpu_submit` series retained |
| R07 | closed | Tensor write/readback health probe; child-process destroyed-device test; no force-bool |
| R08 | closed | Unsupported reports set `measurement_success=false`; illegal combinations rejected |

### Final gate results (2026-09-25)

Logs: `artifacts/runs/rs-2026-002-c1-c4-final-handoff/`

Environment: `POSELIB_ROOT=/Users/tfjiang/Projects/RustScan/third_party/native/PoseLib` (worktree `third_party/native/PoseLib` is empty; **workspace check depends on this `POSELIB_ROOT`**).

| Command | Result |
| --- | --- |
| `cargo fmt --all -- --check` | PASS |
| `cargo check --workspace --all-targets` | PASS with `POSELIB_ROOT` set as above |
| `cargo test -p rustscan-gs --lib --features gpu-wgpu -- --test-threads=1` | PASS: **192** (`profiler_disabled_with_debug_log_skips_profile_readbacks` ok; R04 total readbacks asserted **0**) |
| `cargo test -p rustscan-gs --no-default-features --lib -- --test-threads=1` | PASS: 38 |
| `cargo test -p rustscan-gs --test integration_test --features gpu-wgpu -- --ignored --test-threads=1` | PASS: 1 |
| `cargo clippy -p rustscan-gs --all-targets --features gpu-wgpu --no-deps -- -D warnings` | FAIL: **pre-existing rustscan-gs baseline only**. No diagnostics in `trainer.rs` / `gpu_profiler.rs` / `checkpoint.rs`. |
| `git diff --check` | PASS |

### Clippy baseline (unchanged; not introduced by C1–C4)

duplicated attribute; unused `Int` import; dead_code in `gradient_check` / radix / prefix_sum / device_status / loss / optimizer; `chunks_exact` const size; `too_many_arguments`; `clone_on_copy`; field_reassign_with_default; useless_conversion; complex type; identical if blocks; useless `vec!`.

### Remaining limitations

- Clippy `-D warnings` is not green because of the pre-existing rustscan-gs baseline above.
- Workspace `cargo check` requires `POSELIB_ROOT` (or a vendored PoseLib tree) in this worktree.
- Destroyed-device health regression uses intentional `Device::destroy` in a child process.
- C5–C8 not started. Branch not merged to main.

## Prior: round 4 — R04 diagnostic gate

| Field | Value |
| --- | --- |
| Round-4 code package | `cc4323b57f8aa23bd268583a93c804c58572cf38` |
| Tip after docs | `d878109f42dd67f38bcf88a1894ad70609f1811a` |
| Logs | `artifacts/runs/rs-2026-002-c1-c4-rereview-round4-r04/` |

## Prior: round 3 — R02 / R07 / R04

| Field | Value |
| --- | --- |
| Round-3 code package | `4f539d8af475936022c2a930d856a6c2765a4136` |
| Tip after docs | `eb8f3f3bf866f13a92c0c6e4505c3607dec9f841` |
| Logs | `artifacts/runs/rs-2026-002-c1-c4-rereview-round3/` |

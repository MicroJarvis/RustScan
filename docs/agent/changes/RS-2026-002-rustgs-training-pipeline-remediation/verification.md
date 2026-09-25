# RS-2026-002 Verification

## C5 frame selection — P2 review-fix (pending independent review)

| Field | Value |
| --- | --- |
| Review baseline | `bb4697d3f1407fcbc2b8bca922c26da0c3a2b344` |
| Branch | `agent/RS-2026-002/c5-frame-selection` |
| Worktree | `.worktrees/rs-2026-002-c5-frame-selection` |
| Tip after this package | *(pinned at commit time)* |

C5: `implemented_pending_review`. C6–C8: `not_started`. Not merged; not marked complete by implementer.

### P2 findings closed

| ID | Fix |
| --- | --- |
| static_162 missing-image drift | `list_colmap_frame_candidates` + `static_162_allowed_stable_ids_from_candidates` reconstruct pre-C5 `take(180)` enumerated exclude `76..=93`; filtered-dataset helper refuses; eval suite uses candidates and marks gate `Inapplicable` when unrestorable |
| Resume selection inheritance | `resolve_checkpoint_selection`: provide→validate/match; omit+checkpoint→inherit after dataset verify; v1/v2 stay absent; observer uses resolved meta |

### Gate results

Logs: `artifacts/runs/rs-2026-002-c5-review-fix-p2/`

Environment: `POSELIB_ROOT=/Users/tfjiang/Projects/RustScan/third_party/native/PoseLib`

| Command | Result |
| --- | --- |
| `cargo fmt --all -- --check` | PASS |
| `cargo check --workspace --all-targets` | PASS with `POSELIB_ROOT` |
| `cargo test -p rustscan-gs --all-targets --features gpu-wgpu -- --test-threads=1` | PASS: lib **212**; checkpoint_resume **58** |
| `cargo clippy -p rustscan-gs --all-targets --features gpu-wgpu --no-deps -- -D warnings` | FAIL: **pre-existing baseline only** (gradient_check/radix/prefix_sum/loss/optimizer/autodiff/forward/topology…). No new diagnostics in C5-touched checkpoint/runtime/split/colmap/eval-suite/checkpoint_resume |
| `git diff --check` | PASS |

### Compatibility boundaries

- static_162 allowed IDs require original COLMAP candidates (or explicit manifest); guessing from filtered poses is rejected.
- Missing image inside prefix outside exclude → 161 frames ending at 180 (not 162/181).
- Resume without `with_selection` inherits verified v3 selection; conflict with explicit selection rejects; v2 remains `None`.

### Known limitations

- Clippy `-D warnings` still red on the pre-existing rustscan-gs baseline.
- C6–C8 not started.

## C5 frame selection — prior review-fix package

| Field | Value |
| --- | --- |
| Review baseline | `62739c37d301b95f76c727e921fc96f58b074137` |
| Branch | `agent/RS-2026-002/c5-frame-selection` |
| Worktree | `.worktrees/rs-2026-002-c5-frame-selection` |
| Tip after this package | `a327802ed9dad759d58fd922e6ea9409f513a494` |

C5: `implemented_pending_review`. C6–C8: `not_started`. Not merged; not marked complete by implementer.

### Review findings closed this package

| ID | Fix |
| --- | --- |
| Resume wiring | `prepare_resume_runtime` uses `match_checkpoint_dataset_identity` (same path as `validate_dataset_and_config`); reconstruction/config remain strict |
| Legacy ID mapping | Frozen `assign_pre_c5_enumerated_ids`; gap-free-only auto-accept; missing-image / exclude / oversample cases rejected with explicit reason |
| static_162 | superseded by P2 candidate-list reconstruction (prior filtered-pose pin was insufficient with missing images) |
| Checkpoint selection meta | `TRAINING_CHECKPOINT_VERSION=3` + `CheckpointFrameSelectionMeta`; v2 loads as `V2SelectionMetaAbsent` with `selection=None` (never invented) |

### Gate results

Logs: `artifacts/runs/rs-2026-002-c5-review-fix/`

Environment: `POSELIB_ROOT=/Users/tfjiang/Projects/RustScan/third_party/native/PoseLib`

| Command | Result |
| --- | --- |
| `cargo fmt --all -- --check` | PASS |
| `cargo check --workspace --all-targets` | PASS with `POSELIB_ROOT` |
| `cargo test -p rustscan-gs --all-targets --features gpu-wgpu -- --test-threads=1` | PASS: lib **211**; checkpoint_resume **56**; other targets green |
| `cargo clippy -p rustscan-gs --all-targets --features gpu-wgpu --no-deps -- -D warnings` | FAIL: **pre-existing rustscan-gs baseline only** (see below). No diagnostics in C5-touched `checkpoint.rs` / `runtime.rs` / `split.rs` / `events.rs` / `train_command.rs` / eval suite after fixing one new `clone_on_copy` in the resume unit test |
| `git diff --check` | PASS |

### Clippy baseline vs this package

Pre-existing (unchanged class): duplicated attribute; unused `Int` import; dead_code in gradient_check / radix / prefix_sum / device_status / loss / optimizer; `chunks_exact` const size; `too_many_arguments` in autodiff/forward/project; `clone_on_copy` in topology; field_reassign_with_default; useless_conversion; type_complexity in optimization_report tests; identical if blocks; useless `vec!`.

**New in this package (fixed before handoff):** `clone_on_copy` on `Intrinsics` in `prepare_resume_runtime_accepts_pre_c5_gap_free_dataset_hash` — removed.

### Compatibility boundaries

- Pre-C5 gap-free enumerated identity (unique stable IDs, old IDs exactly `0..n-1`) accepted at resume with warn; re-save persists stable-image_id hash.
- Gapped IDs (missing images), exclude-filtered non-compact IDs, and oversampled duplicates **cannot** be reconstructed from the final pose list alone → hard reject with reason.
- Checkpoint v1/v2 selection fields are **absent** (`None`); resume consistency check only runs when both sides present selection meta.
- Canonical `train_stable_ids` ≠ `train_loader_frame_ids` (oversample/shuffle order).

### Known limitations

- `ColmapConfig::{max_frames,frame_stride}` still apply a max/stride-only `FrameSelection` at load end for API compatibility.
- Clippy `-D warnings` remains red on the pre-existing baseline above.
- C6–C8 not started.

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

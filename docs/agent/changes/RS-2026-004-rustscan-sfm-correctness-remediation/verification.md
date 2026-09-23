# RS-2026-004 Verification

## Record Rules

For every task record the owner, base and final commit, command, operating
system, feature set, native dependencies, fixtures, result, duration, skipped
tests, limitations, and next action. Do not replace unavailable native or GPU
evidence with an ignored/skipped test.

## Review Baseline — 2026-09-20

Planning base commit:
`d2cdb8de7d63e448022f02e69e622c90a501a591` on `main`.

The planning checkout also contained unrelated, pre-existing changes to
`README.md`, `ROADMAP.md`, `docs/current-project-status.md`, and `docs/index.md`.
They are outside this change and must not enter its implementation commits.

| Command | Baseline result |
| --- | --- |
| `cargo test -p rustscan-sfm --test sequence_registration -- --nocapture` | **FAIL:** 61 passed, 11 failed. Ten tests panic at `rustscan-sfm/src/sfm/mapper.rs:422` with filtered frame length 4 and index 4. One private-snapshot test timed out in the parallel run. |
| `cargo test -p rustscan-sfm --test sequence_registration keyframe_database_metadata_uses_private_snapshot_after_source_replacement` | **PASS:** 1 passed when run alone; treat the parallel timeout separately from the mapper indexing defect. |
| `cargo test -p rustscan-sfm --test sequence_registration --no-default-features` | **PASS:** 55 passed. This feature set does not exercise all default-feature sequence paths and is not sufficient final evidence. |
| `cargo test -p rustscan-sfm --lib gpu:: -- --nocapture --test-threads=1` | **PASS with limitation:** 69 passed. Some PnP tests detect the known macOS AGX/XPC shader compiler failure and skip execution, so GPU PnP remains unproved on this host. |
| `cargo test -p rustscan-sfm view_graph_calibration --lib -- --nocapture` | **PASS:** 2 passed, 818 filtered out. Neither test enables `refine_intrinsics`, so the reviewed focal-search defect has no regression coverage. |

The default-feature build emitted dozens of RustSFM warnings, including unused
imports, dead code, and a private-interface warning. Final targeted Clippy with
`-D warnings` is required; pre-existing warnings must be removed or documented
with the narrow allowances permitted by `rust-style.md`.

## Task Results

### T0 — Baseline

Status: complete. Baseline evidence is recorded below. Three focused regressions
were added in `rustscan-sfm/src/sfm/mapper.rs`. They currently fail by panicking
at `mapper.rs:422`; their assertions expect a contextual error that names the
missing image, not a panic. Acceptance conditions were not changed.

Owner: `cursor-agent`  
Base commit: `d2cdb8de7d63e448022f02e69e622c90a501a591`  
Final commit: `c88bca19f216f5c5e213255464afa4422b8f2be4`  
Branch: `agent/RS-2026-004/rustscan-sfm-correctness-remediation`  
Worktree: `/Users/tfjiang/Projects/RustScan/.worktrees/rs-2026-004-rustscan-sfm-correctness`

The worktree HEAD matches the recorded base commit. After the tests, the only
untracked path added there is this change package. Ignored logs live under
`artifacts/runs/rs-2026-004-t0/`. The worktree does not contain the unrelated dirty
`README.md`, `ROADMAP.md`, `docs/current-project-status.md`, or `docs/index.md`
changes, nor the pre-existing dirty `docs/nalgebra-unification-todo.md` change
still present on the main checkout. Those main-checkout changes were not
committed, cleaned, or overwritten.

#### Environment

- OS: macOS 27.0 (build 26A428), Darwin 27.0.0 arm64, Apple M5 Max.
- Toolchain: rustc 1.98.1 (`48a229cea`, 2026-09-01), cargo 1.98.1.
- Default features for every command except the explicit no-default run:
  `ceres-ba`, `vlfeat-sift`, `gpu-wgpu`, `poselib`.
- Ceres Solver 2.2.0, Homebrew `ceres-solver` 2.2.0_5, used through the
  macOS `ceres-ba` system feature (`ceres-solver` 0.5.1 under
  `third_party/rust/ceres-solver`, `ceres-solver-sys` 0.5.3). Also present:
  glog 0.6.0_1, gflags 2.3.0, SuiteSparse 7.12.2.
- Eigen 5.0.1 (`pkg-config eigen3`).
- PoseLib v2.0.5, `7e9f5f53372e43f89655040d4dfc4a00e5ace11c`. The new worktree
  did not have the submodule checked out. `git submodule update --init
  third_party/native/PoseLib` checked out that recorded commit before the
  successful default-feature commands. The first attempt without it failed in
  `rustscan-sfm/build.rs` and is not a test result. Logs:
  `artifacts/runs/rs-2026-004-t0/setup-failure/` in the worktree.
- VLFeat source was already present at `third_party/native/vlfeat`.
- GPU adapter: Apple M5 Max, 40 cores, Metal 4. The shader compiler reports
  `AGXMetalG17X` and `XPC_ERROR_CONNECTION_INTERRUPTED`.
- Logging-only setting: `CARGO_TERM_COLOR=never`. It is not part of the test
  command.
- Full logs: `.worktrees/rs-2026-004-rustscan-sfm-correctness/artifacts/runs/rs-2026-004-t0/`.
  Durations below are `/usr/bin/time -p` real time and include compilation.

#### Commands and results

| Command | Result | Duration | Skips |
| --- | --- | --- | --- |
| `cargo test -p rustscan-sfm --test sequence_registration -- --nocapture` | **FAIL:** 62 passed, 10 failed, 0 ignored. Test time 24.02s. | 58.08s | none |
| `cargo test -p rustscan-sfm --test sequence_registration keyframe_database_metadata_uses_private_snapshot_after_source_replacement` | **PASS:** 1 passed, 71 filtered out. Test time 0.01s. | 0.36s | none |
| `cargo test -p rustscan-sfm --test sequence_registration --no-default-features` | **PASS:** 55 passed, 0 failed, 0 ignored. Test time 0.22s. This feature set still does not exercise every default-feature sequence path. | 26.14s | none |
| `cargo test -p rustscan-sfm --lib colmap::tests -- --nocapture --test-threads=1` | **PASS with an ignored fixture test:** 22 passed, 0 failed, 1 ignored, 797 filtered out. Test time 0.02s. | 17.22s | `real_colmap_sparse_tracks_recover_registered_image_pose_with_pnp` ignored: requires `artifacts/inputs/flowers2_colmap`. This ignore is not import/export evidence. |
| `cargo test -p rustscan-sfm --lib view_graph_calibration -- --nocapture` | **PASS:** 2 passed, 818 filtered out. Tests: `rejects_matches_with_small_triangulation_angle`, `filter_rotation_inconsistent_pairs_removes_outlier_edge`. Neither enables `refine_intrinsics`. | 0.32s | none |
| `cargo test -p rustscan-sfm --lib ba:: -- --nocapture --test-threads=1` | **PASS:** 33 passed, 0 failed, 787 filtered out. Test time 0.08s. This is the existing suite only. It does not add the missing deterministic unusable-solution atomicity coverage required by T5. | 0.40s | none |
| `cargo test -p rustscan-sfm --lib gpu:: -- --nocapture --test-threads=1` | **PASS with limitation:** 69 passed, 0 failed, 0 ignored, 751 filtered out. Test time 152.50s. | 152.82s | Seven PnP-focal tests returned before GPU execution and Cargo still counted them passed. They are not GPU execution evidence. |

The ten default-feature sequence failures are the filtered-index panic
`index out of bounds: the len is 4 but the index is 4` at
`rustscan-sfm/src/sfm/mapper.rs:422`:

- `blank_sequence_frame_returns_unresolved_incomplete_coverage`
- `complete_sequence_registers_all_six_arbitrary_frame_ids_on_cpu`
- `default_pnp_seed_is_stable_across_intervening_registration_calls`
- `narrow_round_does_not_publish_same_round_registrations_as_support`
- `pause_after_sparse_publish_resumes_from_immutable_keyframes`
- `pause_before_sparse_publish_preserves_old_model_byte_for_byte`
- `sequence_memory_no_pending_uses_only_caller_floor`
- `taskflow_sequence_preserves_typed_pause_before_final_ba`
- `wide_round_can_use_tracks_committed_by_narrow_non_keyframe`
- `taskflow_sequence_waits_for_budget_and_runs_final_global_ba_once`

The last test reports the panic at
`rustscan-sfm/tests/sequence_registration.rs:1377` because the test unwraps a
worker `JoinHandle`. The worker itself panics at `mapper.rs:422` with the same
length/index message.

GPU executions that did not run, despite Cargo marking the tests passed:

- `gpu::pnp_focal::tests::wgpu_pnp_focal_p3p_column_zero_candidate_reprojects_sample`
- `gpu::pnp_focal::tests::wgpu_pnp_focal_p3p_reorders_adverse_baseline_before_solving`
- `gpu::tests::wgpu_pnp_focal_noisy_refinement_does_not_reduce_support`
- `gpu::tests::wgpu_pnp_focal_out_of_bounds_focal_returns_none`
- `gpu::tests::wgpu_pnp_focal_p3p_batch_candidate_projects_its_sample`
- `gpu::tests::wgpu_pnp_focal_p3p_candidate_projects_its_sample`
- `gpu::tests::wgpu_pnp_focal_recovers_synthetic_pose_and_focal`

Each printed `known macOS AGX compiler service failure` and
`gpu pnp-focal candidate pipeline creation failed: Validation Error`.
`agx_pipeline_failure_skip_is_macos_only_and_preserves_other_errors` also
prints a skip line, but that test only checks the skip helper; it is not a
solver run.

#### Baseline drift

The planning record was 61 passed and 11 failed, with ten `mapper.rs:422`
panics and one private-snapshot timeout in the parallel run. This rerun is
62 passed and 10 failed. The private-snapshot test passed inside the parallel
suite and passed again when run alone. The ten mapper index panics remain.
This drift is recorded only. Acceptance conditions were not rewritten.

Default-feature `rustscan-sfm` lib compilation emitted 56 warnings, and the
sequence integration test emitted 2 more. The no-default-feature build emitted
43 lib warnings and 7 integration-test warnings. Targeted Clippy with
`-D warnings` was not run; it remains part of the final matrix, not this
baseline.

#### Focused failing regressions

Command:

`cargo test -p rustscan-sfm --lib reports_image_name_instead_of_panicking -- --test-threads=1 --nocapture`

Result: **FAIL**, as required before the fix. 0 passed, 3 failed, 820 filtered
out. Test time 0.05s. Wall time 7.72s, including the lib-test rebuild.
Environment is the same default-feature host recorded above. No tests were
ignored.

Each test panics at `rustscan-sfm/src/sfm/mapper.rs:422` with
`index out of bounds: the len is 2 but the index is 2`:

| Test | Missing image the assertion names |
| --- | --- |
| `filtered_leading_support_reports_image_name_instead_of_panicking` | `a.png` |
| `filtered_middle_support_reports_image_name_instead_of_panicking` | `m.png` |
| `disconnected_target_reports_image_name_instead_of_panicking` | `c.png` |

The tests call `expect_err` and require the error text to contain that image
name. They do not use `#[should_panic]`. `cargo fmt -p rustscan-sfm` passes.

#### Not run

The final `tasks.yaml` matrix was not run: workspace `cargo check`, targeted
Clippy, `--all-features` lib tests, the RustViewer loader test,
`scripts/check_cpu_glam.py`, and `git diff --check`. `cargo fmt --all -- --check`
was not run for the whole workspace; only `cargo fmt -p rustscan-sfm` was run.

#### Next action

T1 should make the three new regressions return contextual errors, then clear
the ten existing sequence panics. Do not change acceptance conditions. Wave 1
tasks T2, T5, and T6 need their own branches and worktrees before parallel
work; this branch is already checked out in the T0 worktree.

### T1 — Mapper Identity

Status: complete. Filtered database images keep a stable retained-image record.
Single-target registration looks up the target and requested supports by name.
A support or target that is absent from the database is a contextual error.
A database image that is present but not match-connected does not panic and
does not replace another image's metadata. Registered reference images that the
cache omits stay in the seed under their own names.

Owner: `cursor-agent`

Base commit: `b01cd442eba0ff0e2716c67473e819c365ec5bf2`

Final commit: `bbedfbc519987177429c080becb762fef41760bb`

Branch: `agent/RS-2026-004/t1-mapper-identity`

Worktree: `/Users/tfjiang/Projects/RustScan/.worktrees/rs-2026-004-t1-mapper-identity`

Changed files:

- `rustscan-sfm/src/sfm/mapper.rs`
- `rustscan-sfm/src/sfm/mapper/reconstruction_input.rs`
- `rustscan-sfm/src/sfm/mapper/image_features_tests.rs`
- `docs/agent/changes/RS-2026-004-rustscan-sfm-correctness-remediation/tasks.md`
- `docs/agent/changes/RS-2026-004-rustscan-sfm-correctness-remediation/tasks.yaml`
- `docs/agent/changes/RS-2026-004-rustscan-sfm-correctness-remediation/verification.md`

The worktree was created from the T0 tip above. `main` and the T0 worktree were
not modified. PoseLib was already checked out at
`7e9f5f53372e43f89655040d4dfc4a00e5ace11c`.

#### Environment

- OS: macOS 27.0 (build 26A428), Darwin 27.0.0 arm64, Apple M5 Max.
- Toolchain: rustc 1.98.1 (`48a229cea`, 2026-09-01), cargo 1.98.1
  (`797e8a9bc`, 2026-08-05).
- Default features except the explicit no-default run: `ceres-ba`,
  `vlfeat-sift`, `gpu-wgpu`, `poselib`.
- Ceres Solver 2.2.0 via Homebrew `ceres-solver` 2.2.0_5. Eigen 5.0.1.
- PoseLib v2.0.5, `7e9f5f53372e43f89655040d4dfc4a00e5ace11c`.
- GPU adapter: Apple M5 Max, Metal 4, `AGXMetalG17X` /
  `XPC_ERROR_CONNECTION_INTERRUPTED`, same host limitation recorded in T0.
- Logging-only setting: `CARGO_TERM_COLOR=never`.
- Logs: `artifacts/runs/rs-2026-004-t1/` in this worktree. Durations are
  `/usr/bin/time -p` real time and include compilation unless noted.

#### Commands and results

Before the fix, on this worktree at the T0 tip:

```text
cargo test -p rustscan-sfm --lib reports_image_name_instead_of_panicking -- --test-threads=1 --nocapture
```

Result: **FAIL**, exit 101. The three T0 regressions still panicked at
`rustscan-sfm/src/sfm/mapper.rs:422` with filtered length 2 and index 2
(`index out of bounds: the len is 2 but the index is 2`). Wall time was about
49s, dominated by the first compile.

After the fix:

| Command | Result |
| --- | --- |
| `cargo fmt --all -- --check` | **PASS**, exit 0, real 1.18s on the final tree. |
| `cargo check --workspace --all-targets` | **PASS**, exit 0, real 3.94s (`check2.log`). Incremental after the earlier full check. |
| `cargo clippy -p rustscan-sfm --all-targets --all-features -- -D warnings` | **pre-existing failure**, exit 101, real 20.20s (`clippy2.log`). `could not compile rustscan-slam (lib) due to 84 previous errors`. The command applies `-D warnings` to path dependencies and stops in `rustscan-slam` before it produces a `rustscan-sfm` result. Those diagnostics are outside T1. No T1 file was edited to hide them. |
| `cargo test -p rustscan-sfm --all-targets` | **blocked**, exit 143, real 905s (`sfm_all.log`, `summary.txt`). The lib test process stayed asleep in `WgpuContext::new_async` (`pollster::block_on`) with worker threads waiting on the same mutex. It produced no final test summary. The process was stopped with SIGTERM so the later commands could run. This is the known parallel GPU-initialization deadlock on this host, not an identity assertion failure. |
| `cargo test -p rustscan-sfm --test sequence_registration -- --test-threads=1 --nocapture` | **PASS**, exit 0, real 60.18s, test time 59.86s (`sequence3.log`). 72 passed, 0 failed, 0 ignored. The ten previous mapper index panics are gone. |
| `cargo test -p rustscan-sfm --no-default-features --lib -- --test-threads=1` | **PASS**, exit 0, real 59.81s, test time 53.79s (`nodefault2.log`). 619 passed, 0 failed, 19 ignored. |
| `git diff --check` | **PASS**, exit 0, after removing trailing whitespace from this record. |

Focused identity tests after the fix, same filter as the baseline:

```text
cargo test -p rustscan-sfm --lib reports_image_name_instead_of_panicking -- --test-threads=1 --nocapture
```

**PASS**, 3 passed. Dropped leading support (`a.png`), dropped middle support
(`m.png`), and disconnected target (`c.png`) return errors that name the
missing image.

Additional regressions in `mapper.rs`:
`database_frames_skip_paths_absent_from_match_connected_cache` checks retained
source indices 0 and 2, and
`retained_reference_setup_keeps_camera_and_seed_after_dropping_middle_image`
checks multi-camera ids 11, 99, and 42 and that the retained `c.png` seed
translation stays 3 rather than the dropped middle image's 5.

#### Known limitations

- `cargo test -p rustscan-sfm --all-targets` did not finish. Parallel GPU
  context initialization deadlocked. Do not treat that run as package-wide
  pass evidence.
- Targeted Clippy with `-D warnings` fails in `rustscan-slam` on pre-existing
  diagnostics. T1 did not clear unrelated RustSFM warnings.
- A target image that exists in the database but is not match-connected still
  returns `Ok(candidate: None)` and names the target in the debug log, so
  sequence registration can record an unresolved attempt. A requested support
  in that situation is an error; see the review remediation below.
- Registered reference images that are not requested supports and are omitted
  by the match cache are still copied back from the reference by name. They
  are not given another image's database id. Requested supports are not
  skipped this way.
- GPU PnP execution on this Mac remains unproved, as in T0.

#### Next action

Start T2 on `agent/RS-2026-004/t2-reconstruction-validation` in
`.worktrees/rs-2026-004-t2-reconstruction-validation`, based on the T1 review
commit recorded below. Do not merge this branch to `main`.

#### T1 review remediation

Status: complete for the `bbedfbc` identity blockers. Follow-up P1s on
`a33ef4a` are recorded in the next subsection.

- A retained image that is not in the reference, when a database cache is also
  present, keeps the database image id and camera id. It is not assigned
  `idx + 1` or camera index 0.
- Frame membership is merged by stable frame id into the reference frame list.
  A database frame index is not written into that list. A frame id that cannot
  be mapped uniquely, including a conflicting rig, returns an error that names
  the image.
- A requested support that exists in the database and the registered reference
  but is absent from the match-connected cache returns an error that names the
  support. The mapper does not skip it.

Owner: `cursor-agent`

Base commit: `bbedfbc519987177429c080becb762fef41760bb`

Final commit: `a33ef4a5ae4c5a70b603be44b7a98ce3febad67f`

Branch: `agent/RS-2026-004/t1-mapper-identity`

Worktree: `/Users/tfjiang/Projects/RustScan/.worktrees/rs-2026-004-t1-mapper-identity`

Changed files:

- `rustscan-sfm/src/sfm/mapper.rs`
- `rustscan-sfm/src/sfm/mapper/reconstruction_input.rs`
- `rustscan-sfm/src/sequence_registration.rs`
- `docs/agent/changes/RS-2026-004-rustscan-sfm-correctness-remediation/tasks.md`
- `docs/agent/changes/RS-2026-004-rustscan-sfm-correctness-remediation/tasks.yaml`
- `docs/agent/changes/RS-2026-004-rustscan-sfm-correctness-remediation/verification.md`

`main` and the other RS-2026-004 worktrees were not modified.

#### Environment

Same host as the T1 record above: macOS 27.0, Darwin 27.0.0 arm64, Apple M5
Max, rustc 1.98.1, cargo 1.98.1. Default features except the no-default run:
`ceres-ba`, `vlfeat-sift`, `gpu-wgpu`, `poselib`. `CARGO_TERM_COLOR=never`.
Durations are `/usr/bin/time -p` real time and include compilation.

#### Commands and results

| Command | Result |
| --- | --- |
| `cargo fmt --all -- --check` | **PASS**, exit 0, real 1.19s. |
| `cargo check --workspace --all-targets` | **PASS**, exit 0, real 4.85s. |
| `cargo test -p rustscan-sfm --lib -- --test-threads=1 reference_database` | **PASS**, 2 passed. |
| `cargo test -p rustscan-sfm --lib -- --test-threads=1 support_present_in_database reports_image_name_instead_of_panicking retained_reference_setup_keeps_camera database_frames_skip_paths` | **PASS**, 6 passed. |
| `cargo test -p rustscan-sfm --test sequence_registration -- --test-threads=1` | **PASS**, exit 0, real 69.90s, test time 63.78s. 72 passed. |
| `cargo test -p rustscan-sfm --no-default-features --lib -- --test-threads=1` | **PASS**, exit 0, real 62.94s, test time 55.74s. 622 passed, 19 ignored. |
| `git diff --check` | **PASS**, exit 0. |

#### Known limitations

- Reference-only setup, with no database cache, still assigns `idx + 1` and
  camera 0 to an image that is not in the reference.
- A disconnected target still returns `Ok(candidate: None)`. A requested
  support in that situation is an error; see the review P1 remediation below.
- `cargo test -p rustscan-sfm --all-targets` and targeted Clippy were not
  re-run in this subsection.

#### T1 review P1 remediation

Status: complete. Two P1 findings on `a33ef4a5ae4c5a70b603be44b7a98ce3febad67f`
are fixed without deleting, ignoring, or weakening existing acceptance.

1. Sequence registration no longer silently drops an unconnected support.
   Controlled degradation is explicit: the attempt diagnostic records
   `controlled support degradation`, names the removed support, keeps the
   original `support_frame_ids`, and continues only when at least one
   match-connected support remains. A lone or fully disconnected support set
   returns an error that names the target, the support, and
   `not match-connected`. Registration success after degradation still carries
   that diagnostic; insufficient remainder cannot report success.
2. Overlapping same-name reference/database images validate `image_id`,
   `camera_id`, and frame/rig identity when both sides provide frame/rig data.
   Any mismatch fails before setup construction and names the image plus both
   ID sets. Non-overlapping images keep the previous identity rules.

Owner: `cursor-agent`

Base commit: `a33ef4a5ae4c5a70b603be44b7a98ce3febad67f`

Final commit: `90f05093fb014bf76a3a96afff76adb84d7158c2`

Branch: `agent/RS-2026-004/t1-mapper-identity`

Worktree: `/Users/tfjiang/Projects/RustScan/.worktrees/rs-2026-004-t1-mapper-identity`

Changed files:

- `rustscan-sfm/src/sfm/mapper.rs`
- `rustscan-sfm/src/sfm/mapper/reconstruction_input.rs`
- `rustscan-sfm/src/sequence_registration.rs`
- `rustscan-sfm/tests/sequence_registration.rs`
- `docs/agent/changes/RS-2026-004-rustscan-sfm-correctness-remediation/tasks.md`
- `docs/agent/changes/RS-2026-004-rustscan-sfm-correctness-remediation/tasks.yaml`
- `docs/agent/changes/RS-2026-004-rustscan-sfm-correctness-remediation/verification.md`

#### Commands and results

Logs: `artifacts/runs/rs-2026-004-t1-review-p1/` in this worktree.

| Command | Result |
| --- | --- |
| `cargo fmt --all -- --check` | **PASS**, exit 0, real 1.18s. |
| `cargo check --workspace --all-targets` | **PASS**, exit 0, real 4.85s. |
| `cargo test -p rustscan-sfm --all-features --lib -- --test-threads=1` | **FAIL**, exit 101, real 253.53s, test time 245.51s. 802 passed, 2 failed, 19 ignored. All T1 identity and support-degradation regressions passed. The two failures are outside T1: `gpu::five_point_f32::tests::five_point_f32_actual_gpu_stages` and `gpu::pnp_focal::tests::wgpu_pnp_focal_p3p_reorders_adverse_baseline_before_solving`. Treat those GPU numerical cases as unavailable on this host. |
| `cargo test -p rustscan-sfm --all-features --test sequence_registration -- --test-threads=1` | **PASS**, exit 0, real 67.97s, test time 60.54s. 70 passed, 0 failed, 0 ignored. `wide_round_can_use_tracks_committed_by_narrow_non_keyframe` requires the controlled-degradation diagnostic and still keeps frame id 505 in `support_frame_ids`. |
| `cargo test -p rustscan-sfm --no-default-features --lib -- --test-threads=1` | **PASS**, exit 0, real 62.15s, test time 55.45s. 631 passed, 0 failed, 19 ignored. |
| `git diff --check` | **PASS**, exit 0. |

#### Known limitations

- Reference-only setup still assigns `idx + 1` / camera 0 when no database cache
  is present.
- Controlled degradation requires at least one remaining match-connected
  support. Empty remainder fails closed and does not report registration
  success.
- `--all-features --lib` GPU numerical failures above are outside T1 and do not
  clear the AGX/XPC adapter limitation recorded in T0.
- Targeted Clippy with `-D warnings` was not re-run; pre-existing
  `rustscan-slam` failures remain outside T1.

#### Next action

Start T2 from final commit `90f05093fb014bf76a3a96afff76adb84d7158c2`. Do not merge this branch to
`main`.

### T2 — Reconstruction Validation And IDs

Status: review_fix. Not reviewed.

Owner: `cursor-agent`

Base commit: `059dc5736bd2a27ffe523e192b9f8f75cbde0922`

Implementation commit: `7546a82a2e26349821b83fcb7e36dd1450fef7cf`

Branch: `agent/RS-2026-004/t2-reconstruction-validation`

Worktree: `/Users/tfjiang/Projects/RustScan/.worktrees/rs-2026-004-t2-reconstruction-validation`

Changed files:

- `rustscan-sfm/src/core/reconstruction_validation.rs`
- `rustscan-sfm/src/core/types.rs`
- `rustscan-sfm/src/lib.rs`
- `rustscan-sfm/src/sequence_registration.rs`
- `rustscan-sfm/src/sfm/mapper.rs`
- `rustscan-sfm/src/sfm/mapper/reconstruction_input.rs`
- `rustscan-sfm/src/sfm/observation_manager.rs`

Scope notes: `sequence_registration.rs` only calls `validate_structure` before
the existing sequence minimum-count checks. `reconstruction_input.rs` uses the
occupied-id allocator for new local image/camera ids and `try_point3d_id` when
seeding points. `mapper.rs` routes pre-export validation through
`validate_for_colmap_export`. `observation_manager.rs` allocates point ids from
the shared occupied-id allocator. Those call sites are the construction and
export gates required by the design; T1 identity behavior was not changed.
`rustscan-sfm/src/io/colmap.rs` is unchanged and remains T3.

Validation rules, in order:

- Parallel image metadata lengths match (`image_names`, `image_paths`,
  `image_ids`, `image_camera_indices`, `image_frame_indices`, `poses`,
  `observations`, `keypoints`). Camera ids match cameras. Point ids match
  points.
- Camera, image, point, rig, and frame ids are unique. Image, camera, rig, and
  frame ids are in `1..2147483647`. Point ids are non-zero `u64`.
- Each image camera index references a camera.
- Each image has the same number of observations and keypoints.
- Every observation references an existing point.
- Every track image and feature index is in range, and each `(image, feature)`
  belongs to at most one track.
- Observations and tracks agree in both directions.
- Frame rig ids, sensor ids, and camera data ids resolve, and image/frame
  links are bidirectional.
- Legacy camera, per-image cameras, registered poses, keypoints, point
  positions, point errors, and frame `rig_from_world` values are finite.
- `validate_for_colmap_export` runs the structural validator first, then
  rejects a track that references an image without a pose.

ID allocator:

- `OccupiedIdAllocator` stores the occupied set and a lower bound that only
  increases. The next id is `max(lower_bound, max(occupied)+1)` using checked
  arithmetic, not a vector index.
- Fresh COLMAP record ids start at 1. Sparse occupied ids such as `{2, 4, 9}`
  allocate `10`, then `11`.
- Replacing the occupied set with a smaller set does not reuse a previously
  issued or observed id. Deleted point id `41` is followed by `42`.
- Id `0` and duplicate occupied ids are errors. A COLMAP record id at
  `2147483646` cannot allocate another id. A point id of `u64::MAX` cannot
  allocate another id.
- `try_image_id`, `try_camera_id_for_image`, `try_camera_for_image`, and
  `try_point3d_id` return errors instead of inventing ids or cameras.
  `image_id`, `camera_id_for_image`, `camera_for_image`, and `point3d_id`
  remain as deprecated compatibility wrappers. Core validation, frame-sensor
  lookup, local setup, seed point ids, and new point allocation do not call
  them. The crate-level `#![allow(deprecated)]` was removed. Remaining
  production calls are listed in the review remediation below. This record
  does not claim that every persistence fallback has been removed.

Commands and results, from
`/Users/tfjiang/Projects/RustScan/.worktrees/rs-2026-004-t2-reconstruction-validation`,
with `POSELIB_ROOT` pointed at the T1 worktree PoseLib checkout because this
worktree does not initialize that submodule:

- `cargo fmt --all -- --check`: pass.
- `cargo check --workspace --all-targets`: pass.
- `cargo clippy -p rustscan-sfm --all-targets --all-features -- -D warnings`:
  fail. Clippy stops in `rustscan-slam` and does not produce a rustscan-sfm
  result. Final compiler line: `error: could not compile rustscan-slam (lib)
  due to 84 previous errors`. The first error is `module_inception` at
  `rustscan-slam/src/config/mod.rs:5` (`pub mod config`). The other 83 are
  pre-existing slam lints (`derivable_impls`, `clone_on_copy`,
  `too_many_arguments`, `manual_is_multiple_of`, and similar). They are outside
  T2.
- `cargo test -p rustscan-sfm --no-default-features --lib -- --test-threads=1`:
  pass. `638 passed; 0 failed; 19 ignored; finished in 54.15s`.
- `cargo test -p rustscan-sfm --all-features --lib -- --test-threads=1`:
  `809 passed; 2 failed; 19 ignored; finished in 244.20s`. This run was on the
  tree immediately before the final seed/test lookup switch from deprecated
  `point3d_id`/`image_id` to `try_point3d_id`/`try_image_id`. That switch
  returns the stored id when one exists; the no-default suite above was
  re-run after it. The two failures are the pre-existing macOS GPU numerical
  tests, not T2, and were not deleted or ignored:
  `gpu::five_point_f32::tests::five_point_f32_actual_gpu_stages` failed
  `sample 0: pivot=0`, left `SingularElimination`, right `Success`.
  `gpu::pnp_focal::tests::wgpu_pnp_focal_p3p_reorders_adverse_baseline_before_solving`
  failed its positive-depth reprojection assertion.
- `git diff --check`: pass.

Known limitations:

- Production code still calls the deprecated fallbacks listed in the review
  remediation. Those calls are not T2-owned construction paths.
- The allocator does not fill gaps below the high-water mark. That is
  intentional: deleted ids are not reused.
- The earlier all-features lib run, before this review fix, had two
  pre-existing GPU numerical failures
  (`five_point_f32_actual_gpu_stages`,
  `wgpu_pnp_focal_p3p_reorders_adverse_baseline_before_solving`). This review
  fix did not delete or ignore them. The review commands below did not rerun
  that full lib suite.

#### T2 review remediation

Status: fixed, not reviewed.

Review-fix commit: `a8ff5059a3795441d751830c5f98b0083ed51ade`

1. `seed_reconstruction_from_reference` returns
   `Result<Option<ReconstructionSeed>>`. A missing point id returns an anyhow
   error containing `reference point index <n>` and `point_id`. It is not
   converted to `None`. `Ok(None)` remains only when no reference image name
   matches the input paths, or when a matched reference has neither a
   registered pose nor a seeded point. Regression:
   `missing_reference_point_id_fails_instead_of_dropping_the_seed`.
2. `validate_structure` now checks each rig before frame links:
   `ref_sensor_id` must name a sensor in that rig; sensor ids are unique and
   in `1..2147483647`; a camera sensor id must be in `camera_ids`; frame data
   sensors must belong to the frame's rig; camera `data_id`s must name an
   image; image/frame links stay bidirectional; `sensor_from_rig` rotation and
   translation must be finite. Regression:
   `rig_sensor_errors_name_the_failing_field`.
3. Deprecated fallback calls that remain in production. Each fires only when
   the corresponding metadata vector is missing or short: `image_id` uses
   `index + 1`, `camera_id_for_image` uses `1`, `camera_for_image` uses the
   legacy `reconstruction.camera`, and `point3d_id` uses `index + 1`.

T3, `rustscan-sfm/src/io/colmap.rs`. These build the COLMAP export records.
T3 replaces them with `try_*` after `validate_for_colmap_export`, so a missing
id is an export error instead of a fabricated one. `cameras_from_reconstruction`
at line 850 uses the same `idx + 1` fabrication without calling the deprecated
method.

- `image_id`: lines 892, 917
- `camera_id_for_image`: line 893
- `point3d_id`: lines 884, 909

T4, camera identity during registration and triangulation. These functions
return cameras or ids into numerical code that does not currently return
`Result`. T4 makes the camera index mandatory, then switches the reads to
`try_*`.

- `rustscan-sfm/src/sfm/mapper/state.rs`: `camera_id_for_image` at 46 and 64;
  `image_id` at 405, 421, 429, 449, and 505
- `rustscan-sfm/src/sfm/incremental_triangulator.rs`: `camera_for_image` at
  638, 749, 852, 1035, and 1115
- `rustscan-sfm/src/sfm/track_triangulation.rs`: `camera_for_image` at 132 and
  213
- `rustscan-sfm/src/sfm/global_mapper.rs`: `camera_for_image` at 739
- `rustscan-sfm/src/sfm/mapper.rs` production calls below the test module:
  `camera_for_image` at 1348, 5781, 6601, 6940, 7165, 7740, 7741, 7761, 7831,
  8163, 8271, 8607, 8642, 8702, 8970, 9193, 9533, 9728, 9776, 10903, 10967,
  11198, 11412, 11465, 11496, 11522, 11903, 11946, 12073, 12118, 12152, 12314,
  12371, 12487, 12581, 12587, 12613, and 12614; `image_id` at 7557, 7558, 7577,
  7578, 7595, 7596, 7958, and 7959; `point3d_id` at 8446

T5, bundle adjustment. Projection and rig matching assume a camera is always
available. T5 switches these to `try_*` after T4, and leaves BA state unchanged
when the lookup fails.

- `rustscan-sfm/src/ba/shared.rs`: `camera_for_image` at 138 and 164
- `rustscan-sfm/src/ba/ceres_problem.rs`: `camera_for_image` at 236
- `rustscan-sfm/src/ba/ceres_support.rs`: `camera_for_image` at 833, 855, and
  869
- `rustscan-sfm/src/sfm/mapper/bundle_adjustment.rs`: `image_id` at 332;
  `camera_id_for_image` at 345

Test-only calls of the deprecated wrappers remain in `ba/ceres.rs` and in
`mapper.rs` below the `#[cfg(test)]` module at line 12642. The compatibility
test in `reconstruction_validation.rs` calls them under a local
`#[allow(deprecated)]` to prove the wrappers still fabricate ids. That allow is
not a crate-wide exemption.

Review commands, same worktree and `POSELIB_ROOT`:

- `cargo fmt --all -- --check`: pass.
- `cargo check --workspace --all-targets`: pass.
- `cargo clippy -p rustscan-sfm --all-targets --all-features -- -D warnings`:
  fail before rustscan-sfm is linted. `error: could not compile rustscan-slam
  (lib) due to 84 previous errors`. First error remains `module_inception` at
  `rustscan-slam/src/config/mod.rs:5`.
- `cargo test -p rustscan-sfm --no-default-features --lib -- --test-threads=1`:
  pass. `640 passed; 0 failed; 19 ignored; finished in 58.28s`.
- `cargo test -p rustscan-sfm --all-features --lib reconstruction_validation -- --test-threads=1`:
  pass. `8 passed; 0 failed; 0 ignored; finished in 0.00s`.
- `cargo test -p rustscan-sfm --all-features --test sequence_registration -- --test-threads=1`:
  pass. `70 passed; 0 failed; 0 ignored; finished in 64.05s`.
- `git diff --check`: pass.

Next action: do not mark T2 reviewed and do not start T3. Do not merge this
branch to `main`.

### T3 — Strict COLMAP IO

Status: reviewed.

Owner: cursor-agent. Branch `agent/RS-2026-004/t3-colmap-io`. Worktree
`/Users/tfjiang/Projects/RustScan/.worktrees/rs-2026-004-t3-colmap-io`. Base
`ad1c6c6847fa60b74caebf5f6d35f3af8bd79bd1`. Implementation commit
`77327ce3ec55a48b22d966f377fbb5764c893c06`. Reviewed tip
`03936f34c5707ed0e59c03a072f1b9aa25dfea87`.

Import validates the complete raw text or binary model before
`Reconstruction` or `ColmapSparseModel` construction. Errors use
`source=<file> record=<type> id=<id> referenced=<id> feature=<index> reason=<reason>`.
`referenced` and `feature` are `-` when they do not apply. Quaternion norms at
or below `1e-8` are rejected. A finite norm above that epsilon is normalized
only after validation.

Text and binary fixtures reject the same cases: duplicate camera, image,
point, rig, and frame IDs; unknown camera, image, and point references;
conflicting observation/track; zero quaternion; non-finite translation;
feature index 50 past the feature list; zero width; non-positive focal.

`export_colmap`, `export_colmap_with_sparse_index`,
`export_colmap_sparse_snapshot`, and `export_colmap_sparse_model` call
`validate_for_colmap_export` before creating directories or files.
`write_colmap_sparse_model`, `write_colmap_sparse_text`, and
`write_colmap_sparse_binary` validate the raw model before `create_dir_all`.
Export ID reads use `try_image_id`, `try_camera_id_for_image`,
`try_camera_for_image`, and `try_point3d_id`.

Removed from the normal import and export path: `ensure_point_tracks_have_observations`,
`ensure_observations_have_point_tracks`, HashMap last-write-wins for camera,
image, point, and frame-data IDs, `idx + 1` camera IDs, the empty-camera
fallback to the legacy `camera` field, and the deprecated reconstruction
accessors in `colmap.rs`. There is no compatibility repair API. COLMAP stores
the reference sensor outside `sensors`; import copies that existing sensor
into `Rig.sensors` so T2 export validation can see it, and export omits an
unposed copy so the wire format is not duplicated.

The mapper rig-seed fixture repeated camera 11 inside `sensors` and referenced
camera 12, which is not in `cameras.txt`, while image 205 still used camera
11. That fixture now keeps one camera-11 reference sensor and assigns both
seed images to it. This is a fixture correction for strict import, not a T4
camera-model change.

Unposed images are still omitted from export after validation. Mapper, BA,
and triangulation still call the deprecated ID and camera fallbacks; those
remain T4 and T5 work.

Commands on macOS, rustc 1.98.1, `POSELIB_ROOT` set to the T1 PoseLib v2.0.5
checkout:

- `cargo fmt --all -- --check`: pass.
- `cargo check --workspace --all-targets`: pass.
- `cargo clippy -p rustscan-sfm --all-targets --all-features -- -D warnings`:
  fail. Clippy stops in `rustscan-slam` before linting `rustscan-sfm`.
  First error: `module_inception` at `rustscan-slam/src/config/mod.rs:5`
  (`pub mod config`). Final line: `error: could not compile rustscan-slam (lib) due to 84 previous errors`.
  `rustscan-slam` was not modified.
- `cargo test -p rustscan-sfm --no-default-features --lib -- --test-threads=1`:
  pass. `643 passed; 0 failed; 19 ignored; finished in 56.65s`.
- `cargo test -p rustscan-sfm --all-features --lib colmap -- --test-threads=1`:
  pass. `172 passed; 0 failed; 19 ignored; finished in 0.65s`.
- `cargo test -p rustscan-sfm --all-features --lib reconstruction_validation -- --test-threads=1`:
  pass. `8 passed; 0 failed; 0 ignored; finished in 0.02s`.
- `git diff --check`: pass.

GPU tests were not required and were not deleted or ignored.

#### T3 review P1 fixes

Status: review_fix applied. Not reviewed.

Code commit `a00b3613a83e68c50909aff9b14a14661bd8e385`.

1. Checked `f64 -> f32` narrowing through `ensure_f64_fits_f32` /
   `f64_to_f32`. Import rejects finite `f64` values that become non-finite
   `f32` for quaternions, translations, 2D/3D coordinates, reprojection
   error, camera parameters, and derived `fx`/`fy`/`cx`/`cy`.
   `validate_quaternion` also rejects a non-finite quaternion norm. No
   silent clamp, zero, or identity substitution.

2. Public production readers now validate before returning:
   `read_camera_model`, `read_colmap_cameras`, `read_colmap_images`,
   `read_colmap_poses`, `read_colmap_points3d`, `read_colmap_rigs`,
   `read_colmap_frames`, `read_colmap_sparse_files`, and
   `read_colmap_sparse_files_with_format`. Camera-only models without
   `points3D` still validate cameras and images, so
   `reference_camera_setup_for_retained` cannot bypass duplicate-ID /
   dimension / focal / pose checks via `read_colmap_cameras` and
   `read_colmap_poses`. Explicit `*_raw` APIs remain for low-level decode
   and parser unit tests only.

Regression coverage: text and binary fixtures for `f32` overflow; camera-only
text and binary fixtures for duplicate camera/image IDs, zero dimensions,
invalid focal, missing camera reference, zero quaternion, and non-finite
translation.

Commands after the review fixes:

- `cargo fmt --all -- --check`: pass.
- `cargo check --workspace --all-targets`: pass.
- `cargo test -p rustscan-sfm --no-default-features --lib -- --test-threads=1`:
  pass. `645 passed; 0 failed; 19 ignored; finished in 56.74s`.
- `cargo test -p rustscan-sfm --all-features --lib colmap -- --test-threads=1`:
  pass. `174 passed; 0 failed; 19 ignored; finished in 0.72s`.
- `cargo test -p rustscan-sfm --all-features --lib reconstruction_validation -- --test-threads=1`:
  pass. `8 passed; 0 failed; 0 ignored; finished in 0.02s`.
- `git diff --check`: pass.

Clippy with `-D warnings` still stops in pre-existing `rustscan-slam`
`module_inception` and was not re-run as a gate for this fix. Next action: do
not mark T3 reviewed and do not start T4, T5, or T6. Do not merge to `main`.

#### T3 review P1 round 2

Status: reviewed.

Code commit `03936f34c5707ed0e59c03a072f1b9aa25dfea87`.

1. Quaternion unitization normalizes in `f64` first. Components such as
   `[1e30; 4]` have a finite `f64` norm and finite per-component `f32`
   conversions, but the `f32` squared norm overflows under
   `UnitQuaternion::new_normalize`. Import divides in `f64`, narrows the
   unit quaternion, checks the `f32` squared norm, then confirms the final
   rotation is finite with a valid norm. No identity substitution.

2. Focal parameters that remain positive in `f64` but become `0.0` after
   `f32` narrowing (for example `1e-50`) are rejected in `validate_cameras`
   and again in `camera_model_from_colmap` for derived `fx`/`fy`.

3. Optional dependency reads no longer use `Err(_) => empty`.
   `read_optional_raw_images` and `read_optional_raw_rigs` return empty only
   when neither `.bin` nor `.txt` exists. A present but malformed file
   propagates the parse error with the file path.

Regressions: `large_finite_quaternion_normalizes_in_f64_before_f32_for_text_and_binary`,
`positive_focal_that_underflows_f32_is_rejected_for_text_and_binary`,
`malformed_optional_dependencies_are_not_treated_as_absent`.

Commands on this worktree with
`POSELIB_ROOT=/Users/tfjiang/Projects/RustScan/third_party/native/PoseLib`
and `CARGO_TERM_COLOR=never`:

- `cargo fmt --all -- --check`: pass, exit 0.
- `cargo check --workspace --all-targets`: pass, exit 0.
- `cargo clippy -p rustscan-sfm --all-targets --all-features -- -D warnings`:
  fail, exit 101. First error remains `module_inception` at
  `rustscan-slam/src/config/mod.rs:5` (`pub mod config`). Final line:
  `error: could not compile rustscan-slam (lib) due to 84 previous errors`.
  `rustscan-slam` was not modified.
- `cargo test -p rustscan-sfm --no-default-features --lib -- --test-threads=1`:
  pass. `648 passed; 0 failed; 19 ignored; finished in 57.70s`.
- `cargo test -p rustscan-sfm --all-features --lib colmap -- --test-threads=1`:
  pass. `177 passed; 0 failed; 19 ignored; finished in 0.83s`.
- `cargo test -p rustscan-sfm --all-features --lib reconstruction_validation -- --test-threads=1`:
  pass. `8 passed; 0 failed; 0 ignored; finished in 0.00s`.
- `git diff --check`: pass, exit 0.

#### Independent review decision

Independent code review accepted tip
`03936f34c5707ed0e59c03a072f1b9aa25dfea87`. Confirmed:

- Quaternions are normalized in `f64` before `f32` narrowing, so large
  finite components cannot overflow the `f32` squared norm into a zero or
  non-finite rotation.
- Positive `f64` focals that underflow to `0.0` as `f32` are rejected.
- Malformed present `images.*` / `rigs.*` are not treated as missing files;
  only true absence of both candidates yields an empty optional list.
- The new text and binary regressions for those cases pass.
- `cargo fmt --all -- --check`, workspace `cargo check`, COLMAP lib tests,
  reconstruction validation tests, and `git diff --check` pass.
- Targeted Clippy with `-D warnings` remains blocked by pre-existing
  unmodified `rustscan-slam` diagnostics. First error:
  `module_inception` at `rustscan-slam/src/config/mod.rs:5`.

Next action: start T4 in its own worktree and branch
`agent/RS-2026-004/t4-camera-invariants`. Do not implement T4 in this T3
worktree. Do not start T5 or T6 until dependencies and the task protocol
allow. Do not merge this branch to `main` from this handoff.

### T4 — Camera Invariants

Status: reviewed.

Owner: `cursor-agent`

Base commit: `ccfc1dab13bccb1c5b3f7f380255dee0485bb72e` (T3 reviewed tip)

Branch: `agent/RS-2026-004/t4-camera-invariants`

Worktree: `/Users/tfjiang/Projects/RustScan/.worktrees/rs-2026-004-t4-camera-invariants`

Implementation commit: `4de9ff7f832a4858a6cc7164fa1b6d9cba7dc56f`

Reviewed tip: `be27ce41843823810c2c4a2f06a0a7e5e2ee6dbf`

Changed files:

- `rustscan-sfm/src/core/types.rs`
- `rustscan-sfm/src/sfm/view_graph_calibration.rs`
- `rustscan-sfm/src/ba/ceres_support.rs`
- `rustscan-sfm/src/core/reconstruction_validation.rs`
- `rustscan-sfm/src/geometry/two_view.rs`
- `rustscan-sfm/src/io/colmap.rs`
- `rustscan-sfm/src/sfm/mapper.rs`
- `rustscan-sfm/tests/adaptive_keyframes.rs`
- `rustscan-sfm/tests/sequence_registration.rs`
- `docs/agent/changes/RS-2026-004-rustscan-sfm-correctness-remediation/tasks.md`
- `docs/agent/changes/RS-2026-004-rustscan-sfm-correctness-remediation/tasks.yaml`
- `docs/agent/changes/RS-2026-004-rustscan-sfm-correctness-remediation/verification.md`

`CameraModel` now owns intrinsics only in COLMAP `params`. `fx`/`fy`/`cx`/`cy`
are derived accessors. Checked mutators are `set_focal_lengths`,
`set_principal_point`, `set_param`, and `scale_focal`. Compatibility
`set_fx`/`set_fy`/`set_cx`/`set_cy` route through those mutators.
`from_colmap` / `try_new_pinhole` reject non-finite or non-positive focals.
Serde keeps legacy mirror fields on the wire but rejects disagreement with
params-derived values.

View-graph focal refinement scales canonical focal parameters before scoring
and uses mean calibrated Sampson cost so grid candidates are distinguishable.
Non-finite or non-positive scales are rejected. Synthetic optimum is scale
`1.05` (not `0.9` or `1.0`).

Commands with `POSELIB_ROOT=/Users/tfjiang/Projects/RustScan/third_party/native/PoseLib`
and `CARGO_TERM_COLOR=never`:

- `cargo fmt --all -- --check`: pass.
- `cargo check --workspace --all-targets`: pass.
- `cargo clippy -p rustscan-sfm --all-targets --all-features -- -D warnings`:
  fail, exit 101. First error: `module_inception` at
  `rustscan-slam/src/config/mod.rs:5`. Final line: `could not compile
  rustscan-slam (lib) due to 84 previous errors`. Slam was not modified.
- `cargo test -p rustscan-sfm --all-features --lib view_graph_calibration -- --test-threads=1`:
  pass. `3 passed`.
- `cargo test -p rustscan-sfm --all-features --lib types::tests -- --test-threads=1`:
  pass. `12 passed`.
- `cargo test -p rustscan-sfm --no-default-features --lib -- --test-threads=1`:
  pass. `652 passed; 0 failed; 19 ignored; finished in 56.66s`.
- `git diff --check`: pass.

Known limitations:

- BA and mapper still write `params[i]` directly in finite-difference and
  solve write-back paths, then call the no-op
  `sync_intrinsics_from_params` seam. Accessors always read from params, so
  observers cannot see a stale mirror, but those writers are not yet forced
  through `set_param` (T5 may tighten BA write-back further).
- Targeted Clippy remains blocked by pre-existing `rustscan-slam` lints.

Historical next action from the implementation handoff (superseded by the
independent review decision below): do not start T5 or T6; do not merge to
`main`.

#### T4 review P1 remediation

Status: reviewed (included in tip `be27ce41843823810c2c4a2f06a0a7e5e2ee6dbf`).

Code commit: `99fe9c8e11130e3a23d248144ee2d96c72e6619d`

1. Checked mutators are atomic. `set_param`, `scale_focal`,
   `set_focal_lengths`, and `set_principal_point` build a candidate parameter
   array, validate (including finite positive focals after `f32` narrowing),
   then commit once. Failures leave params and projection unchanged.
   Regressions: `checked_mutators_leave_camera_unchanged_on_failure`.

2. Mapper / local-camera config overrides collect optional `fx`/`fy`/`cx`/`cy`
   and call `apply_optional_intrinsics` once. For single-focal models the
   shared focal is the mean of the provided pair, so `600` then `601` yields
   `600.5` independent of order. Illegal overrides return contextual
   `anyhow` errors; compatibility setters return `Result` and no longer
   `expect`. Regression:
   `single_focal_optional_intrinsics_average_once_order_independently`.

3. `CameraModel.params` is private. Production BA assemble/write-back, PnP
   focal write-back, and finite-difference paths use `set_param` /
   `set_shared_focal`. Illegal candidates are rejected without mutating the
   committed camera. Test-only corruption uses
   `inject_raw_param_for_test`.

Existing focal-search test still passes and now also checks off-center
projection plus COLMAP params round-trip agreement.

Commands with `POSELIB_ROOT=/Users/tfjiang/Projects/RustScan/third_party/native/PoseLib`
and `CARGO_TERM_COLOR=never`:

- `cargo fmt --all -- --check`: pass.
- `cargo check --workspace --all-targets`: pass.
- `cargo clippy -p rustscan-sfm --all-targets --all-features -- -D warnings`:
  fail, exit 101. First error: `module_inception` at
  `rustscan-slam/src/config/mod.rs:5`. Final:
  `could not compile rustscan-slam (lib) due to 84 previous errors`. Slam
  unmodified.
- `cargo test -p rustscan-sfm --no-default-features --lib -- --test-threads=1`:
  pass. `654 passed; 0 failed; 19 ignored; finished in 57.03s`.
- `cargo test -p rustscan-sfm --all-features --lib types::tests -- --test-threads=1`:
  pass. `14 passed`.
- `cargo test -p rustscan-sfm --all-features --lib view_graph_calibration -- --test-threads=1`:
  pass. `3 passed`.
- `cargo test -p rustscan-sfm --all-features --lib colmap -- --test-threads=1`:
  pass. `178 passed; 0 failed; 19 ignored`.
- `git diff --check`: pass.

The earlier all-targets claim from an interrupted run is not review evidence.
Full `cargo test -p rustscan-sfm --all-targets` was not completed for the
reviewed tip.

Known limitations:

- Targeted Clippy remains blocked by pre-existing `rustscan-slam` lints.
- Full Reconstruction/BA commit atomicity on unusable solutions remains T5.

#### T4 review P1 remediation (apply_optional_intrinsics atomicity)

Status: reviewed.

Code commit: `be27ce41843823810c2c4a2f06a0a7e5e2ee6dbf`.

`apply_optional_intrinsics` no longer calls `set_focal_lengths` then
`set_principal_point` sequentially. It copies `params` to a candidate, writes
optional fx/fy/cx/cy into that candidate only (single-focal mean semantics
unchanged), validates once via `commit_params`, and leaves `self` untouched on
any failure. Compat `set_fx`/`set_fy`/`set_cx`/`set_cy` remain public with
rustdoc directing single-focal configuration to `set_focal_lengths` or
`apply_optional_intrinsics`. Production mapper / reconstruction-input config
paths already use `apply_optional_intrinsics`; remaining `set_fx`/`set_fy`
calls are dual-equal test fixtures only.

Regression: `apply_optional_intrinsics_is_atomic_when_principal_point_is_invalid`
covers legal fx/fy with NaN cx, Inf cy, and single-focal shared-focal retention.
Success-path coverage in
`single_focal_optional_intrinsics_average_once_order_independently` now also
checks PINHOLE dual focals, off-center principal point, params slice, and
projection agreement.

Commands with `POSELIB_ROOT=/Users/tfjiang/Projects/RustScan/third_party/native/PoseLib`
and `CARGO_TERM_COLOR=never`:

- `cargo fmt --all -- --check`: pass.
- `cargo check --workspace --all-targets`: pass.
- `cargo clippy -p rustscan-sfm --all-targets --all-features -- -D warnings`:
  fail, exit 101. First error: `module_inception` at
  `rustscan-slam/src/config/mod.rs:5`. Final:
  `could not compile rustscan-slam (lib) due to 84 previous errors`. Slam
  unmodified. No global allow added.
- `cargo test -p rustscan-sfm --all-features --lib types::tests -- --test-threads=1`:
  pass. `15 passed`.
- `cargo test -p rustscan-sfm --all-features --lib view_graph_calibration -- --test-threads=1`:
  pass. `3 passed`.
- `cargo test -p rustscan-sfm --all-features --lib colmap -- --test-threads=1`:
  pass. `178 passed; 0 failed; 19 ignored`.
- `cargo test -p rustscan-sfm --no-default-features --lib -- --test-threads=1`:
  pass. `655 passed; 0 failed; 19 ignored; finished in 56.41s`.
- `git diff --check`: pass.

Known limitations:

- Targeted Clippy remains blocked by pre-existing `rustscan-slam` lints.
  This is an existing repository issue outside T4, not a new T4 defect.
- Full Reconstruction/BA commit atomicity on unusable solutions remains T5.
- Full `cargo test -p rustscan-sfm --all-targets` was not completed for this
  reviewed tip and is not cited as pass evidence.

#### Independent review decision

Independent code review accepted tip
`be27ce41843823810c2c4a2f06a0a7e5e2ee6dbf`. Confirmed:

- `CameraModel` keeps COLMAP `params` as the sole mutable intrinsics store;
  accessors and checked mutators stay consistent for projection and export.
- Checked mutators and `apply_optional_intrinsics` build a candidate, validate
  once, and commit once. Illegal principal-point updates cannot leave focals
  partially applied, including on single-focal models.
- Mapper / reconstruction-input config overrides use
  `apply_optional_intrinsics`; single-focal mean semantics are order-
  independent.
- T4 acceptance is covered by the implementation and the focused gates below.
  Targeted Clippy failure in unmodified `rustscan-slam` is a pre-existing
  repository limitation, not a T4 finding.

Verification recorded for the reviewed tip (`POSELIB_ROOT` set,
`CARGO_TERM_COLOR=never`):

- `cargo fmt --all -- --check`: pass.
- `cargo check --workspace --all-targets`: pass.
- `cargo test -p rustscan-sfm --all-features --lib types::tests -- --test-threads=1`:
  pass. `15 passed`.
- `cargo test -p rustscan-sfm --all-features --lib view_graph_calibration -- --test-threads=1`:
  pass. `3 passed`.
- `cargo test -p rustscan-sfm --all-features --lib colmap -- --test-threads=1`:
  pass. `178 passed; 19 ignored`.
- `cargo test -p rustscan-sfm --no-default-features --lib -- --test-threads=1`:
  pass. `655 passed; 19 ignored`.
- `git diff --check`: pass.
- `cargo clippy -p rustscan-sfm --all-targets --all-features -- -D warnings`:
  fail, exit 101. First error: `module_inception` at
  `rustscan-slam/src/config/mod.rs:5`. Final:
  `could not compile rustscan-slam (lib) due to 84 previous errors`.
  `rustscan-slam` was not modified.

Full `cargo test -p rustscan-sfm --all-targets` was not completed for this
review round and is not used as pass evidence.

Next action: do not start T5 or T6 from this handoff. Do not merge this
branch to `main`.

### T5 — Atomic BA

Status: review_ready. Not reviewed.

Owner: `cursor-agent`

Base commit: `701d051814a29ab3ee4fbefc01355112f71b5a47` (local `main` with T1–T4
integrated)

Implementation commit: tip of `agent/RS-2026-004/t5-atomic-ba` (this handoff).


Branch: `agent/RS-2026-004/t5-atomic-ba`

Worktree: `/Users/tfjiang/Projects/RustScan/.worktrees/rs-2026-004-t5-atomic-ba`

This start supersedes the earlier “do not start T5” handoff freeze.

Changed files:

- `rustscan-sfm/src/ba/mod.rs`
- `rustscan-sfm/src/ba/ceres_problem.rs`
- `rustscan-sfm/src/sfm/global_mapper.rs`
- `rustscan-sfm/src/sfm/mapper/bundle_adjustment.rs`
- `docs/agent/changes/RS-2026-004-rustscan-sfm-correctness-remediation/tasks.md`
- `docs/agent/changes/RS-2026-004-rustscan-sfm-correctness-remediation/tasks.yaml`
- `docs/agent/changes/RS-2026-004-rustscan-sfm-correctness-remediation/verification.md`

Public Ceres BA now gates Reconstruction mutation on
`should_commit_ba_solution(ceres_summary_usable, termination_type,
parameters_valid)`. `NoConvergence` remains committable when Ceres marks the
solution usable; `Failure` / `UserFailure` never commit. Write-back stages
cameras (via checked `set_param`), poses, and points, then applies once; any
illegal camera/pose/point value rejects the whole commit and forces an
unusable report. Point-error refresh and covariance run only after a
successful commit. Cancel before write-back remains unchanged.

`global_mapper::run_iterative_global_refinement` sets success from
`report.is_solution_usable()` and stops further refinement rounds on an
unusable or absent BA result. Mapper camera-plausibility rollback in
`refine_bundle_adjustment_checked` stays a post-success policy gate.

Deterministic coverage uses `BaCommitTestOverride` on the real solve/write-back
path (not a standalone boolean stub): Failure, UserFailure, NaN/Inf/negative
camera params leave cameras, poses, xyz, point errors, frames, and sensors
unchanged; usable `NoConvergence` still commits; convergence refreshes point
errors; `unusable_global_ba_is_not_success_and_stops_refinement` checks
global success + round stop. Existing taskflow cancel tests continue to cover
cancellation without mutation.

Commands with `POSELIB_ROOT=/Users/tfjiang/Projects/RustScan/third_party/native/PoseLib`
and `CARGO_TERM_COLOR=never`:

- `cargo fmt --all -- --check`: pass.
- `cargo check --workspace --all-targets`: pass.
- `cargo test -p rustscan-sfm --all-targets -- --test-threads=1`: pass
  (lib `846 passed; 19 ignored`, plus integration/example targets; exit 0).
- `cargo test -p rustscan-sfm --no-default-features --lib -- --test-threads=1`:
  pass. `656 passed; 0 failed; 19 ignored`.
- `cargo test -p rustscan-sfm --no-default-features --features ceres-ba --lib -- --test-threads=1`:
  pass. `693 passed; 0 failed; 19 ignored`.
- `cargo test -p rustscan-sfm --features ceres-ba --lib ba:: -- --test-threads=1`:
  pass. `37 passed`.
- `cargo test -p rustscan-sfm --features ceres-ba --lib global_mapper -- --test-threads=1`:
  pass. `9 passed`.
- `cargo clippy -p rustscan-sfm --all-targets --all-features -- -D warnings`:
  fail, exit 101. First error: `module_inception` at
  `rustscan-slam/src/config/mod.rs:5`. Final:
  `could not compile rustscan-slam (lib) due to 84 previous errors`. Slam
  unmodified. No global allow added.
- `git diff --check`: pass.

Known limitations:

- Targeted Clippy remains blocked by pre-existing `rustscan-slam` lints.
- GPU numerical host limitations from earlier tasks are unchanged; this run
  did not treat ignored GPU skips as execution evidence.
- Database transaction atomicity remains T6.

Next action: do not start T6. Do not merge to `main`. Await independent
code review.

### T6 — Database Transactions

Status: pending.

### T7 — nalgebra Ownership

Status: pending.

### T8 — Integration And Final Verification

Status: pending.

## Final Verification Matrix

Copy every command from `tasks.yaml` here with its exact result. Add platform
specific Ceres, PoseLib, GPU adapter, and fixture evidence. Finish with the
reviewer decision, known limitations, integration commit, and next action.


## T1–T4 Integration — 2026-09-23

Owner: Codex. Integration branch/worktree: `main`, repository root.
User explicitly authorized merging T4 and removing the T1–T4 worktrees;
this supersedes the earlier review handoff's instruction not to merge.

- Base: `d2cdb8de7d63e448022f02e69e622c90a501a591`.
- Integrated tip: `0919b73ba0555f2098e28e619540ae5c27f469f8`.
- `git merge --ff-only agent/RS-2026-004/t4-camera-invariants`: PASS.
  The verified ancestry chain is main → T1 → T2 → T3 → T4.
- Integrated files: `git diff --name-status d2cdb8d 0919b73` (21 files).
  This handoff additionally updates this verification record only.
- `git diff --exit-code agent/RS-2026-004/t4-camera-invariants HEAD`
  immediately after fast-forward: PASS; identical trees.
- `cargo fmt --all -- --check`: PASS.
- `POSELIB_ROOT=/Users/tfjiang/Projects/RustScan/third_party/native/PoseLib CARGO_TERM_COLOR=never cargo check --workspace --all-targets`:
  PASS, exit 0; existing warnings remain. Log:
  `artifacts/runs/rs-2026-004-t4-integration-20260923/cargo-check.log`.
- Tests and Clippy were not repeated: integration introduced no implementation
  changes, and the difference from reviewed code tip `be27ce4` to `0919b73`
  contains only task documentation. Prior independent review evidence above
  applies. Full all-target tests remain incomplete and the existing Clippy
  blocker remains; integration does not claim those gates passed.
- `git diff --check d2cdb8d..0919b73`: reports existing Markdown hard-break
  trailing spaces and extra EOF blank lines in the imported task documents.
  No source-code whitespace errors were reported.

Cleanup:

- Verified all T1–T4 worktrees were clean (including untracked source and
  submodule status), and all four branch tips were ancestors of main.
- Removed `.worktrees/rs-2026-004-t1-mapper-identity`,
  `.worktrees/rs-2026-004-t2-reconstruction-validation`,
  `.worktrees/rs-2026-004-t3-colmap-io`, and
  `.worktrees/rs-2026-004-t4-camera-invariants` using Git worktree removal.
  Git required `--force` because of the submodule-bearing worktree;
  no dirty source changes were discarded.
- Preserved and SHA-256-verified 22 local T1 verification files under root
  `artifacts/runs/rs-2026-004-t1/` and
  `artifacts/runs/rs-2026-004-t1-review-p1/` before removal.
- Retained the four task branch refs, other worktrees, and the separate
  `codex/preserve-main-20260923` WIP snapshot.

Next action: publish main when requested, then plan T5 from the integrated
main. T5–T8 remain pending; this is not final RS-2026-004 acceptance.

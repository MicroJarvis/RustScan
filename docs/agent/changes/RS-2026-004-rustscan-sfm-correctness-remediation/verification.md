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

Final commit: this T1 commit. The hash is the commit that adds this record on
`agent/RS-2026-004/t1-mapper-identity`. It is not written into the file, because
that would require a second commit. The handoff reports the hash.

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
- A database image that exists but is not match-connected returns
  `Ok(candidate: None)` instead of a hard error, so sequence registration can
  record an unresolved attempt. A name missing from the database entirely is
  still a contextual error. That split is what keeps the T0 fixtures failing
  closed and the blank-frame sequence test unresolved rather than aborted.
- Registered reference images omitted by the match cache are copied back from
  the reference by name so an existing pose is not dropped or assigned to
  another path. They are not given another image's database id.
- GPU PnP execution on this Mac remains unproved, as in T0.

#### Next action

Start T2 on `agent/RS-2026-004/t2-reconstruction-validation` in
`.worktrees/rs-2026-004-t2-reconstruction-validation`, based on this T1 commit.
Do not merge this branch to `main`.

### T2 — Reconstruction Validation And IDs

Status: pending.

### T3 — Strict COLMAP IO

Status: pending.

### T4 — Camera Invariants

Status: pending.

### T5 — Atomic BA

Status: pending.

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


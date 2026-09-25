# T6 Independent Re-review — 2026-09-25

Task: `RS-2026-004 / T6`

Reviewer: Codex

Branch: `agent/RS-2026-004/t6-database-transactions`

Base commit: `c71f8092778e2abb1feaaf03c9d784ea25434349`

Reviewed HEAD: `f8eb53bc7b2fa7e0b2de1dc39bfec0b114b42083`

Implementation commit: `aa052d34dc728edbdef7f57d90800422dc3242e2`

Previous reviews: `review-t6-2026-09-24.md`, `review-t6-2026-09-25.md`

## Findings

| Severity | Location | Finding | Required action |
|---|---|---|---|
| — | T6 transaction scope and tests | None. The prior P2 findings are resolved and no new correctness or scope issue was found. | None. |

## Review gates

- [x] Changes stay inside the declared T6 scope and review records.
- [x] Database merge, local population, pair-geometry batches, rollback, retry,
      cleanup failure, and bookkeeping acceptance conditions are covered.
- [x] No public API or compatibility impact requiring follow-up was introduced.
- [x] Relevant tests and checks have independent evidence below.
- [x] The review record and T6 task status are updated.
- [x] No vendor-specific Agent dependency was added.

## Verification

Environment: macOS, native dependencies configured,
`POSELIB_ROOT=/Users/tfjiang/Projects/RustScan/third_party/native/PoseLib`.

Independently executed on reviewed HEAD:

- `cargo fmt --all -- --check`: **PASS**.
- `cargo check --workspace --all-targets`: **PASS**, with existing warnings.
- `cargo test -p rustscan-sfm --lib -- --list`: **PASS**; the two previously
  missing tests are registered: `resolve_mapper_database_path_allows_missing_output_for_local_write`
  and `run_incremental_pipeline_reports_success_status`.
- `cargo test -p rustscan-sfm --lib -- --test-threads=1 with_transaction mid_failure retry_after transaction_mid_batch_delete`:
  **PASS**, 12 passed, 0 failed, 0 ignored.
- `cargo test -p rustscan-sfm --all-targets -- --test-threads=1`:
  **PASS**; the main library reported 871 passed, 0 failed, 19 ignored, and
  all remaining targets/examples reported 0 failures. The real GPU example
  tests also completed successfully on this host.
- `cargo clippy -p rustscan-sfm --all-targets --all-features -- -D warnings`:
  **BLOCKED by existing unrelated diagnostics**; Clippy stopped in unchanged
  `rustscan-slam` with 84 errors, beginning at
  `rustscan-slam/src/config/mod.rs:5` (`module_inception`). No diagnostic
  points to the T6 changed files.
- `git diff --check c71f809..HEAD`: **PASS**.

The cleanup-failure regression now keeps the transaction open while injecting
cleanup failure, preserves both the operation and cleanup errors, and checks
that current deletion bookkeeping is not falsely restored. Successful cleanup
tests cover both initial bookkeeping states. Merge and mapper retry fixtures
now compare stable pair/image/camera IDs and complete keypoint, descriptor,
match, geometry, rig, frame, and pose-prior state. The restored legacy tests
are listed and included in the full test run.

Known limitation: 19 tests remain intentionally ignored by the existing test
suite; their ignored status is not used as passing evidence. Clippy remains a
repository-wide pre-existing blocker outside T6.

## Decision

**APPROVE**

T6 satisfies its transaction atomicity acceptance conditions and may enter the
normal integration decision. This review does not merge, push, delete the
worktree, or start T7.

# T5 Independent Re-review — 2026-09-24 Distortion Recheck

Task: RS-2026-004 / T5  
Reviewer: Codex  
Branch: `agent/RS-2026-004/t5-atomic-ba`  
Reviewed HEAD: `82512b9a3b6a4d455b13e1e8dbdd938bfbc7a4ec`  
Implementation commit: `d6287d8bafa6045de9d187a07a41f7e75ba66f9b`  
Prior review: `review-t5-2026-09-24-final.md`  
Decision: **APPROVE**

## Previous finding

The prior P1 is fixed. Candidate error refresh no longer infers projection
failure from an undistorted pinhole expression. `CameraProjectionOutcome` and
`CameraModel::classify_img_from_cam_unchecked` cover all supported COLMAP
models and distinguish finite projections, finite geometric-domain skips, and
non-finite intermediate or final values. `project_point_for_candidate_error`
rejects the latter before the candidate can be installed while retaining the
existing finite behind-camera and model-domain skip policy.

The previous T5 fixes remain present: composed frame/sensor poses are checked
after write-back staging, unusable BA skips the trailing global filter, and
failure-injection hooks remain test-only. The new regression exercises a
usable solve with an overflowing radial distortion parameter and verifies that
the live reconstruction remains bitwise unchanged. Radial, OpenCV, division,
and general camera classification tests also pass.

## Verification

Environment: `POSELIB_ROOT=/Users/tfjiang/Projects/RustScan/third_party/native/PoseLib`.

- `cargo fmt --all -- --check`: **PASS**.
- `cargo check --workspace --all-targets`: **PASS** with existing warnings.
- `cargo test -p rustscan-sfm --no-default-features --features ceres-ba --lib ba:: -- --test-threads=1`: **PASS**, 49 passed, 0 failed.
- `cargo test -p rustscan-sfm --all-features --lib types::tests -- --test-threads=1`: **PASS**, 16 passed, 0 failed.
- `git diff --check 2647492..HEAD`: **PASS**.
- Product source tree has no uncommitted changes; only the review/documentation history is present on this branch.
- `cargo clippy -p rustscan-sfm --no-default-features --features ceres-ba --lib -- -D warnings`: **BLOCKED** by 84 existing `rustscan-slam` diagnostics, beginning with `rustscan-slam/src/config/mod.rs:5`; no T5 source caused or suppressed these errors.

## Next action

T5 has no remaining P1/P2 findings in this review. It is eligible for the
normal integration decision. Do not start T6 or merge to `main` as part of this
review; preserve the worktree and wait for the repository owner to authorize
integration.

# T5 Independent Re-review — 2026-09-24

Task: RS-2026-004 / T5
Reviewer: Codex
Branch: `agent/RS-2026-004/t5-atomic-ba`
Reviewed HEAD: `26474925d24271014643e3295bb4be28b6314516`
Prior review: `review-t5-2026-09-24.md`
Decision: **CHANGES_REQUESTED**

## Previous finding status

The composed-pose finding is fixed. `validate_candidate_poses` now runs on the
fully staged candidate before point-error refresh and rejects non-finite final
image/frame/sensor poses. The prior finite `3e38 + 3e38` composition repro is
covered by regression tests. The T5 production API remains free of the test
hooks, and global BA still skips the trailing filter after an unusable result.

## Remaining P1 — distorted projection failure is misclassified

Location: `rustscan-sfm/src/ba/ceres_problem.rs:1848-1906`.

When `camera.img_from_cam_unchecked` returns `None`, the code decides whether
the result is a numerical overflow by recomputing only the undistorted pinhole
expression:

```text
trial_x = camera.fx() * (u / w) + camera.cx()
trial_y = camera.fy() * (v / w) + camera.cy()
```

That does not represent radial, OpenCV, or fisheye distortion. A finite
`SIMPLE_RADIAL` camera with a finite but extreme distortion parameter can make
the actual distortion calculation overflow to `None` while the pinhole trial
remains finite. The helper then returns `Ok(None)`, and the caller skips the
observation while retaining the old finite `point.error`; a numerically invalid
candidate can therefore pass validation.

Independent repro used `CameraModel::from_colmap` with model
`SIMPLE_RADIAL`, parameters `[50.0, 50.0, 50.0, 1e308]`, point `[2, 0, 1]`,
and identity pose. The actual `img_from_cam_unchecked` returned `None`, while
`project_point_for_candidate_error` returned `Ok(None)`. The assertion that a
numeric distortion failure must be rejected failed with exit 101.

Evidence:
`artifacts/runs/t5-review-2647492/distortion-probe.rs` and
`distortion-probe.log`.

Required action:

- Use a model-aware projection result that distinguishes finite geometric
  domain rejection from non-finite intermediate/result, or expose a structured
  projection error from the camera API. Do not infer all camera models from a
  pinhole fallback.
- Reject non-finite distortion/covariance/projection arithmetic before
  installing the candidate. Preserve the existing finite behind-camera/domain
  skip policy.
- Add a regression for radial/OpenCV/fisheye distortion overflow (at least one
  affected non-pinhole model), and assert candidate rejection plus bitwise
  preservation of the live reconstruction.

## Verification

Environment: `POSELIB_ROOT=/Users/tfjiang/Projects/RustScan/third_party/native/PoseLib`.

- `cargo fmt --all -- --check`: PASS.
- `git diff --check 5593e69..HEAD`: PASS.
- `cargo test -p rustscan-sfm --no-default-features --features ceres-ba --lib ba:: -- --test-threads=1`: PASS, 45 tests.
- `cargo check --workspace --all-targets`: PASS with `POSELIB_ROOT` set.
  Running without that required environment first failed in the build script;
  this is an environment prerequisite, not a source failure.
- `cargo clippy -p rustscan-sfm --all-targets --all-features -- -D warnings`:
  BLOCKED by existing `rustscan-slam` diagnostics, 84 errors beginning at
  `rustscan-slam/src/config/mod.rs:5`; no unrelated fixes or allowances added.
- Distortion overflow repro: FAIL as described above, exit 101.
- No product source changes were made by this review; temporary probe source
  was outside RustSFM and the worktree source diff is unchanged.

## Next action

Fix the model-aware projection classification within T5, add the regression,
record exact verification, and request re-review. Do not start T6 or merge to
main. This review document is currently uncommitted.

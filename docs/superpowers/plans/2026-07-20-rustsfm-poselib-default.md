# RustSFM PoseLib Default Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing PoseLib-backed GR6P, GR8P, and GP3P paths available in normal RustSFM builds, expose their availability and real mapper use, and validate the change on all 960 `flowers2` images plus a RustGS probe.

**Architecture:** Restore the existing PoseLib v2.0.5 gitlink as a valid submodule and add `poselib` to RustSFM's default feature set while retaining `--no-default-features`. Expose one compile-time generalized-pose capability value for the CLI, route the mapper's missing-solver diagnostic through the existing error, and extend session telemetry around the existing structureless solver call without changing pose decisions.

**Tech Stack:** Rust 2021, Cargo features, PoseLib v2.0.5 C++ bridge, Eigen3, Git submodules, GitHub Actions, existing RustSFM mapper tests, COLMAP sparse output, RustGS training probe.

---

### Task 1: Restore PoseLib Delivery and Enable the Default Feature

**Files:**
- Create: `.gitmodules`
- Modify: `RustSFM/Cargo.toml:7-13`
- Modify: `.github/workflows/ci.yml:10,53`
- Verify: `scripts/setup_rustsfm_deps.sh`

- [x] **Step 1: Record the failing dependency state**

Run:

```bash
git ls-files -s third_party/PoseLib
test -f .gitmodules
test -f third_party/PoseLib/PoseLib/solvers/gen_relpose_6pt.cc
```

Expected: the first command reports gitlink `7e9f5f53372e43f89655040d4dfc4a00e5ace11c`, while the two `test` commands fail because the submodule mapping and checkout are absent.

- [x] **Step 2: Restore the submodule mapping**

Create `.gitmodules` with the exact pinned dependency path and upstream URL:

```ini
[submodule "third_party/PoseLib"]
	path = third_party/PoseLib
	url = https://github.com/PoseLib/PoseLib.git
```

Initialize the existing gitlink:

```bash
git submodule sync -- third_party/PoseLib
git submodule update --init --depth 1 third_party/PoseLib
git -C third_party/PoseLib rev-parse HEAD
```

Expected: the final command prints `7e9f5f53372e43f89655040d4dfc4a00e5ace11c`.

- [x] **Step 3: Make PoseLib part of the normal RustSFM build**

Change the feature table to:

```toml
[features]
default = ["ceres-ba", "vlfeat-sift", "gpu-wgpu", "poselib"]
gpu-wgpu = ["dep:bytemuck", "dep:pollster", "dep:wgpu"]
lowe-sift-backend = []
poselib = []
ceres-ba = ["dep:ceres-solver"]
vlfeat-sift = []
```

- [x] **Step 4: Make CI initialize the dependency**

Change both checkout steps in `.github/workflows/ci.yml` to:

```yaml
- uses: actions/checkout@v4
  with:
    submodules: recursive
```

Keep the existing default, explicit `--features poselib`, and `--no-default-features` matrix commands. The explicit feature build remains useful because it documents the supported feature name even though it is now in the default set.

- [x] **Step 5: Verify default and dependency-minimal feature resolution**

Run:

```bash
cargo metadata --format-version 1 --no-deps \
  | jq -e '.packages[] | select(.name == "rustsfm") | .features.default | index("poselib")'
cargo test -p rustsfm --lib poselib_structureless_absolute_pose_reuses_generalized_relative_solver
cargo test -p rustsfm --lib --no-default-features structureless_estimator_reports_missing_gr6p_gr8p_solver
```

Expected: metadata contains `poselib` in the default list, the default build runs and passes the PoseLib solver test, and the no-default build passes its missing-solver test.

- [x] **Step 6: Commit Task 1**

```bash
git add .gitmodules RustSFM/Cargo.toml .github/workflows/ci.yml
git commit -m "build(rustsfm): enable PoseLib solvers by default"
```

### Task 2: Report Compile-Time Solver Capabilities and Accurate Errors

**Files:**
- Modify: `RustSFM/src/geometry/generalized_pose.rs:245-270`
- Modify: `RustSFM/src/cli/commands.rs:1-25,20-24`
- Modify: `RustSFM/src/sfm/mapper.rs:6720-6735`
- Test: `RustSFM/src/geometry/generalized_pose.rs`

- [x] **Step 1: Write the failing capability-format test**

Add this test to `generalized_pose.rs` before adding the production type:

```rust
#[test]
fn generalized_pose_capabilities_report_compile_time_solver_state() {
    let capabilities = generalized_pose_capabilities();
    assert_eq!(capabilities.poselib, cfg!(feature = "poselib"));
    assert_eq!(
        capabilities.structureless_gr6p_gr8p,
        cfg!(feature = "poselib")
    );
    assert_eq!(
        capabilities.format_log(),
        format!(
            "rustsfm_capabilities poselib={} structureless_gr6p_gr8p={}",
            cfg!(feature = "poselib"),
            cfg!(feature = "poselib")
        )
    );
}
```

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
cargo test -p rustsfm generalized_pose_capabilities_report_compile_time_solver_state
```

Expected: compilation fails because `generalized_pose_capabilities` and its return type do not exist.

- [x] **Step 3: Implement the compile-time capability value**

Add above `GeneralizedPoseError`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneralizedPoseCapabilities {
    pub poselib: bool,
    pub structureless_gr6p_gr8p: bool,
}

impl GeneralizedPoseCapabilities {
    pub fn format_log(self) -> String {
        format!(
            "rustsfm_capabilities poselib={} structureless_gr6p_gr8p={}",
            self.poselib, self.structureless_gr6p_gr8p
        )
    }
}

pub const fn generalized_pose_capabilities() -> GeneralizedPoseCapabilities {
    GeneralizedPoseCapabilities {
        poselib: cfg!(feature = "poselib"),
        structureless_gr6p_gr8p: cfg!(feature = "poselib"),
    }
}
```

- [x] **Step 4: Print the capability once for both CLI mapper routes**

Import `rustsfm::generalized_pose::generalized_pose_capabilities` in `commands.rs`. Immediately after logger initialization in `run_reconstruct`, add:

```rust
println!("{}", generalized_pose_capabilities().format_log());
```

`run_colmap_mapper` delegates to `run_reconstruct`, so both `reconstruct` and COLMAP-compatible `mapper` commands produce exactly one line.

- [x] **Step 5: Replace the inaccurate mapper diagnostic**

Change the missing-solver arm in `solve_colmap_structureless_absolute_pose` to:

```rust
Err(err @ GeneralizedPoseError::MissingGeneralizedRelativePoseSolver) => {
    log::debug!("COLMAP structure-less registration skipped: {err}");
    return None;
}
```

The existing `Display` message names `--features poselib`. No source string may contain
`GR6P/GR8P solver is not ported yet` after this change.

- [x] **Step 6: Verify GREEN in enabled and disabled builds**

Run:

```bash
cargo test -p rustsfm generalized_pose_capabilities_report_compile_time_solver_state
cargo test -p rustsfm --no-default-features generalized_pose_capabilities_report_compile_time_solver_state
! rg -n "GR6P/GR8P solver is not ported yet" RustSFM/src
```

Expected: both capability tests pass with their compile-time values and the obsolete diagnostic search returns no matches.

- [x] **Step 7: Commit Task 2**

```bash
git add RustSFM/src/geometry/generalized_pose.rs RustSFM/src/cli/commands.rs RustSFM/src/sfm/mapper.rs
git commit -m "fix(rustsfm): expose generalized pose capabilities"
```

### Task 3: Measure Structureless Solver Outcomes

**Files:**
- Modify: `RustSFM/src/sfm/mapper.rs:3206-3235,3560-3620,6632-6755`
- Test: `RustSFM/src/sfm/mapper.rs:10870-10910`

- [x] **Step 1: Extend the telemetry test first**

Add fields to the existing test fixture only, before production code:

```rust
structureless_estimates: 2,
structureless_accepted: 1,
structureless_solver_ms: 6.25,
```

Require these stable keys:

```rust
"structureless_estimates=2",
"structureless_accepted=1",
"structureless_solver_ms=6.25",
```

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
cargo test -p rustsfm incremental_registration_telemetry_reports_hot_path_stages
```

Expected: compilation fails because the three telemetry fields do not exist.

- [x] **Step 3: Add fields and stable formatting**

Extend `IncrementalRegistrationTelemetry`:

```rust
structureless_estimates: usize,
structureless_accepted: usize,
structureless_solver_ms: f64,
```

Place the new keys immediately after `structureless_attempts` in `format_log`:

```text
structureless_attempts={} structureless_estimates={} structureless_accepted={} structureless_solver_ms={:.2}
```

- [x] **Step 4: Time and count solver estimates**

Pass `&mut IncrementalRegistrationTelemetry` through `solve_structureless_absolute_pose` and `solve_colmap_structureless_absolute_pose`. Measure only the call to `estimate_structureless_absolute_pose`:

```rust
let solve_started = Instant::now();
let estimate_result = estimate_structureless_absolute_pose(&options, problem);
telemetry.structureless_solver_ms += solve_started.elapsed().as_secs_f64() * 1_000.0;
let estimate = match estimate_result {
    Ok(Some(estimate)) => {
        telemetry.structureless_estimates += 1;
        estimate
    }
    Ok(None) => return None,
    Err(err @ GeneralizedPoseError::MissingGeneralizedRelativePoseSolver) => {
        log::debug!("COLMAP structure-less registration skipped: {err}");
        return None;
    }
    Err(err) => {
        log::debug!("COLMAP structure-less registration skipped: {err}");
        return None;
    }
};
```

Update all production and test-only callers with the session telemetry value already in scope or a local default in test compatibility helpers.

- [x] **Step 5: Count accepted structureless choices after all gates**

In `registration_choice_for_image_with_pnp_scorer`, after pair-rotation validation and immediately before returning a structureless `RegistrationChoice`, add:

```rust
if mode == NextImageRegistrationMode::StructureLess {
    telemetry.structureless_accepted += 1;
}
```

This counts only choices that passed solver inlier checks and mapper pair-rotation validation.

- [x] **Step 6: Verify GREEN and mapper integration**

Run:

```bash
cargo test -p rustsfm incremental_registration_telemetry_reports_hot_path_stages
cargo test -p rustsfm default_structureless_path_uses_pair_pose_fallback_without_poselib
cargo test -p rustsfm --no-default-features default_structureless_path_uses_pair_pose_fallback_without_poselib
```

Expected: all tests pass; the default build uses PoseLib and the no-default build retains its fallback behavior.

- [x] **Step 7: Commit Task 3**

```bash
git add RustSFM/src/sfm/mapper.rs
git commit -m "perf(rustsfm): report structureless solver outcomes"
```

### Task 4: Add Robust Structureless Regression Coverage

**Files:**
- Modify: `RustSFM/src/geometry/generalized_pose.rs:2180-2245`
- Modify: `RustSFM/README.md:20-35`
- Modify: `RustSFM/COLMAP_MODULE_PARITY.md:150-160`

- [x] **Step 1: Add a synthetic outlier regression test**

Copy the existing deterministic structureless scene construction into a new feature-gated test named `poselib_structureless_absolute_pose_rejects_outliers`. Append six query observations whose pixels are displaced by at least 80 pixels while retaining the corresponding world observations and camera indices. Use fixed seed 23, `max_error = 1.0`, `min_num_trials = 64`, and `max_num_trials = 512`.

Assert that every true correspondence remains an inlier, at least five of six deliberately bad
correspondences are rejected, and the robust pose remains bounded:

```rust
assert!(estimate.inlier_mask[..inlier_count].iter().all(|&value| value));
assert!(estimate.inlier_mask[inlier_count..]
    .iter()
    .filter(|&&value| !value)
    .count() >= 5);
assert_pose_close(estimate.query_cam_from_world, query_cam_from_world, 7.0e-2);
```

- [x] **Step 2: Run the regression test**

Run:

```bash
cargo test -p rustsfm poselib_structureless_absolute_pose_rejects_outliers -- --nocapture
```

Expected: PASS against the existing robust solver. This is characterization coverage for an already implemented algorithm, so no production change follows unless the test exposes a real defect.

- [x] **Step 3: Update user-facing build documentation**

Change the README to state that PoseLib solvers are enabled in default builds and initialized through the submodule. Document both commands:

```bash
git submodule update --init --recursive
cargo test -p rustsfm --lib
```

Also document the intentional dependency-minimal check:

```bash
cargo test -p rustsfm --lib --no-default-features
```

Update the parity note that currently calls PoseLib optional so it accurately describes the default and no-default paths.

- [x] **Step 4: Run focused and full library verification**

Run:

```bash
cargo fmt --package rustsfm --check
cargo test -p rustsfm --lib -- --skip real_colmap_sparse
cargo test -p rustsfm --lib --no-default-features -- --skip real_colmap_sparse
cargo test -p rustsfm --lib --features poselib -- --skip real_colmap_sparse
cargo build -p rustsfm --release
```

Expected: every command exits zero. The `real_colmap_sparse` exclusions are required because the
ignored worktree fixture no longer points to the original 24-image COLMAP model; this known
environment gap must be reported separately. Warnings from external Ceres headers are acceptable;
Rust compilation errors and included-test failures are not.

- [x] **Step 5: Commit Task 4**

```bash
git add RustSFM/src/geometry/generalized_pose.rs RustSFM/README.md RustSFM/COLMAP_MODULE_PARITY.md
git commit -m "test(rustsfm): cover default structureless solver"
```

### Task 5: Validate All Flowers2 Images and RustGS Consumption

**Files:**
- Create: `docs/superpowers/plans/2026-07-20-rustsfm-poselib-default-benchmark.md`
- Read: `/Users/tfjiang/Projects/RustScan/test_data/flowers2`
- Create output outside git: `/tmp/rustsfm-flower2-poselib-20260720`

- [ ] **Step 1: Verify the release binary capability before the long run**

Run the release CLI against an intentionally invalid reconstruction input only far enough to capture its first output line, or invoke a bounded valid mapper command. Confirm the output contains:

```text
rustsfm_capabilities poselib=true structureless_gr6p_gr8p=true
```

Also verify the binary contains the PoseLib bridge symbol:

```bash
nm target/release/rustsfm | rg rustsfm_poselib_gen_relpose_6pt
```

- [ ] **Step 2: Run all 960 images without sampling**

Reuse the same `flowers2` database and mapper options as the 900-image baseline, changing only the output directory and binary. Capture `/usr/bin/time`, info/debug logs, and summary JSON under `/tmp/rustsfm-flower2-poselib-20260720`.

Do not terminate or modify the previously preserved process PID 73017.

- [ ] **Step 3: Analyze solver and reconstruction outcomes**

Extract for every produced sparse model:

- registered images;
- 3D point count;
- structureless attempts, estimates, and accepted choices;
- structureless solver time;
- PnP calls and registration hot-path timings;
- images absent from the largest model and from all models.

Acceptance requires no obsolete "not ported" messages, at least one real solver estimate, no regression below the 900-image largest-model baseline, finite registered poses, and valid COLMAP sparse files. Report the actual count rather than claiming 960/960 unless measured.

- [ ] **Step 4: Run the bounded RustGS compatibility probe**

Point RustGS at the largest new sparse model and its image directory. Run the existing one-frame, one-iteration, 1,000-Gaussian probe with PLY export and reload. Require no NaN/OOM, successful COLMAP parsing, successful training iteration, and export roundtrip.

- [ ] **Step 5: Record benchmark evidence**

Write the exact commands, commit hash, capability line, elapsed time, model table, telemetry comparison, remaining-image classification, and RustGS probe result to the benchmark document. Include the previous baseline of 900 images and 420,294 points.

- [ ] **Step 6: Run final repository verification and commit evidence**

Run:

```bash
cargo fmt --all --check
git diff --check
git status --short
```

Review that only the benchmark document is uncommitted, then commit:

```bash
git add docs/superpowers/plans/2026-07-20-rustsfm-poselib-default-benchmark.md
git commit -m "docs(rustsfm): record PoseLib flowers2 benchmark"
```

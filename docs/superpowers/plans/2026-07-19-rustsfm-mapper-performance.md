# RustSFM Mapper Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove avoidable mapper work, reject low-quality database edges, and reduce Ceres global BA cost on macOS without weakening the final reconstruction pass.

**Architecture:** Database-backed reconstruction will construct lightweight frames directly from database keypoints and load colors only for registered images. Pair validation will use one quality gate for computed and stored geometry. Global BA will retain local BA plus a final full solve while using a fixed intermediate iteration budget and a minimum registration-growth interval. macOS will use the installed system Ceres build so its SuiteSparse/Accelerate backends and Schur specializations are available.

**Tech Stack:** Rust, rusqlite/COLMAP database format, Ceres Solver 2.2, Cargo target-specific dependencies, Rust unit tests.

---

### Task 1: Database Frame Fast Path and Lazy Colors

**Files:**
- Modify: `RustSFM/src/sfm/mapper/reconstruction_input.rs`
- Modify: `RustSFM/src/sfm/mapper.rs`
- Test: `RustSFM/src/sfm/mapper.rs`

- [x] **Step 1: Write the failing database-frame test**

Add a unit test that creates two image headers and a COLMAP database, then verifies the database preparation path preserves database keypoints while leaving SIFT, wide descriptors, and eager colors empty:

```rust
#[test]
fn database_frame_fast_path_does_not_extract_descriptors_or_colors() -> Result<()> {
    let (paths, database) = test_database_with_images_and_keypoints()?;
    let frames = database_frames(&paths, &database)?;
    assert_eq!(frames[0].keypoints.len(), 4);
    assert!(frames[0].sift.descriptors_u8.is_empty());
    assert!(frames[0].wide_descriptors.data.is_empty());
    assert!(frames[0].colors.is_empty());
    Ok(())
}
```

- [x] **Step 2: Run the focused test and verify RED**

Run:

```bash
cargo test -p rustsfm database_frame_fast_path_does_not_extract_descriptors_or_colors -- --exact
```

Expected: FAIL because `database_frames` does not exist.

- [x] **Step 3: Implement lightweight database frames**

Resolve the database before calling `extract_frames`. When it exists, build `ImageFrame` values from paths, image dimensions, and database keypoints with empty descriptor/color fields. Keep the existing full extraction path when no database exists.

- [x] **Step 4: Add and verify lazy color tests**

Add a test where `frame.colors` is empty but the frame points at a small RGB image, and verify `extract_colors_for_image` samples the observed feature color. Run the test before and after implementation.

- [x] **Step 5: Run mapper database tests**

Run:

```bash
cargo test -p rustsfm mapper_database
cargo test -p rustsfm database_frame
cargo test -p rustsfm extract_colors
```

Expected: PASS.

### Task 2: Unified Pair Quality Gate

**Files:**
- Modify: `RustSFM/src/sfm/mapper.rs`
- Test: `RustSFM/src/sfm/mapper.rs`

- [x] **Step 1: Write failing stored-pair rejection tests**

Add tests proving that a stored non-adjacent edge is rejected when its reprojection error exceeds the mapper threshold or it lacks enough triangulated support, while a geometrically sound wide-baseline edge remains accepted.

```rust
assert!(!keep_pair_for_mapping(&high_error_pair, &config));
assert!(!keep_pair_for_mapping(&weak_pair, &config));
assert!(keep_pair_for_mapping(&supported_pair, &config));
```

- [x] **Step 2: Run focused tests and verify RED**

Run:

```bash
cargo test -p rustsfm stored_database_pair_rejects_high_reprojection_error -- --exact
```

Expected: FAIL because the current stored-pair gate accepts any finite error.

- [x] **Step 3: Implement one gate for stored and computed pairs**

The gate must reject invalid COLMAP configurations, require configured inlier/triangulation support, allow a relaxed error bound for adjacent backbone edges, and require stronger support plus finite bounded reprojection error for non-adjacent edges. Do not reject an edge merely because its relative rotation magnitude is large; rotation consistency belongs to view-graph filtering.

- [x] **Step 4: Run pair/database mapper tests**

Run:

```bash
cargo test -p rustsfm keep_pair
cargo test -p rustsfm estimate_database_pair_geometries
```

Expected: PASS with old weak-gate expectations updated to the unified policy.

### Task 3: Bounded Global BA Scheduling

**Files:**
- Modify: `RustSFM/src/sfm/mapper/bundle_adjustment.rs`
- Modify: `RustSFM/src/sfm/mapper/config.rs`
- Modify: `RustSFM/src/sfm/mapper.rs`
- Modify: `RustSFM/src/cli/mod.rs`
- Test: `RustSFM/src/sfm/mapper.rs`

- [x] **Step 1: Write failing fixed-budget test**

Construct a reconstruction with more than 20,000 observations and assert that `global_ba_iterations_for_reconstruction` still returns the configured 50 iterations.

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
cargo test -p rustsfm global_ba_iteration_budget_does_not_grow_with_observations -- --exact
```

Expected: FAIL with actual value 150.

- [x] **Step 3: Remove observation-count iteration multiplication**

Make all intermediate global BA calls use the configured iteration count. Preserve the existing small-initial-reconstruction Ceres tolerance behavior and the final global BA.

- [x] **Step 4: Write failing minimum-growth scheduling test**

After marking the schedule, register fewer than five frames and assert that ratio-only triggers do not run another global BA; register the fifth and assert that it runs.

- [x] **Step 5: Implement minimum registration growth and safer defaults**

Require at least five newly registered frames for ratio-based image/point triggers, while preserving absolute frequency triggers. Change native defaults for both ratios from `1.1` to `1.5`; explicit COLMAP project values continue to override them.

- [x] **Step 6: Run BA schedule and CLI tests**

Run:

```bash
cargo test -p rustsfm global_ba_schedule
cargo test -p rustsfm global_ba_iteration
cargo test -p rustsfm cli
```

Expected: PASS.

### Task 4: High-Performance macOS Ceres Backend

**Files:**
- Modify: `RustSFM/Cargo.toml`
- Modify: `RustSFM/src/ba/ceres_problem.rs`
- Test: `RustSFM/src/ba/ceres_problem.rs`

- [x] **Step 1: Add a backend-report regression test**

Expose the sparse linear algebra backend selected by Ceres in `BundleAdjustmentReport` or a focused policy helper, and assert that macOS does not select `EIGEN_SPARSE` when the Homebrew Ceres installation is available.

- [x] **Step 2: Run the backend test and verify RED**

Run:

```bash
cargo test -p rustsfm --lib ba::ceres_problem::tests::ceres_sparse_backend_uses_optimized_macos_build -- --exact
```

Expected: FAIL because the bundled source build only provides EigenSparse.

- [x] **Step 3: Select Ceres by target**

Use system Ceres on macOS and the bundled source build on other targets. Keep `ceres-solver` default features disabled so `source` and `system` are never enabled together.

- [x] **Step 4: Verify compile configuration and tests**

Run:

```bash
cargo clean -p ceres-solver-sys
cargo test -p rustsfm ceres_solver_policy
cargo test -p rustsfm --lib -- --skip real_colmap_sparse
```

Expected: PASS; linked Ceres reports SuiteSparse or AccelerateSparse, and the vendored `ceres-solver-src` build is absent from the active macOS feature tree. The Linux feature tree must contain `source` without `system`. The unfiltered suite currently has 19 pre-existing `real_colmap_sparse_*` failures because the local flowers2 fixture no longer matches the tests' 24-image/256-point expectations.

### Task 5: flowers2 Smoke Benchmark

**Files:**
- Modify: `docs/superpowers/plans/2026-07-19-rustsfm-mapper-performance.md`

- [x] **Step 1: Run a bounded database-backed smoke reconstruction**

Run the mapper on a representative contiguous subset of flowers2 using the existing database and record extraction time, retained pair count, global BA count, termination status, registered images, and elapsed time.

- [x] **Step 2: Compare invariants**

Verify database preparation is no longer dominated by SIFT extraction, high-error long edges are removed without isolating the subset, global BA never exceeds 50 configured iterations, and the output remains readable by RustGS.

Benchmark result (release build, `random_seed=0`, first 24 contiguous flowers2 frames, existing 960-image database):

- RustSFM: 24/24 images registered, 6,845 points, one model, 88/89 database pairs retained, zero isolated images.
- Timing: database frame preparation 52.11 ms, pair loading/filtering 9.84 ms, incremental mapping 7,011.14 ms, internal elapsed 7,161.37 ms, wall time 7.82 s. The earlier database-backed run spent about 216 s in eager frame extraction.
- Global BA: seven passes; every pass was configured for 50 iterations. Ceres reports at most 51 attempted entries because its summary includes iteration zero. Each pass terminated at the hard cap as `NoConvergence / MaxIterations`; reconstruction still completed normally with all 24 frames registered.
- Pair quality: mean 648.0 inliers, 0.483 px mean reprojection error, 7.587 degree mean triangulation angle; the retained graph has degree range 4-8.
- RustGS probe: resolved `sparse/0` with one camera, zero missing images, and 6,845 initialization points; a one-frame, one-iteration Metal training probe saved 1,000 Gaussians successfully.
- Outputs were written under `/tmp/rustsfm-flowers2-smoke-20260719/bounded` and are not repository artifacts.

- [x] **Step 3: Run formatting and full default-feature verification**

Run:

```bash
cargo fmt --all -- --check
cargo test -p rustsfm --lib -- --skip real_colmap_sparse
cargo check -p rustgs
```

Expected: PASS. Fresh result: 557 passed, 0 failed, 19 known fixture tests filtered; `cargo fmt --all -- --check` and `cargo check -p rustgs` both passed.

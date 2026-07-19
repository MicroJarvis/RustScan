# RustSFM macOS BA Backend Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose Ceres sparse backend and Schur solver selection, report the actual backend used, and add phase timings so SuiteSparse, Accelerate Sparse, and iterative Schur can be compared on flowers2.

**Architecture:** Add backend-neutral preferences to `ba` and `MapperConfig`, map them to Ceres only inside `ceres_problem.rs`, and preserve the current automatic thresholds when preferences are `Auto`. Capture setup, solve, and postprocess timing in `BundleAdjustmentReport`; mapper log formatting will include those fields without changing mapping decisions.

**Tech Stack:** Rust, clap, serde-free project INI parsing, Ceres Solver 2.2 Rust bindings, SuiteSparse, Apple Accelerate Sparse, Cargo tests, release-mode flowers2 benchmark.

---

## File Map

- Modify `RustSFM/src/ba/mod.rs`: add backend-neutral solver preferences and timing/report fields; keep the native/Ceres public BA API stable.
- Modify `RustSFM/src/ba/ceres_problem.rs`: resolve explicit Ceres sparse backends and solver preferences, return the actual backend, and measure setup/solve/postprocess phases.
- Modify `RustSFM/src/ba/native.rs`: populate new report fields and honor the solver preference in the existing native policy.
- Modify `RustSFM/src/sfm/mapper/config.rs`: store and default the two mapper preferences; re-export parsing types from `RustSFM/src/lib.rs` through the mapper module.
- Modify `RustSFM/src/sfm/mapper/bundle_adjustment.rs`: forward mapper preferences into `BundleAdjustmentOptions`.
- Modify `RustSFM/src/cli/mod.rs`: add native reconstruction flags and COLMAP-compatible mapper flags.
- Modify `RustSFM/src/cli/commands.rs`: forward native reconstruction flag values into `MapperConfig` and parse COLMAP-compatible resolved values.
- Modify `RustSFM/src/cli/project.rs`: resolve project INI values, preserve command-line precedence, and test propagation/defaults.
- Modify `RustSFM/src/sfm/mapper.rs`: include selected solver/backend and BA phase timings in local/global BA log lines.
- Modify `vendor/ceres-solver/src/solver.rs` only if the existing builder lacks a getter needed to report an explicitly selected backend; prefer the existing sparse-backend setter and current-backend getter.

### Task 1: Add backend-neutral preferences and parser tests

**Files:**
- Modify: `RustSFM/src/ba/mod.rs`
- Modify: `RustSFM/src/sfm/mapper/config.rs`
- Modify: `RustSFM/src/lib.rs`
- Test: `RustSFM/src/ba/mod.rs`
- Test: `RustSFM/src/sfm/mapper/config.rs`

- [ ] **Step 1: Write parser tests before implementation**

Add tests requiring case-insensitive parsing and hyphen/underscore aliases:

```rust
assert_eq!("accelerate-sparse".parse(), Ok(BundleAdjustmentSparseLinearAlgebra::AccelerateSparse));
assert_eq!("ACCELERATE_SPARSE".parse(), Ok(BundleAdjustmentSparseLinearAlgebra::AccelerateSparse));
assert_eq!("iterative_schur".parse(), Ok(BundleAdjustmentLinearSolverPreference::IterativeSchur));
assert!("cuda".parse::<BundleAdjustmentSparseLinearAlgebra>().is_err());
```

Also assert both `Default` implementations are `Auto`.

- [ ] **Step 2: Run focused tests to verify RED**

Run `cargo test -p rustsfm bundle_adjustment_preference --lib`. Expected: compile failure because the preference enums and parsers do not yet exist.

- [ ] **Step 3: Implement the preference enums**

In `RustSFM/src/ba/mod.rs`, add `BundleAdjustmentLinearSolverPreference` with `Auto`, `DenseSchur`, `SparseSchur`, and `IterativeSchur`, and `BundleAdjustmentSparseLinearAlgebra` with `Auto`, `SuiteSparse`, `AccelerateSparse`, and `EigenSparse`. Implement `FromStr` using `to_ascii_lowercase().replace('-', "_")`, return an `anyhow::Error` listing accepted values, and implement `Display`, `Default`, `Debug`, `Clone`, `Copy`, `PartialEq`, and `Eq`.

Add the two fields to `BundleAdjustmentOptions` with `Auto` defaults. Add `MapperConfig` fields with the same defaults and re-export the types from `RustSFM/src/lib.rs` so clap can use them.

- [ ] **Step 4: Run preference tests and existing BA option tests**

Run:

```bash
cargo test -p rustsfm bundle_adjustment_preference --lib
cargo test -p rustsfm mapper_global_ba_options --lib
```

Expected: PASS.

- [ ] **Step 5: Commit the preference layer**

```bash
git add RustSFM/src/ba/mod.rs RustSFM/src/sfm/mapper/config.rs RustSFM/src/lib.rs
git commit -m "feat(rustsfm): add BA backend preferences"
```

### Task 2: Map preferences to Ceres and native Schur policy

**Files:**
- Modify: `RustSFM/src/ba/ceres_problem.rs`
- Modify: `RustSFM/src/ba/native.rs`
- Modify: `RustSFM/src/sfm/mapper/bundle_adjustment.rs`
- Test: `RustSFM/src/ba/ceres_problem.rs`
- Test: `RustSFM/src/ba/native.rs`

- [ ] **Step 1: Write failing Ceres policy tests**

Extend the existing policy tests with explicit solver and backend mapping:

```rust
assert_eq!(map_sparse_backend(BundleAdjustmentSparseLinearAlgebra::SuiteSparse), SparseLinearAlgebraLibraryType::SUITE_SPARSE);
assert_eq!(map_sparse_backend(BundleAdjustmentSparseLinearAlgebra::AccelerateSparse), SparseLinearAlgebraLibraryType::ACCELERATE_SPARSE);
assert_eq!(map_requested_solver(BundleAdjustmentLinearSolverPreference::IterativeSchur, 10, true).linear_solver, LinearSolverType::ITERATIVE_SCHUR);
assert_eq!(map_requested_solver(BundleAdjustmentLinearSolverPreference::IterativeSchur, 10, true).preconditioner, Some(PreconditionerType::SCHUR_JACOBI));
```

Test that `Auto` retains the 50/1000 thresholds and that explicit `DenseSchur`, `SparseSchur`, and `IterativeSchur` bypass those thresholds.

- [ ] **Step 2: Run policy tests to verify RED**

Run `cargo test -p rustsfm ceres_solver_policy --lib`. Expected: compile or assertion failure because the current policy has no preference argument or explicit backend mapping.

- [ ] **Step 3: Implement explicit Ceres mapping**

Change the internal policy helper to accept both preferences. Keep `Auto` exactly equivalent to the current implementation. Map the sparse preference to Ceres' `SparseLinearAlgebraLibraryType`; for `Auto`, do not set a backend on the builder. Set the explicit backend before `build()` and record the builder's resolved current backend after setting it. Map explicit solver preferences directly and attach `SCHUR_JACOBI` only to `IterativeSchur`.

Return the Ceres option object, resolved backend, and policy together so the report does not infer the backend from the requested value. If Ceres rejects an explicitly unavailable backend during option validation, return `None` from the existing BA boundary and include the requested backend in the diagnostic string used by the checked mapper wrapper.

- [ ] **Step 4: Implement native policy forwarding**

Change `native_linear_solver_policy` to accept `BundleAdjustmentLinearSolverPreference`. Use the existing size thresholds only for `Auto`; map explicit choices directly and use Schur-Jacobi for explicit iterative Schur. Populate the report's sparse-backend field with `None` for native BA.

- [ ] **Step 5: Forward mapper preferences into BA options**

In `mapper_ba_options`, copy `config.ba_linear_solver` and `config.ba_sparse_backend` into `BundleAdjustmentOptions`. Do not alter existing mapper defaults or thresholds.

- [ ] **Step 6: Run backend and native tests**

Run:

```bash
cargo test -p rustsfm ceres_solver_policy --lib
cargo test -p rustsfm native_linear_solver_policy --lib
cargo test -p rustsfm --lib -- --skip real_colmap_sparse
```

Expected: all focused and non-fixture library tests pass.

- [ ] **Step 7: Commit solver selection**

```bash
git add RustSFM/src/ba RustSFM/src/sfm/mapper/bundle_adjustment.rs
git commit -m "feat(rustsfm): select Ceres BA solver backends"
```

### Task 3: Add BA phase timing and mapper diagnostics

**Files:**
- Modify: `RustSFM/src/ba/mod.rs`
- Modify: `RustSFM/src/ba/ceres_problem.rs`
- Modify: `RustSFM/src/ba/native.rs`
- Modify: `RustSFM/src/sfm/mapper.rs`
- Test: `RustSFM/src/ba/ceres.rs`
- Test: `RustSFM/src/sfm/mapper.rs`

- [ ] **Step 1: Write timing/report tests**

After an existing small Ceres fixture solve, assert:

```rust
assert!(report.setup_ms.is_finite() && report.setup_ms >= 0.0);
assert!(report.solve_ms.is_finite() && report.solve_ms >= 0.0);
assert!(report.postprocess_ms.is_finite() && report.postprocess_ms >= 0.0);
assert!(report.elapsed_ms >= report.solve_ms);
```

Add a mapper log assertion that a local BA line contains `setup_ms=`, `solve_ms=`, `postprocess_ms=`, and `ba_elapsed_ms=`.

- [ ] **Step 2: Run timing tests to verify RED**

Run:

```bash
cargo test -p rustsfm ceres_local_ba_keeps_constant_image_fixed --lib
cargo test -p rustsfm local_ba --lib
```

Expected: compile failure because report timing fields and diagnostic suffixes do not yet exist.

- [ ] **Step 3: Add report timing and selected backend fields**

Extend `BundleAdjustmentReport` with `setup_ms`, `solve_ms`, `postprocess_ms`, `elapsed_ms`, and `sparse_backend: Option<BundleAdjustmentSparseLinearAlgebra>`. Update Ceres and native constructors with real values or zero/total values where a phase is not separately measurable.

- [ ] **Step 4: Measure Ceres phases**

Use `std::time::Instant` in `solve_bundle_adjustment_ceres`: start a total timer before collection, a setup timer around observation/parameter/problem/manifold/solver-option construction, a solve timer around `problem.solve`, and a postprocess timer around solution write-back, point-error refresh, covariance, and summary mapping. Store the resolved sparse backend in the report and keep all existing early-return behavior unchanged.

- [ ] **Step 5: Add diagnostic fields to local/global mapper logs**

Append selected solver, preconditioner, sparse backend, phase timings, and total elapsed time to the existing `local_ba` and `global_ba` format strings. Keep existing key names and values so fixture tests remain compatible.

- [ ] **Step 6: Run timing and log tests**

Run:

```bash
cargo test -p rustsfm ceres_local_ba_keeps_constant_image_fixed --lib
cargo test -p rustsfm local_ba --lib
cargo test -p rustsfm --lib -- --skip real_colmap_sparse
```

Expected: PASS, with finite non-negative phase timings and unchanged reconstruction invariants.

- [ ] **Step 7: Commit diagnostics**

```bash
git add RustSFM/src/ba RustSFM/src/sfm/mapper.rs
git commit -m "perf(rustsfm): report BA backend timings"
```

### Task 4: Add native and COLMAP-compatible CLI/project configuration

**Files:**
- Modify: `RustSFM/src/cli/mod.rs`
- Modify: `RustSFM/src/cli/commands.rs`
- Modify: `RustSFM/src/cli/project.rs`
- Test: `RustSFM/src/cli/mod.rs`
- Test: `RustSFM/src/cli/project.rs`

- [ ] **Step 1: Write failing CLI and project propagation tests**

Add native clap parsing coverage for `--ba-linear-solver iterative-schur --ba-sparse-backend accelerate-sparse`. Add project resolver coverage for:

```ini
[Mapper]
ba_linear_solver=iterative_schur
ba_sparse_backend=accelerate-sparse
```

Assert command-line values override project values and an empty project uses `Auto`.

- [ ] **Step 2: Run CLI tests to verify RED**

Run `cargo test -p rustsfm cli::project --lib` and `cargo test -p rustsfm cli --lib`. Expected: compile failure because the flags and resolved fields do not exist.

- [ ] **Step 3: Add native reconstruction flags**

Add typed clap fields to `ReconstructArgs` with `default_value = "auto"`, using the public preference enums as `FromStr` values. Forward them to `MapperConfig` in `run_reconstruct`.

- [ ] **Step 4: Add COLMAP mapper fields and INI resolution**

Add optional string fields named `Mapper.ba_linear_solver` and `Mapper.ba_sparse_backend` to `ColmapMapperArgs` and `ResolvedColmapMapperArgs`. Resolve command-line value first, then project value, then `auto`; parse the resulting string into the public enums when constructing `MapperConfig`. Preserve existing numeric/bool resolver behavior and error style.

- [ ] **Step 5: Run CLI/project tests**

Run:

```bash
cargo test -p rustsfm cli::project --lib
cargo test -p rustsfm cli --lib
cargo test -p rustsfm --lib -- --skip real_colmap_sparse
```

Expected: PASS.

- [ ] **Step 6: Commit CLI support**

```bash
git add RustSFM/src/cli RustSFM/src/sfm/mapper/config.rs RustSFM/src/sfm/mapper.rs
git commit -m "feat(rustsfm): expose BA backend controls"
```

### Task 5: Format, compile, and benchmark backend choices

**Files:**
- Modify: `docs/superpowers/plans/2026-07-19-rustsfm-macos-ba-backends.md`
- Create: no repository benchmark artifacts; write results only into this plan

- [ ] **Step 1: Run formatting and compile verification**

Run:

```bash
cargo fmt --all -- --check
cargo check -p rustsfm
cargo check -p rustgs
```

Expected: all commands pass.

- [ ] **Step 2: Run focused and non-fixture tests**

Run `cargo test -p rustsfm --lib -- --skip real_colmap_sparse`. Expected: all non-fixture tests pass; the 19 known fixture mismatches remain excluded.

- [ ] **Step 3: Run 96-frame backend benchmark**

Use the existing database-backed flowers2 input, fixed `random_seed=0`, `--threads 1`, and identical mapper limits. Run `auto`, `suite-sparse`, `accelerate-sparse`, and `iterative-schur` where system Ceres accepts the explicit backend. Record registered images, points, BA count, selected backend, phase timings, and wall time in this plan.

- [ ] **Step 4: Run 200-frame benchmark for stable candidates**

Repeat for candidates whose 96-frame output has identical registration and finite costs. Reject any configuration that silently falls back, changes registration count, or produces non-finite termination diagnostics.

- [ ] **Step 5: Run the winning configuration on all 960 frames**

Run only the fastest numerically equivalent configuration on all frames, then load its sparse output with RustGS and verify camera count, image coverage, and initialization point count. Do not change the default backend based on one run; record a recommendation for a separate default-change decision.

- [ ] **Step 6: Complete the plan with benchmark results and commit**

Run:

```bash
git diff --check
git status --short
```

Append actual command lines and measured results to this plan, then commit:

```bash
git add docs/superpowers/plans/2026-07-19-rustsfm-macos-ba-backends.md
git commit -m "docs(rustsfm): record BA backend benchmarks"
```

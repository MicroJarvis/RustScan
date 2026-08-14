# RustSFM Review Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the confirmed RustSFM build, output-reuse, CI, worktree-native-dependency,
and GPU pipeline error-handling problems without changing SfM algorithm behavior.

**Architecture:** Preserve the public `Cache/database.db` and `Cache/keyframe-sparse/0` contracts used by RustViewer. Give each keyframe reconstruction a private, copied `TempDir` snapshot and hold its guard through mapper export, so later source mutations and concurrent writers cannot change the paths being consumed. Serialize the shared database/output portion per output with a persistent fail-fast lock. Keep database reuse for adaptive selection, while ensuring mapper input paths bound the database rows that can participate.

**Tech Stack:** Rust 2021, Cargo features, SQLite/rusqlite, GitHub Actions, Rust integration tests.

---

### Task 1: Restore the minimal-feature build

**Files:**
- Modify: `RustSFM/src/gpu/mod.rs:1166`

- [x] Add `#[cfg(feature = "gpu-wgpu")]` to the two tests that reference `pnp_focal`.
- [x] Run `cargo test -p rustsfm --lib --no-default-features --no-run` and confirm compilation succeeds.

### Task 2: Make the keyframe input an exact request snapshot

**Files:**
- Modify: `RustSFM/src/sequence_registration.rs:194`
- Test: `RustSFM/tests/sequence_registration.rs`

- [x] Add a failing test that seeds an obsolete image in `Cache/keyframes`, invokes the keyframe stage, and proves mapper consumes only the requested files.
- [x] Run the focused test and confirm it fails because the stale image remains.
- [x] Replace the shared input directory with a unique private snapshot whose guard remains alive through mapper export.
- [x] Copy rather than hard-link selected images, reject symlink inputs and destination collisions, and validate the exact staged entry set.
- [x] Serialize each output's keyframe reconstruction with a persistent fail-fast lock and remove the legacy shared directory only after success.
- [x] Ensure staging or downstream failure drops only the private snapshot and preserves the prior legacy directory.
- [x] Run the focused test and the complete sequence-registration integration target.

### Task 3: Execute RustSFM tests in CI

**Files:**
- Modify: `.github/workflows/ci.yml:70`
- Modify: `RustSFM/src/feature/sift.rs:1186`

- [x] Change CI from build-only checks to execute supported default- and minimal-feature lib tests.
- [x] Keep optional external sparse fixtures out of CI's default test run.
- [x] Make the empty-input GPU benchmark return successfully when no compatible adapter is available.
- [x] Run the focused SIFT test where an adapter is available and run the minimal-feature lib suite everywhere.

### Task 4: Make native dependency discovery worktree-safe

**Files:**
- Modify: `RustSFM/build.rs:140`

- [x] Add unit-testable candidate construction that accepts the current manifest directory as an argument.
- [x] Replace compile-time `env!("CARGO_MANIFEST_DIR")` with runtime `env::var_os("CARGO_MANIFEST_DIR")` in `main`-time resolution.
- [x] Emit `rerun-if-env-changed=CARGO_MANIFEST_DIR` and retain explicit `POSELIB_ROOT`/`VLFEAT_ROOT` precedence.
- [x] Run default and minimal-feature RustSFM build checks.

### Task 5: Final verification

**Files:**
- Verify only; no new production changes.

- [x] Run `cargo fmt --package rustsfm --check`.
- [x] Run `cargo test -p rustsfm --lib --no-default-features`.
- [x] Run `cargo test -p rustsfm --test sequence_registration --no-default-features`.
- [x] Run `cargo test -p rustsfm --test adaptive_keyframes --no-default-features`.
- [x] Review `git diff --check`, the final diff, and confirm unrelated untracked files are unchanged.

### Task 6: Return PnP-focal pipeline failures instead of unwinding

**Files:**
- Modify: `RustSFM/src/gpu/pnp_focal.rs`
- Modify: `RustSFM/src/gpu/mod.rs`
- Test: `RustSFM/src/sfm/mapper.rs`

- [x] Reproduce the Metal failure and confirm that `MTLCompilerService` returns
  `XPC_ERROR_CONNECTION_INTERRUPTED` in the `AGXMetalG17X` domain.
- [x] Add a failing regression test proving `WgpuPnPFocalSolver::from_context` unwinds while the
  pipeline error is uncaptured.
- [x] Route all eight PnP-focal compute pipeline creation sites through a shared helper that scopes
  `Validation`, `OutOfMemory`, and `Internal` errors and returns the captured error with pipeline
  context.
- [x] Add test-only handling that skips affected correctness tests only on macOS and only when the
  complete error chain contains both `XPC_ERROR_CONNECTION_INTERRUPTED` and `AGXMetal`.
- [x] Confirm unrelated WGSL/validation errors remain test failures and mapper GPU errors still
  fall back to CPU PnP-f.
- [x] Run the default-feature lib suite: 700 passed, 0 failed, 19 explicitly filtered fixture
  tests.
- [x] Run the minimal-feature lib suite: 574 passed, 0 failed, 19 explicitly filtered fixture
  tests.
- [x] Run the minimal-feature integration targets: adaptive keyframes 10 passed, build support 1
  passed, sequence registration 61 passed.

The local Metal compiler service still cannot compile the P3P candidate shader, so seven numerical
GPU correctness tests and the real-solver mapper test are reported as environment skips on this
machine. Healthy backends still execute those tests, while production pipeline initialization now
returns an error that the existing mapper dispatch converts into its CPU fallback. A follow-up
should cache an unavailable GPU solver for the mapper session so repeated registration attempts do
not retry the same failed pipeline initialization.

### Task 7: Address final review findings

**Files:**
- Modify: `RustSFM/src/sequence_registration.rs`
- Modify: `RustSFM/src/gpu/pnp_focal.rs`
- Modify: `RustSFM/src/build_support.rs`
- Modify: `RustSFM/tests/build_support.rs`
- Modify: `.github/workflows/ci.yml`

- [x] Replace the non-atomic shared keyframe directory with a mapper-pinned private snapshot.
- [x] Add filesystem-equivalence, symlink, source-mutation, lock-contention, and recovery regressions.
- [x] Preserve captured `wgpu::Error` values as typed anyhow sources.
- [x] Make sparse/JSON publication sync platform-aware and clean staged state when the first rename fails.
- [x] Make the production build script call the shared runtime manifest lookup and test that wiring.
- [x] Execute integration targets on Linux and filesystem publication tests on Windows in CI.
- [x] Re-run default/minimal lib suites, integration targets, formatting, and diff validation.

### Task 8: Make RustSFM verification and project status truthful

**Files:**
- Modify: `RustSFM/src/io/colmap.rs`
- Modify: `RustSFM/src/sfm/mapper.rs`
- Modify: `RustSFM/README.md`
- Modify: `.github/workflows/ci.yml`
- Modify: root and `docs/` status/index/architecture documents

- [x] Mark the 19 `real_colmap_sparse_*` tests as ignored because their compatible external
  `test_data/flowers2_colmap` fixture is not distributed by Git or submodules.
- [x] Restore unfiltered RustSFM library commands in CI and document the opt-in `--ignored`
  parity command.
- [x] Verify the default library suite: `708 passed; 0 failed; 19 ignored`.
- [x] Verify the minimal library suite: `581 passed; 0 failed; 19 ignored`.
- [x] Verify the minimal integration suites: adaptive keyframes `10 passed`, build support
  `3 passed` plus one child-process probe intentionally ignored, sequence registration
  `62 passed`.
- [x] Update canonical project documentation so RustSFM appears in workspace structure and
  current status.

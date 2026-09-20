# Crate naming migration verification

- Task: `RS-2026-003`
- Base commit: `41d5b08bf0cc36731e8d5c574b0d3edbfdd76c96`
- Implementation branch: `codex/unify-crate-names`
- Final commit: not created; changes are staged for review.

## Scope and compatibility

Renamed the six first-party component directories and Cargo packages:

| Before | After |
|---|---|
| `rustff` | `rustscan-ff` |
| `rustgs` | `rustscan-gs` |
| `rustmesh` | `rustscan-mesh` |
| `rustsfm` | `rustscan-sfm` |
| `rustslam` | `rustscan-slam` |
| `rust-viewer` | `rustscan-viewer` |

Updated library names/imports, workspace dependencies, root lockfile, CI,
scripts, generator paths, examples, README and maintained documentation.
Removed the stale tracked mesh member lockfile; the root lockfile is the
workspace authority. `AGENTS.md` defines the mandatory naming convention;
`docs/agent/repository-layout.md` explains compatibility exceptions.

Package and import names intentionally change. Existing binaries (`rustgs`,
`rustsfm`, `rustslam`, `rust-viewer`), example targets, feature names, native
symbols, and persisted data paths keep their names. SfM explicitly selects
`rustsfm` as `default-run`; SLAM explicitly declares its existing binary.
Algorithm implementations are unchanged apart from imports and formatting.

Pre-existing staged documentation cleanup was carried into the isolated
worktree from index tree `bcff5ed2168498110451bc0aadcfb976ebb2e32b`.
Integration applies only the naming delta relative to that tree. The resulting
index in the original checkout matches the verified worktree index. Residual
ignored local output, caches and member lockfile are moved to the matching new
directories without overwriting data. No commit or push is performed.

## Environment and commands

Validation ran on macOS / Apple M5 Max. The isolated checkout reused the
original checkout's native sources and Cargo cache with:

```sh
export POSELIB_ROOT=/Users/tfjiang/Projects/RustScan/third_party/native/PoseLib
export VLFEAT_ROOT=/Users/tfjiang/Projects/RustScan/third_party/native/vlfeat
export CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target
```

| Command / check | Result |
|---|---|
| `cargo metadata --no-deps --format-version 1` | PASS: nine packages; component directories, packages and library names agree. All non-library target names/kinds match the baseline. |
| `cargo fmt --all -- --check` | PASS. |
| `cargo check --workspace --all-targets` | PASS before and after migration, including examples; existing warnings remain. |
| `cargo check -p rustscan-sfm --no-default-features --all-targets` | FAIL before and after: GPU examples import disabled GPU types/modules without feature gating. This predates the rename. |
| `cargo test --workspace --lib --bins --tests --no-fail-fast -- --test-threads=1` | FAIL: 1,893 passed, 15 failed, 26 ignored; four failing targets. See baseline comparison below. |
| `cargo test --workspace --doc --no-fail-fast` | PASS: 12 passed, four existing ignored. |
| `cargo clippy --workspace --all-targets -- -D warnings` | FAIL before and after: identical diagnostic text/counts, 280 diagnostics across 79 distinct messages, including dead code, missing safety docs and needless references. |
| `cargo clippy --offline --workspace --all-targets --all-features -- -D warnings` | UNAVAILABLE: `bzip2 v0.4.4` is absent from the local cache. All-feature lint validation is not claimed. |
| `python3 scripts/generate_five_point_wgsl.py --check` | PASS. |
| `python3 scripts/test_generate_five_point_wgsl.py` | PASS: five tests. |
| `python3 scripts/check_cpu_glam.py` | PASS. |
| Changed Python AST / shell syntax checks | PASS. |
| Naming and references audit | PASS: 406 first-party Rust files, no old qualified imports or legacy dependency aliases. 62 resolved documentation links checked; no newly broken tracked-file links. |
| Root lockfile comparison | External package resolutions are identical to the baseline. |
| `git diff --cached --check` | PASS. |

Actual directory-entry casing was checked. A build on a case-sensitive volume
and Linux/Windows execution have not been performed; CI remains the platform
gate. Examples were compiled by the all-targets check, not all executed.

## Baseline failure comparison

The complete migrated workspace test run reports four failing targets:

- SfM library: three failures. Two numerical GPU failures were reproduced
  before renaming with `cargo test -p rustsfm --features gpu-vulkan --lib gpu::
  -- --test-threads=1` (67 passed, two failed):
  `five_point_f32_actual_gpu_stages` and
  `wgpu_pnp_focal_p3p_reorders_adverse_baseline_before_solving`.
  The former also passed on the original default Metal/Wgpu configuration,
  demonstrating the backend-sensitive nature of the failure.
- SfM `sequence_registration`: ten failures at `src/sfm/mapper.rs:422`,
  indexing element four in a four-element array. The baseline reproduces the
  exact ten failing test names (62 passed, ten failed).
- SfM `wgpu_sift_quality`: one failure; both runs report 333 GPU keypoints
  against 1,235 CPU keypoints. Baseline integration command:
  `cargo test -p rustsfm --features gpu-vulkan --test sequence_registration
  --test wgpu_sift_quality --no-fail-fast -- --test-threads=1`.
- Viewer `project_store`: `new_manifest_uses_the_declared_schema_and_config_defaults`
  expects GPU SIFT enabled, while the macOS default disables it. Reproduced
  with `cargo test -p rust-viewer --test project_store
  new_manifest_uses_the_declared_schema_and_config_defaults
  -- --exact --test-threads=1`.

The third SfM library failure,
`wgpu_context_reports_a_real_adapter_when_available`, also reproduces in the
baseline: actual backend `Vulkan`, expected `Wgpu`. Exact command:
`cargo test -p rustsfm --features wgpu/vulkan,wgpu/vulkan-portability --lib
 gpu::tests::wgpu_context_reports_a_real_adapter_when_available
 -- --exact --test-threads=1`. This enables the dependency backend features
without the package's `gpu-vulkan` switch, matching the relevant workspace
feature-unification behavior.

The union of baseline failing test names exactly matches all 15 failures in
the migrated workspace run. No new failing test name was observed.

Additional baseline commands were `cargo check -p rustsfm
--no-default-features --all-targets` and
`cargo clippy --workspace --all-targets -- -D warnings` in the original
checkout. Raw local logs are under `/tmp/rustscan-naming-*.log`; they are
temporary supporting evidence, not required repository inputs.

## Handoff

Next action: review and commit the staged migration together with the
separately identifiable preceding documentation cleanup as appropriate.
Resolve existing test/lint failures in separate scoped tasks, and run CI's
platform gates before release.

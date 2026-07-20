# RustSFM PoseLib Default Solver Design

**Date:** 2026-07-20

**Goal:** Make RustSFM's existing GR6P, GR8P, and GP3P solvers available in normal builds so
structureless registration can recover images that do not yet have enough 2D-3D observations.

## Evidence

The 960-image `flowers2` reconstruction registered 900 images in its largest model. Across two
mapping attempts, 524 structureless registration candidates reached the solver boundary, but
every attempt emitted:

```text
COLMAP structure-less registration skipped: GR6P/GR8P solver is not ported yet
```

That message does not describe the current source tree. RustSFM already contains:

- a PoseLib v2.0.5 C++ bridge for GR6P and GP3P;
- a COLMAP-derived GR8P local refit bridge;
- fixed-seed generalized relative-pose RANSAC and support scoring;
- structureless observation preparation and mapper integration;
- synthetic solver and mapper registration tests.

The focused structureless solver test and mapper integration test both pass when RustSFM is built
with `--features poselib`. The failed full run used a default build, and `poselib` is absent from
the default feature list.

Dependency delivery is also incomplete. `third_party/PoseLib` is recorded as gitlink commit
`7e9f5f53372e43f89655040d4dfc4a00e5ace11c` (PoseLib v2.0.5), but the root repository has no
`.gitmodules` entry. A fresh worktree therefore contains an empty directory, while the primary
workspace succeeds only because it already has a manually initialized PoseLib checkout.

## Scope

This change includes:

1. restoring the PoseLib v2.0.5 submodule contract;
2. enabling the `poselib` feature in RustSFM's default feature set;
3. keeping an explicit no-PoseLib build through `--no-default-features`;
4. replacing the inaccurate "not ported" diagnostic with compile-time capability reporting;
5. reporting structureless solver attempts, estimates, acceptances, and elapsed time;
6. testing both default and no-default feature configurations;
7. rerunning `flowers2` and probing the resulting sparse model with RustGS.

This change does not replace PoseLib with a native Rust solver, move small generalized-pose
solves to wgpu, change registration thresholds, change BA policy, merge independently seeded
models, or change COLMAP/RustGS output formats.

## Alternatives

### A. Keep PoseLib optional and require explicit build commands

This minimizes default dependency changes, but it preserves the failure mode that produced the
900-image run. A successful release build would not imply a feature-complete mapper, and the
missing capability would remain easy to overlook in benchmarks and user workflows.

### B. Port GR6P and GR8P to native Rust

This removes the C++ build dependency, but the polynomial minimal solver and GR8P local refit are
numerically sensitive. A second implementation would require extensive parity fixtures and would
not improve the dominant mapper runtime. It is not justified while the existing bridge is already
tested and compatible with the repository's Ceres, Eigen, and VLFeat native dependencies.

### C. Make the existing PoseLib bridge a complete default capability

This is the selected approach. It fixes dependency delivery, enables the already integrated
solver in normal builds, preserves a deliberate dependency-minimal configuration, and adds enough
telemetry to determine whether structureless registration helps real data.

## Design

### Dependency Delivery

The root `.gitmodules` will map `third_party/PoseLib` to
`https://github.com/PoseLib/PoseLib.git`. The existing gitlink remains pinned to the v2.0.5 commit.
CI checkout will initialize submodules before any RustSFM build.

`scripts/setup_rustsfm_deps.sh` remains an idempotent convenience for existing clones and manual
environments. It must accept an already initialized submodule and retain the v2.0.5 default.

RustSFM's default feature list will include `poselib`. Users that need a dependency-minimal build
can continue using `cargo build -p rustsfm --no-default-features`; that configuration compiles the
existing explicit missing-solver branch.

### Capability Reporting

The generalized-pose module will expose a small compile-time capability query rather than
duplicating `cfg!(feature = "poselib")` throughout the CLI. Reconstruct and mapper entry points
will emit one stable information line before mapping:

```text
rustsfm_capabilities poselib=true structureless_gr6p_gr8p=true
```

The no-PoseLib path will report both values as false. If mapper execution later reaches
structureless registration without the feature, the diagnostic will say that the solver was
disabled at build time and name `--features poselib`; it will no longer say the solver is
unported.

Capability reporting is informational rather than a hard failure because
`--no-default-features` is an intentionally supported configuration and the mapper can still
register images through central PnP or the explicitly enabled pair-pose fallback.

### Structureless Telemetry

The existing session-scoped incremental registration telemetry will add:

- `structureless_estimates`: solver calls that returned a pose hypothesis;
- `structureless_accepted`: estimates that passed inlier and ratio gates and became registration
  choices;
- `structureless_solver_ms`: time spent in generalized relative-pose RANSAC and GR8P refit.

`structureless_attempts` continues counting candidates that reached structureless pose
estimation. This separation distinguishes missing capability, solver failure, acceptance-gate
failure, and successful registration without per-candidate log spam.

Counters are accumulated per mapping attempt and appended to the existing stable
`incremental_registration` diagnostic. They do not modify candidate order, retry state, random
seeds, or pose decisions.

## Correctness Invariants

- Default builds contain the PoseLib bridge and advertise the capability as enabled.
- `--no-default-features` builds without PoseLib and advertise the capability as disabled.
- Fixed non-negative RANSAC seeds remain deterministic.
- Structureless inlier masks still map one-to-one to collected correspondence-graph observations.
- Existing PnP, generalized-frame, and experimental pair-pose fallbacks retain their ordering.
- PoseLib setup does not fetch or update dependencies during `build.rs` execution.
- COLMAP sparse output and RustGS input contracts remain byte-structure compatible.

## Testing

Test-driven changes will cover:

- default feature metadata includes `poselib`;
- the capability line reports the compile-time configuration;
- the no-PoseLib error names the disabled feature rather than claiming missing implementation;
- telemetry formatting includes attempts, estimates, acceptances, and solver time;
- a synthetic structureless scene with outliers produces a valid pose and rejects outliers;
- mapper integration consumes solver inliers and produces a structureless registration choice;
- CI builds default, explicit PoseLib, and no-default configurations from an initialized
  submodule checkout.

Verification will run formatting, targeted red/green tests, the RustSFM library suite with default
features, the no-default build, and a release build. A full `flowers2` run will then verify:

1. capability reporting says GR6P/GR8P is enabled;
2. no "solver is not ported" messages appear;
3. at least one real structureless estimate is produced;
4. the largest model does not regress below 900 registered images and is compared against the
   previous 900-image baseline;
5. exported sparse data can initialize RustGS and complete a bounded training/export probe.

Reaching 960/960 remains the desired outcome, but it is not guaranteed by enabling one solver.
Any remaining images will be classified using matching connectivity, collected correspondence
counts, solver support, acceptance thresholds, and model membership before further algorithm
changes are proposed.

## Rollback

If making PoseLib a default feature causes an unsupported toolchain regression, remove it from the
default feature list while retaining the restored submodule, capability line, diagnostics,
telemetry, and explicit `--features poselib` CI coverage. No sparse output migration is required.

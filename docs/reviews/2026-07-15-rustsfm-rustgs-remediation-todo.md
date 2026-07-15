# RustSFM / RustGS Review Remediation Tracker

Date: 2026-07-15

This document tracks the remediation work produced by the repository-wide RustSFM and
RustGS review. A checkbox is complete only after the implementation, a regression test,
and the listed verification gate all pass.

## Batch A: RustSFM Memory And Database Safety

- [ ] A-01 Replace paired VLFeat `realloc` operations with failure-atomic allocation.
- [ ] A-02 Add allocator-failure coverage for both VLFeat paired-buffer growth paths.
- [ ] A-03 Validate feature-matching inputs before deleting existing database results.
- [ ] A-04 Wrap match and two-view-geometry replacement in a rollback-capable transaction.
- [ ] A-05 Split read-only database opening from schema creation and migration.
- [ ] A-06 Reject negative or overflowing SQLite matrix dimensions before integer conversion.
- [ ] A-07 Validate COLMAP binary and text collection counts before allocation.
- [ ] A-08 Validate `pair_id` in its full-width integer representation before narrowing.
- [ ] A-09 Make FreeImage initialization thread-safe.
- [ ] A-10 Validate vocabulary-tree structure during deserialization.
- [ ] A-11 Propagate explicit reference-model and undistortion failures.
- [ ] A-12 Use target OS, rather than host OS, for native build flags.

## Batch B: COLMAP Cross-Repository Contract

- [ ] B-01 Introduce one authoritative COLMAP camera-model ID and parameter table.
- [ ] B-02 Cover every supported camera model in text and binary fixtures.
- [ ] B-03 Preserve each COLMAP image's `CAMERA_ID` in RustGS.
- [ ] B-04 Reject unsupported multi-intrinsics training explicitly or add per-frame intrinsics.
- [ ] B-05 Reject distorted source images unless an undistortion path is applied.
- [ ] B-06 Correct the RustSFM `DIVISION` extra-parameter indices.
- [ ] B-07 Preserve image names through the end of a COLMAP text image line.
- [ ] B-08 Reject absolute and parent-traversing image names in both repositories.
- [ ] B-09 Make `--no-copy-images` interoperable with an explicit RustGS image root.
- [ ] B-10 Remove relative sparse-directory `unwrap` panics.
- [ ] B-11 Reject camera dimensions that do not fit the internal representation.
- [ ] B-12 Add RustSFM-to-RustGS end-to-end contract fixtures.

## Batch C: RustGS Training Correctness

- [ ] C-01 Apply `max_initial_gaussians` during sparse-point initialization.
- [ ] C-02 Enforce the Gaussian budget during every topology mutation.
- [ ] C-03 Make topology scheduling consume all public densify and prune window settings.
- [ ] C-04 Restore per-parameter Adam scaling after optimizer rebuilds.
- [ ] C-05 Detect non-finite loss before any backward or optimizer operation.
- [ ] C-06 Always report the actual final-step loss.
- [ ] C-07 Restrict training SH degree to the fully supported range.
- [ ] C-08 Remove the raster visibility storage-buffer data race.
- [ ] C-09 Implement or reject every exposed LiteGS feature switch.
- [ ] C-10 Make training failure a terminal event.

## Batch D: Scene IO, Tests, And Quality Gates

- [ ] D-01 Define and document lossy versus lossless splat artifact formats.
- [ ] D-02 Make the default training artifact preserve SH coefficients and metadata.
- [ ] D-03 Strengthen scene round-trip verification beyond Gaussian count.
- [ ] D-04 Require PLY `sh_degree` to match the available `f_rest` properties.
- [ ] D-05 Add RustGS unit tests for configuration, COLMAP IO, topology, optimizer state, and scene IO.
- [ ] D-06 Add a real GPU integration-test target with a deterministic tiny fixture.
- [ ] D-07 Add malformed COLMAP, SQLite, PLY, splat, vocabulary, and path fixtures.
- [ ] D-08 Restore RustGS Clippy with `-D warnings`.
- [ ] D-09 Align README input-format, test-command, and minimum-Rust-version claims with reality.

## Required Gates

- [ ] `cargo fmt -p rustsfm -p rustgs -- --check`
- [ ] `cargo test -p rustsfm --lib --no-fail-fast`
- [ ] `cargo test -p rustgs --all-targets --no-fail-fast`
- [ ] `cargo test -p rustgs --no-default-features --all-targets --no-fail-fast`
- [ ] `cargo clippy -p rustgs --all-targets -- -D warnings`
- [ ] RustGS executes non-zero unit and integration test counts.

## Completion Definition

The remediation is complete only when every checkbox above is checked, all required gates
pass, the public documentation matches the executable behavior, and no review finding is
left as a silent no-op or an undocumented input restriction.

# RustSFM / RustGS Review Remediation Tracker

Date: 2026-07-15

This document tracks the remediation work produced by the repository-wide RustSFM and
RustGS review. A checkbox is complete only after the implementation, a regression test,
and the listed verification gate all pass.

## Batch A: RustSFM Memory And Database Safety

- [x] A-01 Replace paired VLFeat `realloc` operations with failure-atomic allocation.
- [x] A-02 Add allocator-failure coverage for both VLFeat paired-buffer growth paths.
- [x] A-03 Validate feature-matching inputs before deleting existing database results.
- [x] A-04 Wrap match and two-view-geometry replacement in a rollback-capable transaction.
- [x] A-05 Split read-only database opening from schema creation and migration.
- [x] A-06 Reject negative or overflowing SQLite matrix dimensions before integer conversion.
- [x] A-07 Validate COLMAP binary and text collection counts before allocation.
- [x] A-08 Validate `pair_id` in its full-width integer representation before narrowing.
- [x] A-09 Make FreeImage initialization thread-safe.
- [x] A-10 Validate vocabulary-tree structure during deserialization.
- [x] A-11 Propagate explicit reference-model and undistortion failures.
- [x] A-12 Use target OS, rather than host OS, for native build flags.

## Batch B: COLMAP Cross-Repository Contract

- [x] B-01 Introduce one authoritative COLMAP camera-model ID and parameter table.
- [x] B-02 Cover every supported camera model in text and binary fixtures.
- [x] B-03 Preserve each COLMAP image's `CAMERA_ID` in RustGS.
- [x] B-04 Reject unsupported multi-intrinsics training explicitly or add per-frame intrinsics.
- [x] B-05 Reject distorted source images unless an undistortion path is applied.
- [x] B-06 Correct the RustSFM `DIVISION` extra-parameter indices.
- [x] B-07 Preserve image names through the end of a COLMAP text image line.
- [x] B-08 Reject absolute and parent-traversing image names in both repositories.
- [x] B-09 Make `--no-copy-images` interoperable with an explicit RustGS image root.
- [x] B-10 Remove relative sparse-directory `unwrap` panics.
- [x] B-11 Reject camera dimensions that do not fit the internal representation.
- [x] B-12 Add RustSFM-to-RustGS end-to-end contract fixtures.

## Batch C: RustGS Training Correctness

- [x] C-01 Apply `max_initial_gaussians` during sparse-point initialization.
- [x] C-02 Enforce the Gaussian budget during every topology mutation.
- [x] C-03 Make topology scheduling consume all public densify and prune window settings.
- [x] C-04 Restore per-parameter Adam scaling after optimizer rebuilds.
- [x] C-05 Detect non-finite loss before any backward or optimizer operation.
- [x] C-06 Always report the actual final-step loss.
- [x] C-07 Restrict training SH degree to the fully supported range.
- [x] C-08 Remove the raster visibility storage-buffer data race.
- [x] C-09 Implement or reject every exposed LiteGS feature switch.
- [x] C-10 Make training failure a terminal event.

## Batch D: Scene IO, Tests, And Quality Gates

- [x] D-01 Define and document lossy versus lossless splat artifact formats.
- [x] D-02 Make the default training artifact preserve SH coefficients and metadata.
- [x] D-03 Strengthen scene round-trip verification beyond Gaussian count.
- [x] D-04 Require PLY `sh_degree` to match the available `f_rest` properties.
- [x] D-05 Add RustGS unit tests for configuration, COLMAP IO, topology, optimizer state, and scene IO.
- [x] D-06 Add a real GPU integration-test target with a deterministic tiny fixture.
- [x] D-07 Add malformed COLMAP, SQLite, PLY, splat, vocabulary, and path fixtures.
- [x] D-08 Restore RustGS Clippy with `-D warnings`.
- [x] D-09 Align README input-format, test-command, and minimum-Rust-version claims with reality.

## Required Gates

- [x] `cargo fmt -p rustsfm -p rustgs -- --check`
- [x] `cargo test -p rustsfm --lib --no-fail-fast`
- [x] `cargo test -p rustgs --all-targets --no-fail-fast`
- [x] `cargo test -p rustgs --no-default-features --all-targets --no-fail-fast`
- [x] `cargo clippy -p rustgs --all-targets -- -D warnings`
- [x] RustGS executes non-zero unit and integration test counts.

Additional executed gates:

- [x] RustSFM writer to RustGS loader contract test with `rustsfm-contract-tests`.
- [x] Ignored-by-default tiny wgpu integration test executed on a working adapter.
- [x] `cargo check -p rust-viewer` after adding the terminal training-failure event.

## Completion Definition

The remediation is complete only when every checkbox above is checked, all required gates
pass, the public documentation matches the executable behavior, and no review finding is
left as a silent no-op or an undocumented input restriction.

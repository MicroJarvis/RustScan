# RustSFM GPU RANSAC 512-Chunk Experiment Design

## Goal

Evaluate changing the wgpu two-view RANSAC chunk size from 64 to 512. The experiment may be merged
only when it reduces GPU scorer/readback synchronization while preserving every selected pair's
matching and geometry output exactly under a fixed seed.

## Scope

The candidate changes only `GPU_RANSAC_CHUNK_TRIALS` used by Essential, Fundamental, and Homography
GPU RANSAC. It does not change CPU paths, PnP, descriptor matching, RANSAC thresholds, iteration
limits, seeds, samplers, candidate order, local refinement, pair scheduling, database transactions,
progress reporting, or image/keyframe/pair limits. RustSFM continues to use generic `wgpu` without
forcing Metal or Vulkan.

## Approaches Considered

1. **Fixed 512 experiment with a strict output fingerprint (selected).** Add deterministic benchmark
   output fingerprinting while the chunk remains 64, record the baseline, then change the constant
   to 512 and compare. This has the smallest runtime behavior change and proves dataset-level parity.
2. **Expose chunk size as a CLI/runtime option.** This makes sweeps convenient but expands a public
   configuration surface before a useful value is known.
3. **Implement adaptive 128/256/512 sizing immediately.** This may ultimately perform better, but it
   combines policy design with the first measurement and makes result attribution harder.

## Exact-Result Contract

For the selected flowers2 pairs, a canonical result fingerprint covers:

- canonical pair IDs and raw `matches` row dimensions/data;
- `two_view_geometries` existence, dimensions, config, and inlier-match data;
- exact stored F, E, H, relative rotation, and relative translation SQLite blob bytes, read through
  a read-only query without floating-point decoding or normalization.

Rows are ordered by canonical pair ID and encoded with explicit tags and length prefixes before a
BLAKE3 digest is calculated. Timing fields, database page layout, row IDs unrelated to selected
pairs, and SQLite file bytes are excluded. The fingerprint is reported per benchmark repetition and
is serde-defaulted for report compatibility.

The 512 candidate fails the experiment if any repetition fingerprint differs from the 64 baseline,
even when matched-pair, verified-pair, total-match, or inlier counts are unchanged. A failure is
reported and the branch is not merged; no attempt is made to reinterpret a different model as
equivalent.

## Implementation Sequence

1. With the chunk still at 64, add a failing unit test for deterministic selected-pair result
   fingerprinting, then implement the smallest canonical encoder and benchmark report field.
2. Build release and run flowers2 `--window 5 --pair-limit 96 --repetitions 3 --use-gpu
   --random-seed 0`. Require one identical fingerprint across all three repetitions and preserve the
   known `96 matched / 96 verified / 62,409 matches` result.
3. Change the chunk-boundary test to expect 512 and confirm it fails against 64. Change only
   `GPU_RANSAC_CHUNK_TRIALS` to 512 and make the test pass.
4. Run geometry parity/regression tests and the identical flowers2 96x3 command. Compare every
   candidate fingerprint to the recorded 64 baseline, then compare scorer calls, readback calls,
   readback wait, geometry time, and total time. No speed threshold is imposed before measurement.
5. Run the 2,890-pair benchmark only if strict bounded parity passes. Merge only after review and
   complete affected regression checks; otherwise document the failed gate and clean up the
   unmerged experiment worktree/branch.

## Error Handling And Safety

Fingerprint generation fails rather than omitting a selected pair whose matches or geometry cannot
be read. GPU adapter or AGX/XPC failure invalidates that benchmark attempt; it is not treated as a
performance result. Only one flowers2 GPU benchmark runs at a time. Benchmark databases remain
isolated SQLite online-backup copies, and `/tmp` JSON outputs are not committed.

## Verification

- RED/GREEN unit test for canonical result fingerprint changes when matches, inlier data, config, or
  any geometry/pose blob changes and remains stable when insertion order changes.
- RED/GREEN `gpu_ransac_chunk_end` boundary test for 512.
- Existing profiled/non-profiled `PairGeometry` parity test and all `two_view::tests`.
- Match timing, controlled matching, pause/cancel/rollback, benchmark snapshot, and serde tests.
- `cargo check -p rustsfm --no-default-features --features gpu-wgpu`.
- `cargo fmt --all -- --check` and `git diff --check`.
- flowers2 64 and 512 bounded 96x3 fingerprints and result counts; full 2,890 only after the strict
  gate passes.

# GPU five-point f32 large-batch replay — 2026-09-14

## Scope and provenance

- Worktree: `/Users/tfjiang/Projects/RustScan/.worktrees/gpu-five-point-f32`, starting HEAD `4cedae2`; initially no uncommitted changes.
- Only the replay example and its tests changed; no shader or production algorithm changes, no commit. Historical output retained.
- Actual database: `/Users/tfjiang/Projects/RustScan/output/flowers2_960_settlement_20260913/matching.db`.
- Same deterministic 12 pairs × 512 trials = 6144 inputs; fingerprint `3049eeb4981ff41bed23a9cab5adc59ce6c1e642d9b4360144d8aef2d221510c`, matching the historical final replay.
- Apple M5 Max, Wgpu. Release build, `--no-default-features --features gpu-wgpu`; existing Rust CPU five-point solver (not PoseLib).
- Each batch: one warmup for each path, then three measured rounds with rotating execution order. Serial and dedicated Rayon 4-thread pool compared with GPU.
- Independent fixed-prefix candidate generation, NOT production RANSAC or full 960-image pipeline throughput.

## Actual timing

Median host wall-clock milliseconds for all 6144 trials, excluding warmup:

| Batch | GPU solve calls/pass | CPU serial ms | CPU 4threads ms | GPU ms | GPU trials/s |
|---:|---:|---:|---:|---:|---:|
| 64 | 96 | 75.434 | 22.432 | 1109.356 | 5,538 |
| 128 | 48 | 65.979 | 20.808 | 560.590 | 10,960 |
| 256 | 24 | 65.324 | 20.580 | 283.763 | 21,652 |
| 512 | 12 | 68.782 | 21.571 | 158.575 | 38,745 |
| 1024 | 6 | 64.016 | 19.545 | 88.475 | 69,443 |
| 2048 | 3 | 67.984 | 21.139 | 57.884 | 106,143 |
| 6144 | 1 | 64.807 | 19.462 | 39.368 | 156,067 |

At batch 6144 GPU is 1.65× faster than serial but takes 2.02× the 4-thread CPU time. GPU throughput improves 28.18× over batch 64 and 4.03× over batch 512 in this sweep. Three rounds are a small local sample, not a statistical performance guarantee.

### Profiling limitations

Host total timing and solve-call counts only; **no kernel timestamps or per-stage profile**. Existing GPU passes have `timestamp_writes: None`. GPU totals include conversion, allocation, upload, kernels, waits, readback and decode. DB loading, sampling, device initialization, warmup, quality comparisons and JSON output are excluded. CPU totals include solver and ordered collection.

GPU API calls per pass are counted from actual chunking: 96/48/24/12/6/3/1. Including warmup plus three rounds: 384/192/96/48/24/12/4; 760 calls across the sweep. These are host `solve_essential` invocations, not kernel dispatch counts. No claim about which individual stage dominates.

## Quality and reproducibility

- All measured CPU results match serial warmup coefficient bits; every GPU round matches its warmup according to the existing `gpu_bits` check, and all larger batches match batch 64.
- All seven aggregate quality summaries are identical.
- CPU models: 26,490; GPU models: 20,219.
- Model-count mismatch: 2,136/6,144 trials; GPU empty: 974/6,144; CPU models without GPU counterpart: 3,982.
- Rank-deficient/singular elimination: 756 trials.
- CPU→GPU normalized sign-invariant nearest-model distance p50/p90/p99: 0.00007666 / 0.26557 / 1.39182.
- GPU→CPU distance p50/p90/p99: 0.00005240 / 0.002935 / 0.22199.
- GPU constraint residual p99: 3.77979e-6; mask Hamming fraction mean: 0.01619, p99: 0.45415.
- Historical final JSON has the same input fingerprint, model counts, empty/mismatch counts and distance/residual/mask summaries. However `complex_filtered` changed from 29,644 to 1,819 and `unconverged_roots` from 2,174 to 29,999. Cause not established in this task; historical binary/source provenance was not verified. Do not claim full historical diagnostic equivalence.

Larger batches improve throughput but do not resolve the existing f32 quality deficit. This is not an equal-quality replacement or a production acceleration result.

## Reproduction and validation

```sh
cargo test -p rustsfm --release --no-default-features --features gpu-wgpu --example five_point_gpu_replay
cargo run -p rustsfm --release --no-default-features --features gpu-wgpu --example five_point_gpu_replay -- --database /Users/tfjiang/Projects/RustScan/output/flowers2_960_settlement_20260913/matching.db --pairs 12 --trials 512 --rounds 3 --output output/five_point_gpu_batch_sweep_20260914.json
```

Seven example tests passed, including new coverage for all seven batch sizes, per-pass call counts, partial/empty input counts, and global chunk index coverage. Existing tests cover deterministic sampling, ordered CPU replay and coefficient bits, sign-invariant matching, and masks.

The initial test attempt with `gpu-wgpu,poselib` failed because that optional feature could not locate PoseLib source. The successful build uses only `gpu-wgpu`, sufficient for this example. Existing unrelated dead-code warnings remain.

Artifacts (relative to worktree):
- `output/five_point_gpu_batch_sweep_20260914.json`: compact 21,638-byte report, all raw timing rounds/warmups plus aggregate quality; per-trial diagnostics omitted.
- `output/five_point_gpu_batch_sweep_20260914.log`: benchmark build/run log.
- `output/batch_sweep_tests_gpu_20260914.log`: passing tests.
- `output/batch_sweep_tests_20260914.log`: initial optional-feature failure.

Historical 20+ MB JSON files were parsed by Python for selected summaries only, never printed/read into the agent context in full. Existing history was not overwritten.

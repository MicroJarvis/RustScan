# P1: nullspace and algebra/LU workgroup layout (2026-09-14)

## Decision and scope

Retain both single-variable changes in the independent GPU f32 experiment. Nullspace was tested and accepted first; LU was then tested against the retained nullspace version. Neither change alters formulas, thresholds, arithmetic evaluation order, generated expressions, buffer formats, pass order, CPU fallback, or production routing. No commit/push was performed.

**This does not establish a GPU win over CPU8.** The final unchanged eight-round CPU8 harness reported GPU wins **0/8** for both batch sizes. Existing f32-versus-f64 quality limitations remain; bitwise preservation is within the GPU algorithm, not CPU/GPU numerical equivalence.

## Provenance and protected work

- Worktree: `/Users/tfjiang/Projects/RustScan/.worktrees/gpu-five-point-f32`.
- Actual adapter: wgpu, **Apple M5 Max**.
- Database: `/Users/tfjiang/Projects/RustScan/output/flowers2_960_settlement_20260913/matching.db`.
- Same loader and workset as `RustSFM/examples/five_point_gpu_cpu_threads.rs`: 127 pairs × 512 trials = **65,024** real inputs.
- Input digest: `af07459d637b5b8c2b20c76d3ef103f058505fe0e8307578ee7e8750f3e230c5`.
- Complete historical GPU signature: `1f6f3f8d88a17e4e1c42abfc85aad44a66c4b864091805594e086ca80bf9b339`.
- Final CPU8 reference signature: `9e6764b41602d0005710187d2558d046f1a843c5f873b9542dd7b4f6c29c563a`.
- New independent harness: `RustSFM/examples/five_point_gpu_p1_layout.rs`, reusing existing loader/replay/complete-signature helpers, not editing them.

Unique local artifact directory: `output/five-point-p1-20260914-uxspnwdh/`. Created with `mkdtemp`; experiment JSON files refuse overwrite. Before solver edits, preserved exact GPU source, five-point examples, manifests/lockfile, generator and existing CPU comparison document in `source/`, with `source-sha256.json`. Preserved the existing CPU-thread executable as `baseline-cpu-threads`, then built and preserved `baseline-p1` before changing nullspace. `null-p1`/`null-source/` capture the accepted first stage; `lu-p1`/`final-source/` capture the second. `harness.rs` is the final formatted harness; its formatting-only pass occurred after measurements. `artifact-sha256.json` hashes the saved artifacts, sources, binaries and logs present at snapshot time.

SHA-256 checks against the initial snapshot confirmed these pre-existing uncommitted files unchanged:

- `RustSFM/examples/five_point_gpu_capacity_replay.rs`
- `RustSFM/examples/five_point_gpu_cpu_threads.rs`
- `docs/gpu-five-point-cpu-threads-comparison-20260914.md`

## Shader and dispatch audit

The diagnostic dispatcher is in `five_point_f32.rs`; full and profiled solving share `solve_essential_inner` and `complete_pass_timed` in `five_point_f32_complete.rs`. The root-only API uses the same complete-pass helper, but does not invoke nullspace or LU. Shader entry audit and source dispatch inventory are saved locally.

| Pass | Final workgroup | Full dispatch (x, y) | Diagnostic dispatch |
|---|---|---|---|
| constraints | 1 | count, 1 | unchanged count, 1 |
| nullspace/basis | **32** | **ceil(count/32), 1** | **ceil(count/32), 1** |
| pack (inline WGSL) | 1 | count, 1 | not used |
| elimination | 1,32,1 | count, ceil(200/32) | unchanged same layout |
| algebra/LU | **32** | **ceil(count/32), 1** | **ceil(count/32), 1** |
| polynomial | 1 | count, 11 | unchanged same layout |
| validate_polynomial | 1 | count, 1 | unchanged same layout |
| roots | 32 | ceil(count/32), 1 | root-only ceil(count/32), 1 unchanged |
| recover | 1 | count, 1 | not used |

Only nullspace and algebra switch from `workgroup_id` to `global_invocation_id`: one invocation per trial, with an early bounds guard using input storage length divided by its fixed trial stride (45 or 36). Diagnostic dispatch selects these two existing kernel objects; other kernels retain exact previous group counts. No limit expansion or buffer sizing change.

## Correctness gates

Both stages independently passed the actual GPU test:

`cargo test -p rustsfm --lib --no-default-features --features gpu-wgpu five_point_f32_actual_gpu_stages -- --nocapture`

- Nullspace stage: **1 passed**, 29.25 s test runtime.
- LU stage: **1 passed**, 28.75 s test runtime.
- These are actual GPU executions, not merely shader parsing/building or ignored tests.

Each of the twelve benchmark processes checked:

1. Exact input digest and 65,024-trial count.
2. Full historical decoded trial/slot/diagnostic/model bit signature on the initial full replay, all three end-to-end warmups, all three measured end-to-end replays, all three timestamp warmups and all three measured timestamp replays. Result lengths checked first.
3. Prefix sizes **1, 31, 32, 33, 63, 65, 65,023**: full decoded results exactly equal the corresponding full-run prefix, including trial/slot ordering. These exercise partial workgroups and a near-full-capacity nonmultiple.
4. The same sizes through constraint construction, nullspace diagnostics and algebra diagnostics. Hash includes nullspace status/sweeps/off-diagonal/rank/basis bits and algebra status/elimination/solve/determinant/coefficient/pivot bits. Failed bases are represented by zero bases for this independent diagnostic exercise, not CPU fallback.

All seven full-prefix and diagnostic hashes matched the original baseline across **all twelve** processes. No signature failure or regression requiring rollback occurred.

## Timing protocol and results

Per stage, process order was **B0, C0, C1, B1, B2, C2** (three paired rounds, middle pair reversed). LU baseline is `null-p1`, not the original solver. Each process runs three warmups and three measured replays separately for unprofiled end-to-end and timestamp profiling. Table values are within-process medians of three measured samples, in milliseconds. No CPU benchmark or other GPU job was intentionally run concurrently. Build/test gaps occurred before first candidates and between first pair and later pairs.

End-to-end uses a timestamp-disabled context and includes f64→f32 rays, allocation/upload, all kernels, waits, readback, decoding and collection. Excludes DB load/sampling, initialization, signature/boundary checks and final result destruction. Profiling uses a separate timestamp-enabled context; captures all nine pass intervals and raw ticks/period. All reported measured intervals were available. Kernel sums are medians of per-replay sums, not sums of separate pass medians. Warmup signatures are checked but warmup timings are not saved by the P1 harness.

| Variable / pair | End-to-end B → C | Reduction | Changed kernel B → C | Nine-kernel sum B → C |
|---|---:|---:|---:|---:|
| nullspace / 0 | 157.835 → 126.593 | 19.79% | 32.674 → 1.890 | 106.743 → 72.793 |
| nullspace / 1 | 166.266 → 134.946 | 18.84% | 40.861 → 7.692 | 112.384 → 81.409 |
| nullspace / 2 | 165.297 → 134.266 | 18.77% | 39.941 → 7.620 | 111.088 → 81.130 |
| LU / 0 | 131.872 → 113.811 | 13.70% | 21.809 → 1.381 | 81.048 → 60.688 |
| LU / 1 | 130.237 → 102.489 | 21.31% | 22.196 → 1.373 | 73.104 → 55.272 |
| LU / 2 | 125.633 → 105.318 | 16.17% | 23.863 → 1.383 | 79.227 → 52.604 |

All measured candidate end-to-end samples were faster than all baseline samples within their respective pair. Raw end-to-end samples (ms):

| Variable / pair | Baseline | Candidate |
|---|---|---|
| nullspace / 0 | 157.835, 160.511, 154.933 | 122.926, 135.847, 126.593 |
| nullspace / 1 | 166.067, 170.838, 166.266 | 136.054, 133.124, 134.946 |
| nullspace / 2 | 164.651, 165.297, 165.367 | 136.052, 133.879, 134.266 |
| LU / 0 | 129.704, 132.621, 131.872 | 113.811, 114.528, 112.758 |
| LU / 1 | 133.753, 124.628, 130.237 | 101.242, 102.489, 102.501 |
| LU / 2 | 124.508, 134.777, 125.633 | 101.944, 113.958, 105.318 |

There is substantial run-to-run timestamp variation, especially basis (~1.9 versus ~7.6 ms with the same layout). No affinity, thermal or frequency control was applied; cause was not diagnosed. Three local process pairs do not establish a statistical guarantee. Do not combine cross-stage medians into a controlled total speedup, or equate kernel-only timing to end-to-end latency.

Full-precision samples, all nine kernel intervals and raw timestamp evidence remain in compact `null-{baseline,candidate}-{0,1,2}.json` and `lu-{baseline,candidate}-{0,1,2}.json` artifacts. No large historical JSON was read.

## Final CPU8 comparison: no GPU win

Rebuilt and executed the **unchanged** `five_point_gpu_cpu_threads` example with `--cpu-threads 8`. Its existing protocol runs three warmups plus eight measured rounds, alternating CPU→GPU / GPU→CPU, with 100 ms idle before each path and timestamps disabled. Dedicated Rayon pool remains alive. Same 65,024 inputs for each batch configuration; serial CPU calculation remains an offline signature/quality reference only.

| Batch | CPU8 median ms | GPU median ms | GPU wins | Existing stable-win gate |
|---|---:|---:|---:|---|
| 32,768 (2 calls) | 111.052 | 127.128 | 0/8 | false |
| 65,024 (1 call) | 111.735 | 127.013 | 0/8 | false |

Raw measured samples (ms):

| Batch | CPU8 | GPU |
|---|---|---|
| 32,768 | 113.058, 114.001, 107.296, 110.860, 111.402, 111.110, 110.995, 110.839 | 115.761, 119.338, 125.641, 126.661, 131.752, 127.594, 130.313, 129.387 |
| 65,024 | 110.922, 112.939, 109.767, 110.791, 113.345, 113.128, 112.549, 109.474 | 130.615, 128.711, 127.871, 125.366, 126.154, 133.841, 122.412, 122.931 |

Artifact: `final-cpu8.json`, including all warmup/measured records and quality counts. CPU and GPU historical signatures passed throughout. The higher GPU timings than GPU-only P1 runs are a distinct measured protocol, not a solver signature regression; no cause is asserted. The existing stable-win rule requires GPU faster in every pair and median at least 5% lower. No CPU default or production policy was changed.

## Validation and reproduction

All terminal calls used `head_lines=15`, `tail_lines=20`, and a timeout no greater than 180 seconds. No timeout occurred. Build/test logs were retained in the unique local directory; existing unrelated dead-code warnings remain.

Passed:

- Release builds of the P1 baseline, nullspace candidate, LU candidate and final CPU-thread example.
- Actual GPU stage test at each candidate stage (above).
- Example tests: CPU-thread example **10 passed**, P1 example/shared helpers **8 passed**.
- All twelve actual GPU P1 processes and the final CPU8 example.
- `cargo fmt --all -- --check`.
- `git diff --check`.
- `python3 scripts/generate_five_point_wgsl.py --check` (200 + 11 expressions).
- Exact protected-file SHA-256 verification.

Build commands:

```sh
cargo build --release -p rustsfm --no-default-features --features gpu-wgpu --example five_point_gpu_p1_layout
cargo build --release -p rustsfm --no-default-features --features gpu-wgpu --example five_point_gpu_cpu_threads
cargo test -p rustsfm --no-default-features --features gpu-wgpu --example five_point_gpu_p1_layout --example five_point_gpu_cpu_threads
```

The preserved `baseline-p1`, `null-p1`, `lu-p1` executables each accept `--database PATH --output NEW_PATH`. Run them sequentially in B,C / C,B / B,C order with distinct output paths; compare `input_digest`, `signature` and `boundaries` across compact reports before accepting timings. Do not overwrite existing artifacts or restore whole source directories over unrelated work. No production full RANSAC, reconstruction, quality-equivalent throughput claim, or further optimization was undertaken.

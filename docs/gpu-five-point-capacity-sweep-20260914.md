# GPU five-point f32: device-capacity / real-pair sweep — 2026-09-14

## Scope and preserved work

Worktree: `/Users/tfjiang/Projects/RustScan/.worktrees/gpu-five-point-f32`, HEAD `4cedae22f8cfc0d7d3f36931db883a7d4b7a4252`.

At entry, `RustSFM/examples/five_point_gpu_replay.rs` had the uncommitted 37-line batch-sweep diff (+33/-4), and `docs/gpu-five-point-batch-sweep-20260914.md` was untracked. Both were left untouched. This experiment adds the independent `RustSFM/examples/five_point_gpu_capacity_replay.rs`, derived from that replay, and changes only the shared CPU example loader's CLI pair bound and related tests. No shader, production solver, RANSAC, commit, push, or branch change. Historical artifacts were not overwritten; no historical large JSON was read.

Database: `/Users/tfjiang/Projects/RustScan/output/flowers2_960_settlement_20260913/matching.db`, opened through the existing `ColmapDatabase::open_read_only` loader. Raw matches only, not verified inliers. No synthetic or replicated input expansion.

## Actual adapter and complete-solve capacity

Runtime adapter query: **Apple M5 Max, Metal, IntegratedGpu**, 64 GiB system memory (`hw.memsize=68719476736`). The independent query uses the same default instance / HighPerformance / no fallback selection as `WgpuContext`; the solver adapter name and backend are checked. `WgpuContext` requests `required_limits: adapter.limits()` (`RustSFM/src/gpu/context.rs:64–73`). Its private device limits are not separately exposed or queried by this example; the report records the actual adapter limits and this device-request contract rather than inventing device measurements.

Relevant actual limits:

| Limit | Value |
|---|---:|
| `max_compute_workgroups_per_dimension` | 65,535 |
| `max_storage_buffer_binding_size` | 4,294,967,292 B |
| `max_buffer_size` | 41,747,087,360 B |
| `max_compute_invocations_per_workgroup` | 1,024 |
| `max_compute_workgroup_size_x/y/z` | 1,024 each |
| `max_compute_workgroup_storage_size` | 32,768 B |
| `max_storage_buffers_per_shader_stage` | 31 |
| `max_bind_groups` | 8 |
| `max_bindings_per_bind_group` | 65,535 |
| `min_storage_buffer_offset_alignment` | 256 B |

All adapter limits, including unrelated texture/mesh limits, are retained compactly in the JSON. Kernels use one bind group, at most five storage buffers, whole-buffer bindings at offset zero, `@workgroup_size(1)`, and no `var<workgroup>` arrays. Function-local arrays are not workgroup storage. Shader creation and all measured dispatches succeeded.

### All stages in `solve_essential`

Source: `RustSFM/src/gpu/five_point_f32_complete.rs:120–162,210–281`; generated shader confirms the two-dimensional mapping at `shaders/five_point_generated.wgsl:16–24`.

| Stage | Dispatch (X,Y,Z) | Bound storage buffers |
|---|---|---|
| constraints | (N,1,1) | left rays, right rays, constraints |
| basis/nullspace | (N,1,1) | constraints, diagnostics |
| pack | (N,1,1) | diagnostics, basis |
| elimination | **(N,200,1)** | basis, algebra |
| algebra | (N,1,1) | basis, algebra |
| polynomial | **(N,11,1)** | basis, algebra |
| validate polynomial | (N,1,1) | basis, algebra |
| roots | (N,1,1) | constraints, diagnostics, algebra, basis, output |
| recover | (N,1,1) | constraints, diagnostics, algebra, basis, output |

Dispatch limits apply **per dimension**, not to X×Y: elimination requires N≤65,535 and 200≤65,535, not N≤floor(65,535/200). At the measured full batch elimination dispatches 65,024×200=13,004,800 workgroups; polynomial dispatches 715,264. There are nine kernel dispatches per host solve call and 218N workgroups in total. The largest shader record index (352N words) remains far below u32 overflow at this device's dispatch bound.

### All buffers

Each trial has five rays per side, each padded to 16 B. Buffer ceilings below include binding and device-buffer limits (readback has no storage binding).

| Buffer | Bytes/trial | Trial ceiling from buffer limits |
|---|---:|---:|
| left rays | 80 | 53,687,091 |
| right rays | 80 | 53,687,091 |
| constraints (45 f32) | 180 | 23,860,929 |
| diagnostics (40 f32) | 160 | 26,843,545 |
| packed basis (36 f32) | 144 | 29,826,161 |
| algebra (352 f32) | 1,408 | **3,050,402** |
| output (176 f32) | 704 | 6,100,805 |
| readback staging (176 f32) | 704 | 59,299,840 |

Thus the current complete-solve legal trial ceiling on this adapter is:

`min(65535, floor(4294967292/1408), floor(41747087360/1408)) = 65535`, with the independent fixed-Y checks satisfied.

65,536 is illegal even though memory is available. With unchanged 512 trials per pair, the largest whole-pair single-call workset is **127×512=65,024**, leaving 511 trial slots unused. We did not execute the exact 65,535 boundary because that would no longer be a whole-pair 512-trial workset; its legality is calculated from source and actual limits, not claimed as a measured run.

Buffers plus readback total **3,460 B/trial**. The experiment additionally caps this estimate at 512 MiB. Full-batch buffers are 224,983,040 B = **214.561 MiB**. This is a buffer-footprint estimate, **not measured peak RSS**: host inputs, output structs, coefficient-bit vectors, upload staging, driver allocations and allocator retention are additional. The actual run completed on the 64 GiB machine; no claim of a measured GPU peak-memory profile.

## Exact input provenance

The original 12 pairs were reloaded through the **same Rust loader**, not a separately approximated SQL selection. Loader behavior remains: raw-match count≥5; sort by pair ID; midpoint of each of P equal rank strata; same projection-validity filtering; a stateful sampler with seed 1 per pair; 512 trials per pair. `--pairs` now accepts values above 12, while defaults remain 12 pairs / 512 trials and the trials range stays 1–512. Selection still errors if insufficient eligible pairs or fewer than five valid observations; it does not replicate or silently replace pairs.

Original 12 pairs involve **23 unique image IDs**:

`37, 47, 110, 150, 183, 257, 258, 330, 343, 370, 408, 486, 488, 564, 566, 643, 644, 726, 735, 810, 890, 902, 910`.

Original ordered input BLAKE3: `3049eeb4981ff41bed23a9cab5adc59ce6c1e642d9b4360144d8aef2d221510c`, matching the previous report.

Expanded set: **127 distinct pair IDs**, **231 unique image IDs**, from **14,045 eligible pairs**, **65,024 trials**. Changing the number of strata changes the selected pair set; the expanded set is not asserted to contain the original 12 pairs. Every batch processes the same ordered expanded workset once per pass; there is no input cloning to enlarge it (ordinary GPU chunk conversion / sampled quality copies are not workload expansion).

Expanded ordered input BLAKE3: `af07459d637b5b8c2b20c76d3ef103f058505fe0e8307578ee7e8750f3e230c5`.

The JSON retains all selected pair IDs, image IDs, ranks, valid/raw counts and per-pair digests. DB loading took 179.578 ms, GPU initialization 112.743 ms, excluded from timing.

## Measured performance

Release, `--no-default-features --features gpu-wgpu`, existing Rust CPU five-point solver (not PoseLib). Dedicated Rayon pool with four threads. Each batch has one warmup per path and three measured rounds, execution order rotated serial→4threads→GPU / GPU→serial→4threads / 4threads→GPU→serial.

**Every timing is for the same complete 65,024-trial workset**, including the final partial chunk where needed. GPU includes conversion, allocation, upload, all kernels, synchronization, readback and decoding. CPU includes solver and ordered collection. Quality/signature work, input loading, initialization, warmup and post-timing destruction are excluded. No per-stage/kernel timestamps were collected.

Median milliseconds of three rounds:

| Batch | Solve calls/pass | CPU serial ms | CPU 4threads ms | GPU total ms | GPU trials/s | Buffer MiB at batch size |
|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 8 | 677.118 | 200.836 | 404.273 | 160,842 | 27.031 |
| 16,384 | 4 | 668.612 | 198.438 | 354.688 | 183,328 | 54.063 |
| 32,768 | 2 | 667.772 | 198.958 | 332.351 | 195,648 | 108.125 |
| 49,152 | 2 | 666.079 | 197.462 | 330.148 | 196,954 | 162.188 |
| **65,024** | **1** | **667.801** | **198.175** | **323.473** | **201,019** | **214.561** |

Throughput = 65,024 / median seconds, not an average of per-round throughputs.

Three-round accumulated milliseconds (warmup excluded):

| Batch | Serial total ms | 4threads total ms | GPU total ms | GPU warmup ms |
|---:|---:|---:|---:|---:|
| 8,192 | 2,032.208 | 639.617 | 1,208.905 | 414.388 |
| 16,384 | 2,009.895 | 599.473 | 1,066.602 | 366.732 |
| 32,768 | 2,006.848 | 598.558 | 991.903 | 347.451 |
| 49,152 | 2,000.113 | 592.377 | 991.935 | 351.133 |
| 65,024 | 2,002.855 | 596.424 | 960.017 | 336.412 |

Raw measured rounds and all warmups are retained in seconds in JSON. GPU solve calls including warmup are 32 / 16 / 8 / 8 / 4, **68 total**, or **612 kernel dispatches**. GPU timed duration including warmups across the sweep: **7,035.477 ms**. Host solve calls are not dispatch counts; readback copy submissions are additional.

The full single batch is **2.064× faster than CPU serial**, but takes **1.632× CPU 4threads time**. Its throughput is 1.250× batch 8,192; gains above 32,768 are small in this three-round sample. Not a statistical performance guarantee, production acceleration claim, or equal-quality comparison.

## Signatures and quality

Every CPU 4-thread / measured result matched serial warmup coefficient bits in order. Every measured GPU result matched its warmup, and all larger batches matched the 8,192 baseline exactly.

GPU BLAKE3 signature across all five batches:

`1f6f3f8d88a17e4e1c42abfc85aad44a66c4b864091805594e086ca80bf9b339`.

Unlike the older replay's narrower `gpu_bits`, this independent example includes **all fields of the decoded trial/slot structs**, including algebra status, filtering/rejection counters, optional-model presence, and diagnostic f32 bits. It does not hash unused raw GPU-buffer padding. Signature computation occurs outside timing. First batch's cross-batch flag is null (reference); all remaining flags and all repeat checks are true.

Full-workset counts (not sampled), identical across batches:

- CPU models: **288,264**; GPU models: **209,600**.
- GPU empty: **12,012 / 65,024 trials**.
- Model-count mismatch: **26,294 / 65,024 trials**.

Costlier quality checks use **1,016 / 65,024 trials (1.5625%)**, eight per selected pair at local indices **0,64,128,192,256,320,384,448**. This is a deterministic stratified diagnostic sample, not a statistical confidence estimate. Mask comparisons still use **all valid raw observations for each sampled trial's pair**, normalized z=1 coordinates, squared Sampson threshold 1e-6, CPU f64 evaluation of both model sets. Bidirectional distances use nearest models after Frobenius normalization and sign invariance, not slot alignment.

Sample results, identical across batches:

| Metric | Value |
|---|---|
| CPU / GPU models | 4,514 / 3,325 |
| Count mismatch / GPU empty trials | 404 / 171 |
| Rank deficient / singular elimination | 144 / 144 |
| CPU models without GPU counterpart | 724 |
| CPU→GPU distance p50 / p90 / p99 | 0.00016415 / 0.486970 / 1.387821 |
| GPU→CPU distance p50 / p90 / p99 | 0.00009521 / 0.00413853 / 0.119260 |
| GPU constraint residual p99 | 4.15366e-6 |
| Mask Hamming fraction mean / p99 | 0.0212933 / 0.540897 |

Large batches preserve results but **do not repair the existing f32 quality deficit**. No comparison of historical diagnostic counters beyond the reloaded input digest is claimed.

## Reproduction, validation and artifacts

Run each command with a maximum runtime of **180 seconds**:

```sh
cargo test -p rustsfm --release --no-default-features --features gpu-wgpu --example five_point_gpu_capacity_replay --example five_point_gpu_replay --example five_point_replay_probe
cargo run -p rustsfm --release --no-default-features --features gpu-wgpu --example five_point_gpu_capacity_replay -- --database /Users/tfjiang/Projects/RustScan/output/flowers2_960_settlement_20260913/matching.db --inspect-only --output output/five_point_gpu_capacity_inspect_recheck.json
cargo run -p rustsfm --release --no-default-features --features gpu-wgpu --example five_point_gpu_capacity_replay -- --database /Users/tfjiang/Projects/RustScan/output/flowers2_960_settlement_20260913/matching.db --pairs 127 --trials 512 --rounds 3 --output output/five_point_gpu_capacity_127x512_recheck.json
```

All actual terminal calls used `head_lines=15`, `tail_lines=20`, timeout≤180s. **No command timed out.** An initial build exposed that wgpu 29's storage-binding limit is u64, not u32; the example calculation was corrected, then inspection, tests and the complete sweep succeeded. Existing unrelated dead-code warnings remain.

Tests: **8 + 7 + 4 passed** across the new GPU example, preserved GPU example, and CPU loader example. Added checks cover >12 pair CLI acceptance with preserved defaults/trial bounds, distinct expanded strata, independent elimination Y bound, dispatch/binding/buffer ceilings, 3,460-byte footprint, unique-image deduplication, complete chunk coverage and partial final chunks. Existing sampler, ordered CPU results, sign-invariance and masks tests also pass. `git diff --check` passed. No production-wide suite was needed or run for this example-only change.

Artifacts, relative to this worktree:

- `output/five_point_gpu_capacity_127x512_20260914.json`: **65,339 bytes**, full compact provenance/timings/signatures/aggregate and sampled quality; no per-trial diagnostic JSON. Serialization refuses summaries ≥100,000 bytes.
- `output/five_point_gpu_capacity_127x512_20260914.log`: successful release build/run.
- `output/five_point_gpu_capacity_inspect_20260914.json`: first successful actual adapter/original-pair inspection (before aggregate original digest was added; final run JSON includes that digest).
- `output/five_point_gpu_capacity_inspect_20260914.log`: retained initial u64 type-error attempt.
- `output/five_point_gpu_capacity_inspect_run_20260914.log`: successful inspection.
- `output/five_point_gpu_capacity_tests_20260914.log`: passing focused tests.

Output artifacts are locally present under the existing ignored output directory; source and this report remain uncommitted.

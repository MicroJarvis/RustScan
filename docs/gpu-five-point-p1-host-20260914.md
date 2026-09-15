# P1: host allocation and buffer initialization (2026-09-14)

## Decision and scope

Retain both single-variable changes, P1.1 and P1.2, in the independent GPU f32
experiment. Each was measured separately against its own immediately preceding
build. Neither changes formulas, thresholds, arithmetic evaluation order, generated
expressions, buffer formats, pass order, dispatch geometry, shader source, the
public output interface, or production routing. No CPU runtime fallback was added.
No commit or push was performed.

**The complete GPU output signature is unchanged**:
`1f6f3f8d88a17e4e1c42abfc85aad44a66c4b864091805594e086ca80bf9b339`, bit-for-bit,
across all thirteen benchmark processes and every boundary check.

For the first time in this experiment the existing CPU8 comparison harness reports
`gpu_stable_win = true` at both large batches, **8/8 rounds** each. This is an
offline candidate-generation throughput result on output that is still **not**
numerically equivalent to CPU f64: `equal_quality` remains `false`, and the quality
counts are byte-identical to the previous round. It is not a production speedup, not
full RANSAC, and not a 960-image pipeline result.

## Provenance and protected work

- Worktree: `/Users/tfjiang/Projects/RustScan/.worktrees/gpu-five-point-f32`.
- Baseline commit `3ab9c0987be7ea5d2912a862d2c9038d17d5d1a7`, branch
  `gpu-five-point-f32`. The previous round's retained nullspace/LU layout work was
  uncommitted; this round's diff sits on top of it and did not revert it. The exact
  pre-change `git diff` is preserved as `baseline-uncommitted.diff`.
- Hardware: `Mac17,7`, **Apple M5 Max**, 18 logical CPUs, 64 GiB, macOS 26.5.2.
  Adapter reported by the solver: `Wgpu / "Apple M5 Max"`. wgpu 29.0.1.
- Database:
  `/Users/tfjiang/Projects/RustScan/output/flowers2_960_settlement_20260913/matching.db`.
- Fixed input: 127 pairs × 512 trials = **65,024** real inputs, seed 1 per pair,
  **231 unique image IDs** (recorded in every report). Input digest
  `af07459d637b5b8c2b20c76d3ef103f058505fe0e8307578ee7e8750f3e230c5`.
- Unique local artifact directory: `output/five-point-p1host-20260914-ThcNbcf6/`,
  created with `mktemp -d`. Experiment JSON files refuse to overwrite. Preserved:
  full pre-change source snapshot with `source-sha256.txt`, the `baseline-p1host`,
  `p11-p1host`, `p12-p1host` and `final-p1host` executables, the rebuilt but
  source-unchanged `final-cpu-threads`, all thirteen experiment JSONs, the three
  analysis transcripts, the read-only `analyze.py`, the field inventory, and
  `artifact-sha256.txt`.
- SHA-256 verification confirmed these pre-existing files **unchanged** by this
  round: `five_point_gpu_capacity_replay.rs`, `five_point_gpu_cpu_threads.rs`,
  `five_point_gpu_p1_layout.rs`, `five_point_replay_probe.rs`, `gpu/context.rs`,
  `gpu/five_point_f32.rs`, all four five-point WGSL shaders, and both existing
  five-point reports.
- New file: `RustSFM/examples/five_point_gpu_p1_host.rs`, reusing the existing
  loader, replay and signature helpers without editing them.
- Changed files: `gpu/five_point_f32_complete.rs` and `gpu/five_point_f32_tests.rs`
  only. `cargo fmt` left the solver file byte-identical to the measured source
  (`e1db8083…`), so the measured binaries correspond exactly to the final tree. The
  formatting pass touched only the test file and the new harness.

## What changed

### P1.1 — no per-trial temporary slot Vec

`decode` previously built a `Vec<FivePointModelSlot>` per trial through
`collect::<Result<Vec<_>>>()` and then moved it into the fixed `[_; 10]` array. The
`Result` collect shim loses the exact-size hint, so the Vec grew in stages: three
allocations per trial were observed.

Now the ten slot statuses are validated first into a stack `[FivePointSlotStatus; 10]`
and the final array is built in place with `std::array::from_fn`. The only added work
is re-reading ten status floats; there is no extra copy and no new allocation.

Error precedence is deliberately preserved: `upstream_status` and `algebra_status`
are still decoded before any slot status, so a record that is invalid in both places
reports the same error as before. Slot statuses are still validated in ascending slot
order. The status match arms moved verbatim into `FivePointSlotStatus::decode`,
mirroring the existing `FivePointStatus::decode`.

### P1.2 — device-side buffer zero initialization

`complete_buffer` created each of the five per-call buffers with
`create_buffer_init` from a host `vec![0u8; size]`. That is exactly 161.0 MiB of host
zero data per full replay (measured, and equal to 649 floats/trial × 4 × 65,024), plus
a memcpy of the same size into mapped memory.

The zeros are still required. The full field inventory is preserved as
`P1.2-buffer-field-inventory.md`; in summary:

| Buffer | Stride | Zero-dependent fields |
|---|---:|---|
| `constraints` | 45 | **none** — all 45 written unconditionally every trial |
| `diagnostics` | 40 | basis 0..35 on three failure exits; 37/38 when `scale == 0`; 39 when `scale == 0` or not converged |
| `basis` | 36 | all 36 for every failed trial (`pack` returns early) |
| `algebra` | 352 | 200..349 on both `algebra` failure exits; 339..349 whenever `polynomial` is skipped |
| `out` | 176 | throughout: the `+=` counters 1/4/5/6/7/8, the `max()` seeds 10/14, header 2/3/9/15 on early return, and every field of untouched slots (status 0.0 = `Unused`) |

The generated `elimination` kernel was verified to cover all 200 indices with exactly
200 switch cases, so `algebra` never reads stale values in `0..199`.

So the change is not to remove zeroing but to stop preparing it on the host. The
buffers are now created with `create_buffer` and `mapped_at_creation: false`. wgpu
documents "Every `Buffer` starts with all bytes zeroed."
(`wgpu-29.0.1/src/api/buffer.rs:35`), adds the internal `COPY_DST` that the lazy
clear requires (`wgpu-core-29.0.1/src/device/resource.rs:1071-1075`), tracks the new
buffer as uninitialized, registers `MemoryInitKind::NeedsInitializedMemory` over the
whole bound range when it is placed in a bind group (`device/resource.rs:2971-2974`),
and emits `clear_buffer` for the uninitialized ranges before the submission that uses
them (`command/memory_init.rs:161-250`). **It would be wrong to claim that dropping
the host memset yields random data.** No usage flag, buffer size or binding changed.

Because every call still creates fresh buffers, "keep the necessary clearing" is
satisfied by construction here. It becomes an explicit reset obligation only in P2,
when buffers are reused.

## Correctness gates

New deterministic decoder test, no GPU required:
`gpu::five_point_f32::tests::decode_builds_fixed_slots_without_losing_state_or_order`
covers empty input, an all-unused trial, a trial mixing all nine slot statuses with
accepted slots at both ends, a failed→successful→failed three-trial batch checking
that no state leaks between trials, that `essential` is `Some` exactly for `Accepted`
slots, that slot indices and per-slot diagnostics stay in order, five illegal slot
encodings including NaN, an illegal header status, and the header-before-slot error
precedence.

Actual GPU executions (not shader parsing, not ignored tests):

- `five_point_f32_actual_gpu_stages`: **1 passed** after P1.1, and again after P1.2.
- All 62 `gpu::` library tests: **62 passed**.
- Example tests: p1-host **11 passed**, p1-layout **8 passed**, cpu-threads
  **10 passed**, capacity-replay **8 passed**.

Every one of the thirteen benchmark processes independently checked:

1. Exact input digest and the 65,024-trial count before solving.
2. The full historical decoded trial/slot/diagnostic/model bit signature on the
   reference replay, every warmup, every measured end-to-end replay, every profiled
   replay, the accounting replay and every timestamp replay. Lengths are checked first.
3. Prefix sizes **1, 31, 32, 33, 63, 65, 65,023** equal to the corresponding
   full-run prefix including trial and slot ordering, each also **repeated** on the
   same solver instance to show the second call imports nothing from the first.
4. Size alternation **65,024 → 1 → 32,768 → 65 → 512 → 65,023 → 32**, each equal to
   the full-run prefix. This is the stale-data and capacity-edge check for P1.2.
5. A degenerate all-zero-ray batch producing no models, an empty-input call, and then
   a successful 512-trial replay, to show failure→success reuse is clean.

All checks passed in all thirteen processes. No signature failure or rollback occurred.

Other gates passed: `cargo fmt --all -- --check`, `git diff --check`,
`python3 scripts/generate_five_point_wgsl.py --check` (200 + 11 expressions). Clippy
on the changed files reports only two warnings that are present verbatim in the
preserved pre-change snapshot (`len() % 5 == 0`, `u64::from(max_storage_buffer_binding_size)`);
the new harness contributes no warning of its own, and the four attributed to it come
from the shared `five_point_gpu_capacity_replay.rs` module it includes unmodified.

## Timing protocol

New harness `five_point_gpu_p1_host.rs`. Per process, per batch in **512, 32,768,
65,024**: three warmups then five measured end-to-end replays of the whole 65,024-trial
workset, then a separate three-warmup/five-measured series through the profiled entry
point for host phases, then one untimed accounting replay. Tables report within-process
medians of the five measured samples; no round is selected. Process order was
**B0, C0, C1, B1, B2, C2** for P1.1 and **C0, B0, C1, B1, C2, B2** for P1.2, so the
pair order alternates in both.

End-to-end uses the existing shared replay on a timestamp-disabled context and
includes f64→f32 rays, buffer creation, uploads, all kernels, waits, readback,
decoding and ordered collection. It excludes database load and sampling, GPU
initialization, signature and boundary checks, allocation accounting and final result
destruction. Host phases come from the profiled entry point on a timestamp-disabled
context, so no query work is added; they are disjoint host wall times. GPU pass
intervals come only from a separate timestamp-enabled context and are never added to
host phases, because they overlap the host wait.

The allocation counter is a counting global allocator enabled only for the one
untimed replay per batch, covering the solver calls and the harness ray conversion but
not the harness's own verification. It records **requested bytes, not resident pages**.

## P1.1 results

Medians in ms, three baseline processes versus three candidate processes.

| Batch | Metric | Baseline (3 processes) | Candidate (3 processes) | Median change |
|---:|---|---|---|---:|
| 65,024 | decode | 15.571, 14.955, 15.844 | 7.326, 7.505, 7.950 | **−51.8%** |
| 65,024 | end-to-end | 121.795, 108.549, 111.996 | 101.987, 102.884, 100.144 | **−8.9%** |
| 32,768 | decode | 15.765, 15.577, 15.580 | 7.582, 7.660, 7.762 | **−50.8%** |
| 32,768 | end-to-end | 116.825, 116.832, 109.191 | 104.750, 104.084, 103.015 | **−10.9%** |
| 512 | decode | 31.163, 25.982, 31.534 | 8.701, 7.471, 9.950 | **−72.1%** |
| 512 | end-to-end | 1701.635, 1608.617, 1796.319 | 1583.760, 1568.032, 1727.460 | −3.6% |

Every candidate sample beat every baseline sample within the comparison for decode at
all three batches, and for end-to-end at 32,768 and 65,024. At batch 512 the
end-to-end ranges **overlap** (candidate 1727.460 exceeds baseline 1608.617), so no
end-to-end win is claimed at 512; the solver-total series does separate there.

Host allocation per full replay, batch 65,024: **195,498 → 426 allocations**
(3 per trial removed) and **537.6 → 398.7 MiB** requested. Host zeroed bytes are
unchanged at 161.0 MiB, as expected for a decode-side change. Peak live bytes are
unchanged. The nine-kernel timestamp sum is unchanged (53.88/52.13/52.40 → 54.25/54.15/52.35 ms),
confirming this is host-side only.

At batch 512 the candidate's `prepare`, `encode`, `submit` and `wait` also improved,
which a decode-only change does not directly explain. The plausible cause is reduced
allocator pressure across 127 calls per replay, but this was **not isolated** and no
mechanism is asserted.

## P1.2 results

Baseline here is the retained P1.1 build, not the original solver.

| Batch | Metric | Baseline (3 processes) | Candidate (3 processes) | Median change |
|---:|---|---|---|---:|
| 65,024 | prepare | 20.737, 21.224, 19.912 | 2.402, 2.365, 2.264 | **−88.6%** |
| 65,024 | end-to-end | 103.759, 101.737, 100.151 | 80.354, 80.348, 79.876 | **−21.0%** |
| 32,768 | prepare | 18.735, 18.384, 18.902 | 1.858, 2.434, 1.905 | **−89.8%** |
| 32,768 | end-to-end | 102.272, 102.647, 102.933 | 88.207, 86.645, 85.898 | **−15.6%** |
| 512 | prepare | 54.290, 58.492, 52.716 | 12.924, 14.224, 15.652 | **−73.8%** |
| 512 | end-to-end | 1722.346, 1726.692, 1734.348 | 1636.736, 1640.016, 1655.482 | **−5.0%** |

Every candidate sample beat every baseline sample for prepare and end-to-end at all
three batches. Host zeroed bytes went **161.0 MiB → 0.0 MiB** and requested bytes
**398.7 → 237.7 MiB**.

Three honest qualifications:

- **Peak live host bytes did not change** (163.7 MiB at 65,024, 110.1 at 32,768,
  56.9 at 512). The 161 MiB was transient: each zero array was freed before the next
  was allocated, and the peak is set by the readback vector and the decoded results.
  A large reduction in requested bytes is not a reduction in footprint.
- Allocation **count rose slightly**, 426 → 438 at batch 65,024, from wgpu's own
  bookkeeping for the lazy clear.
- The nine-kernel timestamp sum is unchanged (52.56/52.33/52.36 → 52.17/52.20/52.16 ms).
  The device-side clear is real work but occurs outside the nine instrumented compute
  passes, so it is invisible in the kernel sum while still included in end-to-end and
  in `wait`. `wait` fell slightly, 60.4 → 58.8 ms at 65,024. Readback also fell,
  6.04 → 4.33 ms, which a buffer-creation change does not directly explain; not isolated.

## Combined effect, and the batch-512 finding

Original B0 baseline versus the retained P1.1+P1.2 build, median of process medians:

| Batch | End-to-end B → C | Reduction | prepare | decode |
|---:|---:|---:|---:|---:|
| 65,024 | 111.996 → 80.348 ms | **28.3%** | 20.237 → 2.365 | 15.571 → 8.406 |
| 32,768 | 116.825 → 86.645 ms | **25.8%** | 18.750 → 1.905 | 15.580 → 8.220 |
| 512 | 1701.635 → 1640.016 ms | 3.6%, overlapping | 58.640 → 14.224 | 31.163 → 11.527 |

Batch 512 is the batch this round added to the protocol, and it exposes the dominant
remaining cost: at 127 calls per replay the host spends about **1.37–1.43 s in `wait`**
and **0.20 s in readback**, so the whole replay costs roughly twenty times the
single-call 65,024 replay for identical work. Host preparation and decoding are no
longer the limiter there. That is direct evidence for the P2 and P3 items —
persistent sessions and deferred submission/readback — and it means neither P1 change
should be expected to fix the small-batch regime.

## CPU8 comparison: first stable GPU win, on non-equivalent output

The **unchanged** `five_point_gpu_cpu_threads` example was rebuilt (source SHA-256
verified identical to the snapshot) and run with `--cpu-threads 8`. Its existing
protocol is three warmups plus eight measured rounds, alternating CPU→GPU / GPU→CPU,
100 ms idle before each path, timestamps disabled, dedicated Rayon pool kept alive.
The CPU reference signature `9e6764b41602d0005710187d2558d046f1a843c5f873b9542dd7b4f6c29c563a`
matches the previous round exactly, so the CPU path is bit-identical.

| Batch | CPU8 median ms | GPU median ms | CPU/GPU | GPU wins | Stable-win gate |
|---:|---:|---:|---:|---:|---|
| 32,768 (2 calls) | 121.922 | 94.375 | 1.292 | **8/8** | **true** |
| 65,024 (1 call) | 121.647 | 89.304 | 1.362 | **8/8** | **true** |

Raw measured samples (ms):

| Batch | CPU8 | GPU |
|---|---|---|
| 32,768 | 121.935, 120.347, 122.524, 121.909, 122.590, 122.013, 119.869, 121.807 | 94.030, 94.434, 92.800, 94.748, 95.092, 94.948, 94.315, 93.749 |
| 65,024 | 122.186, 121.504, 121.742, 121.527, 120.441, 121.552, 121.928, 122.432 | 89.198, 90.144, 89.174, 89.518, 89.410, 93.249, 88.925, 88.894 |

**The CPU8 numbers drifted.** The previous round measured CPU8 medians of 111.052 and
111.735 ms with identical CPU code and an identical CPU signature; this run measured
121.922 and 121.647 ms, about 9% slower. The cause was not diagnosed; no affinity,
thermal or frequency control was applied. Part of the apparent margin is therefore
environmental, and the ratios above should not be read as a controlled speedup.

The conclusion survives the drift. Substituting the previous round's more favourable
CPU8 samples, every GPU sample still beats every historical CPU sample: at 65,024 the
GPU maximum 93.249 is below the historical CPU minimum 109.474, and at 32,768 the GPU
maximum 95.092 is below the historical CPU minimum 107.296. Against the historical
CPU medians the ratios are 1.251 and 1.177.

CPU4 was not run. CPU8 is the stronger CPU baseline, and it is the configuration the
existing stable-win gate is defined against; CPU4 would not change the retain decision.

**Quality is unchanged and still not equivalent.** The full-workset counts are
byte-identical to the figures already recorded: CPU 288,264 models, GPU 209,600
models, 12,012 GPU-empty trials, 11,967 trials where CPU has a solution and GPU is
empty, 26,294 trials with differing candidate counts, `equal_quality: false`. Eight
local rounds on one device are not a statistical guarantee, and an offline
candidate-generation win at 65,024 aggregated trials is not evidence that production
can supply that much useful work or that it would be quality-equivalent.

## Retain / rollback

**Retain both.** P1.1 and P1.2 each produced a reduction in the phase they target,
with every candidate sample below every baseline sample in that phase, at an unchanged
complete GPU signature and unchanged kernel timings. Nothing was rolled back, and no
failing result was discarded.

## Unresolved risks

- f32 versus f64 quality is untouched: 11,967 trials still lose a CPU-available
  solution and candidate counts differ on 26,294 trials. Q1 and Q2 remain open, and
  this solver must not be routed into the default RANSAC.
- Batch 512 remains roughly twenty times the cost of the single-call large batch for
  the same work, dominated by per-call wait and readback. P1 does not address it.
- The CPU8 host drift between rounds is undiagnosed, so cross-round timing
  comparisons are unreliable even though within-round pairing is not.
- `readback` at 65,024 and several host phases at batch 512 improved by more than the
  changed phase alone explains. Not isolated; no mechanism asserted.
- Peak host footprint is unchanged, so this round buys no headroom for larger batches.

## Reproduction

```sh
cargo build --release --target-dir target -p rustsfm --no-default-features \
  --features gpu-wgpu --example five_point_gpu_p1_host
cargo build --release --target-dir target -p rustsfm --no-default-features \
  --features gpu-wgpu --example five_point_gpu_cpu_threads
cargo test --release --target-dir target -p rustsfm --lib --no-default-features \
  --features gpu-wgpu five_point_f32
```

Note that `CARGO_TARGET_DIR` was set in the environment to a temporary cache;
`--target-dir target` was passed explicitly to every build so the artifacts are
reproducible in the worktree.

The preserved `baseline-p1host`, `p11-p1host`, `p12-p1host` and `final-p1host`
executables each accept `--database PATH --label NAME --output NEW_PATH` and refuse to
overwrite. Run them alternately with distinct output paths and compare
`input_digest`, `signature`, `boundaries` and `size_alternation` before accepting any
timing. `analyze.py` is read-only and takes
`analyze.py NAME baseline...json -- candidate...json`. Do not overwrite existing
artifacts or restore whole source directories over unrelated work.

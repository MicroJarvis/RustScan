# GPU layout vs CPU 4/8-thread replay — 2026-09-14

## Conclusion

Current GPU layout beats CPU four threads in all eight measured paired rounds at both batch sizes, but loses to CPU eight threads in all eight rounds at both sizes. This is fixed-workset candidate-generation throughput, **not equal-quality throughput or production RANSAC acceleration**. No algorithm, threshold, shader, production routing or thread-admission policy changed in this experiment.

## Provenance and protocol

- Worktree branch: `gpu-five-point-f32`, solver commit `3ab9c09`.
- Harness: `rustsfm/examples/five_point_gpu_cpu_threads.rs`; reuses the capacity replay's loader, GPU replay and signature helpers (visibility-only edits to that existing example).
- Apple M5 Max, Metal; host reports 18 physical / 18 logical CPUs. Dedicated Rayon pools of exactly 4 or 8 workers, not affinity-pinned. OS placement on heterogeneous cores is uncontrolled.
- Read-only DB: `/Users/tfjiang/Projects/RustScan/artifacts/runs/flowers2_960_settlement_20260913/matching.db`.
- 127 distinct real pairs, **231 unique images**, 512 sampler-prefix trials per pair, seed 1. Same **65,024 trials** for every measurement. Not the entire 960-image dataset, no repeated-input padding.
- Input digest: `af07459d637b5b8c2b20c76d3ef103f058505fe0e8307578ee7e8750f3e230c5`.
- GPU: current roots group size 32, elimination group size (1,32,1); timestamp queries disabled.
- Three alternating warmup rounds per path/batch, then eight paired measured rounds. Even rounds CPU then GPU; odd rounds GPU then CPU. Each measured path preceded by an equal **100 ms idle interval**, outside its timer, to reduce asymmetric effects of recently active Rayon workers. This is not a CPU/GPU overlap benchmark. Pool stays alive for both paths.
- CPU timer includes f64 solving and ordered collection. GPU timer includes f64-to-f32 input conversion, allocation/upload, all kernels, waits, result readback/decode and collection. Excludes DB loading/sampling, reference calculation, initialization, warmups, idle time, signature/quality checks and final result destruction. Same definition for both CPU-thread configurations.
- Predefined local win gate: GPU faster in every paired round AND GPU median at least 5% below CPU median. Not a statistical confidence interval or cross-device guarantee.
- CPU4 experiment ran first; after both batch sizes passed its gate, CPU8 experiment ran. CPU4 and CPU8 were not interleaved with one another; each was interleaved with GPU.

## Results

All times below cover the full 65,024-trial workset. Even-sized sample medians use the average of the two middle values.

| CPU workers | Batch | GPU solve calls | CPU median ms | GPU median ms | CPU/GPU time ratio | GPU wins | Local win gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 32,768 | 2 | 208.254 | 177.444 | 1.174 | 8/8 | pass |
| 4 | 65,024 | 1 | 210.228 | 179.625 | 1.170 | 8/8 | pass |
| 8 | 32,768 | 2 | 112.498 | 175.929 | 0.639 | 0/8 | fail |
| 8 | 65,024 | 1 | 109.028 | 181.732 | 0.600 | 0/8 | fail |

GPU elapsed time is about 14.6–14.8% lower than CPU4. CPU8 elapsed time is about 36.1–40.0% lower than GPU: CPU8 is 1.56–1.67x faster. The previous approximately 163 ms GPU measurement came from a different protocol/session and must not substitute for the 176–182 ms observed here. Small batch-size differences here do not establish a preferred batch size.

### Raw measured rounds (ms, round 0 through 7)

| CPU workers / batch | CPU | GPU |
|---|---|---|
| 4 / 32768 | 207.849, 221.086, 215.982, 211.286, 207.854, 205.584, 206.623, 208.654 | 180.239, 174.034, 179.115, 173.695, 185.999, 175.774, 192.828, 173.092 |
| 4 / 65024 | 207.863, 218.667, 206.974, 212.277, 209.527, 210.930, 211.596, 206.134 | 181.527, 173.034, 185.631, 178.896, 183.663, 171.017, 180.355, 176.075 |
| 8 / 32768 | 110.972, 112.663, 114.457, 110.819, 111.194, 113.027, 112.333, 114.910 | 170.707, 175.170, 182.263, 176.688, 178.137, 170.551, 183.409, 173.055 |
| 8 / 65024 | 111.137, 109.533, 109.596, 113.543, 108.453, 106.594, 108.524, 107.513 | 180.697, 182.766, 188.911, 178.056, 190.071, 175.341, 191.477, 176.899 |

## Quality gates and limitations

Every CPU warmup/measured result is coefficient-bit/order identical to the serial CPU reference. Every GPU warmup/measured result has the historical complete decoded trial/slot/diagnostic/model signature. Result lengths are checked before signatures. Across both thread-count experiments, input digests, CPU reference signatures, GPU signatures and quality counts match.

- CPU reference signature (harness-specific model-count/row-major coefficient encoding): `9e6764b41602d0005710187d2558d046f1a843c5f873b9542dd7b4f6c29c563a`.
- GPU signature: `1f6f3f8d88a17e4e1c42abfc85aad44a66c4b864091805594e086ca80bf9b339`.
- CPU models: 288,264; GPU models: 209,600.
- GPU-empty trials: 12,012; CPU-nonempty/GPU-empty: 11,967.
- Candidate-count mismatch: 26,294 trials.

These signatures certify repeatability **within each algorithm**, not CPU/GPU equivalence. No new mask-distance study was run; existing f32 quality deficits remain. Serial CPU is an offline reference, never a GPU runtime fallback. No production full RANSAC, reconstruction, RSS comparison, affinity or thermal control. No default production CPU thread count was changed.

## Reproduction and validation

From the worktree root (output paths must not already exist):

```sh
cargo test -p rustsfm --no-default-features --features gpu-wgpu --example five_point_gpu_cpu_threads
cargo build --release -p rustsfm --no-default-features --features gpu-wgpu --example five_point_gpu_cpu_threads

target/release/examples/five_point_gpu_cpu_threads --database /Users/tfjiang/Projects/RustScan/artifacts/runs/flowers2_960_settlement_20260913/matching.db --cpu-threads 4 --output artifacts/runs/five_point_cpu4_gpu_layout_comparison.json
# Run the eight-thread comparison after checking the four-thread local win gate.
target/release/examples/five_point_gpu_cpu_threads --database /Users/tfjiang/Projects/RustScan/artifacts/runs/flowers2_960_settlement_20260913/matching.db --cpu-threads 8 --output artifacts/runs/five_point_cpu8_gpu_layout_comparison.json

cargo fmt --all -- --check
git diff --check
python3 scripts/generate_five_point_wgsl.py --check
```

Actual validation: 10 example/helper tests passed (includes alternating-order/even-median/win-gate tests); release build, both actual GPU replays, format/diff and generator consistency checks passed. Existing unrelated warnings remain. Each command had a 180-second or shorter bound; no timeouts. Compact local JSON artifacts contain all warmups/measured rounds; this report retains the measured results for version control. No commit/push in this experiment.

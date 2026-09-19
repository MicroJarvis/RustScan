# Q4 coefficient precision — resumed verification (2026-09-15)

## Decision

**Retain.** Polynomial-only FMA compensation on Apple M5 Max meets the Q3
acceptance conditions (fixed loss-cohort coefficient p90 ↓84–103×;
MissingNoSlot ↓; quality net non-negative). New quality baseline signatures
are the candidate model/diagnostic hashes below. Upstream bytes (basis/A/solve/B)
remain bit-identical to the pre-Q4 baseline; only coeff11 changes.

This is an experimental retain, not production promotion. LU/B residual error
and the reclassified RecoveryFailed/NotDone cohorts are follow-up work (Q5).

## Status and ownership

**The existing Q4 candidate was already implemented and had completed two real
65,024-trial runs. This continuation reproduced it, rather than developing a
second candidate. Q3's coefficient/MissingNoSlot acceptance conditions pass on
Apple M5 Max. This is an experimental result, not production promotion.**

Read predecessor: `gpu-five-point-q3-recovery-attribution-20260915.md`.
All paths below are relative to this worktree. No commit/push, production
integration, CPU runtime fallback, root/recovery/threshold changes, or LU changes
were made in this continuation.

### Already present at entry

- `scripts/generate_five_point_wgsl.py` and generated
  `RustSFM/src/gpu/shaders/five_point_generated.wgsl`: f32 polynomial-only
  compensation. Each signed triple product carries FMA product residuals;
  magnitude-ordered sum compensation uses explicit FMA subtraction, followed by
  `value + correction`. Term packing/order and upstream arithmetic are retained.
- `RustSFM/examples/five_point_gpu_q4_coefficients.rs`,
  `RustSFM/examples/q4/attribution.rs`, `polynomial-fixtures.json`: offline
  same-basis/same-A/same-solve/same-B attribution, fixed baseline loss cohort,
  binary stage readback, quality sidecars, signature repeats and timings.
- `RustSFM/src/gpu/five_point_q4_tests.rs`, its test-module registration and
  generator checks: required-device arithmetic probe and 12 real B fixtures
  under six signed/power-of-two scales, each dispatched twice.
- `experiments/q4-baseline-20260915/`: saved pre-Q4 patch, old-file hashes,
  generator/shader/test snapshots, baseline runs at both tolerances.
- `experiments/q4-candidate-tol1e{3,2}-20260915.*`: completed candidate runs.
- Earlier failed probes/tests are preserved: ordinary subtraction compensation
  was defeated by measured Metal reassociation. Later FMA-sum tests passed.
  This is empirical compiler behavior, not a portable WGSL guarantee.
- No Q4 report existed; `scripts/summarize_five_point_q4.py` ended in an
  incomplete list comprehension at line 74 and could not execute.

### New in this continuation

- Preserved the incomplete script verbatim as
  `experiments/q4-summary-incomplete-preserved-20260915.py`; completed only its
  aggregation/output tail. It retains existing gates, adds timing/quality
  summaries and artifact SHA256, and refuses to overwrite its output.
- Rebuilt the existing candidate and reran actual-device focused tests.
- Two fresh full replays:
  `experiments/q4-verified-tol1e3-20260915.*` and
  `experiments/q4-verified-tol1e2-20260915.*` (report, bits, trials, stages,
  quality). Old candidate/baseline artifacts were not overwritten.
- `experiments/q4-verified-summary-20260915.json`,
  `experiments/q4-verified-focused-20260915.log`,
  `experiments/q4-verified-source-manifest-20260915.json`, and this report.
- No Rust/shader/solver changes were necessary.

## Preservation and single-variable evidence

Comparing the live tracked patch against `tracked-before.patch`, only four
tracked files differ from pre-Q4: generated WGSL, its generator, generator tests,
and `five_point_f32_tests.rs` (adds the Q4 test module). Earlier P/Q tracked
patch sections, including LU/algebra, roots, recovery, CPU geometry and GPU
integration, are unchanged. The saved existing-file SHA256 inventory also
matches except the deliberately extended test module file. This comparison is
against pre-Q4 work, not against HEAD; the large pre-existing diff was retained.

For each matching tolerance the summarizer compares all 65,024 binary records
(1,544 bytes each): the first 1,500 bytes, representing basis36 + A200 + solve100
+ B39, are **bit-identical in every baseline/candidate record**. Only coeff11 is
allowed to change. Rank-deficient placeholders remain unchanged. This is direct
same-basis and same-B isolation, not merely similar aggregate distributions.

Both fresh reports equal the respective older Q4 reports except the timing
field. SHA256 comparison confirms identical old/new `.bits.json`, `.trials.json`
and `.stages.bin` at both tolerances, plus `.quality.json` at 1e-2. The historical
1e-3 baseline report lacks quality; quality comparisons use the complete 1e-2
baseline sidecar, not invented values. Both baseline root-class tables exactly
match the Q3 JSON tables.

## Fresh GPU results: 127 pairs × 512 = 65,024 trials

Device: `GpuSiftCapabilities { backend: Wgpu, device_name: "Apple M5 Max" }`.
The two tolerances below affect **offline root matching only**, not solver gates.

| Metric | Q3/baseline | Q4 verified |
|---|---:|---:|
| CPU own-basis models | 288,264 | 288,264 |
| f64 reference on GPU basis | 285,371 | 285,371 |
| GPU models | 243,967 | 248,884 (+4,917) |
| CPU minus GPU | 44,297 | 39,380 |
| Same-basis downstream model gap | 41,404 | 36,487 |
| CPU-has/GPU-empty trials | 2,899 | 2,127 |
| RankDeficient trials | 621 | 621 |
| MissingNoSlot, 1e-3 | 37,345 | 10,094 |
| MissingNoSlot, 1e-2 | 21,467 | 3,737 |
| Spurious accepted, 1e-3 | 11,749 | 2,649 |
| Spurious accepted, 1e-2 | 4,267 | 665 |
| Covered, 1e-3 / 1e-2 | 232,218 / 239,700 | 246,235 / 248,219 |
| RecoveryFailed, 1e-3 / 1e-2 | 9,862 / 14,445 | 18,063 / 20,119 |
| NotDone, 1e-3 / 1e-2 | 5,458 / 8,793 | 10,359 / 12,651 |

RecoveryFailed and NotDone increase as many roots now have a nearby slot.
These are end-to-end class counts, not proof that unchanged recovery code became
intrinsically worse. They also mean the overall model deficit is not solved.
No follow-up root or recovery change is included in Q4.

### Coefficient precision and attribution

Error is max absolute coefficient error divided by max absolute reference
coefficient. It is not a per-coefficient relative error bound.

| Cohort/measurement | Baseline | Candidate |
|---|---:|---:|
| Fixed baseline loss cohort p90, 1e-3 (n=20,906) | 0.0842088964 | 0.000997629454 (84.41× lower) |
| Fixed baseline loss cohort p90, 1e-2 (n=19,317) | 0.117181859 | 0.00113743043 (103.02× lower) |
| All valid trials same-B polynomial p90 (n=64,403) | 0.000595403931 | 4.39903101e-8 |
| All valid trials same-B polynomial p99 | 11.2573017 | 2.26749030e-6 |
| All valid trials same-basis total polynomial p90 | 0.000843941115 | 0.000121721986 |
| All valid trials same-basis total polynomial p99 | 11.2678620 | 0.00947228968 |

The 621 unavailable stage references are rank-deficient trials, not silently
counted as zero error. Fixed cohorts avoid selecting a different population
simply because the candidate recovers more roots. For context, candidate's own
1e-3 loss cohort has 17,971 trials and p90 0.00128075574.

The remaining error is primarily upstream: f64 polynomial evaluated on GPU B
versus f64 same-basis polynomial has p90 0.000121722662 (all), 0.00113738764
(fixed 1e-2 baseline-loss cohort), essentially the same as candidate total error.
Same-B compensation greatly reduces polynomial expansion error but does not
repair LU/B error. Same-B p99 is still nonzero, so this is not an exact-arithmetic
or universal robustness claim.

## Re-established model quality (not model-count-only acceptance)

Offline Sampson inliers, unchanged 4-pixel camera-normalized threshold, using
all fixed pair observations. CPU per-trial quality sidecars are identical.

| Trial best GPU inliers | Baseline | Candidate |
|---|---:|---:|
| p50 | 89 | 98 |
| p90 | 1,167 | 1,190 |
| p99 | 2,729 | 2,739 |
| Relative to CPU pair best, p50 | 0.473684 | 0.485207 |
| Relative to CPU pair best, p90 | 0.975460 | 0.979167 |

| GPU trial best / CPU pair best | Baseline trials | Candidate trials |
|---|---:|---:|
| zero | 2,942 | 2,172 |
| (0, .25) | 12,758 | 12,767 |
| [.25, .5) | 18,024 | 18,117 |
| [.5, .75) | 14,895 | 15,009 |
| [.75, 1) | 16,081 | 16,633 |
| >=1 | 324 | 326 |

Direct candidate-versus-baseline trial quality: **5,701 better, 4,200 worse,
55,123 equal**. Pair-best quality: **4 better, 5 worse, 118 equal**. Zero-based
pair indices with regression: 45 (347→346), 73 (372→371), 77 (3644→3642),
88 (2569→2568), 104 (28→27). There is no universal pair/trial dominance claim.
The zero-inlier bucket is a quality measure, not definitionally an empty-model
count. This is not end-to-end RANSAC or the 960-image production pipeline.

## Signatures and time

Input digest:
`af07459d637b5b8c2b20c76d3ef103f058505fe0e8307578ee7e8750f3e230c5`

CPU f64 (unchanged):
`9e6764b41602d0005710187d2558d046f1a843c5f873b9542dd7b4f6c29c563a`

Baseline model:
`472124c8f1a9c6299d659e2f835dc11c682ff8a593757f0b145f73454f51d6ea`

Candidate model:
`26e825cf97002be4f7fc36b77ce4fed682445af37746504253444e0103c5e716`

Baseline diagnostic:
`3d04bb3edaef183049e07d6c3841e296f963da3b279bcbd99d878e1dc090c050`

Candidate diagnostic:
`d9e911ac6b6e2be18f7e3b8b8615417efba6257f955ce2bbf21f943a1c7f358a`

Each fresh run checks three full replay repetitions at batch 512 and three at
batch 65,024 against the initial candidate model and diagnostic signatures.
The summarizer additionally checks bits equality across tolerance runs.

Medians of six samples per batch (three from each tolerance run):

| Full 65,024-trial diagnostic replay, partitioned by batch | Baseline seconds | Fresh candidate seconds | C/B |
|---|---:|---:|---:|
| 512 | 1.561989 | 1.545985 | 0.98975 |
| 65,024 | 0.0822393 | 0.0810818 | 0.98592 |

These are real wall-clock full diagnostic replay measurements, including host
and readback, excluding offline f64 attribution/quality scoring. Raw samples
are in the summary. Baseline is historical, not an interleaved rebuild/run:
**the ~1% median differences do not establish a speedup or a rigorous regression
bound**. No kernel-only or production-throughput claim follows.

## Actual commands and outcomes

All terminal calls used timeout <=180,000 ms and head15/tail20 output limits.
Large JSON was inspected only through Python summaries. No device/dependency
blocker or timed-out verification occurred. Run from this worktree:

```sh
cargo test -p rustsfm --release --lib --no-default-features --features gpu-wgpu five_point -- --nocapture --test-threads=1
cargo build -p rustsfm --release --no-default-features --features gpu-wgpu --example five_point_gpu_q4_coefficients
python3 -m unittest discover -s scripts -p test_generate_five_point_wgsl.py

target/release/examples/five_point_gpu_q4_coefficients --database /Users/tfjiang/Projects/RustScan/output/flowers2_960_settlement_20260913/matching.db --candidate --match-tol 0.001 --baseline-trials experiments/q4-baseline-20260915/tol1e3.trials.json --output experiments/q4-verified-tol1e3-20260915.json

target/release/examples/five_point_gpu_q4_coefficients --database /Users/tfjiang/Projects/RustScan/output/flowers2_960_settlement_20260913/matching.db --candidate --match-tol 0.01 --baseline-trials experiments/q4-baseline-20260915/tol1e2.trials.json --output experiments/q4-verified-tol1e2-20260915.json

python3 scripts/summarize_five_point_q4.py --baseline-dir experiments/q4-baseline-20260915 --candidate-1e3 experiments/q4-verified-tol1e3-20260915.json --candidate-1e2 experiments/q4-verified-tol1e2-20260915.json --output experiments/q4-verified-summary-20260915.json
```

- Focused tests: **11 passed, 0 failed, 0 ignored**, twice (0.89 s / 0.84 s
  test execution, excluding build). Required-device Q4 tests do not skip.
  All 72 same-B/sign/scale cases run twice and exceed the required 10× gain.
  Arithmetic residual probe passes on the actual GPU.
- Build: success, release build 13.71 s; existing unrelated warnings remain
  (32 library warnings, plus test warnings).
- Generator tests: **5 passed**, 0.146 s; includes generated artifact
  reproducibility, strict expression/packing checks and compensation checks.
- Both full replays: exit 0, **65,024 trials each**, expected input and CPU
  signatures, repeat stability, same-basis attribution and fresh quality.
- Summary: exit 0; fixed-cohort precision, MissingNoSlot, spurious, empty-trial,
  unchanged upstream bytes/CPU quality and repeat bits gates pass.
- One exploratory Python print failed with `KeyError: model_quality` when
  trying the older 1e-3 baseline; corrected inspection used its complete 1e-2
  counterpart. This was not a solver/test failure. `diff -u` exit 1 in audit
  calls means expected file differences, not a validation failure.

Outputs intentionally refuse overwrite: use new names for any later rerun.
The source/binary SHA256 manifest and artifact hashes tie this proof to the
current candidate. Scope is complete for this Q4 verification; production
promotion, cross-device testing, LU improvements and root/recovery follow-ups
remain separate decisions.

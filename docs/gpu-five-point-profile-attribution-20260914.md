# GPU five-point f32: pass timing and loss attribution — 2026-09-14

## Scope / preservation

Worktree: `/Users/tfjiang/Projects/RustScan/.worktrees/gpu-five-point-f32`.
No shader, solver algorithm, threshold, production matching/RANSAC routing, CPU runtime fallback, commit or push changes.

Added optional `WgpuContext::try_new_experimental_timestamps()` (requests only adapter-supported `TIMESTAMP_QUERY`), `timestamp_queries_enabled()`, and experimental `solve_essential_profiled()`. Default constructors still request **empty features**. Both solver APIs share the same nine dispatches, buffers and decoding; profile off allocates no query set. The experimental solver's shared host path now also takes a few `Instant` measurements; it does not change production matching.

New independent example: `rustsfm/examples/five_point_gpu_profile_replay.rs`. It imports the existing CPU read-only loader and generated f64 algebra reference, without editing them. The five pre-existing dirty/untracked capacity/batch-sweep files were SHA-256 checked before/after and are unchanged:

- `rustsfm/examples/five_point_gpu_replay.rs`
- `rustsfm/examples/five_point_replay_probe.rs`
- `rustsfm/examples/five_point_gpu_capacity_replay.rs`
- `docs/gpu-five-point-batch-sweep-20260914.md`
- `docs/gpu-five-point-capacity-sweep-20260914.md`

Preservation manifest: `artifacts/runs/five_point_profile_preserved.sha256`.

## Input and measurement contract

Actual adapter: **Apple M5 Max / Metal**, timestamp queries supported. Default and experimental contexts were checked for matching adapter name/backend; the default was checked not to enable timestamp queries.

Database, opened read-only:
`/Users/tfjiang/Projects/RustScan/artifacts/runs/flowers2_960_settlement_20260913/matching.db`.

Same **127 pairs × 512 = 65,024** real sampled trials, from 14,045 eligible raw-match pairs; no replication or synthetic workload expansion. Loader selection, filtering, per-pair seed 1 and stateful sampler are unchanged. All pair provenance and sample input digest remain in the independent JSON. Both batch sizes replay the entire ordered workset, including the final partial chunk at 32,768.

Input BLAKE3, hard-checked:
`af07459d637b5b8c2b20c76d3ef103f058505fe0e8307578ee7e8750f3e230c5`.

GPU signature, hard-checked against the capacity sweep in **every warmup and measured run**, all decoded trial/slot fields and coefficient bits:
`1f6f3f8d88a17e4e1c42abfc85aad44a66c4b864091805594e086ca80bf9b339`.

Three modes rotate order each round:

0. Default context, profile off.
1. Timestamp-enabled context, profile off.
2. Timestamp-enabled context, profile on.

**3 warmup + 3 measured rounds per mode per batch**. Final experiment: 36 complete-workset replays, 54 solve calls, 486 compute dispatches; 18 profiled solve calls. CPU quality analysis and representative stage re-dispatches are outside these timings.

Outer time includes f64→f32 conversion, allocation/upload preparation, encoding/submission, waiting, result readback/decoding, collection and destruction inside replay. Excludes DB loading, initialization, output signatures, CPU comparison, representative diagnostics and JSON writing. Host prepare includes validation, padding, buffer/query creation; actual GPU transfer/initialization work is not claimed to occur entirely in that host interval. Encode includes bind groups and encoder finish; submit is queue submission; wait includes GPU execution and scheduling. Readback includes result copy submission, map, wait and host buffer decoding; decode separately constructs trial structs. Timestamp readback includes its separate resolve submission/wait plus the final timestamp copy/map/wait.

**Host intervals are disjoint wall times. GPU pass intervals overlap host waiting: do not add GPU and host time.** Per-field medians do not sum exactly to median total, and the outer timer also includes conversion/collection/cleanup not inside the solver profile.

## Timestamp anomaly and bounded diagnostic isolation

18 boundary queries (start/end for each pass), no within-pass timestamp feature needed. No intermediate stage readbacks or per-stage waits are used for timing.

Initial same-compute-submission resolve produced zero `recover` timestamps. Retaining raw ticks established these were actually **[0,0]**, not merely a rounded short duration. In the second same-submission experiment, 5/6 measured recover chunks at batch 32,768 and 2/3 at 65,024 were invalid. Those artifacts remain available; they are **not** the final timing basis.

The final implementation waits for the **whole compute submission**, reads results, then resolves all 18 queries in one separate submission and reads timestamps once. On the final real workload, all nine pass samples were valid in **all 18 profiled solves including warmups**. This bounded change to diagnostic resolve ordering removed the observed anomaly on this adapter; it does **not establish a Metal driver root cause** or claim all adapters need this workaround. It adds two final submissions relative to profile off (query resolve and timestamp copy), explicitly included in profile-on totals.

Raw ticks and timestamp period are retained. Equal boundary ticks become `null`, never zero-cost work; nonmonotonic boundaries fail the diagnostic. On an adapter lacking timestamp support, the experimental constructor requests no unsupported feature, the report explicitly uses host-only measurements, and GPU pass times are `null`. Host wait is not a substitute for per-pass execution time. Both supported and feature-disabled paths were actually tested on this adapter; a physically timestamp-incapable adapter was not available.

**wgpu guarantees initialized buffer contents. Nothing here supports a claim that old uncleared buffers caused random numerical errors.** Existing explicit initialization and shaders were not changed.

## Final actual timings

Authoritative artifact:
`artifacts/runs/five_point_gpu_profile_127x512_20260914_separate_resolve.json` (**76,628 bytes**).

Milliseconds, median of three complete-workset rounds:

| Batch | Default/off | Timestamp context/off | Timestamp context/on | On vs same-context off |
|---:|---:|---:|---:|---:|
| 32,768 | 328.276708 | 332.111750 | 337.059000 | +1.4896% |
| 65,024 | 323.428625 | 327.867000 | 326.308208 | −0.4754% |

Timestamp-context/off vs default/off: +1.1682% and +1.3723%. The small negative full-batch on/off difference is sample noise/order/system-state evidence, **not a profiling speedup**. Three rounds cannot certify zero perturbation; result signatures do certify exact decoded-result invariance for these runs.

Raw round milliseconds, default/off / timestamp-context/off / on:

- 32,768: `[328.276708,329.502500,328.138750]` / `[324.949292,332.993416,332.111750]` / `[331.091958,337.059000,337.421292]`.
- 65,024: `[327.764083,323.428625,316.307959]` / `[326.908666,327.867000,329.478958]` / `[326.308208,325.607125,327.353292]`.

GPU milliseconds: sum each pass across chunks within each round, then median across rounds:

| Pass | Batch 32,768 | Batch 65,024 |
|---|---:|---:|
| constraints | 0.076083 | 0.068542 |
| basis/nullspace | 32.801375 | 32.884541 |
| pack | 0.165626 | 0.158709 |
| elimination | 81.113166 | 81.948000 |
| algebra | 22.444708 | 23.598625 |
| polynomial | 3.579708 | 3.780666 |
| validate polynomial | 0.145417 | 0.141750 |
| roots | **126.492417** | **119.464000** |
| recover | 6.515501 | 6.482541 |

These are pass-boundary counter observations, not certified exclusive hardware activity. Gaps, transfers, resolve and driver work are outside pass durations. `roots` is the largest measured pass; elimination is second.

Host milliseconds in profile-on mode, same aggregation:

| Interval | Batch 32,768 | Batch 65,024 |
|---|---:|---:|
| prepare | 19.943374 | 20.557292 |
| encode | 0.292917 | 0.154792 |
| submit | 0.054709 | 0.028916 |
| wait | 283.394916 | 278.388167 |
| result readback | 5.538000 | 5.954709 |
| trial decode | 16.328209 | 16.035792 |
| timestamp resolve/readback | 5.211792 | 2.633625 |

## Full-workset loss accounting (not sampled)

CPU original-f64-ray models **288,264**; GPU models **209,600**. Count mismatch **26,294 trials**. GPU empty **12,012 trials**, of which CPU nonempty **11,967**; CPU empty/GPU nonempty **2 trials**. Empty GPU sets leave **51,188 CPU candidates** without any target; empty CPU sets leave **3 GPU candidates** without any target. Candidate differences are not inferred just by subtracting total counts.

### Trial counters (overlap explicitly retained)

| Event | Trials | CPU nonempty / GPU empty within group |
|---|---:|---:|
| rank failure | 10,035 | 10,028 |
| algebra failure | 10,036 | 10,029 |
| invalid polynomial | 0 | 0 |
| any NotConverged slot | 52,672 | 1,900 |
| root iterations reach 384 | 5,629 | 464 |
| any recovery rejection | 11,246 | 1,579 |

All 10,035 rank failures also report singular algebra. There is **one additional algebra-only failure**. Never sum these marginals as independent lost trials. JSON includes disjoint joint groups (bit order: rank, algebra, polynomial, NotConverged, recovery, iteration limit) and full CPU/GPU missing-candidate association counts in each group.

### Root-slot counters

54,988 trials entered root iterations; sum of reported degrees / non-unused slots **549,479**. Allocated slots **650,240**, including **100,761 unused**. These are slot counts, not trial counts or certified distinct mathematical roots:

| Slot outcome | Slots |
|---|---:|
| Accepted | 209,600 |
| NotConverged | 297,822 |
| Complex | 21,468 |
| DuplicateRoot | 398 |
| NullVectorFailure | 7,733 |
| InvalidEssential | 12,458 |
| DuplicateModel / remaining RealRoot | 0 / 0 |

Recovery rejects total **20,191 root slots** across 11,246 trials. Example assertions reconcile header counters with slot statuses and accepted model counts.

**NotConverged is not a pure iteration-convergence counter.** Shader `five_point_recovery.wgsl` checks the root's real-axis polynomial residual before testing `done` and imaginary magnitude (lines 81–87), and later includes real-polish rejection (line 102). Thus complex roots may receive this status before Complex classification. 297,822 cannot be called “297,822 failed Aberth iterations” or “297,822 missing real CPU roots”. No shader change was made.

### Bidirectional candidate absence associations

Nearest unit-Frobenius, sign-invariant distance in each direction; not slot matching or a bijection. Empty target sets count as missing. These thresholds are **offline diagnostic association tolerances only**, never solver thresholds.

| Distance tolerance | CPU candidates missing GPU neighbor | GPU candidates missing CPU neighbor |
|---:|---:|---:|
| 0.001 | 125,083 | 46,419 |
| 0.01 | 90,198 | 11,522 |
| 0.1 | 80,668 | 2,232 |

At 0.01, missing-neighbor trials: CPU→GPU **29,306**, GPU→CPU **7,690**. Rank-failed trials contain **43,556** missing CPU candidates (48.2893% of the 90,198 at 0.01) and account for **10,028 / 11,967 = 83.7971%** of CPU-nonempty/GPU-empty trials. These are precise associations / observed rejection boundaries, not proof that changing rank handling would recover those models with acceptable quality.

Even the disjoint no-failure-flag group (1,259 trials) contains **222** CPU candidates missing a GPU neighbor at 0.01. Failure counters alone do not explain candidate-set differences.

## Selected same-input / same-basis diagnostics

First-trial selection by event, deduplicated to global trials **0,1,2,3,720**. Small untimed re-dispatches read GPU constraints, basis diagnostics and algebra. JSON retains IDs, sample indices and per-trial input digests. No old large per-trial JSON was loaded.

- **Trial 2, rank failure:** CPU original rays and CPU f32-quantized rays each produce 4 models; GPU produces 0, reports rank 4. On the exact GPU constraint matrix promoted to f64, smallest singular value is **0.0019073504269987102**; smallest squared-singular/trace ratio is **7.275971167240017e-7**, already below the unchanged **1e-6** numerical-rank cutoff. This sample is not evidence of merely erroneous f32 eigenvalue rounding: even the f64 ratio meets the rejection policy. Nor does mathematical nonzero rank establish numerical reliability.
- **Trial 1, roots/conditioning example:** CPU original and quantized rays each produce 6 models; GPU produces 1. With the **same GPU basis**, relative max elimination expansion error is **1.0312562381792655e-7**, solved-matrix error **1.4270079362852856e-4**, final coefficient error **0.22351172767151198** against f64 generated expansion + nalgebra partial-pivot LU. Using the **same GPU B** for f64 polynomial expansion still gives relative max coefficient error **0.22323883224565283**. This localizes substantial polynomial sensitivity/roundoff in this selected example; it does not prove which missing candidates it caused. Changing input precision alone did not change these CPU model counts, but equal counts are not candidate equivalence.
- Trial 3 has 4 CPU / 3 GPU models despite same-basis coefficient relative error only **9.672465315565732e-7**. A single global error magnitude does not explain all recovery loss.

The reference uses generated CPU algebra and nalgebra f64 LU, not a claim of bit-identical execution of the CPU solver's Eigen path. Rank failures have no valid exported GPU basis; none is fabricated. No root/recovery counterfactual or exhaustive causal proof was performed. Selection is illustrative, not a prevalence sample.

## Validation and reproduction

All actual commands used timeout **≤180 seconds**, `head_lines=15`, `tail_lines=20`; no timeout occurred. Run in the worktree above. New report paths must not already exist; the example refuses overwrites and refuses JSON ≥100,000 bytes.

```sh
cargo test -p rustsfm --release --no-default-features --features gpu-wgpu --example five_point_gpu_profile_replay --example five_point_gpu_capacity_replay --example five_point_gpu_replay --example five_point_replay_probe
cargo test -p rustsfm --release --no-default-features --features gpu-wgpu --lib five_point_f32_actual_gpu_stages -- --test-threads=1
cargo run -p rustsfm --release --no-default-features --features gpu-wgpu --example five_point_gpu_profile_replay -- --database /Users/tfjiang/Projects/RustScan/artifacts/runs/flowers2_960_settlement_20260913/matching.db --output artifacts/runs/five_point_gpu_profile_127x512_recheck.json
shasum -a 256 -c artifacts/runs/five_point_profile_preserved.sha256
git --no-pager diff --check
```

Focused example tests: **8 + 6 + 7 + 4 passed**. Library actual-stage test: **1 passed**. New profile example's actual-GPU test requires adapter creation (no skip), checks default feature behavior, both optional-query paths, result signature invariance, timestamp validity semantics, empty/invalid inputs, and rank/algebra overlap versus candidate counts. Existing library stage test covers valid synthetic/random full solves, same-basis f64 references, roots, failure slots and input bounds. No full production suite or full RANSAC was run. Existing unrelated dead-code warnings remain.

Artifacts (ignored `artifacts/runs/`, locally present):

- `five_point_gpu_profile_127x512_20260914_separate_resolve.json`: authoritative final measurement.
- `five_point_profile_measurement_separate_resolve.log`: final release run.
- `five_point_gpu_profile_127x512_20260914.json`: initial same-submission investigation, zero durations not trustworthy.
- `five_point_gpu_profile_127x512_20260914_final.json`: intermediate despite its filename; same-submission resolve, raw [0,0] evidence / null recovery samples; **not authoritative**.
- `five_point_profile_all_example_tests.log`, `five_point_profile_stage_tests.log`: focused tests before resolve-order isolation; final revalidation is recorded in `five_point_profile_final_tests.log` and `five_point_profile_final_stage_tests.log`.

## One next optimization suggestion (not implemented)

**Target only the roots pass's execution/workgroup layout in a separately approved experiment**, with unchanged iteration cap, numerical tests and a hard result-signature gate. It is the largest measured GPU pass (**119.464 ms** at full batch), whereas recovery is only 6.483 ms. This is an evidence-backed performance target, not a promise to fix missing candidates, and it does not justify weakening thresholds, treating all NotConverged slots as iteration failures, or deploying the solver.

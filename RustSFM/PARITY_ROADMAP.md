# RustSFM COLMAP Parity Roadmap (non-GUI)

Target: 100% COLMAP behavior parity excluding Qt GUI and Python bindings.
GPU: **wgpu** only (no CUDA/SiftGPU).

## Phase 0 — Parity harness (done)

- [x] Stage-based `compare` (`features`, `matches`, `twoview`, `registration`, `tracks`, `ba`)
- [x] flowers2 + flowers2_colmap fixtures
- [x] CI build matrix: default / `poselib` / `--no-default-features`

## Phase 1 — Sparse SfM core → 100% (in progress)

| Module | Est. | Next closure |
|--------|-----:|--------------|
| ObservationManager | 99% | initial-pair + structureless fixtures (added 2026-06-24) |
| IncrementalTriangulator | 99% | logic line-verified vs COLMAP (2026-06-27); len-2 gap is bit-exact-SVD only |
| Incremental mapper | 89% | exact stored two-view rows still choose a different initial pair; align initialization/registration order |
| optim/RANSAC | 90% | recorded FIFO verifier schedule now replays exactly; default parallel `random_seed=-1` remains schedule-realization dependent |
| Two-view geometry | 96% | recorded COLMAP verifier trace now matches config/inliers/masks exactly; default schedule replay remains diagnostic |
| Pose solvers | 65% | Ceres generalized refinement/covariance |
| BA orchestration | 70% | exact local/global BA scheduling and final-all pose drift after exact two-view propagation |
| BA/Ceres | 88% | large-scene numerical parity |

**Phase 1 exit:** flowers2 sparse model matches COLMAP within tolerance on poses, registration order, and point counts.

### flowers2 end-to-end measured parity (2026-06-27)

Measured against official COLMAP CPU on the 24-image `flowers2` set
(`rustsfm compare`, `--random-seed 0`):

| Stage | RustSFM vs COLMAP | Verdict |
|-------|-------------------|---------|
| Features | 118006/118006 keypoints, `pct_exact=1.0` | exact |
| Matches | 89/89 pairs, 0 missing/extra, 59954/59954 raw matches in fixed-match re-estimation | exact set |
| Two-view config | agreement 0.955, inlier-set overlap 0.954, 4/89 boundary mismatches | numerical boundary |
| Registration | 24/24 images, identical set | exact |
| Tracks | 6190 vs 6449 points (len2: 77 vs 240); obs 33489 vs 34234 | numerical |
| BA poses | rot err mean 0.054°/max 0.14°, trans RMSE 0.0067 (sim-aligned) | numerical-level |

**Verified root cause of the residual sparse-SfM gap (2026-06-27).** The
incremental triangulator (`Create` incl. recursive split, `CompleteImage`,
`Continue`, `Merge`, `Complete`, `Retriangulate`, the reprojection/angle/track
filters, all option defaults) and `estimate_triangulation` LORANSAC were
line-checked against COLMAP `sfm/incremental_triangulator.cc` and
`optim/loransac.h` and match. `classify_calibrated_two_view` matches
`EstimateCalibratedTwoViewGeometry`, and the E/F/H RANSAC control flow
(sampler, dynamic trials, LO, abort gate, support ordering) is COLMAP-shaped.
The remaining differences are **byte-level numerical divergence in the minimal
solvers / SVD** (`nalgebra` is not bit-for-bit Eigen): two-view `inlier_count
pct_exact` is only 2.2% while `inlier_set_overlap` is 95%, and one boundary
pair flips config with an identical final inlier count (62), i.e. only the
E-vs-F inlier ratio crosses the 0.95 threshold. The same marginal-inlier
sensitivity drives the len-2 track gap through the recursive-`Create`
`size - track_length >= 3` split gate. Closing this to exact COLMAP requires
bit-exact Eigen-equivalent linear algebra and remaining support-scoring details;
Eigen JacobiSVD/QR/companion-root bridges are now used when Eigen is available,
so the remaining work is exact minimal-solver behavior and COLMAP scoring edge
cases, not a high-level logic change.

**2026-06-28 update.** Added an optional Eigen bridge for right-nullspace/SVD
and companion-root paths, plus parity-only CLI hooks:
`match-features --use-existing-matches` refreshes `two_view_geometries` from
fixed raw matches, and `reconstruct --ignore-database-two-view-poses` forces the
mapper to re-estimate pair geometry instead of reusing stored `qvec/tvec`. The
two-view verifier default threshold is now aligned with COLMAP 3.13
`TwoViewGeometry.max_error = 4px` (instead of RustSFM's older 2px). On
`flowers2`, fixed-match two-view re-estimation against COLMAP now gives exact
features/matches (`118006/118006`, `59954/59954`),
`config_agreement=0.966`, `inlier_set_overlap_mean=0.993`,
`max_inlier_diff=4`, and 3 E/F boundary config mismatches:
`0002-0018`, `0005-0021`, `0006-0022`. With fixed `--random-seed 0`, the same
path reaches `config_agreement=0.978` with 2 boundary mismatches, which confirms
the remaining gap is dominated by exact COLMAP PRNG stream / minimal-solver
numerics rather than high-level classifier logic. The LORANSAC abort gate was
also rechecked against COLMAP and now fires after each candidate model support
evaluation; on this fixture it leaves the 4px numbers unchanged, so it is
source-shape parity rather than the remaining mismatch source. An experimental
COLMAP-like thread-local RANSAC stream (`RUSTSFM_COLMAP_SHARED_RANSAC_STREAM=1`)
improves the default parallel fixed-match run to `config_agreement=0.978`,
`max_inlier_diff=3`, 2 mismatches (`0002-0018`, `0005-0021`), but a forced
single-thread stream regresses to `config_agreement=0.955` with 4 mismatches;
therefore the exact residual gap still depends on COLMAP worker scheduling plus
minimal-solver numerics, and the stream mode remains an experiment rather than
the default.

**2026-06-29 trace update.** `debug-twoview --output-json` now includes
RANSAC trace diagnostics for E/F/H: best-support update history (trial,
candidate-model index, sample indices, raw/LO support, dynamic max trials),
termination reason, fallback flag, and the final residuals closest to the
threshold. The trace now also records LO-internal `local_updates` for each
best-support update, i.e. the local trial/model that actually changed the best
support. This is diagnostic-only; fixed raw-match verification is unchanged at
exact features/matches and `twoview config_agreement=0.966`,
`max_inlier_diff=4`, `inlier_overlap_mean=0.993`, 3 mismatches. On the two
current boundary pairs:
`0002-0018` selects E with `E=58/F=58`; the E winner arrives at trial 142
(`sample=[67,82,63,17,22]`, dynamic cap 544), while F reaches 58 after LO at
trial 610 via local updates `54->57->58` (dynamic cap 2183). The nearest E
residuals straddle the normalized
threshold (`idx=83/7` just inside, `idx=93` just outside), so this pair remains
a sample/solver boundary case rather than classifier logic.
`0005-0021` selects F with `E=46/F=49`; E's last improvement is trial 76
(`raw=44`, `LO=46` through local updates `44->45->46`, dynamic cap 1071),
F's last improvement is trial 1197 (`raw=48`, `LO=49` through local updates
`48->49->49`, dynamic cap 3582). F's nearest pixel residual (`idx=6`, 15.31
vs threshold 16) is an inlier with margin ~0.69px², which makes the
remaining 48/49 flip sensitive to COLMAP's exact seven-point/root/SVD behavior
and sampling stream.

**2026-06-29 COLMAP probe update.** Added an offline COLMAP 3.13 trace probe at
`tools/colmap_trace_probe/` that links Homebrew COLMAP estimators and emits the
same E/F/H LORANSAC best-update and LO-update JSON shape. With `--random-seed 0`
on the fixed raw-match DB, COLMAP and RustSFM agree on the final supports and
winner samples for the active boundary pairs: `0002-0018` is `E=56/F=61`
selecting F at F trial 272, and `0005-0021` is `E=46/F=49` selecting F at F
trial 3. Residuals differ only at floating tail precision. Intermediate
best-update histories still diverge in candidate-model indices / residual-sum
ties, so this is not byte-level trace parity, but it proves the current
high-impact mismatch is not the fixed-seed solver/control flow. The remaining
default-run gap is now narrowed to COLMAP's batch verifier behavior:
worker-local PRNG streams start at seed 0 and are consumed continuously across
multiple queued pairs when `random_seed=-1`, whereas RustSFM's default path still
uses per-pair/model derived seeds unless the experimental shared-stream mode is
enabled. Next closure step: make fixed-match re-estimation use a COLMAP-like
FIFO verifier worker pool with worker-local shared PRNG streams, then compare
against the reference DB before revisiting minimal-solver byte ordering.

**2026-06-29 FIFO verifier experiment.** Added an opt-in fixed-match verifier
path gated by `RUSTSFM_COLMAP_FIFO_VERIFIER=1` and
`RUSTSFM_COLMAP_SHARED_RANSAC_STREAM=1`. It uses a FIFO job queue,
worker-local shared RANSAC streams, COLMAP's existing-match batch size default
(`1000`), controller-style batch push/output collection, and a verifier early
gate on `min_inliers` rather than RustSFM's normal `min_num_matches`. Failed
existing-match verification now writes the same default empty two-view geometry
row shape as COLMAP instead of deleting/omitting the pair. The `match-features`
JSON report can record optional verifier scheduling trace events (worker id,
dequeue order, completion order, pair, config, inliers).
On the fixed raw-match `flowers2` DB, the FIFO path now reads matches in
`read_num_matches()` order and matches COLMAP's stored pair order on the test
fixture; trace on/off does not change the result. The current best measured run
is `config_agreement=0.978`, `max_inlier_diff=3`,
`inlier_overlap_mean=0.994`, 2 mismatches (`0004-0020`, `0006-0022`), with
4 threads and the same result under the default thread count. This means the
remaining gap has narrowed to verifier worker scheduling/PRNG assignment and/or
minimal-solver numerical boundary behavior, rather than pair enumeration, batch
count, failed-row handling, or score thresholds.

**2026-06-29 official verifier repeatability probe.** RustSFM's database layer
now keeps a nullable `pose_priors.image_id` compatibility column, synced for
camera pose priors, so the Homebrew COLMAP 3.13 `geometric_verifier` binary can
open the fixed raw-match DB without a manual `ALTER TABLE`. This only unblocks
empty-pose-prior verifier/open compatibility with older COLMAP SQL
(`SELECT ... WHERE image_id = ?`); it is not full legacy pose-prior read/write
interop for non-empty pose-prior tables because newer sensor pose priors have a
different schema. Re-running official `colmap geometric_verifier` four times on
the same `flowers2` raw-match DB with `num_threads=4`, `batch_size=1000`, and
`TwoViewGeometry.random_seed=-1` produced COLMAP-side `twoview` mismatch counts
`2/3/4/3` against the stored reference DB and total inlier-row sums
`56495/56485/56496/56485`. The flipping boundary pairs include
`0004-0020`, `0002-0018`, `0006-0022`, and `0001-0017`. This confirms the
stored COLMAP reference is one verifier schedule realization, not a unique
deterministic target under default parallel scheduling. For first-tier parity,
the strong target is exact features/raw matches plus fixed-seed two-view
agreement; default `random_seed=-1` parallel verification should be judged by
bounded boundary drift unless a recorded COLMAP worker/PRNG schedule is replayed.

**2026-06-29 schedule replay update.** Added diagnostic replay for fixed raw-match
verification through `RUSTSFM_COLMAP_FIFO_VERIFIER_REPLAY_TRACE=/path/to/trace.json`.
The replay path consumes a COLMAP verifier trace (`events[]`) or RustSFM
`match-features` report (`verifier_trace.events[]`), groups jobs by recorded
`worker_id`, runs each worker's recorded `dequeue_order` sequence on one thread,
and emits `verifier_trace.mode = colmap_fifo_shared_ransac_stream_replay`.
Replaying `colmap_verifier_trace_run1.json` on `flowers2` gave exact
features/raw matches and matched the recorded COLMAP trace on all two-view
configs (`0` config mismatches versus that trace). The remaining trace-level
delta was only 10 non-boundary pairs with `1-2` inlier count difference
(`max_abs_inlier_diff=2`, total inlier delta `-1`), while the active boundary
pairs `0004-0020` and `0006-0022` matched the replayed COLMAP trace exactly in
config and inlier count. This moves the residual default-run gap from verifier
scheduling/PRNG assignment to solver/support-scoring tail parity.

**2026-06-30 five-point Eigen closure.** Extended the Eigen-only
`colmap_eigen` bridge to cover the remaining COLMAP 3.13 five-point essential
solver hotspots without linking COLMAP's static estimator libraries:
5-sample nullspace now uses `fullPivHouseholderQr().matrixQ().rightCols<4>()`,
the 10x10 elimination solve uses `partialPivLu().solve(...)`, and the per-root
`Bz` null vector uses `JacobiSVD(..., ComputeFullV).matrixV().rightCols<1>()`.
The bridge stays optional and falls back to the Rust/nalgebra path when Eigen is
unavailable. With the same run7 COLMAP verifier trace plus emitted E/F/H model
matrices (`--include-models`) and the enhanced `twoview-trace-diff` residual
diagnostics, replaying `colmap_verifier_trace_models_run7.json` on the fixed
raw-match `flowers2` DB improved from the previous seven-point-only bridge
baseline (`config_mismatch_count=0`, `inlier_mismatch_count=7`,
`mask_mismatch_count=10`, `max_abs_inlier_delta=3`, `total_inlier_delta=-5`) to
exact trace parity: `config_mismatch_count=0`, `inlier_mismatch_count=0`,
`mask_mismatch_count=0`, `max_abs_inlier_delta=0`, `total_inlier_delta=0`.
This closes the recorded two-view verifier path for first-tier parity; remaining
first-tier work should now focus on replaying/controlling default COLMAP worker
schedule realizations and measuring how exact two-view rows propagate through
tracks, triangulation, mapper registration order, and BA.

**2026-06-30 exact two-view propagation update.** Reconstructed `flowers2` from
the exact run7 replay DB
(`output/rustsfm_flowers2_fifo_models_replay_run7_fivepoint_eigen/database.db`)
without re-estimating pair poses. Runtime was `411360ms` total:
`timing_extract_ms=34100`, `timing_pairs_ms=443`, and
`timing_incremental_ms=376451`. Pair ingestion stayed exact
(`pair_config CALIBRATED=80`, `F/E/H/qvec/tvec/pose=80`), and registration set
matched COLMAP (`ref=24`, `cand=24`, `common=24`, no missing/extra images).
The sparse model still diverges after mapper initialization: RustSFM's strict
and relaxed-min-inlier initialization stages found no pair, then the relaxed
tri-angle stage initialized at `frame_0005.jpg -> frame_0013.jpg`
(`159` inliers/triangulated). Compared with the COLMAP sparse reference,
tracks are `6761` vs `6449` points, `35490` vs `34234` observations,
mean track length `5.249` vs `5.308`, mean point error `0.710` vs `0.639`,
and the histogram shifts from COLMAP `len2=240,len3_4=3388,len5_9=2182,len10+=639`
to RustSFM `len2=129,len3_4=3728,len5_9=2272,len10+=632`. Similarity-aligned
BA pose error is now much larger than the earlier non-exact-two-view baseline:
translation RMSE `0.03125`, rotation mean/RMSE/max `2.140/2.451/4.052 deg`,
adjacent relative rotation mean `0.356 deg`, and adjacent translation-angle mean
`0.262 deg` with similarity scale `0.948947`. This pins the next first-tier gap
on mapper initialization order, track creation/completion/filtering under that
order, and BA scheduling/normalization, not on pair verification.

**2026-06-30 sparse binary compatibility note.** While running the above
`tracks`/`ba` comparison, `compare` exposed an import bug in COLMAP
`points3D.bin`: binary track elements store `IMAGE_ID` and `POINT2D_IDX` as
32-bit unsigned integers, while RustSFM was reading/writing `POINT2D_IDX` as
64-bit. `read_points3d_bin` now reads the index as `u32` and widens internally,
`write_raw_points3d_bin` writes COLMAP-compatible `u32` with an overflow check,
and the binary-points3D unit fixture covers the format.

## Phase 2 — Feature pipeline (in progress)

- [x] `extract-features` CLI (COLMAP `feature_extractor` subset)
- [x] `compare-extract` CLI (fresh VLFeat vs reference database keypoints)
- [x] COLMAP FreeImage grayscale loader (`colmap_image.rs`)
- [x] Per-DOG-level feature limiting (match COLMAP `sift.cc`)
- [x] flowers2 SIFT keypoint counts match COLMAP CPU (`pct_exact=1.0`, 118006/118006)
- [ ] **wgpu** `use_gpu` SIFT backend
- [ ] Matching: FAISS IVF CPU + **wgpu** brute-force/IVF
- [ ] Guided matching end-to-end parity

## Phase 3 — Retrieval

- [x] classical vocab-tree build/query (`retrieval.rs`: hierarchical k-means +
      TF-IDF inverted index + BoW cosine query, deterministic MT19937)
- [x] `build_vocab_tree_pairs` / `generate_vocab_tree_pairs` candidate pairing
- [x] wire `vocab_tree` strategy into `MatchFeatures`/mapper pair generation + CLI
      (`MatchingPairStrategy::VocabTree`, `--matching-strategy vocab-tree`,
      `--vocab-tree-num-images`, descriptor-aware `vocab_tree_pairs_from_frames`)
- [x] on-disk vocab-tree (de)serialization (`VocabTree::save`/`load`, JSON)
- [ ] FAISS IVF index + Hamming embedding + vote-and-verify (COLMAP `VisualIndex`)

## Phase 4 — Scene semantics

- `f64` internal storage or lossless export
- pose prior + rig refinement full pipeline

## Phase 5 — Global mapper / GLOMAP

- [x] global rotation averaging core (`rotation_averaging.rs`: maximum-spanning-tree
      init + IRLS Gauss-Newton on so(3) + Huber weighting + `PairGeometry` adapter,
      deterministic, 5 tests)
- [x] global positioning / translation averaging (`global_positioning.rs`:
      BATA-style alternating center/depth solver + Huber IRLS + scale gauge,
      deterministic, 5 tests)
- [x] global mapper orchestrator (`global_mapper.rs`: view graph → rotation
      averaging → global positioning → assembled per-view `SE3` poses, 3 tests)
- [x] track establishment (`track_establishment.rs`: union-find multi-view track
      fusion + same-image conflict rejection + length filtering, deterministic, 6 tests)
- [x] track triangulation from global poses (`track_triangulation.rs`: multi-view
      DLT + angle/cheirality/reprojection filters, 2 tests)
- [x] `run_global_reconstruction` → sparse `Reconstruction` + optional global BA
- [x] joint camera+point global positioning (`joint_global_positioning.rs`: GLOMAP
      ray constraints; **Ceres LM** default via `joint_global_positioning_ceres.rs`,
      alternating fallback; 6 tests)
- [x] multi-pass global BA + reprojection filtering (`GlobalRefinementOptions`
      in `global_mapper.rs`, wired from `MapperConfig::global_ba_max_refinements`)
- [x] retriangulation between global-BA rounds (`IncrementalTriangulator`
      complete/merge/retriangulate in `run_iterative_global_refinement`, 1 test)
- [x] view-graph calibration (`view_graph_calibration.rs`: match filtering,
      optional focal refine + pose re-estimation, rotation-consistency filter;
      wired before rotation averaging in `run_global_reconstruction`, 2 tests)
- [x] connected-component splitting (`view_graph_splitting.rs`: covisibility
      graph union-find, per-component remap + `run_global_reconstructions`, wired
      from `MapperConfig::multiple_models`, 4 tests)
- [x] expose as a selectable global mapper pipeline in `mapper`/CLI (`--global-mapper`
      on `reconstruct`; skips legacy `pose_graph` post-processing when active)
- [ ] deprecate/replace experimental `pose_graph.rs` heuristic once global mapper is validated on real data

## Phase 6 — Dense MVS (wgpu)

- workspace + undistortion
- **wgpu** PatchMatch stereo
- fusion + meshing

## Phase 7 — CLI / controllers / tools

- Mirror COLMAP `exe` commands (`feature_extractor`, `mapper`, `patch_match_stereo`, …)
- Controller orchestration for full pipeline

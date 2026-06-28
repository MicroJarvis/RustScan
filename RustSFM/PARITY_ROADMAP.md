# RustSFM COLMAP Parity Roadmap (non-GUI)

Target: 100% COLMAP behavior parity excluding Qt GUI and Python bindings.
GPU: **wgpu** only (no CUDA/SiftGPU).

## Phase 0 — Parity harness (done)

- [x] Stage-based `compare` (`features`, `matches`, `twoview`, `registration`, `tracks`, `ba`)
- [x] flowers2 + flowers2_colmap fixtures
- [ ] CI matrix: default / `poselib` / `--no-default-features`

## Phase 1 — Sparse SfM core → 100% (in progress)

| Module | Est. | Next closure |
|--------|-----:|--------------|
| ObservationManager | 99% | initial-pair + structureless fixtures (added 2026-06-24) |
| IncrementalTriangulator | 99% | logic line-verified vs COLMAP (2026-06-27); len-2 gap is bit-exact-SVD only |
| Incremental mapper | 89% | reconstruction-manager retries, byte-level pose RANSAC |
| optim/RANSAC | 82% | **bit-exact Eigen SVD / 5-pt eigensolve** (root cause of two-view + track residual), CHOLMOD sparse |
| Two-view geometry | 80% | classifier verified vs COLMAP; 5 boundary pairs need bit-exact E inlier counts |
| Pose solvers | 65% | Ceres generalized refinement/covariance |
| BA orchestration | 70% | exact local bundle selection, final-all fixtures |
| BA/Ceres | 88% | large-scene numerical parity |

**Phase 1 exit:** flowers2 sparse model matches COLMAP within tolerance on poses, registration order, and point counts.

### flowers2 end-to-end measured parity (2026-06-27)

Measured against official COLMAP CPU on the 24-image `flowers2` set
(`rustsfm compare`, `--random-seed 0`):

| Stage | RustSFM vs COLMAP | Verdict |
|-------|-------------------|---------|
| Features | 118006/118006 keypoints, `pct_exact=1.0` | exact |
| Matches | 89/89 pairs, 0 missing/extra | exact set |
| Two-view config | agreement 0.943, inlier-set overlap 0.952, 5/89 boundary mismatches | numerical boundary |
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
bit-exact Eigen-equivalent linear algebra (JacobiSVD, 5-point companion
eigensolve, RMS point normalization), not a logic change.

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

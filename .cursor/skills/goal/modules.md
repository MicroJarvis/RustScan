# COLMAP 模块复刻清单

与 `RustSFM/COLMAP_MODULE_PARITY.md` 同步。完成模块后更新本文件状态列。

**图例**：✅ 100% · 🟡 partial · ❌ 未开始

## 推荐复刻顺序

依赖顺序：底层数据/几何 → 特征/估计 → SfM 核心 → BA → 控制器 → 长期模块。

| 序 | COLMAP 模块 | RustSFM | 状态 | 进度 | 100% boundary 说明 |
|----|-------------|---------|------|------|-------------------|
| — | Sparse model I/O | `colmap.rs` | ✅ | 100% | text/binary cameras/images/points3D/rigs/frames codec |
| — | Database + CorrespondenceGraph | `database.rs`, `correspondence_graph.rs` | ✅ | 100% | SQLite schema/cache/graph API |
| — | geometry/triangulation 原语 | `triangulation.rs` | ✅ | 100% | DLT/midpoint/multi-view/optimal/angles |
| 1 | optim BA / Ceres | `ba/` | 🟡 | ~72% | quaternion manifold、完整 iteration table 解析 |
| 2 | sfm triangulation + filtering | `incremental_triangulator.rs`, `triangulation_estimator.rs`, `visibility_pyramid.rs` | ✅ | 100% | TriangulateImage/CompleteImage, seeded CombinationSampler |
| 3 | sfm observation manager | `observation_manager.rs` | ✅ | 100% | incremental event paths, VisibilityPyramid, embedded graph |
| 4 | sfm incremental mapper | `mapper.rs` | 🟡 | ~85% | graph-based registration, separate structureless trials |
| 5 | estimators two-view + RANSAC | `two_view.rs`, `optim` | 🟡 | ~62–80% | LORANSAC sampler、CALIBRATED_RIG |
| 6 | absolute/generalized pose | `generalized_pose.rs`, PoseLib | 🟡 | ~63% | Ceres refinement、covariance |
| 7 | scene/sensor 语义 | `types.rs`, `mapper.rs` | 🟡 | ~63% | reconstruction manager、pose prior |
| 8 | feature SIFT | `sift.rs` | 🟡 | ~30% | VLFeat-equivalent 后端 |
| 9 | feature matching | `feature_matching.rs` | 🟡 | ~45% | vocab-tree、FAISS/GPU |
| 10 | controllers / pipeline | `main.rs`, `parity.rs` | 🟡 | ~42% | 全 pipeline parity harness |
| 11 | sfm global mapper | `pose_graph.rs` | 🟡 | ~12% | COLMAP GlobalMapper 编排 |
| 12 | retrieval | — | ❌ | 0% | vocab-tree |
| 13 | mvs | — | ❌ | 0% | PatchMatch/fusion/meshing |
| 14 | ui / exe / tools | `main.rs` CLI only | ❌ | ~5% | GUI/tools |

## COLMAP 源码树 → RustSFM 速查

| COLMAP | RustSFM |
|--------|---------|
| `src/colmap/scene`, `sensor` | `types.rs`, `colmap.rs`, `observation_manager.rs` |
| `src/colmap/feature` | `sift.rs`, `wide.rs`, `feature_matching.rs` |
| `src/colmap/estimators`, `geometry` | `two_view.rs`, `five_point.rs`, `generalized_pose.rs`, `triangulation.rs`, `triangulation_estimator.rs` |
| `src/colmap/optim` | `ba/`, RANSAC in `two_view.rs` / `mapper.rs` |
| `src/colmap/sfm` | `mapper.rs`, `incremental_triangulator.rs`, `observation_manager.rs` |
| `src/colmap/controllers` | `main.rs`, `mapper.rs` |
| `src/colmap/retrieval` | — |
| `src/colmap/mvs` | — |

## 测试基线（更新于 2026-06-20）

```bash
cd RustSFM && cargo test --lib                    # 304 (ceres-ba default)
cd RustSFM && cargo test --lib --features poselib # 309
cd RustSFM && cargo test --lib --no-default-features # 301
```

PoseLib 为 optional feature；启用后 COLMAP structureless 注册走 PoseLib GR6P/GR8P，无需 `--experimental-structureless-pair-pose-fallback`。

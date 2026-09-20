# RustSLAM

RustSLAM is the workspace's pure-Rust visual SLAM crate. It contains feature
extraction and matching, visual odometry, mapping, loop-closing primitives,
bundle-adjustment support, video input, and optional fusion/mesh utilities.

RustSFM is the preferred COLMAP-style sparse reconstruction path for the
current RustViewer workflow. RustSLAM remains available for SLAM-specific
experiments and pipeline integration.

## Modules

- `core`: frames, keyframes, map points, maps, cameras, and poses.
- `features`: ORB, Harris/FAST, descriptors, and feature matching.
- `tracker`: visual odometry and pose/triangulation solvers.
- `mapping` and `optimizer`: local mapping and bundle-adjustment support.
- `loop_closing`: vocabulary, loop detection, and relocalization components.
- `fusion`: Gaussian, TSDF, marching-cubes, and mesh export utilities.
- `pipeline` and `cli`: checkpoints, realtime orchestration, and the optional
  command-line pipeline.

The source tree and feature gates are the authority for implementation status;
this README intentionally avoids dated completion percentages and test counts.

## Features

- `default = ["slam-pipeline"]`: enables the video/CLI pipeline dependencies.
- `slam-pipeline`: enables the CLI, video decoding, frame cache, and system
  information support.
- `viewer-types`: enables the lightweight types consumed by RustViewer.
- `opencv`: enables the optional OpenCV integration.
- `deep-learning`: enables the optional `tch` integration.
- `image`: enables the optional image-loading dependency.

## Build And Test

From the workspace root:

```bash
cargo build -p rustslam
cargo test -p rustslam --lib
```

For the library without the default video/CLI dependencies:

```bash
cargo check -p rustslam --no-default-features --lib
cargo test -p rustslam --no-default-features --lib
```

On 2026-08-15, the dependency-minimal library suite completed with 244 passing
tests and one known failure:
`tracker::vo::tests::test_initialize_keeps_relocalized_pose_in_global_frame`.
Treat the command above as a regression check until that failure is resolved.

The source example `src/examples/run_vo.rs` is an internal example module,
not a Cargo `--example` target. Supported Cargo examples include:

```bash
cargo run -p rustslam --example load_tum_dataset --features image -- /path/to/tum/dataset
cargo run -p rustslam --example e2e_slam_to_mesh --features image
```

## Design And Current Status

- [Design notes](DESIGN.md)
- [Workspace documentation index](../docs/index.md)
- [Current project status](../docs/current-project-status.md)

The current cross-crate status, external fixtures, and verification dates live
in the workspace documents above rather than in this crate overview.

## License

MIT License - see the repository license file.

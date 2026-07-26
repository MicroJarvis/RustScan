# RustViewer

Interactive 3D visualization GUI for RustScan SLAM results.

## Overview

RustViewer is a desktop application for visualizing SLAM reconstruction results, including:

- **Camera trajectory** — Polylines showing camera motion path
- **Sparse point cloud** — Map points from SLAM reconstruction
- **Gaussian point cloud** — 3DGS scene from training
- **Mesh** — Extracted mesh with solid/wireframe rendering

## Reconstruction Workflow

1. Select **Import Images** and choose a folder with at least two JPEG or PNG images.
2. Select **Run Reconstruction**. Each RustSFM result is written below the input folder at `<input>/.rustviewer/rustsfm/<run-id>/`; older runs are never overwritten.
3. RustSFM uses automatic single-camera intrinsics. After it produces a valid COLMAP sparse model, RustViewer loads it, starts RustGS 3DGS training automatically, and displays training snapshots.

Runtime boundaries:

- On macOS, wgpu uses Metal rather than native Vulkan.
- This release cannot cancel a running RustSFM reconstruction.
- Existing RustGS training cancellation remains available.

## Features

- **Offline file loading** — Load `pipeline.json`, `scene.ply`, `mesh.obj/ply`
- **3D navigation** — Arcball camera with mouse orbit/pan/zoom
- **Layer visibility** — Toggle trajectory, points, Gaussians, mesh
- **Apple HIG design** — Clean, native-feeling UI

## Reconstruction Workflow

RustViewer can create a managed `.rustscanproject` from an image sequence. The workbench then
runs RustSFM keyframe reconstruction and full-frame registration through the persistent project
pipeline. It writes the verified COLMAP sparse model and normalized images into the committed
FullFramePnP artifact. RustGS training is enabled only after every imported image has a pose; its
latest Gaussian snapshot is rendered in the RustViewer wgpu viewport.

The desktop workbench currently accepts image sequences only. Video import and video PnP are
intentionally disabled in the UI until the dedicated video reconstruction path has the same
complete-pose and artifact-validation guarantees.

## Architecture

```
RustViewer/
├── Cargo.toml              # Crate config with eframe/egui deps
├── src/
│   ├── main.rs             # Binary entry point
│   ├── lib.rs              # Library root, module declarations
│   ├── app.rs              # Main eframe app struct
│   ├── reconstruction.rs   # RustSFM runner and output validation
│   ├── loader/             # File loading utilities
│   │   ├── mod.rs          # Loader trait and exports
│   │   ├── checkpoint.rs   # Checkpoint JSON loader
│   │   ├── gaussian.rs     # Gaussian PLY loader
│   │   └── mesh.rs         # OBJ/PLY mesh loader
│   ├── renderer/           # 3D rendering
│   │   ├── mod.rs          # Renderer trait and scene graph
│   │   ├── camera.rs       # Arcball camera controller
│   │   ├── scene.rs        # Scene graph and data buffers
│   │   └── pipelines.rs    # wgpu render pipelines
│   └── ui/                 # User interface
│       ├── mod.rs          # UI panel exports
│       ├── panel.rs        # Side panel with controls
│       ├── viewport.rs     # 3D viewport widget
│       └── theme.rs        # egui theme/styling
└── tests/                  # Integration tests
```

## Usage

```bash
# Run the viewer
cargo run -p rust-viewer

# Build release
cargo build -p rust-viewer --release
```

## Dependencies

- **eframe/egui** — Immediate mode GUI framework
- **wgpu** — Cross-platform GPU rendering (via eframe)
- **glam** — SIMD-accelerated math library
- **rustslam** — SLAM library with `viewer-types` feature
- **rustsfm** — Image pose solving and COLMAP sparse-model export
- **rustgs** — 3D Gaussian Splatting training

## Notes

- RustViewer uses `viewer-types` feature from RustSLAM to avoid heavy dependencies (ffmpeg, candle)
- GPU rendering is handled through eframe's wgpu integration
- The crate compiles without the `--features default` flag and still functions correctly

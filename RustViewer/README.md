# RustViewer

RustViewer is the desktop reconstruction workbench and 3D result viewer for
RustScan. It manages image-sequence projects, orchestrates RustSFM and RustGS,
and displays sparse points, trained splats, meshes, and camera/navigation state.

## Current Workflow

1. **Import Images** creates a managed `.rustscanproject` from image files.
2. **Run Reconstruction** executes keyframe RustSFM followed by full-frame pose
   registration through the persistent project pipeline.
3. A successful pose stage publishes a validated COLMAP sparse model and
   normalized image artifacts.
4. RustViewer loads the committed COLMAP output, starts RustGS training, and
   displays snapshots in the wgpu viewport.

Image sequences are the supported reconstruction input in the current UI.
Video ingestion/project support exists in the media layer, but video pose solve
remains explicitly unavailable in the workbench until it has complete pose and
artifact-validation coverage.

The viewer can also open existing `.rustscanproject` packages and offline
COLMAP, checkpoint, Gaussian (`.ply`/`.splat`), and mesh (`.obj`/`.ply`) assets.

## Module Boundaries

- `app.rs`: application state, dialogs, project activation, and action routing.
- `project/`: manifest, artifact, project-store, library, and session state.
- `pipeline/`: persistent stage coordinator and RustSFM worker boundaries.
- `media/`: image and video ingestion plus keyframe selection.
- `loader/`: COLMAP, checkpoint, Gaussian, and mesh loading.
- `training/`: RustGS session, preview, and shared GPU viewport integration.
- `renderer/`: scene, camera, and wgpu render pipelines.
- `ui/`: workbench, panels, viewport, and theme.

## Run And Verify

From the workspace root:

```bash
cargo run -p rust-viewer
cargo build -p rust-viewer --release
cargo test -p rust-viewer --all-targets
```

On macOS, the wgpu stack uses the Metal adapter selected by eframe/wgpu. The
RustGS and RustSFM crate documentation owns their backend and fixture
prerequisites.

## Related Docs

- [Workspace documentation index](../docs/index.md)
- [Current project status](../docs/current-project-status.md)
- [RustSFM overview](../RustSFM/README.md)
- [RustGS overview](../RustGS/README.md)

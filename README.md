# RustScan

<p align="center">
  <img src="https://img.shields.io/badge/Rust-1.75+-dea584?style=for-the-badge&logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Status-Active%20Development-blue?style=for-the-badge" alt="Status">
</p>

RustScan is a Rust workspace for 3D reconstruction tooling: visual SLAM, Gaussian splatting, mesh extraction, mesh processing, and visualization.

This README is intentionally brief. Current status lives in a small set of canonical documents so the repo does not keep drifting copies of the same state.

## Workspace

- `rustscan-types`: shared data structures used across crates.
- `RustSLAM`: visual SLAM, sparse mapping, loop closing, video IO, and mesh extraction.
- `RustGS`: Gaussian splatting training and rendering.
- `RustMesh`: mesh connectivity, IO, processing algorithms, OpenMesh comparison tooling.
- `RustViewer`: visualization and inspection UI.
- `RustFF`: feed-forward reconstruction experiments.
- `RustSFM`: COLMAP-style incremental structure-from-motion.

## Current Verification

RustSFM was verified in this workspace on 2026-08-14:

- Default RustSFM library suite: `709 passed; 0 failed; 19 ignored external-fixture tests`
- Minimal RustSFM library suite: `581 passed; 0 failed; 19 ignored external-fixture tests`
- Minimal `sequence_registration` integration suite: `62 passed; 0 failed`

The ignored `real_colmap_sparse_*` tests require the external
`test_data/flowers2_colmap` fixture, which is not distributed through Git or
submodules. See the RustSFM README for the explicit parity command.

## Documentation

- Workspace overview: [`docs/index.md`](./docs/index.md)
- Project summary: [`docs/project-overview.md`](./docs/project-overview.md)
- RustMesh crate overview: [`RustMesh/README.md`](./RustMesh/README.md)
- RustSFM crate overview: [`RustSFM/README.md`](./RustSFM/README.md)
- RustSFM COLMAP parity roadmap: [`RustSFM/PARITY_ROADMAP.md`](./RustSFM/PARITY_ROADMAP.md)
- RustMesh `rm-opt` status: [`docs/RustMesh-OpenMesh-Progress-2026-04-05.md`](./docs/RustMesh-OpenMesh-Progress-2026-04-05.md)
- Forward roadmap: [`ROADMAP.md`](./ROADMAP.md)

## Getting Started

```bash
# Build the workspace
cargo build --release

# RustMesh
cargo test --manifest-path RustMesh/Cargo.toml --lib

# RustSLAM
cargo test --manifest-path RustSLAM/Cargo.toml --lib

# RustSFM
cargo test -p rustsfm --lib
cargo test -p rustsfm --lib --no-default-features
```

## Notes

- The compatibility entry points under `docs/README.md`, `docs/RustMesh-README.md`, and `docs/ROADMAP.md` are intentionally thin wrappers around the canonical docs above.
- For branch-specific OpenMesh parity work, use the progress and roadmap docs instead of older planning artifacts.

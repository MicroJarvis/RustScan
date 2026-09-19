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

The maintained verification snapshot, dates, fixtures, feature flags, and
limitations are recorded in
[`docs/current-project-status.md`](./docs/current-project-status.md). This
README intentionally does not duplicate dated test counts.

## Documentation

- Workspace overview: [`docs/index.md`](./docs/index.md)
- Current project status: [`docs/current-project-status.md`](./docs/current-project-status.md)
- Workspace architecture: [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md)
- RustMesh crate overview: [`RustMesh/README.md`](./RustMesh/README.md)
- RustSLAM crate overview: [`RustSLAM/README.md`](./RustSLAM/README.md)
- RustSFM crate overview: [`RustSFM/README.md`](./RustSFM/README.md)
- RustSFM COLMAP parity roadmap: [`RustSFM/PARITY_ROADMAP.md`](./RustSFM/PARITY_ROADMAP.md)
- RustGS crate overview: [`RustGS/README.md`](./RustGS/README.md)
- RustViewer crate overview: [`RustViewer/README.md`](./RustViewer/README.md)
- RustFF experiment overview: [`RustFF/README.md`](./RustFF/README.md)
- Forward roadmap: [`ROADMAP.md`](./ROADMAP.md)

## Getting Started

```bash
# Build the workspace
cargo build --release

# RustMesh
cargo test --manifest-path RustMesh/Cargo.toml --lib

# RustSLAM
cargo test --manifest-path RustSLAM/Cargo.toml --lib

# RustSFM. --no-default-features is a compile gate, not a matching pipeline test.
cargo test -p rustsfm --lib --features gpu-wgpu,vlfeat-sift
cargo check -p rustsfm --no-default-features --all-targets
```

## Notes

- The documents above are the maintained entry points. Dated plans and review records
  under `docs/plans/`, `docs/reviews/`, and `docs/superpowers/` are historical context,
  not current API or status contracts.

# RustMesh

RustMesh is a mesh processing crate in pure Rust, inspired by OpenMesh and built around a half-edge connectivity core plus SoA-oriented storage.

This README is the canonical RustMesh overview for the current workspace. The
workspace status index is [`../docs/index.md`](../docs/index.md).

## Verified Status

Validated on 2026-08-15 in the current workspace:

- `cargo test -p rustmesh --lib --quiet`: `210 passed; 0 failed`

The focused commands below are the source of truth for individual algorithm
areas and should be rerun when those areas change.

## Core Capabilities

### Data Structures

- Half-edge mesh connectivity
- Smart typed handles for vertices, halfedges, edges, and faces
- SoA and attribute-aware kernels
- Status flags and smart range iteration helpers

### IO

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| OBJ | Yes | Yes | normals, texcoords, colors |
| PLY | Yes | Yes | ASCII and binary |
| STL | Yes | Yes | ASCII and binary |
| OFF | Yes | Yes | includes `read_off_openmesh_parity()` helper |

### Algorithms

| Area | Status | Notes |
|------|--------|-------|
| Decimation | Implemented | quadric decimation plus OpenMesh comparison tooling |
| Decimation modules | Implemented | modular constraints including quadric, normal, aspect ratio, boundary |
| Smoothing | Implemented | uniform and tangential paths |
| Subdivision | Implemented | Loop, Catmull-Clark, sqrt3, midpoint, butterfly |
| Hole filling | Implemented | mesh repair support |
| Mesh repair | Implemented | topology cleanup utilities |
| Dualization | Implemented | includes boundary-aware dualization |
| Analysis | Implemented | curvature, quality, area, volume, edge-length stats |
| Circulators | Implemented | vertex/face/edge plus HH and EE circulators |
| Remeshing | Implemented and regression-covered on the shared primitive path | split/collapse/flip/valence/isotropic remesh are present; representative acceptance tests now validate topology and long-edge threshold behavior |
| Progressive mesh | Partial | simplify, exact-record refine, reset, `simplification_progress()`, vertex split, and `get_lod(level)` exist; `get_lod(level)` still resets to `original` instead of navigating incrementally from current state |

## OpenMesh Comparison

RustMesh does not claim whole-crate feature or numerical parity with OpenMesh.
The comparison examples provide reproducible, area-specific diagnostics. The
decimation trace defaults to `OpenMeshParity` import mode; `standard` remains
available for debugging different import semantics.

Use these examples for comparison work:

```bash
cargo run --release --example openmesh_compare_decimation
cargo run --release --example openmesh_compare_decimation_trace -- 10
env RUSTFLAGS=-Awarnings cargo run --manifest-path RustMesh/Cargo.toml --release --example openmesh_compare_normals --quiet
cargo run --release --example openmesh_compare_smoothing
cargo run --release --example openmesh_compare_io
```

## Current Gaps

- The OpenMesh comparison examples are diagnostic tools; they do not imply whole-crate parity.
- `AttribSoAKernel` dynamic properties now have typed per-entity handles, automatic resize, supported PLY round-trips for `f32`, `i32`, and `Vec3`, and deterministic propagation on the maintained `collapse` / `split_edge` / triangle `split_face` path; the remaining scope decision is whether rebuild-backed n-gon fallbacks should gain the same propagation contract, while `Vec2` / `Vec4` persistence still fails explicitly.
- `split_edge()` and triangle `split_face()` now use local half-edge surgery on the maintained topology path, while `triangulate_face()` and non-triangle `split_face()` still use controlled rebuild-backed baselines.
- Remeshing acceptance on the shared split/collapse/flip path is now regression-covered; the remaining topology gap is whether non-triangle `split_face()` / `triangulate_face()` should stay rebuild-backed or gain deeper local surgery.
- Vertex-normal semantics and refresh policy are now explicit: RustMesh defaults to area-weighted accumulation, `VertexNormalWeighting::FaceAverage` provides an OpenMesh-compatible equal-face-weight path, maintained topology edits do not auto-refresh normals, and rebuild-backed topology paths drop face-normal storage until explicit refresh; the remaining normals gap is durable comparison coverage rather than raw speed.
- Progressive mesh now exposes exact refine / `vertex_split()` replay records plus monotonic LOD regression coverage, but `get_lod(level)` still resets to `original` because incremental current-state navigation is not wired yet.
- OpenMesh verification is strongest around decimation; broader algorithm-by-algorithm comparison coverage is still selective.
- Some helper/test-data paths still contain older TODO markers that do not affect the verified library surface.

## Key Commands

```bash
# Build
cargo build --manifest-path RustMesh/Cargo.toml --release

# Full library test suite
cargo test --manifest-path RustMesh/Cargo.toml --lib

# Focused RustMesh areas
cargo test --manifest-path RustMesh/Cargo.toml --lib tools::decimation::tests
cargo test --manifest-path RustMesh/Cargo.toml --lib tools::remeshing::tests
cargo test --manifest-path RustMesh/Cargo.toml --lib tools::vdpm::tests

# Reproducible normals parity check
env RUSTFLAGS=-Awarnings cargo run --manifest-path RustMesh/Cargo.toml --release --example openmesh_compare_normals --quiet
```

## Related Docs

- Workspace overview: [`../README.md`](../README.md)
- Workspace status: [`../docs/current-project-status.md`](../docs/current-project-status.md)
- Documentation index: [`../docs/index.md`](../docs/index.md)

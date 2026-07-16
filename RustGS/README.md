# RustGS

RustGS is the RustScan 3D Gaussian Splatting training crate. It provides a wgpu-based
training path, host-side scene I/O, a COLMAP dataset loader, and evaluation/rendering
utilities. The library also accepts an already constructed `TrainingDataset`; the CLI
currently accepts COLMAP directories only.

The minimum supported Rust version is 1.92, matching the current Burn/CubeCL dependency
stack.

## Features

- `default = ["gpu", "cli"]`: builds the library and `rustgs` CLI.
- `gpu`: enables Burn/wgpu training, PLY scene load/save, and evaluation rendering.
- `cli`: enables the `rustgs` binary and CLI-only dependencies.
- `gpu-wgpu`: compatibility alias for `gpu`.
- `rustsfm-contract-tests`: enables the RustSFM writer to RustGS loader contract test.

The library can be checked without optional features:

```sh
cargo check --no-default-features --all-targets
```

Examples and tests that require GPU APIs are gated through Cargo features.

## Training

Training requires a COLMAP sparse model containing `cameras`, `images`, and `points3D` in
binary or text form. RustGS currently supports one shared camera ID using `SIMPLE_PINHOLE`
or `PINHOLE`; distorted camera models must be undistorted before training. Image names must
be relative paths without parent traversal.

By default, images are resolved from the dataset's `images` directory. For a sparse-only
export, pass `--image-root` to the original image directory. This is the matching RustGS
option for a RustSFM export made with `--no-copy-images`.

```sh
cargo run --bin rustgs -- train \
  --input /path/to/colmap-dataset \
  --output output/scene.ply \
  --iterations 30000
```

Useful flags:

- `--max-frames` and `--frame-stride` select directory-backed dataset frames.
- `--image-root` resolves image paths outside the COLMAP export directory.
- `--render-scale` controls training target resolution and is validated in `[0.0625, 1.0]`.
- `--eval-after-train` runs a post-training PSNR evaluation pass.
- `--eval-json` prints the evaluation summary as JSON.

## Scene Artifacts

- `.ply` is the lossless training artifact. It preserves positions, log scales, rotations,
  opacity logits, all SH coefficients through degree 3, and training metadata.
- `.splat` is the legacy 32-byte viewer format. RGB, opacity, and rotation are quantized;
  higher-order SH coefficients and training metadata are discarded. Do not use it as a
  training checkpoint or for fidelity comparisons.

When `--output` is omitted, training writes `output/scene.ply`. The CLI still accepts an
explicit `.splat` output for legacy consumers, and parity reporting marks that export as
intentionally lossy.

## Rendering

Render a saved scene with a camera JSON:

```sh
cargo run --bin rustgs -- render \
  --input output/scene.ply \
  --camera camera.json \
  --output output/render.png
```

`camera.json` format:

```json
{
  "intrinsics": {
    "fx": 500.0,
    "fy": 500.0,
    "cx": 320.0,
    "cy": 240.0,
    "width": 640,
    "height": 480
  },
  "pose": {
    "rotation": [0.0, 0.0, 0.0, 1.0],
    "translation": [0.0, 0.0, 0.0]
  },
  "pose_is_world_to_camera": false
}
```

By default, `pose` is interpreted as camera-to-world, matching `ScenePose`. Set
`pose_is_world_to_camera` to `true` when the pose is already a view transform.

## Verification

```sh
cargo fmt -p rustgs -- --check
cargo test -p rustgs --all-targets --no-fail-fast
cargo test -p rustgs --no-default-features --all-targets --no-fail-fast
cargo clippy -p rustgs --all-targets -- -D warnings
cargo test -p rustgs --features rustsfm-contract-tests --lib \
  io::colmap_dataset::tests::loads_sparse_text_fixture_written_by_rustsfm
cargo test -p rustgs --test integration_test -- --ignored
```

The ignored integration target runs a deterministic three-step wgpu training job, compiles
the actual shaders, and exercises an optimizer rebuild. It requires a working wgpu adapter.

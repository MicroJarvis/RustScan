# RustFF

RustFF is an experimental feed-forward 3D reconstruction library. Its current
Spann3R path is scaffolding for sequential ONNX inference, not a complete
end-to-end reconstruction backend.

## Current Status

Implemented:

- Inference configuration, pointmap result, and error types.
- Image resize/normalization and a bounded sequential memory bank.
- Weighted Procrustes helpers and pointmap-to-pose estimation.
- Model analysis/export and Procrustes reference scripts.

Not implemented:

- Spann3R decoder ONNX execution. `process_frame()` currently returns an
  explicit error after encoder/memory processing when `onnx-ort` is enabled.
- A complete model-backed inference acceptance test. The `onnx-ort` encoder
  session/input path compiles against the pinned ORT 2.0 RC API, but no
  Spann3R ONNX fixtures are distributed with the repository and the decoder
  remains unimplemented.
- A Candle ONNX inference path. The `onnx-candle` feature currently exposes
  dependencies only.
- A RustFF CLI binary or integration into the active RustViewer pipeline.

## Features

- `default = []`: data types, preprocessing, memory, and geometry helpers only.
- `onnx-ort`: enables ONNX Runtime model/session support.
- `onnx-candle`: enables Candle ONNX dependencies; inference is not wired yet.
- `cli`: enables CLI dependencies; no binary target is currently defined.

## Build And Test

```bash
cargo test -p rustff --lib
cargo test -p rustff --lib --features onnx-ort
cargo check -p rustff --features onnx-candle
```

Verified on 2026-09-03: the default library suite passed 2 tests and the
`onnx-ort` suite passed 3, including a compile-only ORT constructor API contract
that does not require a distributed model fixture or runtime initialization.
The feature compiles, but the decoder remains explicitly unfinished, so it is
not yet an end-to-end Spann3R inference backend.

Model export helpers live in `RustFF/scripts/`. Exported models are not stored
in the repository, and successful export does not make the unfinished decoder
path production-ready.

## License

MIT License - see the repository license file.

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
- The `onnx-ort` feature does not currently compile against the pinned ORT
  release because `spann3r.rs` imports the pre-2.0 `Session`,
  `SessionBuilder`, and `Value` API names.
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
cargo check -p rustff --features onnx-candle
```

Verified on 2026-08-15: the default library suite passed 2 tests. Do not use
the `onnx-ort` feature until its ORT 2.0 API integration has been updated.

Model export helpers live in `RustFF/scripts/`. Exported models are not stored
in the repository, and successful export does not make the unfinished decoder
path production-ready.

## License

MIT License - see the repository license file.

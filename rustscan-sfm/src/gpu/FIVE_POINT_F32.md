# Independent five-point f32 GPU experiment

Complete GPU candidate generation through polynomial roots and normalized essential
matrices is implemented. No production call site, CPU runtime fallback, pose
(R/t) decomposition, or RANSAC integration. CPU f64 is used only as an independent
reference in focused tests and replay. Real replay results and limitations are in
[FIVE_POINT_F32_REPLAY.md](FIVE_POINT_F32_REPLAY.md).

## Layout / readback contract

`WgpuFivePointF32` exposes:

- `solve_essential`: rays upload → constraints → Jacobi → basis packing → generated
  elimination → pivoted solve/B → coefficients → roots → null3/E recovery → final
  readback. **No intermediate host readback or CPU solving.** Returns one
  `FivePointTrialResult` per input trial, with ten fixed `FivePointModelSlot`s;
  failures do not compact trial or slot identity. Only Accepted slots contain E.
- `compute_polynomial_roots`: independent GPU root diagnostics, returning RealRoot
  slots before E recovery; uses the same root kernel as the complete solver.

- `compute_constraint_matrices`: row-major 5×9, `b ⊗ a`, matching CPU geometry.
- `compute_nullspace_diagnostics`: optional column-major 9×4 basis, status,
  completed Jacobi sweeps, relative off-diagonal norm (maximum off-diagonal /
  trace), numerical rank. `compute_nullspace_basis` errors if any sample fails.
- `compute_algebra`: takes that exact basis unchanged; returns column-major
  10×20 elimination matrix, column-major 10×10 `left^-1 * right` (no minus sign),
  column-major 13×3 determinant matrix (39 entries), and 11 unnormalized
  coefficients in **descending** powers `z^10 .. z^0`.

Only `Success` validates all algebra arrays. Other statuses are diagnostic;
zero-filled arrays on failed samples are not valid solutions. Nonfinite input
and buffer/dispatch-limit violations return errors before dispatch. Shader
validation/pipeline creation failures and readback errors are propagated.

The nullspace kernel scales A before forming AᵀA. It uses at most 64 cyclic
Jacobi sweeps, checks convergence (`max off-diagonal <= 2e-8 * trace`), requires
numerical rank five (`eigenvalue > 1e-6 * trace`), and checks both A*N and Nᵀ*N.
This is deliberately conservative f32 numerical rank, not an exact rank test.
AᵀA squares the condition number: difficult near-degenerate samples can fail
rather than silently returning a zero vector or an unchecked basis.

The solve uses partial row pivoting, forward elimination and back substitution.
Pivots at or below `1e-7 * max(abs(left))` are rejected. This is not a condition
number estimate; successful status does not guarantee accurate polynomial roots.

## Reproducible generation

From this worktree root:

```sh
python3 scripts/generate_five_point_wgsl.py
python3 scripts/generate_five_point_wgsl.py --check
python3 -m unittest discover -s scripts -p 'test_generate_five_point_wgsl.py' -v
```

The generator reads `src/geometry/five_point_generated.rs` under `rustsfm`,
checks the supported arithmetic AST, array bounds, unique complete assignments,
and known initialization/return syntax. It preserves source indices and operation
order, embeds a source SHA256, and does not require a build.rs hook or dependencies.
The checked-in WGSL is sufficient for Cargo builds.

Each of the 200 elimination expressions runs in its own invocation. Determinant
expressions are mechanically lowered into a WGSL constant table of signed triple
products, preserving multiply association and add/subtract order. Each coefficient
is evaluated on the GPU, followed by a GPU finite/nonzero check. Separate compute
passes provide storage dependencies; there is no intermediate CPU readback inside
`compute_algebra`. Shader modules are shared across their pipelines. Monolithic
expansion and repeated large shader-module creation exceeded bounded Metal test
runs during development; the split/table approach completes the focused test.
GPU compilers may still fuse/reassociate floating-point arithmetic: bit parity
with sequential CPU f32 is not promised.

## Actual GPU validation

```sh
cargo check -p rustscan-sfm --lib --no-default-features --features gpu-wgpu
cargo test -p rustscan-sfm --lib --no-default-features --features gpu-wgpu five_point_f32_actual_gpu_stages -- --nocapture --test-threads=1
```

The test is not ignored and does **not** skip when a GPU is unavailable. It tests
16 seeded samples (8 geometric scenes, 8 random correspondences), constraints,
Jacobi convergence, A*N and orthonormality, scale invariance, zero/duplicate/
near-duplicate constraints, nonfinite input, singular elimination, and empty or
malformed batches. The f64 reference uses the **read-back GPU basis**, not an
independently chosen CPU nullspace. It checks every algebra stage, solve backward
error, and coefficient order via independent determinant evaluation. Twelve of
the 16 reference systems require a first-column row pivot.

Measured on Apple M5 Max (2026-09-13): focused GPU test passed, about 29 seconds
excluding Rust build. Maximum normalized A*N residual: `7.61e-7`. Maximum
infinity-norm relative errors against same-basis CPU f64:

| Stage | Maximum relative error |
|---|---:|
| Elimination matrix | 1.61e-7 |
| Pivoted solve | 9.02e-5 |
| 39-entry determinant matrix | 9.43e-5 |
| 11 polynomial coefficients | 1.37e-3 |

Polynomial cancellation is material: isolated expansion using the same read-back
B reaches `1.40e-3` relative error. The test also checks each coefficient against
an absolute-term roundoff bound (`128 * f32 epsilon * sum(abs(triple products))`),
so cancellation does not make an arbitrary relative threshold the only gate.
This is experimental f32 behavior, **not** production f64 parity or a performance
claim. The subsequent real DB replay exposes material errors and missing models;
other adapters and broader ill-conditioned sample distributions remain unvalidated.

The main workspace has PoseLib at
`/Users/tfjiang/Projects/RustScan/third_party/native/PoseLib`, and Eigen is discoverable
via `pkg-config eigen3`. The commands above deliberately avoid default PoseLib,
VLFeat and Ceres features; no native dependency setup was changed.

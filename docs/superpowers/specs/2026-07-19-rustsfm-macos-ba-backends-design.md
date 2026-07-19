# RustSFM macOS BA Backend Selection Design

**Status:** Approved for planning

**Goal:** Make RustSFM bundle adjustment select and report Ceres sparse linear algebra and Schur solver backends so SuiteSparse, Apple Accelerate Sparse, and iterative Schur can be compared on the same flowers2 reconstruction.

## Scope

This change adds backend selection and timing to the existing Ceres BA implementation. It does not implement Metal or wgpu BA, change local/global BA scheduling, alter reconstruction acceptance thresholds, or change which observations enter BA.

The native `reconstruct` command and COLMAP-compatible `mapper` command must expose equivalent controls. Existing commands remain behaviorally compatible because both controls default to `auto`.

## User Interface

The native reconstruction command gains:

```text
--ba-linear-solver <auto|dense-schur|sparse-schur|iterative-schur>
--ba-sparse-backend <auto|suite-sparse|accelerate-sparse|eigen-sparse>
```

The COLMAP-compatible mapper gains:

```text
--Mapper.ba_linear_solver <auto|dense-schur|sparse-schur|iterative-schur>
--Mapper.ba_sparse_backend <auto|suite-sparse|accelerate-sparse|eigen-sparse>
```

The same `Mapper.ba_linear_solver` and `Mapper.ba_sparse_backend` keys are accepted from a project INI file. Command-line values override project values.

`auto` preserves the current behavior: dense Schur through 50 pose entities, sparse Schur through 1000 pose entities when a sparse backend is available, and iterative Schur with Schur-Jacobi otherwise.

Explicit solver selection affects local and global BA. `iterative-schur` always uses the Schur-Jacobi preconditioner in this change. More advanced preconditioners are outside this scope.

## Configuration Types

The backend-neutral BA module owns two public enums:

```rust
pub enum BundleAdjustmentLinearSolverPreference {
    Auto,
    DenseSchur,
    SparseSchur,
    IterativeSchur,
}

pub enum BundleAdjustmentSparseLinearAlgebra {
    Auto,
    SuiteSparse,
    AccelerateSparse,
    EigenSparse,
}
```

Both implement `Default`, `Display`, and `FromStr`. Parsing is case-insensitive and accepts hyphens or underscores. `MapperConfig` and `BundleAdjustmentOptions` store both preferences, with `Auto` defaults.

The BA report continues to expose the selected Schur solver and preconditioner. It additionally reports the actual Ceres sparse library selected after resolving `auto`; `None` represents a Ceres build with no sparse library or the native Rust BA backend.

## Ceres Selection

`ceres_solver_options` applies an explicit sparse backend to `SolverOptionsBuilder::sparse_linear_algebra_library_type`. For `auto`, it leaves the Ceres default untouched. The current system Ceres default remains SuiteSparse because Ceres ranks available backends as SuiteSparse, Accelerate Sparse, Eigen Sparse, then no sparse backend.

The Schur solver policy is split into two steps:

1. Resolve the sparse backend and determine whether sparse direct solving is available.
2. Resolve the requested Schur solver, using the existing size thresholds only for `auto`.

An explicit unavailable or incompatible backend must fail option validation and propagate as a BA skip reason; it must not silently fall back to another backend. `auto` may use Ceres' normal fallback behavior.

The native non-Ceres BA path honors the solver preference when its implementation supports the requested solver and reports no Ceres sparse library.

## Timing And Diagnostics

Each `BundleAdjustmentReport` records:

```rust
pub setup_ms: f64,
pub solve_ms: f64,
pub postprocess_ms: f64,
pub elapsed_ms: f64,
```

For Ceres BA:

- `setup_ms` covers observation collection, parameter block construction, residual construction, manifolds, constants, and solver option validation.
- `solve_ms` covers the `problem.solve` call.
- `postprocess_ms` covers solution write-back, point-error refresh, optional covariance, and summary mapping.
- `elapsed_ms` covers the complete BA call and may be slightly larger than the sum because it includes boundary bookkeeping.

Native BA records the same fields around its equivalent phases where practical; unsupported phase separation uses zero for the phase while preserving a valid total.

Local and global mapper log lines include:

```text
solver=SparseSchur preconditioner=None sparse_backend=SuiteSparse setup_ms=... solve_ms=... postprocess_ms=... ba_elapsed_ms=...
```

This change does not add timings for PnP, triangulation, or filtering. Those belong to the separate mapper-stage telemetry change.

## Error Handling

Invalid CLI or project values fail during parsing with the accepted value set in the message. A Ceres backend that was requested explicitly but is not compiled into the current Ceres installation produces a clear solver-option error containing the requested backend.

The existing `Option` boundary from the BA layer is insufficient to explain backend validation failures. The checked mapper wrapper must distinguish invalid solver configuration from a generic solver-returned-none skip so logs remain actionable.

## Testing

Unit tests cover:

- case-insensitive and hyphen/underscore parsing for both preferences;
- default `auto` behavior and the existing 50/1000 thresholds;
- explicit dense, sparse, and iterative Schur policy selection;
- explicit SuiteSparse, Accelerate Sparse, and Eigen Sparse mapping;
- Schur-Jacobi selection for explicit iterative Schur;
- CLI and project-file propagation into `MapperConfig`;
- report timing fields being finite, non-negative, and internally consistent;
- mapper logs containing actual solver, sparse backend, and BA timing fields.

On macOS, the system-Ceres test verifies that both SuiteSparse and Accelerate Sparse are accepted by option construction. Real performance claims require release-mode flowers2 runs rather than unit-test timing.

## Benchmark

Use the same fixed-seed, database-backed contiguous flowers2 subsets for each backend. Run at 96 and 200 frames first with `--threads 1`, then run 960 frames only for configurations that register the same image count and remain numerically stable.

Compare:

- registered images and points;
- total mapper wall time;
- number of local and global BA calls;
- cumulative setup, solve, postprocess, and BA elapsed time;
- iterations, termination reasons, selected Schur solver, and sparse backend;
- RustGS loading of the winning output.

Backend comparisons are valid only when the registered image count is identical and sparse point count differs by no more than 1% for the fixed seed. The selected default is not changed as part of this implementation; changing defaults requires benchmark evidence and a separate decision.

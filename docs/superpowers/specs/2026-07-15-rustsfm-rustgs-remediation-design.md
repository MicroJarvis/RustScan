# RustSFM / RustGS Review Remediation Design

## Goal

Remove the confirmed memory-safety, data-corruption, interoperability, training-correctness,
and quality-gate failures identified in the 2026-07-15 deep review of RustSFM and RustGS.

## Scope And Decomposition

The work spans four independently verifiable subsystems:

1. RustSFM native memory and database safety.
2. The COLMAP data contract between RustSFM and RustGS.
3. RustGS optimizer, topology, shader, and reporting correctness.
4. Scene IO, automated tests, static analysis, and public documentation.

The batches execute in that order. Batch B depends on the validation primitives introduced
in Batch A. Batch D runs last because its integration fixtures exercise the contracts fixed
by B and the training behavior fixed by C.

## Design Decisions

### Failure-Atomic Native And Database Operations

Native paired buffers will never be grown with independent `realloc` calls. Growth allocates
replacement buffers, copies existing elements, and commits ownership only after all
allocations succeed. Database replacement operations validate all source data first and use
one SQLite transaction, so an error leaves the previous database contents intact.

Database opening gains explicit read-only and read-write/migrating modes. Read paths must not
create tables, alter schemas, change pragmas that persist, or require write permission.

### Bounded External Inputs

Every external count is checked before conversion or allocation. Binary counts are bounded
by remaining file length and a documented resource ceiling. SQLite signed dimensions must
be non-negative and fit `usize`. Camera dimensions must fit `u32`, be non-zero, and obey a
pixel budget before image or GPU allocation.

Paths from COLMAP or SQLite are relative logical image names. Absolute paths, root prefixes,
and parent traversal are rejected before joining with the configured image root.

### Authoritative COLMAP Contract

RustSFM's complete camera-model definitions become the source of truth exposed through a
small public metadata API. RustGS consumes that metadata instead of maintaining a divergent
ID table. RustGS preserves image `CAMERA_ID`; until the training data model supports varying
intrinsics, a dataset containing more than one effective camera calibration is rejected with
a precise error.

RustGS training consumes undistorted images only. Distorted models with non-zero distortion
parameters are rejected rather than silently approximated as pinhole cameras.

### RustGS Training Invariants

The following invariants hold for every training step:

- Gaussian count never exceeds the configured hard budget.
- Topology actions occur only inside configured windows and cadence.
- Adam always has parameter-group scaling after initialization or rebuild.
- Non-finite loss cannot reach backward propagation or optimizer state.
- The final report contains the final iteration's loss.
- SH degree is within the shader and scene-format range.
- Cross-invocation visibility aggregation uses race-free storage.
- Every public feature flag either changes behavior or is rejected during validation.

### Artifact And Event Contracts

PLY is the lossless default training artifact. The legacy 32-byte `.splat` format remains an
explicitly lossy interchange format and does not claim metadata or SH round-trip fidelity.
Round-trip gates compare all lossless fields within numeric tolerance.

Every emitted `RunStarted` event is followed by exactly one terminal event: completed,
cancelled, or failed.

## Testing Strategy

Each production change follows a red-green-refactor cycle. Tests use small deterministic
fixtures and assert externally visible behavior rather than implementation details.

- RustSFM tests cover allocator failure, transaction rollback, read-only hashes, negative
  SQLite values, malformed collection counts, invalid references, and concurrent image init.
- Cross-repository tests cover every camera ID, multi-camera ownership, distortion rejection,
  image-name handling, and no-copy image roots.
- RustGS tests cover configuration, budget boundaries, schedule boundaries, optimizer rebuild,
  loss reporting, scene round-trip, shader validation, and terminal events.
- Malformed-input tests remain CPU-only. GPU integration tests use a tiny ignored fixture when
  no adapter is available and execute normally on a compatible adapter.

## Compatibility

Existing valid single-camera, undistorted COLMAP PINHOLE datasets remain accepted. Inputs that
were previously accepted only by silently discarding information now fail explicitly. This is
an intentional correctness break and will be documented in the RustGS README.

The legacy `.splat` byte layout remains readable and writable. Its lossy nature becomes part
of the public documentation, while lossless training output uses PLY.

## Verification

Completion requires the master tracker gates, non-zero RustGS tests, no known silent feature
switches, and a clean Clippy run for RustGS. RustSFM Clippy results are reported separately if
the RustSLAM dependency still prevents package-local analysis.

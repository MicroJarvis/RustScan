# Rust Style and Engineering Rules

Read this file before changing Rust, Cargo manifests, FFI, GPU boundaries,
build scripts, examples, or Rust tests. The root `AGENTS.md` contains the
rules that apply to every task.

## Edition, formatting, and linting

- Keep first-party crates on the workspace edition (`2021`) unless a dedicated
  migration task approves a newer edition for the complete workspace.
- `rustfmt` is authoritative. Run `cargo fmt --all -- --check` before handoff.
  Keep formatting-only changes separate from behavior changes when practical.
- New or modified code must pass targeted Clippy with warnings denied:
  `cargo clippy -p <package> --all-targets --all-features -- -D warnings`.
  Use the narrowest package and feature set that covers the change, then run
  the workspace command for cross-crate changes.
- Do not silence a lint globally. An allowance must be as narrow as possible,
  explain the invariant or external constraint, and be limited to a
  compatibility or generated-code boundary. Never add
  `#[allow(clippy::all)]` to make a check pass.
- Existing warnings outside the changed scope do not permit new warnings.
  Record unavoidable pre-existing warnings in the handoff.

## API design and documentation

- Use the smallest visibility that satisfies the caller. Keep helpers private
  or `pub(crate)` unless they are a documented cross-crate contract.
- Public structs, enums, traits, functions, type aliases, constants, and
  modules need rustdoc covering purpose, units or coordinate frame where
  relevant, ownership/lifetime expectations, and failure or panic behavior.
  Document safety invariants on every `unsafe` item.
- Preserve the existing public API unless the task explicitly authorizes a
  breaking change. Prefer compatibility wrappers or a documented deprecation
  path when an API must evolve.
- Use domain types for units, pose direction, coordinate conventions,
  ownership, and lifecycle state when a type or name can express the
  invariant. Use `#[must_use]` when ignoring a returned value can silently
  discard a required result.

## Ownership and error handling

- Prefer borrowing (`&T`, `&mut T`, slices, and string slices) when the callee
  does not need ownership. Avoid `clone`, `to_vec`, and `collect` unless they
  make an ownership, performance, or lifetime requirement explicit.
- Use `Option<T>` for absence and `Result<T, E>` for recoverable failure. Do
  not encode multiple states in sentinels or unrelated booleans.
- Library crates use structured errors (`thiserror` or an existing crate error
  type). Application and CLI boundaries may add context with `anyhow`. Errors
  should preserve their source and identify the failed operation and input.
- `unwrap` and `expect` are forbidden in production paths unless a local
  invariant proves they cannot fail and a short comment explains it. Tests and
  one-time static initialization may use them for intentional failure.
- Do not discard errors with `let _ = ...` or an empty match. Handle, return,
  propagate with `?`, or document why an error is intentionally ignored.
- Make device, precision, layout, color-space, coordinate-frame, and
  serialization changes explicit at subsystem boundaries. Do not hide them in
  unrelated conversions or `From` implementations.

## Unsafe, FFI, and generated code

- Safe Rust is the default. Every new `unsafe` block needs a task-level reason,
  the smallest possible scope, and a `// SAFETY:` comment proving its
  preconditions at the call site.
- Keep C/C++ declarations, ownership rules, ABI conversions, and native
  cleanup inside the owning adapter. Never expose raw FFI pointers or native
  matrix layout through a general CPU API.
- Check pointers, lengths, alignment, lifetimes, thread affinity, and native
  error codes at the FFI boundary. Make destruction and callback ownership
  explicit.
- Generated Rust, WGSL, bindings, and fixtures must identify their generator
  and source of truth. Change the generator or input, regenerate, and run its
  reproducibility check instead of editing generated output directly.

## Concurrency, async, and GPU boundaries

- Use the narrowest synchronization primitive that expresses ownership. Avoid
  `Arc<Mutex<T>>` as a default; prefer message passing, scoped borrowing, or
  ownership transfer where possible.
- Do not hold a mutex, filesystem handle, or blocking native call across an
  `.await`. Run blocking work through the established blocking-worker boundary.
- Shared state needs one documented owner and a clear shutdown/error path. Do
  not introduce process-global mutable state, leaked threads, or detached tasks
  without an explicit lifecycle owner.
- Keep CPU and GPU representations separate. GPU buffers, Burn tensors, WGSL
  values, and staging arrays must cross boundaries with explicit shape, scalar
  type, alignment, and row/column-layout metadata.
- Parallel numerical code must be deterministic where results feed geometry,
  tests, checkpoints, or evidence. Seed random generators and make reduction
  order explicit when floating-point order is observable.

## Dependencies and features

- Prefer an existing workspace dependency and feature set. Add a crate only
  after checking the standard library and current workspace dependencies.
- Put versions in `[workspace.dependencies]` when a dependency is shared by
  multiple first-party crates. Keep feature flags minimal and document why an
  opt-in feature exists.
- Do not add duplicate versions of foundational crates such as `nalgebra`,
  `serde`, `wgpu`, or the async runtime without an explicit compatibility
  reason and task record.
- Default features should support a clean CPU build. GPU, native, platform,
  and external-dataset requirements are opt-in unless a package documents them
  as mandatory.
- Update `Cargo.lock` with Cargo when resolution changes; never hand-edit
  lockfile checksums or generated dependency metadata.

## Tests and verification

- Every behavior change needs a focused test at the narrowest layer that can
  detect the regression. Cover success, boundary, and failure paths. For
  numerical code use non-symmetric, non-identity data and round-trips through
  affected representation boundaries.
- Unit tests belong beside implementation; integration tests belong in a
  crate's `tests/` directory and exercise its public contract. Examples must
  remain buildable.
- Tests must be deterministic and independent of home directory, current
  directory, local GPU selection, and unstated environment variables. Use
  fixtures under `artifacts/inputs/` and generated data under `artifacts/runs/`.
- Do not mark a test `#[ignore]` to hide a failure. An ignored test must state
  the required capability, why normal CI cannot run it, and the exact command
  or workflow that runs it.
- For cross-crate work, run the workspace gates in addition to targeted checks:

  ```text
  cargo fmt --all -- --check
  cargo check --workspace --all-targets
  cargo clippy --workspace --all-targets --all-features -- -D warnings
  cargo test -p <changed-package> --all-targets
  ```

  Also run feature, platform, native, GPU, generator, and data checks from
  the affected crate and CI workflow. Record commands, results, and known
  pre-existing failures in the handoff.

# RustScan Repository Guidelines

This file contains only rules that apply to every repository task. Read the
conditional documents below when the task touches their subject. User
instructions always take precedence over repository defaults.

## Rule precedence and scope

Rules apply to first-party workspace code and documentation. The nearest
`AGENTS.md` applies to a path; then use the active task file under
`docs/agent/tasks/` or `docs/agent/changes/`. Historical reports are evidence,
not new task instructions. `third_party/` is externally owned and requires an
explicit task scope before modification.

Before changing Rust, Cargo, FFI, GPU boundaries, or Rust tests, read
[`docs/agent/rust-style.md`](docs/agent/rust-style.md). Before adding,
moving, or generating files, read
[`docs/agent/repository-layout.md`](docs/agent/repository-layout.md). For
multi-agent work, worktrees, handoffs, or task status, read
[`docs/agent/task-protocol.md`](docs/agent/task-protocol.md).

## CPU mathematical types

All CPU-side matrices, vectors, points, quaternions, poses, and related
linear-algebra APIs MUST use `nalgebra` as the owning type library.

- Use `Matrix*`/`SMatrix` for fixed dimensions and `DMatrix` for runtime
  dimensions.
- Use `Vector*` for vectors, `Point*` for positions, and `UnitQuaternion` for
  rotations. Reuse the shared pose type; do not add another `SE3`.
- CPU code MUST NOT introduce `glam` mathematical types, another matrix crate,
  or a hand-rolled general-purpose linear-algebra type.
- Burn tensors, WGSL `mat*`, Eigen, graphics-library values, and fixed arrays
  are allowed only at explicit GPU, FFI, serialization, or UI boundaries.
  Convert there with explicit scalar type, shape, layout, and convention.
- Unless a boundary documents otherwise, indexing is `(row, column)`, vectors
  are column vectors, and transforms use `p' = M * p`. State pose direction
  (`world_from_camera`, `camera_from_world`) in names or API docs.

## Repository invariants

- The root package is `rustscan`. All first-party component package names and
  directories MUST match `rustscan-<component>` in lowercase kebab-case
  (e.g. `rustscan-sfm`, `rustscan-viewer`, `rustscan-types`). Library targets
  and imports use `rustscan_<component>`; do not retain legacy dependency
  aliases or library names. Existing CLI binary names are compatibility
  contracts and may differ from package names; declare them explicitly.
  Rust items use `snake_case`, `UpperCamelCase`, and `SCREAMING_SNAKE_CASE`.
- `main` is for coordination and integration. Do not overwrite another task's
  dirty worktree, use destructive Git cleanup, or change a public API without
  an explicit task scope.
- Repository-level inputs, runs, and evidence belong under `artifacts/`; do
  not recreate the removed top-level `test_data/`, `output/`, or `experiments/`
  directories.
- `.codex/`, `.claude/`, `.vscode/`, and `.ua/` are local tool/runtime state,
  not project rules or source. Do not create project protocol there or rely on
  those directories for portability.

## Minimum handoff gate

Every implementation handoff must state the changed files, verification
commands and results, known limitations, and next action. For Rust changes,
the required baseline is:

```text
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo test -p <changed-package> --all-targets
```

Run the targeted Clippy, feature, platform, native, GPU, generator, and data
checks required by [`docs/agent/rust-style.md`](docs/agent/rust-style.md) and
the affected crate. A task is not complete without recorded evidence or a
clear unavailable/blocked reason.

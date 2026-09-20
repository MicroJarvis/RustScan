# RustScan Repository Guidelines

## CPU linear-algebra representation

All CPU-side matrices, vectors, points, quaternions, and related
linear-algebra code in this repository MUST use `nalgebra` as the only
mathematical type library.

Use the appropriate `nalgebra` type for the problem:

- `Matrix2`, `Matrix3`, `Matrix4`, and other fixed-size matrix types for
  small, statically sized matrices.
- `SMatrix<T, R, C>` for fixed-size non-square matrices and Jacobians.
- `DMatrix<T>` for matrices whose dimensions are known only at runtime.
- `Vector2<T>`, `Vector3<T>`, and `Vector4<T>` for mathematical vectors.
- `Point2<T>` and `Point3<T>` for positions, where point/vector semantics
  matter.
- `UnitQuaternion<T>` for rotations; use `Quaternion<T>` only for raw,
  potentially non-unit quaternion algebra.

This rule applies to public CPU APIs, internal CPU implementations, geometry,
SfM, SLAM, bundle adjustment, mesh processing, numerical solvers, and tests.
New CPU code MUST NOT introduce or use `glam::Mat*`, `glam::Vec*`, `glam::Quat`,
another mathematics crate, or a hand-rolled general-purpose linear-algebra
type.

`glam` may remain only in an explicit graphics/GPU or third-party adapter
boundary. It MUST NOT be used as the owning type for CPU vectors, points,
matrices, or rotations. Convert explicitly at that boundary when necessary.

## Boundary exceptions

The following representations are permitted only at an explicit boundary and
MUST NOT leak into general CPU linear-algebra APIs:

- Burn `Tensor` values in GPU training code.
- WGSL `mat*` values in shader code.
- Eigen matrices in C/C++ FFI and native helper code.
- `glam` values required by a graphics or third-party API adapter.
- Fixed arrays such as `[[f32; 4]; 4]` or flat buffers used for serialization,
  GPU uniforms, wire formats, or FFI.
- UI-library types such as `egui::Vec2` when they represent screen/layout
  dimensions rather than mathematical geometry.

Boundary conversions MUST make layout and convention explicit. Use names such
as `from_row_major_slice`, `to_row_major_array`, or
`to_column_major_array`; do not pass an unlabeled matrix-shaped array between
subsystems.

## Layout and convention

Unless a boundary API explicitly documents otherwise:

- Matrix indexing is `(row, column)`.
- Vectors are column vectors and transforms use `p' = M * p`.
- CPU numerical code uses `f64` when precision is material; use `f32` where
  the surrounding algorithm or hardware interface requires it.
- Pose direction MUST be stated in the type or API documentation, such as
  `world_from_camera` or `camera_from_world`.

## Shared pose and geometry types

Do not add another `SE3` implementation. Reuse the shared pose type from
`rustscan-types` or an existing crate-level re-export. An `SE3` rotation MUST
use `nalgebra::UnitQuaternion` and its translation MUST use
`nalgebra::Vector3`; keep conversion helpers in one place.

When changing matrix, vector, point, or quaternion code, add or update tests
with a non-symmetric rotation, non-zero translation, point/vector operations,
and a round-trip through every affected boundary. Identity-only tests are
insufficient to detect transposition, row/column-layout, point/vector, and
quaternion-component-order errors.

## Tool-neutral Agent protocol

This repository does not require Codex, Claude Code, OpenSpec, Spec Kit, or
any other vendor-specific Agent plugin. Repository work MUST be
possible with the files in this repository plus standard Git, shell, Rust,
Cargo, rustfmt, and CI commands.

Rules are applied in this order:

1. Explicit user instructions.
2. The nearest applicable `AGENTS.md`.
3. The active task file under `docs/agent/tasks/` or change package under
   `docs/agent/changes/`.
4. Historical documents and experiment records.
5. Local Agent-tool configuration, which MUST NOT override repository rules.

Every non-trivial Agent task MUST declare:

- a unique task ID and one owner;
- the base commit, branch, and worktree;
- files or directories in scope and explicitly out of scope;
- acceptance conditions and exact verification commands;
- a handoff containing changed files, results, limitations, and next action.

The `main` worktree is for coordination and integration. Agents MUST use an
isolated task worktree for implementation, MUST NOT overwrite another task's
dirty changes, and MUST NOT run destructive cleanup such as `git reset --hard`
or `git clean` unless the user explicitly authorizes that exact action.

Small tasks MAY use only one task YAML file. Cross-crate, public API,
numerical, GPU/CPU boundary, or parallel tasks SHOULD use a change package
under `docs/agent/changes/<task-id>/`. The repository protocol remains valid
when OpenSpec or any other CLI is not installed.

## Artifact layout

All repository-level data lives under `artifacts/` and MUST keep these
ownership boundaries:

- `artifacts/inputs/` contains reusable input datasets and test fixtures.
- `artifacts/runs/` contains generated outputs, caches, databases, logs,
  models, profiles, and other reproducible run artifacts.
- `artifacts/evidence/` contains small, versioned experiment summaries,
  logs, scripts, and evidence referenced by repository documentation.

Agents MUST NOT recreate the former top-level `test_data/`, `output/`, or
`experiments/` directories. Large or regenerable files MUST NOT be added to
`artifacts/evidence/`; put them in the ignored `artifacts/runs/` tree. Treat
hard-linked images under `artifacts/runs/legacy/` as read-only and copy them
before modifying their contents.

## Repository directory map

Keep the following ownership boundaries when locating or adding files:

- `Cargo.toml`, `Cargo.lock`, and the root `src/` form the workspace-level
  `rustscan` orchestration CLI. Root CLI code belongs in `src/`; it is not a
  replacement for the individual crates.
- The workspace crates are `rustscan-taskflow`, `rustscan-types`, `rustff`,
  `rustgs`, `rustmesh`, `rustsfm`, `rustslam`, and `rust-viewer`. Keep each
  crate's implementation, tests, examples, and package-specific configuration
  inside that crate. Shared CPU data types belong in `rustscan-types` or an
  explicitly documented crate-level API.
- `scripts/` contains human- or CI-invoked maintenance, conversion,
  generation, evaluation, and verification entry points. `scripts/agent/`
  contains the tool-neutral task checks. Scripts MUST write inputs, evidence,
  and generated runs to the corresponding `artifacts/` subdirectory.
- `.github/` contains CI and repository automation. CI checks MUST reinforce
  the rules in this file; a local helper or workflow MUST NOT silently define a
  second directory layout or Agent protocol.
- `docs/` contains maintained architecture, design, status, and experiment
  records. `docs/agent/` is the active repository workflow and task protocol.
  Historical design records are non-authoritative and MUST NOT be used as new
  task entry points.
- `third_party/native/` and `third_party/rust/` contain external native and
  Rust dependencies, including submodules and local patches. Do not move or
  rewrite them as ordinary application code; changes require an explicit task
  scope and dependency/build verification.
- `rustgs/experiments/` contains package-local explanatory records. It is not a
  second run-artifact store; new generated data still belongs under
  `artifacts/runs/` or `artifacts/evidence/`. Likewise, ignored
  `rustgs/output/` and `rustsfm/output/` are crate-local runtime directories
  and must not be confused with the removed top-level `output/` directory.

The following are local or generated state, not repository source:

- `target/` and crate-local build directories are Cargo output and MUST remain
  ignored.
- `.worktrees/` contains local Git worktrees and is not an implementation
  directory. Agents MUST use Git worktree commands and must not edit another
  task's worktree.
- `.ua/` is local Agent-app runtime state and is ignored; it is not a project
  protocol or source directory.
- `.codex/`, `.claude/`, and `.vscode/` are not repository rule or task entry
  points. Do not create project work there or rely on them for portability
  between Agent models.

When a new directory does not fit this map, classify its ownership and output
type in the active task file before adding it. Do not create another top-level
input, output, experiment, vendor, or Agent-configuration directory without
updating this map and the relevant ignore rules.

## Rust and Cargo naming

Use these naming conventions for new packages, targets, paths, and Rust items:

- Cargo package names and published crate names use lowercase kebab-case,
  such as `rustscan-types` or `rust-viewer`. Do not introduce uppercase
  letters or mixed-case package names.
- A library target derived from a hyphenated package name is imported with
  underscores: package `rustscan-taskflow` is imported as
  `rustscan_taskflow`, and package `rust-viewer` as `rust_viewer`. This Cargo
  conversion is expected and is not a naming inconsistency.
- Binary and example target names use lowercase kebab-case when a separator is
  useful; Rust module paths and identifiers use `snake_case`, types and traits
  use `UpperCamelCase`, and constants use `SCREAMING_SNAKE_CASE`.
- New workspace crate directories SHOULD match their package name exactly in
  lowercase kebab-case. Acronyms are treated as words in paths and package
  names (`rustgs`, `rustsfm`, `rustslam`), not as PascalCase directory names.
- The workspace crate directories now use the lowercase package-aligned paths
  `rustff/`, `rustgs/`, `rustmesh/`, `rustsfm/`, `rustslam/`, and
  `rust-viewer/`. Keep these paths stable; a future rename requires a dedicated
  path-migration task that updates workspace members, path dependencies, CI
  working directories, scripts, documentation, and case-sensitive Git history
  together.
- Renaming a package name is a dependency and published-API change. Renaming a
  directory is a repository-wide path change. Both require an explicit task,
  a complete reference search, and verification on a case-sensitive filesystem.

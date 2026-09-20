# Repository Layout and File Ownership

Read this file before adding, moving, deleting, or generating files. The root
`AGENTS.md` keeps only the invariants that apply to every task.

## Workspace and crate ownership

- `Cargo.toml`, `Cargo.lock`, and root `src/` form the workspace-level
  `rustscan` orchestration CLI. Root CLI code belongs in `src/`; it is not a
  replacement for the individual crates.
- Workspace crates are `rustscan-taskflow`, `rustscan-types`, `rustscan-ff`,
  `rustscan-gs`, `rustscan-mesh`, `rustscan-sfm`, `rustscan-slam`, and
  `rustscan-viewer`. Keep each crate's implementation, tests, examples, and
  package configuration inside that crate. Shared CPU data types belong in `rustscan-types` or a documented
  crate-level API.
- `scripts/` contains human- or CI-invoked maintenance, conversion,
  generation, evaluation, and verification entry points. Scripts must write
  inputs, evidence, and generated runs to the matching `artifacts/` subtree.
- `.github/` contains CI and repository automation. CI must reinforce this
  layout and must not silently define a second directory or Agent protocol.
- `docs/` contains maintained architecture, design, status, and experiment
  records. `docs/agent/` is the active workflow and task protocol.
  Historical design records are not new task entry points.
- `third_party/native/` and `third_party/rust/` contain external native and
  Rust dependencies, submodules, and local patches. Do not rewrite them as
  ordinary application code; changes need explicit scope and dependency/build
  verification.
- `rustscan-gs/experiments/` contains package-local explanatory records. New
  generated data still belongs under `artifacts/runs/` or `artifacts/evidence/`.
  Ignored `rustscan-gs/output/` and `rustscan-sfm/output/` are crate-local runtime
  directories, not the removed top-level `output/` directory.

## Artifact ownership

All repository-level data lives under `artifacts/`:

- `artifacts/inputs/` contains reusable input datasets and test fixtures.
- `artifacts/runs/` contains generated outputs, caches, databases, logs,
  models, profiles, and other reproducible run artifacts. Large or regenerable
  files belong here and normally remain ignored.
- `artifacts/evidence/` contains versioned experiment summaries, reports, logs,
  scripts, and evidence referenced by documentation. Long-form reports may be
  grouped under a topic directory such as `gpu-five-point/` when they are
  historical evidence; raw and regenerable outputs still belong in
  `artifacts/runs/`.

Do not recreate the former top-level `test_data/`, `output/`, or `experiments/`
directories. Treat hard-linked images under `artifacts/runs/legacy/` as
read-only and copy them before modification.

## Local and generated state

- `target/` and crate-local build directories are Cargo output and remain
  ignored.
- `.worktrees/` contains local Git worktrees, not implementation code. Use Git
  worktree commands and never edit another task's worktree.
- `.ua/` is local Agent-app runtime state and is not project protocol or source.
- `.codex/`, `.claude/`, and `.vscode/` are not repository rule or task entry
  points. They must not be required for a portable checkout.

When a new directory does not fit this map, classify its ownership and output
type in the active task file before adding it. Do not create another top-level
input, output, experiment, vendor, or Agent-configuration directory without
updating this map and the relevant ignore rules.

## Path and naming migrations

- Component package names and crate directories MUST match `rustscan-<component>`
  in lowercase kebab-case; the root package remains `rustscan`. Rust imports
  derived from hyphenated packages use underscores, for example
  `rustscan-viewer` → `rustscan_viewer`.
- The current crate directories are `rustscan-ff/`, `rustscan-gs/`, `rustscan-mesh/`,
  `rustscan-sfm/`, `rustscan-slam/`, and `rustscan-viewer/`. Keep them stable.
- A package rename changes dependencies and potentially published APIs. A
  directory rename changes repository paths. Either requires a dedicated task,
  a complete reference search, workspace/CI/script/documentation updates, and
  verification on a case-sensitive filesystem.
- Do not confuse product or historical prose such as “RustSFM” with a path
  that must be changed. Update a reference only when it denotes a real path,
  package, target, or import.

The root `Cargo.lock` is authoritative for workspace builds. Do not add member
lockfiles; they are ignored by Cargo when the member belongs to this workspace.
Existing CLI names (`rustgs`, `rustsfm`, `rustslam`, `rust-viewer`), feature
flags, native symbols, and persisted data formats are compatibility contracts.
Rename them only in a separately scoped compatibility migration. For example:

```sh
cargo run -p rustscan-sfm --bin rustsfm -- --help
cargo run -p rustscan-gs --bin rustgs -- --help
```

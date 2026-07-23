# RustViewer Project Store Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist `.rustscanproject` packages without allowing malformed paths, concurrent writers,
partial artifact replacement, or process crashes to corrupt the last committed project state.

**Architecture:** `ProjectStore` is the only manifest writer. A worker writes a complete package-
relative payload beneath one stage attempt workspace. Commit validates and fsyncs that tree, then
atomically renames the entire tree into an immutable committed-attempt directory. One atomic
`project.json` replacement switches the authoritative artifact references and stage state; old
committed directories remain valid until later garbage collection. Opening a project holds an
exclusive package lock and deterministically recovers either the staging or unreferenced committed
directory for an interrupted lease.

**Tech Stack:** Rust 2021, serde/serde_json, uuid, blake3 streaming hashes, fs2 file locking,
tempfile-based integration tests.

---

## Invariants

1. `project.json` is the only orchestration authority and is written only by typed `ProjectStore`
   operations.
2. A committed `ArtifactRef` points into `Artifacts/{stage}/attempt-{attempt:08}/...`; committed
   attempt directories are immutable.
3. A stage commit performs exactly one directory rename and one manifest replacement. It never
   overwrites the directory referenced by the current manifest.
4. Before the manifest switch, a crash leaves the old manifest and old artifacts usable. After the
   switch, the new manifest references a fully synced committed directory.
5. At most one active lease and one package writer exist. Read-only project summaries do not acquire
   the writer lock or hash large artifacts.
6. General callers cannot replace `project.json`, mutate project identity, or persist configuration
   changes without the corresponding invalidation.
7. Schema v1 is the first published schema. Schema 0 has no invented migration semantics and returns
   `MigrationUnavailable`; future versions are rejected before typed deserialization.

## File Map

- Modify `RustViewer/Cargo.toml`: add only `blake3` and `fs2` for this plan.
- Modify `Cargo.lock`: resolved dependency metadata.
- Modify `RustViewer/src/project/mod.rs`: export the read-only and typed store API.
- Create `RustViewer/src/project/store.rs`: package core, lock, typed manifest persistence.
- Create `RustViewer/src/project/artifacts.rs`: streaming validation and immutable attempt commit.
- Create `RustViewer/src/project/events.rs`: append-only diagnostic JSONL records.
- Create `RustViewer/src/project/library.rs`: summaries, duplicate, delete, and reveal.
- Modify `RustViewer/src/project/manifest.rs`: add only validation needed for package-wide active-state
  and lease consistency.
- Modify `RustViewer/src/project/state.rs`: crate-only transition operations used by the store.
- Modify `RustViewer/tests/project_store.rs`: package, transaction, recovery, and library tests.

### Task 2A: Package Core, Locking, and Typed Manifest Persistence

**Files:**
- Modify: `RustViewer/Cargo.toml`
- Modify: `Cargo.lock`
- Create: `RustViewer/src/project/store.rs`
- Modify: `RustViewer/src/project/mod.rs`
- Modify: `RustViewer/src/project/manifest.rs`
- Modify: `RustViewer/src/project/state.rs`
- Test: `RustViewer/tests/project_store.rs`

- [x] **Step 1: Preserve the current package-core RED tests and add writer-lock coverage**

Keep the existing tests for suffix validation, empty destinations, package directories, atomic JSON,
future schema dispatch, malformed manifests, and path/symlink containment. Add a test that holds one
store open and requires a second writer open to fail:

```rust
#[test]
fn project_store_allows_only_one_writer() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("Flowers.rustscanproject");
    let first = ProjectStore::create(&path, create_request("Flowers")).unwrap();
    assert!(matches!(
        ProjectStore::open(&path),
        Err(ProjectStoreError::AlreadyOpen { .. })
    ));
    drop(first);
    ProjectStore::open(&path).unwrap();
}
```

Also prove that no public API can replace `project.json` with another UUID. The public API should not
expose generic manifest writes; integration tests should use only typed operations.

- [x] **Step 2: Run the package-core tests and verify RED**

Run:

```bash
cargo test -p rust-viewer --test project_store project_store_create_
cargo test -p rust-viewer --test project_store project_store_open_
cargo test -p rust-viewer --test project_store project_store_allows_only_one_writer
```

Expected: FAIL because the exclusive package lock and final typed API do not exist.

- [x] **Step 3: Implement package creation, open, and the exclusive lock**

Use `fs2::FileExt::try_lock_exclusive` on `project.lock`. `ProjectStore` owns the open lock file for
its full lifetime:

```rust
pub struct ProjectStore {
    root: PathBuf,
    manifest: ProjectManifest,
    lock_file: File,
}
```

Creation may populate an explicitly empty destination but must remove newly created partial package
contents when initialization fails. `open` rejects a symlink package root, canonicalizes the root,
acquires the lock before reading `project.json`, reads `schema_version` as `serde_json::Value`, runs
the sequential dispatcher, deserializes, and calls `ProjectManifest::validate()`.

`migrate_one_version` must explicitly return `MigrationUnavailable { from: 0, to: 1 }`; add a comment
that schema v1 is the first published format. Do not fabricate defaults for a schema that never
existed.

- [x] **Step 4: Restrict manifest writes to typed operations**

Make the generic JSON helper crate-private and refuse `project.json`; only this private function may
write the manifest:

```rust
fn write_manifest_atomic(&self, manifest: &ProjectManifest) -> Result<(), ProjectStoreError> {
    manifest.validate()?;
    if manifest.id != self.manifest.id {
        return Err(ProjectStoreError::ProjectIdentityMismatch {
            expected: self.manifest.id,
            found: manifest.id,
        });
    }
    write_bytes_atomic(&self.root.join("project.json"), &serde_json::to_vec_pretty(manifest)?)?;
    Ok(())
}
```

Creation uses a separate private bootstrap writer before `ProjectStore` has an existing identity.
Replace public `replace_manifest` with typed crate-only operations. Configuration updates take a
closure or explicit snapshot plus `ChangeKind`, call `invalidate`, validate, persist, then update
memory. Stage transition methods remain crate-only so Task 6's coordinator becomes the public state
authority.

- [x] **Step 5: Validate package-wide active-state consistency**

Extend manifest validation so persisted input cannot contain two active stages, an active stage
without the matching lease, or a lease while another stage is active:

```rust
let active = ProjectStage::ORDER
    .into_iter()
    .filter(|stage| matches!(self.try_stage(*stage)?.state(),
        StageState::Running | StageState::PauseRequested | StageState::CancelRequested))
    .collect::<Vec<_>>();
```

Require `active.len() <= 1`; require `lease.is_some()` exactly when one active stage exists, and
require stage/attempt/project identity equality. Add malformed JSON fixtures for every violation.

- [x] **Step 6: Stream physical artifact validation on open**

For every stage artifact plus `active_scene` and `final_scene`, reject missing files, symlinks,
directories, canonical paths outside the package, byte-length mismatch, and BLAKE3 mismatch. Hash
with a fixed-size buffer:

```rust
fn hash_reader(mut reader: impl Read) -> io::Result<(u64, blake3::Hash)> {
    let mut hasher = blake3::Hasher::new();
    let mut total = 0_u64;
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 { break; }
        hasher.update(&buffer[..count]);
        total = total.checked_add(count as u64).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "artifact length overflow")
        })?;
    }
    Ok((total, hasher.finalize()))
}
```

Do not hash artifacts in `list_summaries`; only a fully opened project pays this cost.

- [x] **Step 7: Run Task 2A verification and commit**

Run:

```bash
cargo test -p rust-viewer --test project_store project_store_create_
cargo test -p rust-viewer --test project_store project_store_open_
cargo test -p rust-viewer --test project_store project_store_rejects_
cargo test -p rust-viewer --lib project::state::tests
cargo fmt --all -- --check
git diff --check
```

Expected: all selected tests pass. The known unrelated COLMAP loader failure remains outside scope.

Commit:

```bash
git add RustViewer/Cargo.toml Cargo.lock RustViewer/src/project RustViewer/tests/project_store.rs
git commit -m "feat(viewer): persist locked project packages"
```

### Task 2B: Immutable Artifact Commit, Lease Recovery, and Events

**Files:**
- Create: `RustViewer/src/project/artifacts.rs`
- Create: `RustViewer/src/project/events.rs`
- Modify: `RustViewer/src/project/store.rs`
- Modify: `RustViewer/src/project/mod.rs`
- Test: `RustViewer/tests/project_store.rs`

- [x] **Step 1: Replace destination-swap tests with immutable-attempt RED tests**

A stage workspace mirrors package-relative payload paths:

```text
Cache/.staging/import-1/
  Sources/source.json
  Cache/frames/frames.json
```

After commit, one directory rename produces:

```text
Artifacts/import/attempt-00000001/
  Sources/source.json
  Cache/frames/frames.json
```

The manifest references the latter paths. Add tests proving the old committed attempt directory and
old manifest references remain unchanged when validation, rename, fsync, or manifest persistence is
injected to fail.

- [x] **Step 2: Add deterministic commit failpoints and verify RED**

Under `cfg(test)`, route the internal commit through:

```rust
enum CommitFailpoint {
    None,
    AfterWorkspaceSync,
    AfterAttemptRename,
    BeforeManifestWrite,
}
```

For every failpoint: start from a previously succeeded artifact set, invalidate and begin a new
attempt, inject failure, drop the store, reopen, and assert the prior artifact bytes and references
remain valid while the abandoned attempt is preserved under `Logs/recovery`.

- [x] **Step 3: Implement streaming validation and one-directory commit**

Replace `StagedArtifact { source, destination }` with one package-relative payload path and a
validation kind. Reject empty declarations, duplicate paths, unsafe components, symlinks, missing
files, malformed JSON, and undeclared files when strict mode is enabled. Stream BLAKE3 and length;
never call `fs::read` for `ReadableFile`.

Fsync every declared file and containing directory, then rename the workspace directory once to
`Artifacts/{stage}/attempt-{attempt:08}` and fsync the stage artifact parent. Construct final
`ArtifactRef` values only after the rename.

- [x] **Step 4: Switch manifest authority once**

Build one new manifest in memory that commits validated artifacts, sets the stage to `Succeeded`,
clears the lease, and refreshes readiness. Persist it with one atomic `project.json` replacement,
then update the in-memory manifest. If the write fails, keep the old manifest in memory and leave the
new attempt directory unreferenced for deterministic recovery. Never write an intermediate Running
manifest containing new artifact references.

- [x] **Step 5: Implement typed stage control and JSONL events**

Add crate-only store methods for begin, pause request, cancel request, paused, cancelled, failed, and
success. `begin_stage` rejects any existing lease, securely creates an empty workspace, then persists
the new Running state and lease before it returns the workspace to a worker. If lease persistence
fails, preserve or remove the unused workspace without changing the in-memory manifest. Never return
a workspace to a worker before the lease is durable.

Append events only after the authoritative manifest operation succeeds. Before appending, detect a
file that does not end in `\n` and add one separator byte so a corrupt trailing line is preserved and
the next valid event remains parseable. Flush and `sync_data` after every record. Manifest changes
remain successful if diagnostic logging fails; expose such failures through a non-fatal warning list
instead of returning an error that invites the caller to repeat a committed operation.

- [x] **Step 6: Implement deterministic interrupted recovery**

On open, an active lease means interrupted work because the exclusive writer lock has already been
acquired. For its exact stage/attempt:

- Move `Cache/.staging/{stage}-{attempt}` to a collision-safe `Logs/recovery/...` if present.
- Move an unreferenced `Artifacts/{stage}/attempt-{attempt:08}` there if the crash occurred after
  rename but before the manifest switch.
- Preserve every artifact referenced by the old manifest.
- Transition only Running/PauseRequested/CancelRequested to Failed with code `interrupted`.
- Clear the lease, atomically persist, and append a recovery event.
- Do not queue retry automatically.

- [x] **Step 7: Run Task 2B verification and commit**

Run:

```bash
cargo test -p rust-viewer --test project_store project_store_commits_
cargo test -p rust-viewer --test project_store project_store_preserves_
cargo test -p rust-viewer --test project_store project_store_logs_
cargo test -p rust-viewer --test project_store project_store_recovers_
cargo fmt --all -- --check
git diff --check
```

Commit:

```bash
git add RustViewer/src/project RustViewer/tests/project_store.rs
git commit -m "feat(viewer): commit recoverable project artifacts"
```

### Task 2C: Project Library Operations

**Files:**
- Create: `RustViewer/src/project/library.rs`
- Modify: `RustViewer/src/project/store.rs`
- Modify: `RustViewer/src/project/mod.rs`
- Test: `RustViewer/tests/project_store.rs`

- [ ] **Step 1: Write summary, duplicate, delete, and reveal RED tests**

Define a lightweight result that does not load or hash artifacts:

```rust
pub struct ProjectSummary {
    pub id: Uuid,
    pub display_name: String,
    pub root: PathBuf,
    pub updated_unix_ms: u64,
    pub stages: BTreeMap<ProjectStage, StageState>,
    pub thumbnail: Option<ArtifactRef>,
    pub status: ProjectSummaryStatus,
}

pub enum ProjectSummaryEntry {
    Project(ProjectSummary),
    Invalid { root: PathBuf, error: String },
}
```

Test valid and corrupt packages in one library, duplicate UUID replacement, refusal to duplicate an
active lease, exclusion of `Cache/.staging`, `project.lock`, and old logs, exact-ID deletion, nested or
self destinations, symlink roots, and canonical reveal paths.

- [ ] **Step 2: Run the library tests and verify RED**

Run:

```bash
cargo test -p rust-viewer --test project_store project_library_
```

Expected: FAIL because `library.rs` and typed operations do not exist.

- [ ] **Step 3: Implement lightweight summaries**

Scan only immediate `.rustscanproject` children. Read `project.json` as a value, reject future schema,
deserialize and run structural manifest validation, but do not canonicalize/hash every artifact.
Return one `ProjectSummaryEntry` per candidate so a corrupt project does not hide valid projects.
Sort newest first, then case-insensitive display name, then UUID for determinism.

- [ ] **Step 4: Implement atomic duplicate**

Reject self, descendant, symlink, non-suffixed, and non-empty destinations. Refuse duplication while
the source has an active lease. Copy through a sibling temporary package, skipping `project.lock`,
`Cache/.staging`, `Logs`, and unreferenced artifact attempt directories. Generate a new UUID, clear
the lease, validate all copied committed artifact references, write a fresh event log containing a
`duplicated_from` record, fsync, then rename the temporary package into place.

- [ ] **Step 5: Implement exact-ID delete and reveal**

`delete(self, confirmation_id)` consumes the locked store, compares the exact UUID, verifies the
canonical root still has the `.rustscanproject` suffix and is not a symlink, releases the lock, and
removes only that root. `reveal_path(&self)` returns the canonical root without invoking Finder.

- [ ] **Step 6: Run Task 2C and full Task 2 verification**

Run:

```bash
cargo test -p rust-viewer --test project_store project_library_
cargo test -p rust-viewer --test project_store
cargo test -p rust-viewer --lib project::state::tests
cargo fmt --all -- --check
git diff --check
```

Expected: all Task 1 and Task 2 tests pass. A full `cargo test -p rust-viewer` may still report only
the pre-existing `loader::colmap::tests::loads_colmap_and_maps_scene` failure.

- [ ] **Step 7: Commit Task 2C**

```bash
git add RustViewer/src/project RustViewer/tests/project_store.rs
git commit -m "feat(viewer): manage project library packages"
```

After each of Tasks 2A, 2B, and 2C, perform independent specification review followed by independent
code-quality review. Do not begin the next task until both reviews pass.

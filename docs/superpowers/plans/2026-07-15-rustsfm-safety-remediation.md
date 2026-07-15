# RustSFM Safety Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make RustSFM native allocation, SQLite mutation, external count parsing, and one-time native initialization failure-safe.

**Architecture:** Introduce narrow validation and transaction boundaries around existing APIs. Keep public reconstruction behavior unchanged for valid inputs while turning malformed or unsupported inputs into contextual `Result` errors.

**Tech Stack:** Rust 2021, rusqlite, anyhow, C11, FreeImage, VLFeat, tempfile.

---

### Task 1: Failure-Atomic VLFeat Buffer Growth

**Files:**
- Modify: `RustSFM/src/native/vlfeat_sift.c`
- Modify: `RustSFM/src/native/vlfeat_sift.h`
- Modify: `RustSFM/src/sift.rs`

- [ ] **Step 1: Add a failing allocator-injection regression test**

Add a test-only native entry point that runs both paired-growth operations through an
allocator which fails on the second allocation. The Rust test must assert that the entry
point reports failure and that every allocated block is released exactly once.

```rust
#[test]
fn vlfeat_paired_growth_is_failure_atomic() {
    let report = unsafe { rustsfm_vlfeat_test_paired_growth_failure() };
    assert_eq!(report.result, 0);
    assert_eq!(report.live_allocations, 0);
    assert_eq!(report.double_frees, 0);
}
```

- [ ] **Step 2: Run the regression test and confirm the old paired realloc path fails**

Run: `cargo test -p rustsfm --lib vlfeat_paired_growth_is_failure_atomic -- --nocapture`

Expected: the assertion detects a double free, dangling owner, or non-zero live allocation.

- [ ] **Step 3: Replace paired realloc with allocate-copy-commit**

Use the same ownership sequence for `LevelData` buffers and the top-level level arrays:

```c
void* first = alloc(first_bytes);
if (first == NULL) return 0;
void* second = alloc(second_bytes);
if (second == NULL) {
    dealloc(first);
    return 0;
}
memcpy(first, old_first, used_first_bytes);
memcpy(second, old_second, used_second_bytes);
dealloc(old_first);
dealloc(old_second);
owner->first = first;
owner->second = second;
```

- [ ] **Step 4: Run the fault test and normal VLFeat extraction tests**

Run: `cargo test -p rustsfm --lib vlfeat_paired_growth -- --nocapture`

Expected: all selected tests pass with zero leaked or multiply freed allocations.

### Task 2: Atomic Feature-Matching Database Replacement

**Files:**
- Modify: `RustSFM/src/feature/feature_matching_db.rs`
- Modify: `RustSFM/src/io/database.rs`

- [ ] **Step 1: Add a failing preservation test**

Create a temporary database containing a valid pre-existing match and only one usable image.
Call `match_features_to_database` with `clear_existing=true` and assert the call fails while
the original match and two-view geometry remain unchanged.

```rust
let before = db.read_matches(image1, image2)?;
assert!(match_features_to_database(path, &options).is_err());
let after = ColmapDatabase::open(path)?.read_matches(image1, image2)?;
assert_eq!(after, before);
```

- [ ] **Step 2: Run the test and verify existing rows are deleted**

Run: `cargo test -p rustsfm --lib matching_failure_preserves_existing_database -- --nocapture`

Expected: FAIL because the current implementation clears before validating image count.

- [ ] **Step 3: Validate before mutation and add rollback support**

Move image, camera, descriptor, and pair-batch preparation before destructive statements.
Execute clear and replacement statements inside a transaction helper which rolls back on
drop unless explicitly committed.

- [ ] **Step 4: Verify success replacement and failure rollback**

Run: `cargo test -p rustsfm --lib feature_matching_db::tests -- --nocapture`

Expected: both preservation and existing success-path tests pass.

### Task 3: Explicit Read-Only Database Opening

**Files:**
- Modify: `RustSFM/src/io/database.rs`
- Modify call sites under: `RustSFM/src/compare/`, `RustSFM/src/sfm/`

- [ ] **Step 1: Add a failing read-only hash test**

Create and populate a database, close it, mark it read-only, record its bytes, open it through
the new `open_read_only` API, read cameras and images, then assert the bytes are unchanged.

- [ ] **Step 2: Run the test and confirm the current open path requires mutation**

Run: `cargo test -p rustsfm --lib database_read_only_open_preserves_bytes -- --nocapture`

Expected: FAIL because `open` executes schema setup and migration statements.

- [ ] **Step 3: Add separate open modes**

Implement `open_read_only` with `SQLITE_OPEN_READ_ONLY | SQLITE_OPEN_URI`, and keep migrations
only in `open` or a named `open_writable` constructor. Update read-only command paths.

- [ ] **Step 4: Run database and mapper tests**

Run: `cargo test -p rustsfm --lib database::tests mapper::tests --no-fail-fast`

Expected: all tests pass and the read-only byte comparison remains equal.

### Task 4: Bounded Counts And Checked Integer Conversion

**Files:**
- Modify: `RustSFM/src/io/colmap.rs`
- Modify: `RustSFM/src/io/database.rs`
- Modify: `RustSFM/src/core/correspondence_graph.rs`

- [ ] **Step 1: Add malformed-input tests**

Cover `u64::MAX` top-level and nested COLMAP counts, negative SQLite rows, oversized camera
dimensions, and a pair ID whose quotient would wrap through `u32`.

- [ ] **Step 2: Run tests and confirm panic or aliasing behavior**

Run: `cargo test -p rustsfm --lib malformed_count negative_keypoint pair_id_rejects -- --nocapture`

Expected: current code panics or returns the aliased `(1, 2)` pair.

- [ ] **Step 3: Introduce checked count helpers**

The binary helper must validate both a resource ceiling and the number of records possible
from the remaining file bytes before reserving:

```rust
fn checked_count(raw: u64, remaining: u64, minimum_record_bytes: u64, label: &str) -> Result<usize> {
    let count = usize::try_from(raw).with_context(|| format!("{label} does not fit usize"))?;
    if raw > remaining / minimum_record_bytes {
        bail!("{label} count {raw} exceeds remaining input capacity");
    }
    Ok(count)
}
```

Use `u32::try_from`, `usize::try_from`, and non-negative checks instead of `as` at external
input boundaries. Validate pair IDs before calculating or narrowing either component.

- [ ] **Step 4: Run COLMAP, database, and correspondence tests**

Run: `cargo test -p rustsfm --lib colmap::tests database::tests correspondence_graph::tests --no-fail-fast`

Expected: malformed inputs return errors and all valid fixtures continue to pass.

### Task 5: Thread-Safe FreeImage Initialization

**Files:**
- Modify: `RustSFM/src/native/colmap_image.c`
- Modify: `RustSFM/src/sfm/mapper/image_features.rs`

- [ ] **Step 1: Add a concurrent first-use test**

Start multiple threads at a barrier and have every thread load the same temporary image
through `load_colmap_grayscale_u8`.

- [ ] **Step 2: Run the test under ThreadSanitizer where available**

Run: `cargo test -p rustsfm --lib freeimage_concurrent_first_use -- --nocapture`

Expected: normal runs complete; the pre-fix C static flag is reported by TSAN.

- [ ] **Step 3: Replace the static flag with a platform once primitive**

Use C11 `call_once` when available, with one initializer function that calls
`FreeImage_Initialise(FALSE)`. Do not expose initialization state to callers.

- [ ] **Step 4: Re-run concurrent and image-loading tests**

Run: `cargo test -p rustsfm --lib colmap_image -- --nocapture`

Expected: all tests pass and repeated concurrent calls remain stable.

### Task 6: Batch A Verification

**Files:**
- Modify: `docs/reviews/2026-07-15-rustsfm-rustgs-remediation-todo.md`

- [ ] **Step 1: Run the full RustSFM library suite**

Run: `cargo test -p rustsfm --lib --no-fail-fast`

Expected: all tests pass.

- [ ] **Step 2: Run formatting**

Run: `cargo fmt -p rustsfm -- --check`

Expected: exit code 0.

- [ ] **Step 3: Check completed Batch A tracker items**

Only mark items whose regression tests and implementation are both present.

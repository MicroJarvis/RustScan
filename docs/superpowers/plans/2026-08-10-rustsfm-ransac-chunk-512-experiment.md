# RustSFM GPU RANSAC 512-Chunk Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test a fixed GPU two-view RANSAC chunk size of 512 against the current size of 64, merging it only when fixed-seed SQLite match and geometry outputs are bit-for-bit identical.

**Architecture:** Add a canonical BLAKE3 fingerprint for the selected pairs' raw `matches` and `two_view_geometries` SQLite rows, expose it in each isolated benchmark run, and record a 64 baseline before changing the chunk constant. Compare the 512 bounded and full runs against binaries built from the same fingerprint implementation, with no public chunk-size option and no changes to matching semantics beyond the experimental constant.

**Tech Stack:** Rust, rusqlite, BLAKE3, serde, wgpu, Cargo tests, jq, the existing `benchmark-match-pairs` CLI.

---

### Task 1: Canonical Selected-Pair Output Fingerprint

**Files:**
- Modify: `RustSFM/Cargo.toml`
- Modify: `RustSFM/src/io/database.rs`
- Test: `RustSFM/src/io/database.rs`

- [ ] **Step 1: Add the direct BLAKE3 dependency and write the failing fingerprint test**

Add to `RustSFM/Cargo.toml`:

```toml
blake3 = "1"
```

In the `database.rs` test module, add these helpers and a test. The test calls the wished-for method
with forward and reversed pair-list order, proves the fingerprint is lowercase/order-independent,
then mutates and restores every covered raw column:

```rust
fn seed_fingerprint_pair(
    db: &ColmapDatabase,
    left: ImageId,
    right: ImageId,
    offset: u32,
) -> Result<()> {
    let matches = vec![m(offset, offset + 1), m(offset + 2, offset + 3)];
    db.write_matches(left, right, &matches)?;
    db.write_two_view_geometry(
        left,
        right,
        &ColmapTwoViewGeometry {
            config: COLMAP_TWO_VIEW_CALIBRATED,
            inlier_matches: vec![matches[0].clone()],
            f_matrix: Some([1.0 + offset as f64; 9]),
            e_matrix: Some([2.0 + offset as f64; 9]),
            h_matrix: Some([3.0 + offset as f64; 9]),
            qvec: Some([1.0, 0.0, 0.0, offset as f64]),
            tvec: Some([offset as f64, 1.0, 2.0]),
        },
    )?;
    Ok(())
}

fn assert_raw_pair_column_affects_fingerprint(
    db: &ColmapDatabase,
    pair_id: i64,
    table: &str,
    column: &str,
    replacement: rusqlite::types::Value,
    pairs: &[(ImageId, ImageId)],
    baseline: &str,
) -> Result<()> {
    let select = format!("SELECT {column} FROM {table} WHERE pair_id = ?1");
    let update = format!("UPDATE {table} SET {column} = ?1 WHERE pair_id = ?2");
    let original = db
        .conn
        .query_row(&select, params![pair_id], |row| row.get::<_, rusqlite::types::Value>(0))?;
    db.conn.execute(&update, params![replacement, pair_id])?;
    assert_ne!(db.selected_pair_output_fingerprint(pairs)?, baseline);
    db.conn.execute(&update, params![original, pair_id])?;
    assert_eq!(db.selected_pair_output_fingerprint(pairs)?, baseline);
    Ok(())
}

#[test]
fn selected_pair_output_fingerprint_is_order_independent_and_bit_exact() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("database.db");
    let db = ColmapDatabase::open(&path)?;
    let pairs = [(1, 2), (2, 3)];
    seed_fingerprint_pair(&db, 1, 2, 3)?;
    seed_fingerprint_pair(&db, 2, 3, 4)?;

    let baseline = db.selected_pair_output_fingerprint(&pairs)?;
    assert_eq!(baseline.len(), 64);
    assert!(baseline
        .bytes()
        .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()));
    assert_eq!(
        baseline,
        db.selected_pair_output_fingerprint(&[(3, 2), (2, 1)])?
    );

    let pair_id = image_pair_to_pair_id(1, 2).unwrap() as i64;
    for (table, column, replacement) in [
        ("matches", "rows", rusqlite::types::Value::Integer(17)),
        ("matches", "cols", rusqlite::types::Value::Integer(19)),
        ("matches", "data", rusqlite::types::Value::Blob(vec![0x01])),
        ("two_view_geometries", "rows", rusqlite::types::Value::Integer(23)),
        ("two_view_geometries", "cols", rusqlite::types::Value::Integer(29)),
        ("two_view_geometries", "data", rusqlite::types::Value::Blob(vec![0x02])),
        ("two_view_geometries", "config", rusqlite::types::Value::Integer(31)),
        ("two_view_geometries", "F", rusqlite::types::Value::Blob(vec![0x03; 72])),
        ("two_view_geometries", "E", rusqlite::types::Value::Blob(vec![0x04; 72])),
        ("two_view_geometries", "H", rusqlite::types::Value::Blob(vec![0x05; 72])),
        ("two_view_geometries", "qvec", rusqlite::types::Value::Blob(vec![0x06; 32])),
        ("two_view_geometries", "tvec", rusqlite::types::Value::Blob(vec![0x07; 24])),
    ] {
        assert_raw_pair_column_affects_fingerprint(
            &db,
            pair_id,
            table,
            column,
            replacement,
            &pairs,
            &baseline,
        )?;
    }
    Ok(())
}
```

Do not compare whole SQLite file bytes.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features database::tests::selected_pair_output_fingerprint \
  -- --nocapture
```

Expected: compilation fails because `selected_pair_output_fingerprint` does not exist.

- [ ] **Step 3: Implement the raw-row canonical encoder**

Add private length-prefixed hashing helpers near the database blob encoders:

```rust
fn fingerprint_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn fingerprint_optional_blob(hasher: &mut blake3::Hasher, blob: Option<&[u8]>) {
    match blob {
        Some(bytes) => {
            hasher.update(&[1]);
            fingerprint_bytes(hasher, bytes);
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn fingerprint_i64(hasher: &mut blake3::Hasher, value: i64) {
    hasher.update(&value.to_le_bytes());
}
```

Add this public read-only method on `ColmapDatabase`:

```rust
pub fn selected_pair_output_fingerprint(
    &self,
    image_pairs: &[(ImageId, ImageId)],
) -> Result<String> {
    let mut pair_ids = image_pairs
        .iter()
        .map(|&(left, right)| {
            image_pair_to_pair_id(left, right).map_err(|err| anyhow::anyhow!("{err:?}"))
        })
        .collect::<Result<Vec<_>>>()?;
    pair_ids.sort_unstable();
    pair_ids.dedup();
    if pair_ids.len() != image_pairs.len() {
        bail!("selected-pair fingerprint requires unique canonical pairs");
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update(b"rustsfm-selected-pair-output-v1\0");
    hasher.update(&(pair_ids.len() as u64).to_le_bytes());
    for pair_id in pair_ids {
        hasher.update(&(pair_id as u64).to_le_bytes());
        let matches = self
            .conn
            .query_row(
                "SELECT rows, cols, data FROM matches WHERE pair_id = ?1",
                params![pair_id as i64],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, Option<Vec<u8>>>(2)?,
                    ))
                },
            )
            .optional()?;
        match matches {
            None => {
                hasher.update(&[0]);
            }
            Some((rows, cols, data)) => {
                hasher.update(&[1]);
                fingerprint_i64(&mut hasher, rows);
                fingerprint_i64(&mut hasher, cols);
                fingerprint_optional_blob(&mut hasher, data.as_deref());
            }
        }

        let geometry = self
            .conn
            .query_row(
                "SELECT rows, cols, data, config, F, E, H, qvec, tvec
                 FROM two_view_geometries WHERE pair_id = ?1",
                params![pair_id as i64],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, Option<Vec<u8>>>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, Option<Vec<u8>>>(4)?,
                        row.get::<_, Option<Vec<u8>>>(5)?,
                        row.get::<_, Option<Vec<u8>>>(6)?,
                        row.get::<_, Option<Vec<u8>>>(7)?,
                        row.get::<_, Option<Vec<u8>>>(8)?,
                    ))
                },
            )
            .optional()?;
        match geometry {
            None => {
                hasher.update(&[0]);
            }
            Some((rows, cols, data, config, f, e, h, qvec, tvec)) => {
                hasher.update(&[1]);
                fingerprint_i64(&mut hasher, rows);
                fingerprint_i64(&mut hasher, cols);
                fingerprint_optional_blob(&mut hasher, data.as_deref());
                fingerprint_i64(&mut hasher, config);
                for blob in [&f, &e, &h, &qvec, &tvec] {
                    fingerprint_optional_blob(&mut hasher, blob.as_deref());
                }
            }
        }
    }
    Ok(hasher.finalize().to_hex().to_string())
}
```

Use these exact read-only queries, preserving `NULL` versus empty blob:

```sql
SELECT rows, cols, data FROM matches WHERE pair_id = ?1;
SELECT rows, cols, data, config, F, E, H, qvec, tvec
FROM two_view_geometries WHERE pair_id = ?1;
```

Hash row absence with tag `0` and row presence with tag `1`. Do not decode matches, matrices, or
poses, and do not hash timing or unrelated pairs.

- [ ] **Step 4: Verify GREEN and the database regressions**

Run:

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features database::tests::selected_pair_output_fingerprint \
  -- --nocapture
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features database::tests::matches_roundtrip \
  -- --nocapture
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features database::tests::two_view_geometry_roundtrip \
  -- --nocapture
cargo fmt --all -- --check
git diff --check
```

Expected: all focused tests and checks pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add RustSFM/Cargo.toml Cargo.lock RustSFM/src/io/database.rs
git commit -m "feat(rustsfm): fingerprint pair benchmark outputs"
```

### Task 2: Benchmark Fingerprint Reporting

**Files:**
- Modify: `RustSFM/src/diagnostics/match_pair_benchmark.rs`
- Test: `RustSFM/src/diagnostics/match_pair_benchmark.rs`

- [ ] **Step 1: Write failing benchmark and serde tests**

Add a serde-defaulted field:

```rust
#[serde(default)]
pub result_fingerprint: String,
```

to the wished-for `MatchPairBenchmarkRun` in the tests first. Extend
`match_pair_benchmark_repeats_without_modifying_source_database` to assert:

```rust
assert!(report.runs.iter().all(|run| run.result_fingerprint.len() == 64));
assert_eq!(
    report.runs.iter().map(|run| &run.result_fingerprint).collect::<BTreeSet<_>>().len(),
    1
);
```

Add `match_pair_benchmark_run_deserializes_without_fingerprint` using a legacy JSON run and assert
that `result_fingerprint` defaults to the empty string.

- [ ] **Step 2: Run tests and verify RED**

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features match_pair_benchmark -- --nocapture
```

Expected: compile failure because the production run DTO does not contain `result_fingerprint`.

- [ ] **Step 3: Compute the fingerprint before each temporary run database is dropped**

Add the serde-defaulted field to `MatchPairBenchmarkRun`. Immediately after matching and before
constructing the run DTO, open the working database read-only and compute:

```rust
let result_fingerprint = ColmapDatabase::open_read_only(&working_database)?
    .selected_pair_output_fingerprint(&image_pairs)
    .with_context(|| format!("fingerprint benchmark run {} outputs", run_index + 1))?;
```

Store it in the run DTO. Do not include fingerprint time in `matching_seconds` or any timing field.

- [ ] **Step 4: Verify and commit Task 2**

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features match_pair_benchmark -- --nocapture
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --bin rustsfm --no-default-features benchmark_match_pairs -- --nocapture
cargo fmt --all -- --check
git diff --check
git add RustSFM/src/diagnostics/match_pair_benchmark.rs
git commit -m "feat(rustsfm): report benchmark result fingerprint"
```

Expected: library and CLI tests pass; the source database preservation test remains byte-for-byte
unchanged.

### Task 3: Record The 64-Chunk Baseline

**Files:**
- Modify: `docs/superpowers/plans/2026-08-10-rustsfm-ransac-chunk-512-experiment.md`

- [ ] **Step 1: Prove the source still uses 64 and run pre-baseline regressions**

```bash
rg -n "GPU_RANSAC_CHUNK_TRIALS: usize = 64" RustSFM/src/geometry/two_view.rs
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu gpu_ransac_chunk_ -- --nocapture
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo check -p rustsfm \
  --no-default-features --features gpu-wgpu
```

Expected: the constant search finds exactly one line, the chunk test passes, and the check succeeds.

- [ ] **Step 2: Build and preserve the 64 release binary**

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo build -p rustsfm \
  --release --no-default-features --features gpu-wgpu
cp /Users/tfjiang/Projects/RustScan/target/release/rustsfm \
  /tmp/rustsfm-gpu-ransac-chunk-64
```

Verify `/tmp/rustsfm-gpu-ransac-chunk-64` is executable. This binary includes fingerprint reporting
and retains chunk 64 even after the source changes.

- [ ] **Step 3: Ensure no benchmark is running and run the 64 bounded baseline**

Run exactly one non-sandboxed generic-wgpu benchmark:

```bash
/tmp/rustsfm-gpu-ransac-chunk-64 benchmark-match-pairs \
  --database /Users/tfjiang/Projects/RustScan/test_data/flowers2/out9/Untitled.rustscanproject/Cache/.staging/keyframe_sfm-1/rustsfm/Cache/database.db \
  --window 5 --pair-limit 96 --repetitions 3 --use-gpu --random-seed 0 \
  --output-json /tmp/rustsfm-gpu-ransac-chunk-64-96x3.json
```

- [ ] **Step 4: Validate and record the baseline**

```bash
jq -e '
  .pair_count == 96 and .repetitions == 3 and
  all(.runs[]; .matched_pairs == 96 and .verified_pairs == 96 and
      .total_matches == 62409 and (.result_fingerprint | length) == 64) and
  ([.runs[].result_fingerprint] | unique | length) == 1 and
  all(.. | numbers; (isnan | not) and (isinfinite | not))
' /tmp/rustsfm-gpu-ransac-chunk-64-96x3.json
```

Append the exact fingerprint, three matching times, geometry/scorer calls, readback waits, and binary
SHA-256 to this plan. Do not commit the binary or JSON.

#### Task 3 evidence (2026-08-10)

- Source proof: `rg` found exactly `2198:const GPU_RANSAC_CHUNK_TRIALS: usize = 64;`.
  The targeted chunk test passed (`1 passed; 0 failed; 658 filtered out`), and the requested
  `cargo check` and release `cargo build` both exited 0; all three commands emitted only dead-code
  warnings.
- Preserved executable: `/tmp/rustsfm-gpu-ransac-chunk-64` (executable Mach-O arm64), SHA-256
  `880c7d726f789621d641f89ad6dbae7c990137adb29cda6789dbda57d8792541`. It was built from commit
  `c8f28b425b7611f7b51acfd11bd2c5ca14d1e1b1` with source tree
  `a587bd507e70bb2cdb4e6734f36b53a2d10da1e0`.
- Benchmark: generic wgpu with no forced backend, source database from the command above, `window=5`,
  `pair_limit=96`, `repetitions=3`, GPU enabled, and `random_seed=0`; the process gate found no
  running `benchmark-match-pairs` or `rustsfm`. The sandbox probe had no compatible adapter, so the
  required non-sandboxed run produced `/tmp/rustsfm-gpu-ransac-chunk-64-96x3.json` using a
  635285504-byte database snapshot. The baseline JSON SHA-256 is
  `4c56571897acfd0dd8d95079d26ad78cb0196fd5bce15fc28f1cf2b28cdea159`; the source database
  SHA-256 is `dcf79fa307a6294195a8e5db1cddb185bbc1baca2ee490061b89f2a5961a052c`. Backend was
  `wgpu_match_and_score` in every run.
- Selected-pair database output was bit-identical in runs 1-3: `matched_pairs=96`, `verified_pairs=96`,
  `total_matches=62409`, fingerprint
  `5e05ca629b63c98ae63c95ce0f37fe49a43eb870760e598352c8f8ef3d84e8ed`.
- Run 1: `matching_seconds=18.969614875`, descriptor/geometry seconds
  `6.5157951700000005`/`12.074102664999998`; matcher direction/readback calls `192`/`192`, matcher
  readback wait `6.199151996999999`. Essential, fundamental, and homography scorer
  score/mask/readback calls were `192/249/441`, `192/180/372`, and `5066/656/5722`; their readback
  waits were `0.670638922`, `0.5658994169999998`, and `8.644388290000006` seconds.
- Run 2: `matching_seconds=18.899943125`, descriptor/geometry seconds
  `6.506449707000002`/`11.989519454999998`; matcher direction/readback calls `192`/`192`, matcher
  readback wait `6.199279754000002`. Scorer call counts matched run 1; essential, fundamental, and
  homography readback waits were `0.6716153760000001`, `0.566310135`, and `8.638828708` seconds.
- Run 3: `matching_seconds=18.926185916`, descriptor/geometry seconds
  `6.503120993999999`/`12.053072455000002`; matcher direction/readback calls `192`/`192`, matcher
  readback wait `6.190427672000002`. Scorer call counts matched run 1; essential, fundamental, and
  homography readback waits were `0.672624138`, `0.5658347960000001`, and `8.655256307` seconds.
- The exact `jq -e` gate above returned `true` with exit 0, including the finite-number check and the
  single-fingerprint check.

- [ ] **Step 5: Commit Task 3 evidence**

```bash
git add docs/superpowers/plans/2026-08-10-rustsfm-ransac-chunk-512-experiment.md
git commit -m "docs(rustsfm): record 64-chunk fingerprint baseline"
```

### Task 4: Change The GPU RANSAC Chunk To 512 With TDD

**Files:**
- Modify: `RustSFM/src/geometry/two_view.rs`
- Test: `RustSFM/src/geometry/two_view.rs`

- [ ] **Step 1: Change only the boundary test and verify RED**

Change the first assertion in `gpu_ransac_chunk_end_applies_dynamic_limits_at_boundaries` and add a
post-first-chunk dynamic-stop assertion:

```rust
assert_eq!(gpu_ransac_chunk_end(0, 10_000, 10_000, 100), 512);
assert_eq!(gpu_ransac_chunk_end(512, 10_000, 24, 100), 101);
assert_eq!(gpu_ransac_chunk_end(9_980, 10_000, usize::MAX, 100), 10_000);
```

Run:

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu gpu_ransac_chunk_ -- --nocapture
```

Expected: FAIL because the first result remains 64.

- [ ] **Step 2: Make the minimal production change and verify GREEN**

Change exactly:

```rust
const GPU_RANSAC_CHUNK_TRIALS: usize = 512;
```

Run the same command. Expected: the boundary test passes.

- [ ] **Step 3: Run GPU geometry regressions**

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu gpu_geometry_profiled \
  -- --nocapture
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu two_view::tests \
  -- --nocapture
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu controlled_computed_matching_ \
  -- --nocapture
cargo fmt --all -- --check
git diff --check
```

Expected: all deterministic CPU tests pass; adapter-required tests either pass on a real adapter or
take their existing explicit skip path. No pair ordering, transaction, or progress test changes.

- [ ] **Step 4: Commit Task 4**

```bash
git add RustSFM/src/geometry/two_view.rs
git commit -m "perf(rustsfm): test 512-trial gpu ransac chunks"
```

### Task 5: Strict 512 Benchmark Gate

**Files:**
- Modify: `docs/superpowers/plans/2026-08-10-rustsfm-ransac-chunk-512-experiment.md`

Before any 512 benchmark or full comparison, reverify all three preserved baseline inputs and stop if
any value differs:

```bash
shasum -a 256 /tmp/rustsfm-gpu-ransac-chunk-64
shasum -a 256 /tmp/rustsfm-gpu-ransac-chunk-64-96x3.json
shasum -a 256 /Users/tfjiang/Projects/RustScan/test_data/flowers2/out9/Untitled.rustscanproject/Cache/.staging/keyframe_sfm-1/rustsfm/Cache/database.db
```

Require, in command order,
`880c7d726f789621d641f89ad6dbae7c990137adb29cda6789dbda57d8792541`,
`4c56571897acfd0dd8d95079d26ad78cb0196fd5bce15fc28f1cf2b28cdea159`, and
`dcf79fa307a6294195a8e5db1cddb185bbc1baca2ee490061b89f2a5961a052c`.

- [ ] **Step 1: Build the 512 release CLI**

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo build -p rustsfm \
  --release --no-default-features --features gpu-wgpu
```

- [ ] **Step 2: Run the bounded candidate with no concurrent benchmark**

```bash
/Users/tfjiang/Projects/RustScan/target/release/rustsfm benchmark-match-pairs \
  --database /Users/tfjiang/Projects/RustScan/test_data/flowers2/out9/Untitled.rustscanproject/Cache/.staging/keyframe_sfm-1/rustsfm/Cache/database.db \
  --window 5 --pair-limit 96 --repetitions 3 --use-gpu --random-seed 0 \
  --output-json /tmp/rustsfm-gpu-ransac-chunk-512-96x3.json
```

- [ ] **Step 3: Enforce the strict bounded gate**

```bash
jq -e --slurpfile baseline /tmp/rustsfm-gpu-ransac-chunk-64-96x3.json '
  def numeric_fields($fields):
    . as $object |
    ($object | type) == "object" and
    all($fields[]; . as $field |
        ($object | has($field)) and ($object[$field] | type) == "number");
  def scorer_shape:
    numeric_fields(["score_calls", "mask_calls", "readback_calls", "readback_wait_seconds"]);
  def geometry_detail_shape:
    . as $detail |
    ($detail | type) == "object" and
    all(["essential", "fundamental", "homography"][]; . as $model |
        ($detail | has($model)) and ($detail[$model] | type) == "object" and
        ($detail[$model] | has("scorer")) and ($detail[$model].scorer | scorer_shape));
  def timings_shape:
    numeric_fields([
      "backend_initialization_seconds", "database_prepare_seconds", "pair_compute_seconds",
      "database_commit_seconds", "event_sink_seconds", "unclassified_seconds",
      "attempted_pairs", "produced_pair_reports", "committed_batches",
      "gpu_descriptor_match_seconds", "gpu_geometry_seconds", "gpu_descriptor_pack_seconds",
      "gpu_buffer_prepare_seconds", "gpu_submit_seconds", "gpu_readback_total_seconds",
      "gpu_readback_copy_submit_seconds", "gpu_readback_wait_seconds",
      "gpu_readback_map_decode_seconds", "gpu_cpu_postprocess_seconds",
      "gpu_match_direction_calls", "gpu_readback_calls", "gpu_readback_bytes"
    ]) and has("gpu_geometry_detail") and (.gpu_geometry_detail | geometry_detail_shape);
  def run_shape:
    numeric_fields([
      "run_index", "pair_count", "matched_pairs", "verified_pairs", "total_matches",
      "matching_seconds"
    ]) and (.backend | type) == "string" and (.result_fingerprint | type) == "string" and
    has("timings") and (.timings | timings_shape);
  def report_shape:
    . as $report |
    ($report | type) == "object" and
    ($report | numeric_fields(["pair_count", "repetitions"])) and
    ($report.runs | type) == "array" and ($report.runs | length) == 3 and
    all($report.runs[]; run_shape);

  ($baseline | type) == "array" and ($baseline | length) == 1 and
  report_shape and ($baseline[0] | report_shape) and
  .pair_count == 96 and .repetitions == 3 and
  all(.runs[]; .pair_count == 96 and .matched_pairs == 96 and .verified_pairs == 96 and
      .total_matches == 62409 and .backend == "wgpu_match_and_score" and
      (.result_fingerprint | test("^[0-9a-f]{64}$"))) and
  ([.runs[].result_fingerprint] | unique | length) == 1 and
  .runs[0].result_fingerprint == $baseline[0].runs[0].result_fingerprint and
  all(.. | numbers; (isnan | not) and (isinfinite | not))
' /tmp/rustsfm-gpu-ransac-chunk-512-96x3.json
```

If this command fails, record both fingerprints and counts, do not run full benchmarks, do not merge
the code, and proceed directly to review/documentation and unmerged cleanup.

- [ ] **Step 4: If bounded parity passes, run sequential 64 and 512 full benchmarks**

First run the preserved 64 binary:

```bash
/tmp/rustsfm-gpu-ransac-chunk-64 benchmark-match-pairs \
  --database /Users/tfjiang/Projects/RustScan/test_data/flowers2/out9/Untitled.rustscanproject/Cache/.staging/keyframe_sfm-1/rustsfm/Cache/database.db \
  --window 5 --pair-limit 2890 --repetitions 1 --use-gpu --random-seed 0 \
  --output-json /tmp/rustsfm-gpu-ransac-chunk-64-2890.json
```

After it exits, run the 512 binary:

```bash
/Users/tfjiang/Projects/RustScan/target/release/rustsfm benchmark-match-pairs \
  --database /Users/tfjiang/Projects/RustScan/test_data/flowers2/out9/Untitled.rustscanproject/Cache/.staging/keyframe_sfm-1/rustsfm/Cache/database.db \
  --window 5 --pair-limit 2890 --repetitions 1 --use-gpu --random-seed 0 \
  --output-json /tmp/rustsfm-gpu-ransac-chunk-512-2890.json
```

Require both runs to have 2,890 matched/verified pairs, 2,958,062 matches, and identical
`result_fingerprint` values. A mismatch fails the experiment regardless of speed.

- [ ] **Step 5: Record the decision and commit evidence**

Append exact bounded/full fingerprints, counts, timings, score/mask/readback calls, waits, and the
merge/no-merge decision to this plan. Run `git diff --check`, then commit only the plan:

```bash
git add docs/superpowers/plans/2026-08-10-rustsfm-ransac-chunk-512-experiment.md
git commit -m "docs(rustsfm): record 512-chunk experiment"
```

#### Task 5 evidence (2026-08-10)

- Before building or running the 512 candidate, the stop-on-mismatch preflight was armed and freshly
  verified the preserved 64 binary SHA-256
  `880c7d726f789621d641f89ad6dbae7c990137adb29cda6789dbda57d8792541`, preserved 64 bounded JSON
  SHA-256 `4c56571897acfd0dd8d95079d26ad78cb0196fd5bce15fc28f1cf2b28cdea159`, and source database
  SHA-256 `dcf79fa307a6294195a8e5db1cddb185bbc1baca2ee490061b89f2a5961a052c`. The baseline jq gate and
  its supplemental `wgpu_match_and_score` backend assertion both exited 0, and the process preflight
  found no benchmark conflict.
- The strict bounded gate **FAILED solely on output fingerprint parity**. The preserved 64-chunk
  baseline fingerprint was
  `5e05ca629b63c98ae63c95ce0f37fe49a43eb870760e598352c8f8ef3d84e8ed`; all three 512-chunk
  repetitions produced the internally stable fingerprint
  `d8d08eb30c53210f388c24b9f15ab3e59d30afb4fa349c175a49b3e38108decd`. Therefore these results
  do not demonstrate output parity.
- Every other strengthened bounded predicate passed: both real reports satisfied the exact root,
  run, timing/counter, GPU geometry-detail, and nested scorer shape/type gate; `pair_count=96`,
  `repetitions=3`, and exactly three runs were present; and each candidate run had
  `pair_count=96`, `matched_pairs=96`, `verified_pairs=96`, `total_matches=62409`, backend
  `wgpu_match_and_score`, a lowercase hexadecimal 64-character fingerprint, and only finite numeric
  values. The strengthened exact plan `jq --slurpfile` gate remained `false` with exit 1 because the
  candidate fingerprint did not equal the baseline fingerprint. A diagnostic jq gate with only that
  equality predicate removed returned `true` with exit 0. The same exact gate rejected a synthetic
  report with one structurally incomplete run, returning `false` with exit 1.
- Exactly one real non-sandboxed generic-wgpu bounded run was executed, with no forced Metal,
  Vulkan, or other wgpu backend setting and no concurrent benchmark process. No 2,890-pair
  benchmark was run.

| Run | Matching seconds | Descriptor / geometry seconds | Matcher direction / readback calls / wait | Essential score / mask / readback calls / wait | Fundamental score / mask / readback calls / wait | Homography score / mask / readback calls / wait |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `10.413036542` | `6.226976296999998` / `3.876795419` | `192` / `192` / `6.086479377999997` | `96` / `386` / `482` / `0.6124437380000002` | `96` / `272` / `368` / `0.46784329800000013` | `682` / `657` / `1339` / `1.7043951580000005` |
| 2 | `10.462031292` | `6.238785755000001` / `3.9246934209999997` | `192` / `192` / `6.090471703999998` | `96` / `386` / `482` / `0.6139514099999998` | `96` / `272` / `368` / `0.46852012899999984` | `682` / `657` / `1339` / `1.704563249000001` |
| 3 | `10.425532583` | `6.250543418` / `3.882857169` | `192` / `192` / `6.107325626000003` | `96` / `386` / `482` / `0.6126029610000001` | `96` / `272` / `368` / `0.467680545` | `682` / `657` / `1339` / `1.703518597` |

- The candidate was built from HEAD `c037f175899995598affd9db72ee62b9f7a6a8ae`, source tree
  `7570eee5452b624771963de85d938e9c00fdbcb1`. Its release binary SHA-256 was
  `4a3768dae529402ff4971f9c5233eb14225b0ab654b97f715902144a4ef3d254`; the bounded candidate JSON
  SHA-256 was `f001782380776a03191d46be2f69494174f82cfb45641d161a162213efb96486`.
- The source database SHA-256 remained
  `dcf79fa307a6294195a8e5db1cddb185bbc1baca2ee490061b89f2a5961a052c` after the run.
- Decision: **NO MERGE**. The 2,890-pair full benchmarks were skipped after the bounded parity
  failure, and integration/merge-dependent Task 6 work is skipped/not applicable. Task 6 review,
  applicable regression/documentation work, and unmerged cleanup proceed.

### Task 6: Review, Regression, Integration, And Cleanup

**Files:**
- Modify only when a reviewer identifies a verified defect.

- [ ] **Step 1: Dispatch independent specification and code-quality reviewers**

Review all experiment commits against the design. Critical/Important findings block integration.
Reviewers must verify fingerprint coverage, no public backend forcing, no default cap, no extra GPU
dispatch/readback beyond the changed chunk behavior, and no source-database mutation.

- [ ] **Step 2: Resolve findings with RED/GREEN tests and run final verification**

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm \
  --lib --no-default-features --features gpu-wgpu -- --nocapture
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo check -p rustsfm \
  --no-default-features --features gpu-wgpu
cargo fmt --all -- --check
git diff --check
```

Document the exact known external-fixture or adapter prerequisite failures; do not describe a
non-zero test command as fully passing.

- [ ] **Step 3: Integrate only after strict bounded and full parity**

If both strict gates passed, synchronize the branch with current `main`, rerun affected tests, merge
locally into `main`, and verify the merged constant/fingerprint tests. If either gate failed, do not
merge any experiment commit.

- [ ] **Step 4: Clean up the isolated worktree and branch**

After preserving the final decision in the user report (and in `main` only when the experiment is
merged), remove `.worktrees/ransac-chunk-512` and delete `codex/ransac-chunk-512-experiment` with
`git branch -d` when merged or `git branch -D` only after explicit confirmation when unmerged.

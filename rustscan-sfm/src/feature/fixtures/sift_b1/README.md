# B1 frozen real SIFT descriptor regression

These are **unaltered descriptor bytes**, not generated/artificial descriptors.
Normal Rust unit tests embed `left.bin` and `right.bin` with `include_bytes!`;
no images, extraction backend, ignored `artifacts/runs/`, Python, network, or regeneration
step is required to run them. Format: consecutive 128-byte uint8 descriptors,
no header, in ascending original row order (indices are zero-based).

## Origin and deterministic selection

Source: `rustscan-sfm/output/sift_distance_kernel/corpus/{left.bin,right.bin,metadata.json}`,
frozen by `rustscan-sfm/examples/sift_search_bench.rs extract` using default SIFT
extraction options except `max_num_features = 8192`. The recorded images are
`artifacts/inputs/flowers/images/frame_0001.jpg` and `frame_0002.jpg` (flowers, not flowers2).
Original counts: **5,865 left / 5,548 right**.

Selection from historical `rustscan-sfm/output/sift_distance_kernel/after.json`:

1. Its `default_match_bits` contains 1,718 ordered triples. Select positions
   `floor(k * (1718 - 1) / 63)` for `k = 0..63` (64 positions including endpoints).
2. For each side with `N` rows, take the union of that side's indices from those
   match triples and `floor(k * (N - 1) / 63)` for `k = 0..63`.
3. Deduplicate, sort indices ascending, and copy each original 128-byte slice
   verbatim. No re-quantization, normalization, perturbation, or extraction.

This retains real correspondences plus broadly spaced distractors in only
**128 left / 127 right rows**, **32,640 bytes total**. It is intentionally biased
for useful regression coverage, not a representative performance benchmark.
`provenance.json` records every selected row, selected historical match position,
historical report SHA-256, original corpus counts/SHA-256, and the corpus/image
BLAKE3 hashes recorded in the source metadata. The historical match triples are
used **only for selection**, never as the expected subset matcher output: removing
other descriptors can change neighbors and ratio decisions.

Original corpus BLAKE3 (from source metadata):

- left: `2f0595830e0ed2047ad3b892e1bfff79b0572301a79edd9c1304a8d8b15d1fe4`
- right: `5a3e308b60cec518262483e2e1977a31f21e62232102fd65496d727452e7a249`

Fixture BLAKE3 (asserted by the normal test):

- left: `4b2c698a07022a1e49bd3c9100652fa2c22388b9a133d7d5e25bf1dfd40ccea6`
- right: `71bd4d0903fda84df8e6f38cd2c5d0ba08d380f1f977dc490f77903091ad92e1`

Fixture SHA-256 (also recorded/verified by the reproduction script):

- left: `1f94e51d94b1754ba216c4bcc1f435bb86c1c78ab797d2781d64dd283d301fd3`
- right: `5f2cff310fcc52ae4356634793448861da1ff9de71684f9ddd35c61caaf30b19`

Optional provenance check, from repository root, when the original corpus exists:

```sh
python3 rustscan-sfm/src/feature/fixtures/sift_b1/reproduce.py rustscan-sfm/output/sift_distance_kernel/corpus
```

Add `--write` to restore the two binaries from verified source slices. The script
uses only Python's standard library and the checked-in row list; it verifies
original SHA-256/counts and fixture SHA-256 before writing. No historical report
or original images are needed for that byte-for-byte reproduction.

## Independent proof and frozen complete output

`../../sift_regression_tests.rs` uses a scalar f32 distance loop, fully sorts
candidates by `(squared distance, original index)`, and independently specifies:
strict ratio comparison, inclusive squared maximum distance, reverse **ratio**
filtering plus mutuality, distance ordering with explicit query-index tie order,
and truncation after filtering/sorting (`0` means unlimited). It never calls
production distance, nearest-neighbor, filtering, or finalization helpers to
calculate expected results. Nearest/second-nearest indices and f32 bits are
checked bidirectionally, including empty/single/tied artificial cases.

Every complete matcher vector `(query_idx, train_idx, distance.to_bits())` is
compared against the oracle across 1- and 4-thread Rayon pools, indexed/brute,
and plain/profiled uint8 paths, including public dispatch. The entire new module
shares the complete uint8 backend's cfg (`vlfeat-sift` without
`lowe-sift-backend`). No-default validation runs the existing index tests and a
library check; it does not apply uint8 matcher expectations to Lowe BBF.

Real scenarios, in fingerprint order:

| cross_check | max_num_matches | output count |
|---|---:|---:|
| false | 0 | 67 |
| false | 7 | 7 |
| true | 0 | 66 |
| true | 7 | 7 |

The default configuration (including limit 32,768) is also checked against the
66-match oracle output. Each fingerprint scenario encodes one cross-check byte,
a u32 little-endian limit, a u32 little-endian match count, then **every** ordered
triple as three u32 little-endian values. Hash input is actual production output,
collected only after full oracle parity, not a production-only expected hash.
Frozen combined BLAKE3:

`fbb55fac40c5e1a1d2504548b5246f331a7f4936c46735551440d32579ac6e52`

Explicit artificial expected triples independently anchor empty/single behavior,
zero/nonzero ties, exact ratio rejection and next-f32 acceptance, exact maximum
distance acceptance and previous-f32 rejection, reverse-ratio rejection,
nonmutual rejection, stable sorting, and effective truncation of four matches to
one/three in both cross-check modes (including a cut through a distance tie).
The earlier artificial hash test remains unchanged.

## Focused validation

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo test -p rustscan-sfm --release --lib --offline b1_ -- --test-threads=1
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo test -p rustscan-sfm --release --lib --offline --no-default-features sift_index:: -- --test-threads=1
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo check -p rustscan-sfm --release --lib --offline --no-default-features
```

These are matching regressions, not extraction, guided matching, GPU, Lowe BBF,
full-corpus parity, reconstruction-quality, or performance claims. Native thread
limits do not suppress the explicitly constructed 1/4-thread Rayon pools.

#!/bin/bash
# Provision the external RustSFM `flowers2_colmap` parity fixture.
#
# The fixture is the official-COLMAP sparse text export of the 24-image
# flowers2 subset (frame_0001.jpg .. frame_0024.jpg) that the ignored
# `real_colmap_sparse_*` RustSFM tests read from
# `test_data/flowers2_colmap/sparse/text`. It is intentionally not
# distributed through Git or submodules (see RustSFM/README.md).
#
# Provenance: generated 2026-06-30 and archived in the workspace `output/`
# parity directories:
#   output/flowers2_colmap_ref_text_20260630  (reference COLMAP text export)
#   output/flowers2_colmap_txt_20260630       (identical archived copy)
#
# The fixture content is pinned by SHA-256 so a mismatched source is refused
# instead of silently changing the parity reference.
#
# Usage (from the workspace root):
#   ./scripts/provision_flowers2_colmap_fixture.sh [SOURCE_DIR]
#
# SOURCE_DIR defaults to output/flowers2_colmap_ref_text_20260630 and must
# contain the five text files whose digests match the pinned values below.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE="${1:-output/flowers2_colmap_ref_text_20260630}"
DEST="test_data/flowers2_colmap/sparse/text"

case "$SOURCE" in
  /*) SRC_DIR="$SOURCE" ;;
  *) SRC_DIR="$ROOT/$SOURCE" ;;
esac

if [ ! -d "$SRC_DIR" ]; then
  echo "error: fixture source directory not found: $SRC_DIR" >&2
  exit 1
fi

check_file() {
  file="$1"
  expected="$2"
  if [ ! -f "$SRC_DIR/$file" ]; then
    echo "error: missing fixture file: $SRC_DIR/$file" >&2
    exit 1
  fi
  actual="$(shasum -a 256 "$SRC_DIR/$file" | awk '{print $1}')"
  if [ "$actual" != "$expected" ]; then
    echo "error: digest mismatch for $file" >&2
    echo "  expected: $expected" >&2
    echo "  actual:   $actual" >&2
    exit 1
  fi
}

check_file cameras.txt 6372d9e8057fed5a6789469653735c8371586996e846b1a8ef9f8c3821326569
check_file frames.txt   254a9d354202b0e29c6919093cbd6e83e34a2e4030f06da569de882c674a159e
check_file images.txt   85edcb2cb6d5cab54ba6ed52d2f031946278721b5d40a403f97d2dd3de140f4c
check_file points3D.txt e7f47af4d7699f78de0f722f0576b65eeaecd3156cb84a9ae9a1ea57b0dcff0b
check_file rigs.txt     8472bec629acf5f8204018b1b34b699709d2b4d203898facf500c0edec35ca68

mkdir -p "$ROOT/$DEST"
cp "$SRC_DIR/cameras.txt" "$SRC_DIR/frames.txt" "$SRC_DIR/images.txt" \
   "$SRC_DIR/points3D.txt" "$SRC_DIR/rigs.txt" "$ROOT/$DEST/"

echo "provisioned: $ROOT/$DEST"
echo "run parity tests with: cargo test -p rustsfm --lib -- --ignored"

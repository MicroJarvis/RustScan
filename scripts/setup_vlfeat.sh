#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLFEAT_DIR="${VLFEAT_ROOT:-$ROOT/third_party/vlfeat}"
COLMAP_TAG="${COLMAP_TAG:-3.13.0}"

if [[ -f "$VLFEAT_DIR/sift.c" ]]; then
  echo "VLFeat already present at $VLFEAT_DIR"
  exit 0
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

git clone --depth 1 --filter=blob:none --sparse --branch "$COLMAP_TAG" \
  https://github.com/colmap/colmap.git "$TMP_DIR/colmap"
(
  cd "$TMP_DIR/colmap"
  git sparse-checkout set src/thirdparty/VLFeat
)

mkdir -p "$(dirname "$VLFEAT_DIR")"
cp -R "$TMP_DIR/colmap/src/thirdparty/VLFeat" "$VLFEAT_DIR"
echo "COLMAP VLFeat $COLMAP_TAG installed at $VLFEAT_DIR"

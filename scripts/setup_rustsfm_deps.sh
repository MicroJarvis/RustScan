#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/setup_vlfeat.sh"

if [[ ! -f /opt/homebrew/opt/freeimage/include/FreeImage.h ]] \
  && [[ ! -f /usr/local/opt/freeimage/include/FreeImage.h ]] \
  && ! pkg-config --exists freeimage 2>/dev/null; then
  echo "Note: install FreeImage for COLMAP-parity JPEG loading (e.g. brew install freeimage)"
fi

POSELIB_DIR="${POSELIB_ROOT:-$ROOT/third_party/PoseLib}"
POSELIB_TAG="${POSELIB_TAG:-v2.0.5}"

if [[ -f "$POSELIB_DIR/PoseLib/solvers/gen_relpose_6pt.cc" ]]; then
  echo "PoseLib already present at $POSELIB_DIR"
  exit 0
fi

mkdir -p "$(dirname "$POSELIB_DIR")"
git clone --depth 1 --branch "$POSELIB_TAG" https://github.com/PoseLib/PoseLib.git "$POSELIB_DIR"
echo "PoseLib $POSELIB_TAG installed at $POSELIB_DIR"

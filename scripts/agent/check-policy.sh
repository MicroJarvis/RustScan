#!/usr/bin/env bash
set -euo pipefail

base_commit=${1:-HEAD~1}
git diff --unified=0 "$base_commit" -- '*.rs' |
  grep -E '^\+[^+]' |
  grep -E 'glam::(Mat|Vec|Quat)|use[[:space:]]+glam(::|;)' && {
    echo "policy violation: new CPU Rust code references glam linear-algebra types" >&2
    exit 1
  } || true

echo "policy: PASS"

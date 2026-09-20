#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

exec cargo run --release \
  --manifest-path "${repo_root}/rustgs/Cargo.toml" \
  --example rustgs_eval_suite \
  --features gpu,cli \
  -- "$@"

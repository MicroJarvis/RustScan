#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 <task-file> [base-commit]" >&2
  exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || usage
git_root=$(git rev-parse --show-toplevel)
task_file=$1
base_commit=${2:-HEAD~1}

[[ -f "$git_root/$task_file" ]] || {
  echo "task file not found: $task_file" >&2
  exit 1
}

scope=$(awk '
  /^scope:[[:space:]]*$/ {section="scope"; next}
  /^out_of_scope:[[:space:]]*$/ {section="out"; next}
  /^[^[:space:]-][^:]*:/ {section=""; next}
  section == "scope" && /^[[:space:]]*-[[:space:]]/ {sub(/^[[:space:]]*-[[:space:]]*/, ""); print}
' "$git_root/$task_file")
out_of_scope=$(awk '
  /^out_of_scope:[[:space:]]*$/ {section="out"; next}
  /^[^[:space:]-][^:]*:/ {if ($0 !~ /^out_of_scope:/) section=""; next}
  section == "out" && /^[[:space:]]*-[[:space:]]/ {sub(/^[[:space:]]*-[[:space:]]*/, ""); print}
' "$git_root/$task_file")

[[ -n "$scope" ]] || {
  echo "task scope is empty: $task_file" >&2
  exit 1
}

matches_prefix() {
  local path=$1 prefix=$2
  prefix=${prefix%/}
  [[ "$path" == "$prefix" || "$path" == "$prefix"/* ]]
}

failed=0
while IFS= read -r path; do
  [[ -z "$path" ]] && continue
  allowed=0
  while IFS= read -r prefix; do
    [[ -z "$prefix" ]] && continue
    if matches_prefix "$path" "$prefix"; then allowed=1; break; fi
  done <<EOF
$scope
EOF
  if [[ $allowed -eq 0 ]]; then
    echo "out of scope: $path" >&2
    failed=1
  fi
  while IFS= read -r prefix; do
    [[ -z "$prefix" ]] && continue
    if matches_prefix "$path" "$prefix"; then
      echo "explicitly forbidden path changed: $path" >&2
      failed=1
    fi
  done <<EOF
$out_of_scope
EOF
done < <(
  git -c core.quotePath=false diff --name-only "$base_commit" --
  git -c core.quotePath=false ls-files --others --exclude-standard
)

if [[ $failed -ne 0 ]]; then
  exit 1
fi
echo "scope: PASS"

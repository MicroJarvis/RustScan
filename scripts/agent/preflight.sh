#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 <task-file>" >&2
  exit 2
}

[[ $# -eq 1 ]] || usage
task_file=$1

git_root=$(git rev-parse --show-toplevel)
[[ -f "$git_root/$task_file" ]] || {
  echo "task file not found: $task_file" >&2
  exit 1
}

branch=$(git branch --show-current)
[[ -n "$branch" ]] || {
  echo "task worktree must be on a branch" >&2
  exit 1
}
[[ "$branch" != "main" ]] || {
  echo "implementation tasks must not run on main" >&2
  exit 1
}

base_commit=$(awk -F': ' '$1 == "base_commit" {print $2; exit}' "$git_root/$task_file")
[[ -n "$base_commit" ]] || {
  echo "task file has no base_commit: $task_file" >&2
  exit 1
}
git rev-parse --verify "$base_commit^{commit}" >/dev/null || {
  echo "base commit does not resolve: $base_commit" >&2
  exit 1
}

declared_branch=$(awk -F': ' '$1 == "branch" {print $2; exit}' "$git_root/$task_file")
[[ -z "$declared_branch" || "$declared_branch" == "$branch" ]] || {
  echo "current branch ($branch) differs from task branch ($declared_branch)" >&2
  exit 1
}

echo "preflight: PASS"
echo "git_root: $git_root"
echo "branch: $branch"
echo "base_commit: $base_commit"
echo "task: $task_file"

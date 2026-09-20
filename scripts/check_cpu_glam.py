#!/usr/bin/env python3
"""Forbid CPU-side glam math types outside an explicit allowlist.

Scans controlled Rust sources and crate Cargo.toml files for:
  - use glam::... / pub use glam::... / extern crate glam
  - fully-qualified glam::{Vec,DVec,Mat,DMat,Quat,DQuat}*
  - glam dependencies in Cargo.toml

Skips third_party/rust/, target/, artifacts/runs/, and .worktrees/. Line and doc comments are
ignored so historical prose does not false-positive.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST_PATH = Path(__file__).resolve().parent / "cpu-glam-allowlist.txt"

SKIP_DIRS = {"target", "artifacts", "output", ".worktrees", ".git"}
SKIP_PATHS = {"third_party/rust"}

USE_GLAM_RE = re.compile(
    r"\b(?:pub\s+)?use\s+glam(?:\s+as\s+\w+|\s*::|\s*\{)"
)
EXTERN_GLAM_RE = re.compile(r"\bextern\s+crate\s+glam\b")
QUALIFIED_GLAM_RE = re.compile(
    r"\bglam::(?:D?Vec\d*|D?Mat\d*|D?Quat)\b"
)
# Alias form: `use glam as gm;` then later `gm::Vec3`
ALIAS_IMPORT_RE = re.compile(r"\b(?:pub\s+)?use\s+glam\s+as\s+(\w+)\s*;")
CARGO_GLAM_KEY_RE = re.compile(r"(?m)^\s*glam\s*=")

FIX_HINT = (
    "CPU glam types are not allowed; convert them to nalgebra "
    "(see AGENTS.md). Graphics/GPU boundaries must be allowlisted in "
    "scripts/cpu-glam-allowlist.txt."
)


def load_allowlist(path: Path) -> dict[str, str]:
    """Map repo-relative path -> reason. Blank lines and # comments ignored."""
    allowed: dict[str, str] = {}
    if not path.is_file():
        return allowed
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            rel, reason = line.split("#", 1)
            allowed[rel.strip()] = reason.strip()
        else:
            allowed[line] = "allowlisted"
    return allowed


def strip_line_comment_and_strings(line: str) -> str:
    """Blank string literals and drop // comments (best-effort, keeps length)."""
    out: list[str] = []
    in_string = False
    quote = ""
    escaped = False
    i = 0
    while i < len(line):
        ch = line[i]
        if in_string:
            if escaped:
                out.append(" ")
                escaped = False
            elif ch == "\\":
                out.append(" ")
                escaped = True
            elif ch == quote:
                out.append(" ")
                in_string = False
            else:
                out.append("\n" if ch == "\n" else " ")
            i += 1
            continue
        if ch in ('"', "'"):
            in_string = True
            quote = ch
            out.append(" ")
            i += 1
            continue
        if ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
            break
        out.append(ch)
        i += 1
    return "".join(out)


def strip_block_comments(text: str) -> str:
    """Replace /* ... */ with spaces, preserving newlines for line numbers."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end < 0:
                block = text[i:]
                out.append("".join("\n" if c == "\n" else " " for c in block))
                break
            block = text[i : end + 2]
            out.append("".join("\n" if c == "\n" else " " for c in block))
            i = end + 2
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def iter_controlled_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skipped directories in-place so os.walk does not descend.
        current = Path(dirpath).relative_to(root).as_posix()
        dirnames[:] = [
            d
            for d in dirnames
            if d not in SKIP_DIRS
            and (d if current == "." else f"{current}/{d}") not in SKIP_PATHS
        ]
        for name in filenames:
            if name.endswith(".rs") or name == "Cargo.toml":
                yield Path(dirpath) / name


def check_rust_file(path: Path, rel: str) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    cleaned = strip_block_comments(text)
    violations: list[tuple[int, str]] = []
    aliases: set[str] = set()

    for lineno, raw_line in enumerate(cleaned.splitlines(), start=1):
        code = strip_line_comment_and_strings(raw_line)
        if not code.strip():
            continue

        for m in ALIAS_IMPORT_RE.finditer(code):
            aliases.add(m.group(1))
            violations.append((lineno, code.strip()))

        if USE_GLAM_RE.search(code) or EXTERN_GLAM_RE.search(code):
            if not any(v[0] == lineno for v in violations):
                violations.append((lineno, code.strip()))
            continue

        if QUALIFIED_GLAM_RE.search(code):
            violations.append((lineno, code.strip()))
            continue

        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}::(?:D?Vec\d*|D?Mat\d*|D?Quat)\b", code):
                violations.append((lineno, code.strip()))
                break

    return violations


def check_cargo_toml(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    violations: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if CARGO_GLAM_KEY_RE.match(line):
            violations.append((lineno, stripped))
    return violations


def scan_repo(root: Path, allowlist: dict[str, str]) -> list[tuple[str, int, str]]:
    findings: list[tuple[str, int, str]] = []
    for path in iter_controlled_files(root):
        rel = path.relative_to(root).as_posix()
        if rel in allowlist:
            continue
        if path.suffix == ".rs":
            for lineno, snippet in check_rust_file(path, rel):
                findings.append((rel, lineno, snippet))
        else:
            for lineno, snippet in check_cargo_toml(path):
                findings.append((rel, lineno, snippet))
    return findings


def run_self_test() -> int:
    samples = {
        "use_brace.rs": "use glam::{Vec3, Mat4};\n",
        "use_alias.rs": "use glam as gm;\nlet b: gm::Vec3 = gm::Vec3::ZERO;\n",
        "pub_use.rs": "pub use glam::Quat;\n",
        "qualified.rs": "let a: glam::Vec3 = glam::Vec3::ZERO;\n",
        "comment_ok.rs": "// glam::Vec3 historical note\n/// docs glam::Mat4\nlet x = 1;\n",
        "string_ok.rs": 'let s = "glam::Vec3";\n',
        "Cargo.toml": '[dependencies]\nglam = "0.29"\n',
    }

    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, content in samples.items():
            path = root / name
            path.write_text(content, encoding="utf-8")

        # Positive cases must violate.
        for name in ("use_brace.rs", "use_alias.rs", "pub_use.rs", "qualified.rs", "Cargo.toml"):
            path = root / name
            if name.endswith(".toml"):
                hits = check_cargo_toml(path)
            else:
                hits = check_rust_file(path, name)
            if not hits:
                print(f"SELF-TEST FAIL: expected violation in {name}", file=sys.stderr)
                failed += 1
            else:
                print(f"SELF-TEST OK: {name} -> {hits[0][1]}")

        # Negative cases must be clean.
        for name in ("comment_ok.rs", "string_ok.rs"):
            hits = check_rust_file(root / name, name)
            if hits:
                print(f"SELF-TEST FAIL: false positive in {name}: {hits}", file=sys.stderr)
                failed += 1
            else:
                print(f"SELF-TEST OK: {name} clean")

        # Alias must catch later uses.
        alias_hits = check_rust_file(root / "use_alias.rs", "use_alias.rs")
        if len(alias_hits) < 2:
            print(
                "SELF-TEST FAIL: alias import should flag import and gm::Vec3 use",
                file=sys.stderr,
            )
            failed += 1

    if failed:
        print(f"SELF-TEST: {failed} failure(s)", file=sys.stderr)
        return 1
    print("SELF-TEST: all passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root to scan (default: RustScan root)",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=ALLOWLIST_PATH,
        help="Allowlist file (repo-relative paths)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in detector self-tests and exit",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    root = args.root.resolve()
    allowlist = load_allowlist(args.allowlist)
    findings = scan_repo(root, allowlist)
    if not findings:
        print("check_cpu_glam: OK (no CPU glam usage found)")
        return 0

    for rel, lineno, snippet in findings:
        print(f"{rel}:{lineno}: {snippet}", file=sys.stderr)
    print(FIX_HINT, file=sys.stderr)
    print(f"check_cpu_glam: {len(findings)} violation(s)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

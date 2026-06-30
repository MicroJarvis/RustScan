#!/usr/bin/env python3
"""Compare COLMAP and RustSFM command-line surfaces.

The script is intentionally lightweight: it shells out to both binaries,
parses command names from top-level help, parses long option names from command
help, and writes a JSON report that can be tracked over time.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


COLMAP_TO_RUSTSFM_COMMANDS = {
    "feature_extractor": "feature_extractor",
    "exhaustive_matcher": "exhaustive_matcher",
    "sequential_matcher": "sequential_matcher",
    "vocab_tree_matcher": "vocab_tree_matcher",
    "geometric_verifier": "geometric_verifier",
    "mapper": "mapper",
    "model_converter": "model_converter",
    # Native RustSFM names for the same sparse-pipeline capabilities.
    "extract-features": "extract-features",
    "match-features": "match-features",
    "reconstruct": "reconstruct",
}


@dataclass
class HelpResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int

    @property
    def text(self) -> str:
        return "\n".join(part for part in (self.stdout, self.stderr) if part)


def run_help(argv: list[str], timeout: int) -> HelpResult:
    try:
        proc = subprocess.run(
            argv,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        return HelpResult(proc.returncode == 0, proc.stdout, proc.stderr, proc.returncode)
    except FileNotFoundError as exc:
        return HelpResult(False, "", str(exc), 127)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return HelpResult(False, stdout, f"{stderr}\ntimeout", 124)


def parse_colmap_commands(help_text: str) -> list[str]:
    commands: list[str] = []
    in_commands = False
    for raw_line in help_text.splitlines():
        line = raw_line.rstrip()
        if line.strip() == "Available commands:":
            in_commands = True
            continue
        if not in_commands:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("-"):
            continue
        if re.match(r"^[A-Za-z0-9_-]+$", stripped):
            commands.append(stripped)
    return commands


def parse_rustsfm_commands(help_text: str) -> list[str]:
    commands: list[str] = []
    in_commands = False
    for raw_line in help_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped == "Commands:":
            in_commands = True
            continue
        if stripped == "Options:":
            in_commands = False
            continue
        if not in_commands or not stripped:
            continue
        match = re.match(r"^([A-Za-z0-9_-]+)\b", stripped)
        if match:
            commands.append(match.group(1))
    return commands


def parse_long_options(help_text: str) -> list[str]:
    options = set()
    for match in re.finditer(r"--([A-Za-z0-9_.-]+)", help_text):
        options.add(match.group(1))
    return sorted(options)


def option_aliases(command: str, option: str) -> set[str]:
    aliases = {option}
    aliases.add(option.replace("_", "-"))
    aliases.add(option.replace("-", "_"))
    if "." in option:
        tail = option.split(".")[-1]
        aliases.add(tail)
        aliases.add(tail.replace("_", "-"))
        aliases.add(tail.replace("-", "_"))
    command_prefixes = {
        "feature_extractor": ["database_path", "image_path"],
        "exhaustive_matcher": ["database_path"],
        "sequential_matcher": ["database_path"],
        "vocab_tree_matcher": ["database_path"],
        "geometric_verifier": ["database_path"],
        "mapper": ["database_path", "image_path", "output_path"],
        "model_converter": ["input_path", "output_path", "output_type"],
    }
    for name in command_prefixes.get(command, []):
        if option == name:
            aliases.add(name.replace("_path", ""))
    return aliases


def has_option(candidate_options: Iterable[str], command: str, colmap_option: str) -> bool:
    candidates = set(candidate_options)
    return bool(option_aliases(command, colmap_option) & candidates)


def command_help(binary: str, command: str, flavor: str, timeout: int) -> HelpResult:
    if flavor == "colmap":
        return run_help([binary, command, "--help"], timeout)
    return run_help([binary, command, "--help"], timeout)


def build_report(args: argparse.Namespace) -> dict:
    colmap_top = run_help([args.colmap, "help"], args.timeout)
    rustsfm_top = run_help([args.rustsfm, "--help"], args.timeout)
    colmap_commands = parse_colmap_commands(colmap_top.text) if colmap_top.ok else []
    rustsfm_commands = parse_rustsfm_commands(rustsfm_top.text) if rustsfm_top.ok else []
    rustsfm_command_set = set(rustsfm_commands)

    command_reports = {}
    missing_commands = []
    for colmap_command in colmap_commands:
        mapped = COLMAP_TO_RUSTSFM_COMMANDS.get(colmap_command, colmap_command)
        implemented = mapped in rustsfm_command_set
        if colmap_command == "help" and implemented:
            command_reports[colmap_command] = {
                "mapped_rustsfm_command": mapped,
                "implemented": True,
                "colmap_option_count": 0,
                "rustsfm_option_count": 0,
                "missing_options": [],
                "option_coverage": 1.0,
                "colmap_help_ok": True,
                "rustsfm_help_ok": True,
            }
            continue
        if not implemented:
            missing_commands.append(colmap_command)
            command_reports[colmap_command] = {
                "mapped_rustsfm_command": mapped,
                "implemented": False,
                "missing_options": [],
                "option_coverage": 0.0,
            }
            continue

        colmap_help = command_help(args.colmap, colmap_command, "colmap", args.timeout)
        rustsfm_help = command_help(args.rustsfm, mapped, "rustsfm", args.timeout)
        colmap_options = parse_long_options(colmap_help.text) if colmap_help.ok else []
        rustsfm_options = parse_long_options(rustsfm_help.text) if rustsfm_help.ok else []
        missing_options = [
            option
            for option in colmap_options
            if not has_option(rustsfm_options, colmap_command, option)
        ]
        coverage = 1.0
        if colmap_options:
            coverage = (len(colmap_options) - len(missing_options)) / len(colmap_options)
        command_reports[colmap_command] = {
            "mapped_rustsfm_command": mapped,
            "implemented": True,
            "colmap_option_count": len(colmap_options),
            "rustsfm_option_count": len(rustsfm_options),
            "missing_options": missing_options,
            "option_coverage": coverage,
            "colmap_help_ok": colmap_help.ok,
            "rustsfm_help_ok": rustsfm_help.ok,
        }

    implemented_colmap_commands = len(colmap_commands) - len(missing_commands)
    command_coverage = (
        implemented_colmap_commands / len(colmap_commands) if colmap_commands else 0.0
    )
    return {
        "colmap_binary": args.colmap,
        "rustsfm_binary": args.rustsfm,
        "colmap_top_help_ok": colmap_top.ok,
        "rustsfm_top_help_ok": rustsfm_top.ok,
        "colmap_command_count": len(colmap_commands),
        "rustsfm_command_count": len(rustsfm_commands),
        "implemented_colmap_commands": implemented_colmap_commands,
        "command_coverage": command_coverage,
        "missing_colmap_commands": missing_commands,
        "colmap_commands": colmap_commands,
        "rustsfm_commands": rustsfm_commands,
        "commands": command_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colmap", default="colmap", help="COLMAP binary path")
    parser.add_argument(
        "--rustsfm",
        default="target/debug/rustsfm",
        help="RustSFM binary path, e.g. target/debug/rustsfm",
    )
    parser.add_argument("--output-json", type=Path, help="Write JSON report")
    parser.add_argument("--timeout", type=int, default=20, help="Per-help timeout in seconds")
    args = parser.parse_args()

    report = build_report(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Persistent native SIFT cold-once gate (POSIX host, no Cargo/dependencies).

Before editing sift.c, run --capture-baseline with a fresh --report directory.
Then use its baseline.bin via --baseline for all candidate runs. Never regenerate
that oracle after editing. Each --report must be new; commands, failures and
binaries are retained. --tsan is an actual instrumented build/run, not a probe
whose failure can be silently skipped. --count-init adds constructor-only exp
counting; also run without it to validate the uninstrumented production source.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    oracle = parser.add_mutually_exclusive_group(required=True)
    oracle.add_argument("--capture-baseline", action="store_true")
    oracle.add_argument("--baseline", type=Path)
    parser.add_argument("--tsan", action="store_true")
    parser.add_argument("--count-init", action="store_true")
    parser.add_argument("--no-threads", action="store_true", help="serial VL_DISABLE_THREADS branch")
    parser.add_argument("--sift-source", type=Path, help="saved pre-fix sift.c for negative control")
    parser.add_argument("--cold-runs", type=int, default=10)
    args = parser.parse_args()
    if args.cold_runs < 1:
        parser.error("--cold-runs must be positive")
    tools = Path(__file__).resolve().parent
    root = tools.parents[1]
    vlfeat = Path(os.environ.get("VLFEAT_ROOT", root / "third_party/vlfeat")).resolve()
    report = args.report.resolve()
    report.mkdir(parents=False, exist_ok=False)
    baseline = report / "baseline.bin" if args.capture_baseline else args.baseline.resolve()
    env = dict(os.environ, VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", BLIS_NUM_THREADS="1",
               RAYON_NUM_THREADS="1", TSAN_OPTIONS="halt_on_error=1:exitcode=66")
    cc = shlex.split(os.environ.get("CC", "clang"))
    flags = ["-std=c11", "-O3", "-g", "-DNDEBUG", "-DVL_DISABLE_AVX",
             "-DVL_DISABLE_SSE2", "-DVL_DISABLE_OPENMP", "-pthread"]
    if args.count_init:
        flags += ["-DCOUNT_INIT"]
    if args.no_threads:
        flags += ["-DVL_DISABLE_THREADS"]
    if args.tsan:
        flags += ["-fsanitize=thread", "-fno-omit-frame-pointer"]
    if args.sift_source:
        # Only the test include is redirected, leaving the working tree untouched.
        import shutil
        shutil.copyfile(args.sift_source, report / "sift.c")
        flags += ["-I", str(report)]
    flags += ["-I", str(vlfeat)]
    records = []

    def run(command):
        print(shlex.join(command), flush=True)
        start = time.monotonic()
        record = {"command": command}
        records.append(record)
        with (report / f"{len(records):02d}.log").open("wb") as log:
            try:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=90, check=False)
                record["exit_code"] = result.returncode
                print(f"exit={result.returncode} log={log.name}", flush=True)
                result.check_returncode()
            except subprocess.TimeoutExpired:
                record["timeout_seconds"] = 90
                raise
            finally:
                record["seconds"] = round(time.monotonic() - start, 3)
                (report / "commands.json").write_text(json.dumps(records, indent=2) + "\n")

    run(cc + ["--version"])
    sources = [vlfeat / (name + ".c") for name in ["generic", "host", "mathop", "imopv", "random"]]
    exe = report / "sift-once"
    run(cc + flags + [str(tools / "vlfeat_sift_once_test.c")] + list(map(str, sources)) +
        ["-lm", "-o", str(exe)])
    if args.capture_baseline:
        run([str(exe), "baseline", str(baseline)])
    else:
        run([str(exe), "serial", str(baseline)])
        if not args.no_threads:
            for _ in range(args.cold_runs):
                run([str(exe), "cold", str(baseline)])
    (report / "baseline-sha256.txt").write_text(
        hashlib.sha256(baseline.read_bytes()).hexdigest() + "  " + str(baseline) + "\n")


if __name__ == "__main__":
    main()

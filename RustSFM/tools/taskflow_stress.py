#!/usr/bin/env python3
"""Run a prebuilt taskflow_stress example; stdlib only, macOS and Linux.

Example: python3 taskflow_stress.py --binary /path/to/taskflow_stress --output run.json
No compilation is performed. Each trial, including warmups, uses a fresh process.
"""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import tempfile
import time


MODES = ("uncontrolled", "fixed", "taskflow")
KINDS = ("large_ba", "small_ba", "sift")
THREAD_ENV = {
    "VECLIB_MAXIMUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
}
POLL_SECONDS = 0.020


def positive_int(value):
    try:
        result = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def nonnegative_int(value):
    try:
        result = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from None
    if result < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return result


def positive_float(value):
    try:
        result = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be finite and positive") from None
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


class UniqueModes(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) != len(set(values)):
            raise argparse.ArgumentError(self, "duplicate modes are not allowed")
        setattr(namespace, self.dest, values)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--binary", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--rounds", type=positive_int, default=5)
    result.add_argument("--cycles", type=positive_int, default=3)
    result.add_argument("--workflows", type=positive_int, nargs="+", default=[4, 8, 16])
    result.add_argument("--modes", nargs="+", choices=MODES + ("fixed4",),
                        default=list(MODES), action=UniqueModes,
                        help="modes to rotate (default: uncontrolled fixed taskflow)")
    result.add_argument("--warmups", type=nonnegative_int, default=1)
    result.add_argument("--timeout", type=positive_float, default=120.0, help="seconds per child (default: 120)")
    return result


def nearest_rank(values, quantile):
    """Empirical nearest-rank quantile, without interpolation; empty -> None."""
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1] if ordered else None


def distribution(values):
    return {
        "samples": len(values),
        "p50": nearest_rank(values, 0.50),
        "p95": nearest_rank(values, 0.95),
        "max": max(values) if values else None,
    }


def rss_mib(native_rss, system):
    if system == "Darwin":
        return native_rss / (1024 * 1024)
    if system == "Linux":
        return native_rss / 1024
    raise ValueError("OS resource measurement supports only macOS and Linux")


def mode_order(round_index, modes=MODES):
    offset = round_index % len(modes)
    return modes[offset:] + modes[:offset]


def validate_child(child, mode, workflows, cycles):
    if not isinstance(child, dict):
        raise ValueError("child JSON must be an object")
    for key, expected in (("mode", mode), ("workflows", workflows), ("cycles", cycles)):
        if type(child.get(key)) is not type(expected) or child[key] != expected:
            raise ValueError("child JSON has unexpected " + key)

    def duration(value, name, positive=False):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(name + " must be a finite " + ("positive" if positive else "nonnegative") + " number")

    duration(child.get("workload_ms"), "workload_ms", positive=True)
    jobs = child.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != workflows * cycles * 4:
        raise ValueError("child jobs must contain workflows * cycles * 4 entries")
    by_workflow = {}
    for job in jobs:
        if not isinstance(job, dict) or job.get("kind") not in KINDS:
            raise ValueError("child job has an invalid kind")
        for key in ("workflow", "sequence", "solver_threads"):
            if type(job.get(key)) is not int or job[key] < 0:
                raise ValueError("job " + key + " must be a nonnegative integer")
        for key in ("queue_ms", "elapsed_ms", "compute_ms"):
            duration(job.get(key), "job " + key)
        by_workflow.setdefault(job["workflow"], []).append(job)
    expected = Counter({"large_ba": cycles, "small_ba": 2 * cycles, "sift": cycles})
    if len(by_workflow) != workflows:
        raise ValueError("child jobs have an unexpected number of workflows")
    for jobs_in_workflow in by_workflow.values():
        if Counter(job["kind"] for job in jobs_in_workflow) != expected:
            raise ValueError("each workflow must contain cycles repetitions of largeBA, smallBA, SIFT, smallBA")
        if len({job["sequence"] for job in jobs_in_workflow}) != len(jobs_in_workflow):
            raise ValueError("job sequences must be unique within each workflow")


def kill_and_reap(process):
    """Kill the private process group, then reap the child with resource usage."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    while True:
        try:
            return os.wait4(process.pid, 0)
        except InterruptedError:
            continue


def reject_json_constant(value):
    raise ValueError("nonfinite JSON number: " + value)


def finite_json_float(value):
    result = float(value)
    if not math.isfinite(result):
        reject_json_constant(value)
    return result


def run_trial(binary, mode, workflows, cycles, timeout, system):
    trial = {"mode": mode, "workflows": workflows, "cycles": cycles, "status": "failed"}
    env = os.environ.copy()
    env.update(THREAD_ENV)
    with tempfile.TemporaryDirectory(prefix="taskflow-stress-") as directory:
        directory = Path(directory)
        child_path = directory / "child.json"
        stdout_path = directory / "stdout.txt"
        stderr_path = directory / "stderr.txt"
        command = [str(binary), "--mode", mode, "--workflows", str(workflows),
                   "--cycles", str(cycles), "--output", str(child_path)]
        trial["command"] = command
        process = None
        reaped = False
        timed_out = False
        started = time.monotonic()
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                           stdin=subprocess.DEVNULL, env=env, start_new_session=True)
                deadline = started + timeout
                while True:
                    pid, status, usage = os.wait4(process.pid, os.WNOHANG)
                    if pid:
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        pid, status, usage = kill_and_reap(process)
                        break
                    time.sleep(min(POLL_SECONDS, remaining))
                reaped = True
                process.returncode = os.waitstatus_to_exitcode(status)
                wall = time.monotonic() - started
                cpu = usage.ru_utime + usage.ru_stime
                trial.update({
                    "returncode": process.returncode,
                    "timed_out": timed_out,
                    "wall_seconds": wall,
                    "cpu_user_seconds": usage.ru_utime,
                    "cpu_system_seconds": usage.ru_stime,
                    "cpu_seconds": cpu,
                    "avg_cpu_cores": cpu / wall,
                    "peak_rss_native": usage.ru_maxrss,
                    "peak_rss_native_unit": "bytes" if system == "Darwin" else "KiB",
                    "peak_rss_mib": rss_mib(usage.ru_maxrss, system),
                })
            if child_path.exists():
                trial["child_output_text"] = child_path.read_text(encoding="utf-8", errors="replace")
                try:
                    trial["child"] = json.loads(trial["child_output_text"],
                                                                   parse_constant=reject_json_constant,
                                                                   parse_float=finite_json_float)
                except ValueError as error:
                    trial["child_parse_error"] = str(error)
                else:
                    del trial["child_output_text"]
            if timed_out:
                raise ValueError("child exceeded timeout of {} seconds".format(timeout))
            if process.returncode != 0:
                raise ValueError("child exited with status {}".format(process.returncode))
            if "child" not in trial:
                raise ValueError("child did not produce valid JSON output")
            validate_child(trial["child"], mode, workflows, cycles)
            trial["jobs_per_second"] = len(trial["child"]["jobs"]) * 1000 / trial["child"]["workload_ms"]
            trial["status"] = "ok"
        except (OSError, ValueError) as error:
            trial["error"] = "{}: {}".format(type(error).__name__, error)
        finally:
            if process is not None and not reaped:
                _, status, _ = kill_and_reap(process)
                process.returncode = os.waitstatus_to_exitcode(status)
            for name, path in (("stdout", stdout_path), ("stderr", stderr_path)):
                trial[name] = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    return trial


def summarize(trials):
    groups = {}
    for trial in trials:
        if trial["phase"] == "measured" and trial["status"] == "ok":
            groups.setdefault((trial["workflows"], trial["cycles"], trial["mode"]), []).append(trial)
    summary = []
    for (workflows, cycles, mode), group in groups.items():
        jobs = [job for trial in group for job in trial["child"]["jobs"]]
        summary.append({
            "workflows": workflows, "cycles": cycles, "mode": mode,
            "trial_count": len(group), "job_count": len(jobs),
            "workload_ms_median": statistics.median(t["child"]["workload_ms"] for t in group),
            "jobs_per_second_median": statistics.median(t["jobs_per_second"] for t in group),
            "avg_cpu_cores_median": statistics.median(t["avg_cpu_cores"] for t in group),
            "peak_rss_mib_median": statistics.median(t["peak_rss_mib"] for t in group),
            "peak_rss_mib_max": max(t["peak_rss_mib"] for t in group),
            "kinds": {kind: {
                "elapsed_ms": distribution([j["elapsed_ms"] for j in jobs if j["kind"] == kind]),
                "queue_ms": distribution([j["queue_ms"] for j in jobs if j["kind"] == kind]),
            } for kind in KINDS},
        })
    return summary


def save_report(stream, report):
    report["summary"] = summarize(report["trials"])
    stream.seek(0)
    json.dump(report, stream, indent=2, allow_nan=False)
    stream.write("\n")
    stream.truncate()
    stream.flush()


def print_summary(report):
    print("workflows mode         trials jobs workload_ms jobs/s cores RSS_MiB(med/max)")
    for row in report["summary"]:
        print("{workflows:9d} {mode:12s} {trial_count:6d} {job_count:4d} "
              "{workload_ms_median:11.2f} {jobs_per_second_median:6.2f} "
              "{avg_cpu_cores_median:5.2f} {peak_rss_mib_median:.1f}/{peak_rss_mib_max:.1f}".format(**row))
        for kind, metrics in row["kinds"].items():
            elapsed, queue = metrics["elapsed_ms"], metrics["queue_ms"]
            print("  {} n={} elapsed p50/p95/max={:.2f}/{:.2f}/{:.2f} ms; "
                  "queue={:.2f}/{:.2f}/{:.2f} ms".format(
                      kind, elapsed["samples"], elapsed["p50"], elapsed["p95"], elapsed["max"],
                      queue["p50"], queue["p95"], queue["max"]))
    print(report["measurement_notes"]["quantiles"])
    print(report["measurement_notes"]["resources"])


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    system = platform.system()
    if system not in ("Darwin", "Linux") or not hasattr(os, "wait4"):
        cli.error("requires macOS or Linux with os.wait4")
    binary = args.binary.expanduser().resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        cli.error("--binary must name an existing executable file")
    report = {
        "schema_version": 1, "status": "running",
        "platform": {"system": system, "release": platform.release(), "machine": platform.machine(),
                     "description": platform.platform(), "python": platform.python_version()},
        "config": {"binary": str(binary), "rounds": args.rounds, "warmups": args.warmups,
                   "cycles": args.cycles, "workflows": args.workflows, "timeout_seconds": args.timeout,
                   "modes": args.modes, "environment_overrides": THREAD_ENV, "poll_seconds": POLL_SECONDS},
        "measurement_notes": {
            "quantiles": "P95 is descriptive: pooled, correlated closed-loop job samples, not a confidence guarantee; empirical nearest-rank quantiles.",
            "resources": "OS wait4 CPU and peak RSS cover the whole child including setup; workload_ms excludes setup. avg_cpu_cores = (user + system CPU seconds) / observed process wall seconds.",
            "wall": "Monotonic wall time covers spawn through observed exit/reap; polling can add up to about 20 ms plus scheduling delay.",
            "aggregation": "Only successful measured trials enter aggregates; warmups and failed trials remain in trials. Jobs/sec uses each child's workload_ms.",
            "order": "For each workflow count: warmup rounds, then measured rounds. Mode order rotates cyclically each round, continuing across phases.",
        },
        "expected_binary_policy": {
            "workflow": "cycles repetitions of largeBA, smallBA, SIFT, smallBA, with rotated starting offsets",
            "fixed": "FIFO, max 2 active tasks, BA 2 threads",
            "fixed4": "FIFO, max 4 active tasks, BA 1 thread",
            "taskflow": "global CPU 4, memory 1 GiB, max BA 4 threads, scratch 256 MiB",
            "uncontrolled": "every BA 4 threads, no aggregate cap",
            "ba_gate": "default 50000, small BA 1",
            "note": "Implemented by the supplied binary, not enforced by this runner.",
        },
        "trials": [], "summary": [],
    }
    try:
        stream = args.output.expanduser().open("x", encoding="utf-8")
    except OSError as error:
        cli.error("cannot exclusively create output: " + str(error))
    exit_code = 0
    with stream:
        try:
            save_report(stream, report)
            for workflows in args.workflows:
                for round_index in range(args.warmups + args.rounds):
                    phase = "warmup" if round_index < args.warmups else "measured"
                    phase_round = round_index if phase == "warmup" else round_index - args.warmups
                    for mode in mode_order(round_index, args.modes):
                        report["active_trial"] = {"workflows": workflows, "mode": mode,
                                                  "phase": phase, "round": phase_round + 1}
                        save_report(stream, report)
                        trial = run_trial(binary, mode, workflows, args.cycles, args.timeout, system)
                        trial.update(report.pop("active_trial"))
                        report["trials"].append(trial)
                        save_report(stream, report)
                        if trial["status"] != "ok":
                            raise RuntimeError(trial["error"])
            report["status"] = "completed"
        except (Exception, KeyboardInterrupt) as error:
            exit_code = 130 if isinstance(error, KeyboardInterrupt) else 1
            report["status"] = "failed"
            report["error"] = "{}: {}".format(type(error).__name__, error)
            print(report["error"], file=sys.stderr)
        finally:
            save_report(stream, report)
    print_summary(report)
    print("{}: {} ({} raw trials)".format(report["status"], args.output, len(report["trials"])))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

"""Focused runner tests; no Rust binary, build, or performance run is invoked."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import taskflow_stress as runner


def child_result(mode="fixed", workflows=1, cycles=1):
    return {
        "mode": mode, "workflows": workflows, "cycles": cycles, "workload_ms": 100.0,
        "jobs": [
            {"workflow": workflow, "sequence": sequence, "kind": kind,
             "queue_ms": float(sequence), "elapsed_ms": float(sequence + 1),
             "compute_ms": 0.5, "solver_threads": 2, "quality": {"cost": 0.01}}
            for workflow in range(workflows)
            for sequence, kind in enumerate(("large_ba", "small_ba", "sift", "small_ba") * cycles)
        ],
    }


def successful_trial(binary, mode, workflows, cycles, timeout, system):
    child = child_result(mode, workflows, cycles)
    return {"mode": mode, "workflows": workflows, "cycles": cycles, "status": "ok",
            "child": child, "jobs_per_second": len(child["jobs"]) * 10,
            "avg_cpu_cores": 2.0, "peak_rss_mib": 32.0}


class QuantileTests(unittest.TestCase):
    def test_empty(self):
        self.assertIsNone(runner.nearest_rank([], 0.95))
        self.assertEqual(runner.distribution([]), {"samples": 0, "p50": None, "p95": None, "max": None})

    def test_small_samples(self):
        self.assertEqual(runner.nearest_rank([7], 0.95), 7)
        self.assertEqual(runner.nearest_rank([9, 1], 0.50), 1)
        self.assertEqual(runner.nearest_rank([9, 1], 0.95), 9)
        self.assertEqual(runner.nearest_rank(list(range(1, 21)), 0.95), 19)
        self.assertEqual(runner.nearest_rank([3, 3, 3], 1), 3)

    def test_invalid_quantile(self):
        for value in (0, -1, 1.01, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.nearest_rank([1], value)

    def test_rss_units(self):
        self.assertEqual(runner.rss_mib(1048576, "Darwin"), 1)
        self.assertEqual(runner.rss_mib(1024, "Linux"), 1)
        with self.assertRaises(ValueError):
            runner.rss_mib(1024, "Windows")


class ValidationTests(unittest.TestCase):
    def parse(self, *extra):
        return runner.parser().parse_args(["--binary", "example", "--output", "out.json", *extra])

    def test_defaults(self):
        args = self.parse()
        self.assertEqual((args.rounds, args.cycles, args.workflows, args.warmups, args.timeout),
                         (5, 3, [4, 8, 16], 1, 120.0))
        self.assertEqual(self.parse("--warmups", "0").warmups, 0)
        self.assertEqual(tuple(args.modes), ("uncontrolled", "fixed", "taskflow"))

    def test_selected_modes(self):
        self.assertEqual(self.parse("--modes", "fixed4", "taskflow").modes,
                         ["fixed4", "taskflow"])
        self.assertEqual(self.parse("--modes", "taskflow", "fixed", "uncontrolled", "fixed4").modes,
                         ["taskflow", "fixed", "uncontrolled", "fixed4"])
        self.assertEqual(self.parse("--modes", "fixed4").modes, ["fixed4"])

    def test_invalid_modes(self):
        for modes in ((), ("invalid",), ("fixed4", "invalid"), ("Fixed4",)):
            with self.subTest(modes=modes), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                self.parse("--modes", *modes)
            self.assertEqual(error.exception.code, 2)

    def test_duplicate_modes(self):
        for modes in (("fixed4", "fixed4"), ("fixed4", "taskflow", "fixed4"),
                      ("uncontrolled", "fixed", "fixed"), ("taskflow", "taskflow")):
            stderr = io.StringIO()
            with self.subTest(modes=modes), redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                self.parse("--modes", *modes)
            self.assertEqual(error.exception.code, 2)
            self.assertIn("duplicate", stderr.getvalue())

    def test_default_mode_order(self):
        expected = [("uncontrolled", "fixed", "taskflow"),
                    ("fixed", "taskflow", "uncontrolled"),
                    ("taskflow", "uncontrolled", "fixed")]
        for index in range(6):
            self.assertEqual(runner.mode_order(index), expected[index % 3])

    def test_selected_mode_order(self):
        modes = ["fixed4", "taskflow"]
        for index in range(6):
            self.assertEqual(list(runner.mode_order(index, modes)),
                             modes if index % 2 == 0 else ["taskflow", "fixed4"])
        self.assertEqual(modes, ["fixed4", "taskflow"])
        self.assertEqual(list(runner.mode_order(3, ["fixed4"])), ["fixed4"])

    def test_invalid_inputs(self):
        cases = [(flag, value) for flag in ("--rounds", "--cycles", "--workflows")
                 for value in ("0", "-1", "1.2", "bad")]
        cases += [("--timeout", value) for value in ("0", "-1", "nan", "inf", "-inf", "bad")]
        cases += [("--warmups", "-1"), ("--warmups", "1.5")]
        for flag, value in cases:
            with self.subTest(flag=flag, value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.parse(flag + "=" + value)

    def test_required_options(self):
        for args in ([], ["--binary", "example"], ["--output", "out.json"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.parser().parse_args(args)

    def test_child_schema_and_quality(self):
        child = child_result()
        runner.validate_child(child, "fixed", 1, 1)
        self.assertEqual(child["jobs"][0]["quality"], {"cost": 0.01})
        for key, value in (("mode", "wrong"), ("workflows", True), ("workload_ms", float("nan")),
                           ("workload_ms", 0), ("jobs", [])):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                runner.validate_child(dict(child, **{key: value}), "fixed", 1, 1)
        child["jobs"][0]["queue_ms"] = -1
        with self.assertRaises(ValueError):
            runner.validate_child(child, "fixed", 1, 1)

    def test_nonfinite_json_rejected(self):
        for text in ("NaN", "Infinity", "-Infinity", "1e999"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                json.loads(text, parse_constant=runner.reject_json_constant,
                           parse_float=runner.finite_json_float)


class ProcessTests(unittest.TestCase):
    def run_mock_child(self, status=0, timeout=False, mode="fixed"):
        process = Mock(pid=123, returncode=None)
        captured = {}

        def spawn(command, **kwargs):
            captured.update(kwargs)
            captured["command"] = command
            self.assertEqual(command[1:7], ["--mode", mode, "--workflows", "1", "--cycles", "1"])
            Path(command[-1]).write_text(json.dumps(child_result(mode)), encoding="utf-8")
            kwargs["stdout"].write(b"child stdout\n")
            kwargs["stderr"].write(b"child stderr\n")
            return process

        usage = SimpleNamespace(ru_utime=1.25, ru_stime=0.75, ru_maxrss=2048)
        waits = [(0, 0, None), (123, status, usage)] if timeout else [(123, status, usage)]
        times = [0.0, 1.0, 2.0] if timeout else [0.0, 2.0]
        with patch.object(runner.subprocess, "Popen", side_effect=spawn), \
                patch.object(runner.os, "wait4", side_effect=waits) as wait4, \
                patch.object(runner.os, "killpg") as killpg, \
                patch.object(runner.time, "monotonic", side_effect=times), \
                patch.object(runner.time, "sleep") as sleep:
            trial = runner.run_trial(Path(sys.executable), mode, 1, 1, 0.1, "Linux")
        self.assertEqual(trial["command"], captured["command"])
        for key, value in runner.THREAD_ENV.items():
            self.assertEqual(captured["env"][key], value)
        self.assertTrue(captured["start_new_session"])
        self.assertEqual(captured["stdin"], runner.subprocess.DEVNULL)
        self.assertTrue(captured["stdout"].closed)
        self.assertTrue(captured["stderr"].closed)
        self.assertEqual(wait4.call_args_list[0].args, (123, runner.os.WNOHANG))
        if timeout:
            killpg.assert_called_once_with(123, signal.SIGKILL)
            self.assertEqual(wait4.call_args_list[-1].args, (123, 0))
        else:
            killpg.assert_not_called()
        sleep.assert_not_called()
        self.assertEqual(trial["stdout"], "child stdout\n")
        self.assertEqual(trial["stderr"], "child stderr\n")
        self.assertEqual(process.returncode, runner.os.waitstatus_to_exitcode(status))
        return trial

    def test_resource_measurements_and_raw_jobs(self):
        trial = self.run_mock_child()
        self.assertEqual(trial["status"], "ok")
        self.assertEqual(trial["cpu_seconds"], 2)
        self.assertEqual(trial["wall_seconds"], 2)
        self.assertEqual(trial["avg_cpu_cores"], 1)
        self.assertEqual(trial["peak_rss_mib"], 2)
        self.assertEqual(trial["jobs_per_second"], 40)
        self.assertEqual(trial["child"], child_result())

    def test_fixed4_command_and_raw_result(self):
        trial = self.run_mock_child(mode="fixed4")
        self.assertEqual(trial["status"], "ok")
        self.assertEqual(trial["mode"], "fixed4")
        self.assertEqual(trial["command"][1:3], ["--mode", "fixed4"])
        self.assertEqual(trial["child"], child_result("fixed4"))

    def test_timeout_kills_and_reaps(self):
        trial = self.run_mock_child(status=signal.SIGKILL, timeout=True)
        self.assertEqual(trial["status"], "failed")
        self.assertTrue(trial["timed_out"])
        self.assertIn("timeout", trial["error"])
        self.assertEqual(trial["cpu_seconds"], 2)

    def test_nonzero_exit(self):
        trial = self.run_mock_child(status=3 << 8)
        self.assertEqual(trial["status"], "failed")
        self.assertEqual(trial["returncode"], 3)
        self.assertIn("child", trial)


class OrchestrationTests(unittest.TestCase):
    def invoke(self, output, side_effect=successful_trial, *extra):
        with patch.object(runner, "run_trial", side_effect=side_effect) as run, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = runner.main(["--binary", sys.executable, "--output", str(output),
                                "--workflows", "1", "--cycles", "1", "--rounds", "2", *extra])
        return code, run

    def test_rotation_warmups_and_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            code, run = self.invoke(output)
            report = json.loads(output.read_text())
        self.assertEqual(code, 0)
        self.assertEqual(run.call_count, 9)
        self.assertEqual([call.args[1] for call in run.call_args_list],
                         [mode for index in range(3) for mode in runner.mode_order(index)])
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["config"]["modes"], ["uncontrolled", "fixed", "taskflow"])
        self.assertEqual(len(report["trials"]), 9)
        self.assertEqual(len(report["summary"]), 3)
        for row in report["summary"]:
            self.assertEqual(row["trial_count"], 2)
            self.assertEqual(row["job_count"], 8)
            self.assertEqual(row["kinds"]["small_ba"]["elapsed_ms"]["samples"], 4)
            self.assertEqual(row["kinds"]["small_ba"]["elapsed_ms"]["p95"], 4)

    def test_selected_fixed4_taskflow_rotation_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            code, run = self.invoke(output, successful_trial, "--modes", "fixed4", "taskflow")
            report = json.loads(output.read_text())
        self.assertEqual(code, 0)
        self.assertEqual(run.call_count, 6)
        expected = ["fixed4", "taskflow", "taskflow", "fixed4", "fixed4", "taskflow"]
        self.assertEqual([call.args[1] for call in run.call_args_list], expected)
        self.assertEqual([trial["mode"] for trial in report["trials"]], expected)
        self.assertEqual([trial["phase"] for trial in report["trials"]],
                         ["warmup"] * 2 + ["measured"] * 4)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["config"]["modes"], ["fixed4", "taskflow"])
        self.assertEqual(report["expected_binary_policy"]["fixed4"],
                         "FIFO, max 4 active tasks, BA 1 thread")
        self.assertEqual({row["mode"] for row in report["summary"]}, {"fixed4", "taskflow"})
        self.assertEqual(len(report["summary"]), 2)
        for row in report["summary"]:
            self.assertEqual(row["trial_count"], 2)
            self.assertEqual(row["job_count"], 8)

    def test_existing_output_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            output.write_text("keep me")
            with self.assertRaises(SystemExit):
                self.invoke(output)
            self.assertEqual(output.read_text(), "keep me")

    def test_failure_preserves_partial_results(self):
        calls = []

        def fail_second(*args):
            calls.append(args)
            trial = successful_trial(*args)
            if len(calls) == 2:
                trial.update(status="failed", error="simulated child failure")
            return trial

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            code, run = self.invoke(output, fail_second, "--warmups", "0")
            report = json.loads(output.read_text())
        self.assertEqual(code, 1)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["trials"]), 2)
        self.assertEqual(len(report["summary"]), 1)
        self.assertEqual(report["summary"][0]["trial_count"], 1)
        self.assertIn("simulated child failure", report["error"])


if __name__ == "__main__":
    unittest.main()

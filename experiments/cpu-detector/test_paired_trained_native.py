# SPDX-License-Identifier: Apache-2.0
"""Auditor protocol tests. Native compiler/measurement calls are controlled
stand-ins, never training, real model consumption or performance evidence."""
import contextlib
import csv
import io
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import paired_trained_native as paired

REAL_RUN = subprocess.run


@unittest.skipUnless(platform.system() == "Linux" and platform.machine() == "x86_64",
                     "The paired native driver supports Linux x64 only")
class PairedProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / ".gitignore").write_text("/target/\n")
        c = self.root / "experiments/cpu-detector/paired_trained_native.c"
        c.parent.mkdir(parents=True)
        c.write_bytes((Path(__file__).parent / "paired_trained_native.c").read_bytes())
        self.source = c
        self.git("init", "--quiet")
        self.git("add", ".")
        self.commit("Isolated auditor test input, not project history")
        self.head = self.git("rev-parse", "HEAD").strip()
        self.target = self.root / "target"
        self.target.mkdir()
        self.base = self.target / "CONTROL-base.not-a-library"
        self.current = self.target / "CONTROL-current.not-a-library"
        for path in (self.base, self.current):
            path.write_text("UNIT PROTOCOL ONLY; no native library is loaded.\n")
        self.training = self.target / "training"
        self.native = []
        self.output = self.target / "evidence"
        self.native_calls = 0
        self.fail_at = None
        self.differ_at = None
        self.drift_at_end = False
        for directory, backend in (("burn", "burn-flex-0.22.0"),
                                   ("candle", "candle-cpu-0.11.0")):
            model_dir = self.training / directory
            model_dir.mkdir(parents=True)
            # Size-only unit input. Deliberately NOT a checksum-valid model;
            # the external native boundary is replaced for auditor testing.
            model = model_dir / "model.fytd"
            model.write_bytes(bytes(51284))
            report = {"passed": True, "source_commit": self.head,
                      "backend": backend, "steps": 2000, "test_seed": 909,
                      "test_samples": 128,
                      "test_fingerprint": paired.TEST_FINGERPRINT,
                      "predictions": [{"index": i} for i in range(6)],
                      "test": {"unit_protocol_fixture": True}}
            (model_dir / "metrics.json").write_text(json.dumps(report))
            native = self.target / ("native-" + directory)
            cases = native / "cases"
            cases.mkdir(parents=True)
            summary = {"passed": True, "source_commit": self.head,
                       "training_backend": backend,
                       "artifact_sha256": paired.sha256(model),
                       "heldout_samples": 6, "c_link_modes": ["shared", "static"],
                       "managed_execution_modes": ["jit", "aot"],
                       "runtime_identifier": "linux-x64"}
            (native / "summary.json").write_text(json.dumps(summary))
            for i in range(6):
                (cases / f"sample-{i}.rgb").write_bytes(bytes(12288))
                (cases / f"expected-{i}.txt").write_text("0\n")
            self.native.append(native)

    def git(self, *args):
        result = REAL_RUN(["git", "-C", str(self.root), *args],
                          text=True, capture_output=True, check=True)
        return result.stdout

    def commit(self, message):
        self.git("-c", "user.name=Isolated test fixture",
                 "-c", "user.email=fixture@example.invalid",
                 "commit", "--quiet", "-am", message)

    @staticmethod
    def emit(stream, text):
        if stream not in (None, subprocess.PIPE, subprocess.DEVNULL):
            stream.write(text)
            stream.flush()

    def external_run(self, args, **kwargs):
        argv = list(map(str, args))
        if argv[0] == "git":
            return REAL_RUN(args, **kwargs)
        status, out, err = 0, "", ""
        if "-std=c11" in argv:
            pass  # No native compiler/library execution in this unit control.
        elif argv[1:] == ["--version"]:
            out = "UNIT-COMPILER-FIXTURE\n"
            if self.drift_at_end:
                self.source.write_text(self.source.read_text() + "\n/* drift */\n")
                self.commit("Second isolated source-identity input")
        else:
            self.native_calls += 1
            if self.native_calls == self.fail_at:
                status, out, err = 7, "CONTROL partial stdout\n", "CONTROL failure\n"
            else:
                bits = [0] * 6
                if self.native_calls == self.differ_at:
                    bits[1] = 1
                out = json.dumps({"calls": 512, "wall_ns": 2000, "cpu_ns": 1000,
                                  "count": 0, "prediction_bits": bits,
                                  "process_maxrss_kib": 1}) + "\n"
        stdout, stderr = kwargs.get("stdout"), kwargs.get("stderr")
        self.emit(stdout, out)
        self.emit(stderr, err)
        return subprocess.CompletedProcess(
            argv, status, out if stdout == subprocess.PIPE else None,
            err if stderr == subprocess.PIPE else None)

    def invoke(self):
        args = ["paired_trained_native.py", *map(str, [
            self.base, self.current, self.training, *self.native, self.output])]
        with patch.object(paired, "REPO", self.root), \
                patch.object(paired.sys, "argv", args), \
                patch.dict(os.environ, {"FACTSYOLO_BUILD_ID": self.head}), \
                patch.object(subprocess, "run", side_effect=self.external_run), \
                contextlib.redirect_stdout(io.StringIO()):
            paired.main()

    def test_complete_unit_protocol_keeps_all_252_matched_pairs(self):
        self.invoke()
        with (self.output / "raw_pairs.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 252)
        self.assertEqual(sum(r["backend"] == "burn" for r in rows), 126)
        self.assertEqual(sum(r["backend"] == "candle" for r in rows), 126)
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertIs(summary["passed"], True)

    def test_other_backend_version_is_rejected(self):
        p = self.native[0] / "summary.json"
        value = json.loads(p.read_text())
        value["training_backend"] = "burn-flex-0.22.1"
        p.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            self.invoke()
        self.assertEqual(self.native_calls, 0)

    def test_matching_wrong_report_and_native_backend_are_rejected(self):
        p = self.native[0] / "summary.json"
        value = json.loads(p.read_text())
        value["training_backend"] = "candle-cpu-0.11.0"
        p.write_text(json.dumps(value))
        p = self.training / "burn/metrics.json"
        value = json.loads(p.read_text())
        value["backend"] = "candle-cpu-0.11.0"
        p.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            self.invoke()
        self.assertEqual(self.native_calls, 0)



    def test_completed_and_interrupted_pairs_survive_child_failure(self):
        self.fail_at = 4  # One complete pair, then one successful mode.
        with self.assertRaises(RuntimeError):
            self.invoke()
        self.assertTrue((self.output / "raw_pairs.csv").is_file())
        with (self.output / "raw_pairs.csv").open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 1)
        self.assertIs(json.loads((self.output / "summary.json").read_text())["passed"], False)
        events = [json.loads(line) for line in
                  (self.output / "pair_records.jsonl").read_text().splitlines()]
        failed = events[-1]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(sum(t is not None for t in failed["timings"]), 1)
        commands = [json.loads(line) for line in
                    (self.output / "command_records.jsonl").read_text().splitlines()]
        self.assertEqual(commands[-1]["exit"], 7)
        self.assertEqual((self.output / commands[-1]["stdout_log"]).read_text(),
                         "CONTROL partial stdout\n")
        self.assertEqual((self.output / commands[-1]["stderr_log"]).read_text(),
                         "CONTROL failure\n")

    def test_parity_failure_preserves_both_successful_child_outputs(self):
        self.differ_at = 2
        with self.assertRaises(ValueError):
            self.invoke()
        self.assertIs(json.loads((self.output / "summary.json").read_text())["passed"], False)
        events = [json.loads(line) for line in
                  (self.output / "pair_records.jsonl").read_text().splitlines()]
        self.assertEqual(events[-1]["status"], "failed")
        self.assertTrue(all(t is not None for t in events[-1]["timings"]))
        commands = [json.loads(line) for line in
                    (self.output / "command_records.jsonl").read_text().splitlines()
                    if json.loads(line)["status"] == "completed"]
        self.assertEqual(len([c for c in commands if c["exit"] == 0]), 4)



    def test_assume_unchanged_modified_source_is_rejected_before_sampling(self):
        self.git("update-index", "--assume-unchanged",
                 str(self.source.relative_to(self.root)))
        self.source.write_text(self.source.read_text() + "\n/* hidden unit change */\n")
        self.assertEqual(self.git("status", "--porcelain"), "")
        with self.assertRaises(RuntimeError):
            self.invoke()
        self.assertFalse(self.output.exists())

    def test_skip_worktree_modified_source_is_rejected_before_sampling(self):
        self.git("update-index", "--skip-worktree",
                 str(self.source.relative_to(self.root)))
        self.source.write_text(self.source.read_text() + "\n/* hidden unit change */\n")
        self.assertEqual(self.git("status", "--porcelain"), "")
        with self.assertRaises(RuntimeError):
            self.invoke()
        self.assertFalse(self.output.exists())

    def test_ending_head_drift_retains_all_pairs_but_never_passes(self):
        self.drift_at_end = True
        with self.assertRaises(RuntimeError):
            self.invoke()
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertIs(summary["passed"], False)
        self.assertEqual(summary["completed_pairs"], 252)
        with (self.output / "raw_pairs.csv").open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 252)


class RecordedCommandTests(unittest.TestCase):
    def record_failure(self, program, timeout=900):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            evidence = paired.Evidence(output, "unit-command-control")
            with self.assertRaises((RuntimeError, subprocess.TimeoutExpired)) as failure:
                paired.command([sys.executable, "-c", program], recorder=evidence,
                               label="unit-child", timeout=timeout)
            evidence.fail(failure.exception)
            events = [json.loads(line) for line in
                      (output / "command_records.jsonl").read_text().splitlines()]
            return (events, json.loads((output / "summary.json").read_text()),
                    (output / events[-1]["stdout_log"]).read_text(),
                    (output / events[-1]["stderr_log"]).read_text())

    def test_nonzero_child_records_original_stdout_stderr_and_exit(self):
        events, summary, stdout, stderr = self.record_failure(
            "import sys; print('original stdout'); "
            "sys.stderr.write('original stderr\\n'); sys.exit(7)")
        self.assertEqual(events[-1]["exit"], 7)
        self.assertEqual(stdout, "original stdout\n")
        self.assertEqual(stderr, "original stderr\n")
        self.assertIs(summary["passed"], False)

    def test_timeout_records_partial_output_and_incomplete_status(self):
        events, summary, stdout, _ = self.record_failure(
            "import time; print('before timeout', flush=True); time.sleep(5)",
            timeout=0.2)
        self.assertEqual(events[-1]["status"], "timeout")
        self.assertIsNone(events[-1]["exit"])
        self.assertEqual(stdout, "before timeout\n")
        self.assertIs(summary["passed"], False)


if __name__ == "__main__":
    unittest.main()


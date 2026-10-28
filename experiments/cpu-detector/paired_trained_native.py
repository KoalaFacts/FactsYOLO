#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare exact historical and current native C ABI libraries on freshly trained TD1.

This is a Linux x64 evidence gate, not a user-facing release or speed guarantee.
The model and pixels remain ephemeral; only scalar measurements and hashes survive.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

from verify import require_clean_source_tree

REPO = Path(__file__).resolve().parents[2]
BASE_SHA = "1a5e0a3e236cc1057304fb47c3a3a5a7a05c3837"
TEST_FINGERPRINT = "188366b67882eb3599300ac6b056c31b5b28e41733ab801fef1362bb05354c33"
BACKEND_IDS = {"burn": "burn-flex-0.22.0", "candle": "candle-cpu-0.11.0"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Evidence:
    """Incremental audit trail; a partial run can never carry passed=true."""
    def __init__(self, output, head):
        self.output = output
        self.head = head
        self.count = 0
        self.sequence = 0
        self.active_pair = None
        for name in ("command_records.jsonl", "pair_records.jsonl", "raw_pairs.csv"):
            (output / name).touch(exist_ok=False)
        self.write_summary({"passed": False, "status": "running",
                            "base_source": BASE_SHA, "current_source": head,
                            "completed_pairs": 0})

    def append(self, name, value):
        with (self.output / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def write_summary(self, value):
        temporary = self.output / "summary.json.tmp"
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.output / "summary.json")

    def start_command(self, argv, label):
        self.sequence += 1
        label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
        stem = f"{self.sequence:04d}-{label}"
        record = {"command": self.sequence, "argv": argv, "cwd": str(REPO),
                  "status": "running", "exit": None,
                  "stdout_log": stem + ".stdout.log",
                  "stderr_log": stem + ".stderr.log",
                  "started_unix_seconds": time.time()}
        self.append("command_records.jsonl", record)
        return record

    def finish_command(self, record, status, exit_code, error=None):
        record = dict(record, status=status, exit=exit_code,
                      finished_unix_seconds=time.time())
        if error is not None:
            record["error"] = str(error)
        self.append("command_records.jsonl", record)

    def begin_pair(self, context, order):
        self.active_pair = dict(context, order=list(order), status="running",
                                timings=[None, None])
        self.append("pair_records.jsonl", self.active_pair)

    def mode(self, mode, timing):
        self.active_pair["timings"][mode] = timing
        self.append("pair_records.jsonl", self.active_pair)

    def complete_pair(self, row):
        with (self.output / "raw_pairs.csv").open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if self.count == 0:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        self.count += 1
        self.active_pair["status"] = "completed"
        self.append("pair_records.jsonl", self.active_pair)
        self.active_pair = None

    def fail(self, error):
        if self.active_pair is not None:
            self.active_pair.update(status="failed", error=str(error))
            self.append("pair_records.jsonl", self.active_pair)
        self.write_summary({"passed": False, "status": "failed",
                            "base_source": BASE_SHA, "current_source": self.head,
                            "completed_pairs": self.count,
                            "error_type": type(error).__name__, "error": str(error)})

    def succeed(self, summary):
        self.write_summary(dict(summary, status="completed",
                                completed_pairs=self.count))


def command(args, output=None, *, recorder=None, label=None, timeout=900):
    argv = [str(a) for a in args]
    if recorder is None:
        result = subprocess.run(argv, cwd=REPO, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=timeout)
        stdout, stderr = result.stdout, result.stderr
    else:
        record = recorder.start_command(argv, label or Path(argv[0]).name)
        stdout_path = recorder.output / record["stdout_log"]
        stderr_path = recorder.output / record["stderr_log"]
        try:
            # Stream directly to files so partial output survives an interrupted
            # child instead of remaining only in the parent's capture buffer.
            with stdout_path.open("w", encoding="utf-8") as out, \
                    stderr_path.open("w", encoding="utf-8") as err:
                result = subprocess.run(argv, cwd=REPO, text=True,
                                        stdout=out, stderr=err, timeout=timeout)
        except BaseException as error:
            status = "timeout" if isinstance(error, subprocess.TimeoutExpired) else "interrupted"
            recorder.finish_command(record, status, None, error)
            raise
        recorder.finish_command(record, "completed", result.returncode)
        stdout = stdout_path.read_text(encoding="utf-8")
        stderr = stderr_path.read_text(encoding="utf-8")
    if output is not None:
        output.write_text(stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"exit {result.returncode}: {argv[0]}: {stderr[-1200:]}")
    return stdout


def main():
    if len(sys.argv) != 7:
        raise ValueError("usage: paired_trained_native.py BASE.so CURRENT.so "
                         "training_evidence burn_native candle_native output_dir")
    require(platform.system() == "Linux" and platform.machine() == "x86_64",
            "paired native C benchmarking currently supports Linux x64 only")
    base, current, training, burn_native, candle_native, output = (
        Path(p).resolve() for p in sys.argv[1:])
    head = command(["git", "rev-parse", "HEAD"]).strip()
    require(head != BASE_SHA and os.environ.get("FACTSYOLO_BUILD_ID") == head,
            "new source SHA and FACTSYOLO_BUILD_ID must match the checked-out commit")
    require_clean_source_tree(REPO, head)
    require(base.is_file() and current.is_file(), "missing baseline or current TD1 library")
    require(not output.exists(), "evidence directory already exists")
    output.mkdir(parents=True)
    evidence = Evidence(output, head)

    try:
        cpu = None
        try:
            allowed = os.sched_getaffinity(0)
            cpu = min(allowed)
            os.sched_setaffinity(0, {cpu})
        except (AttributeError, OSError, ValueError):
            cpu = None

        compiler = os.environ.get("CC", "cc")
        binaries = []
        provenance = []
        for label, library in (("base", base), ("current", current)):
            binary = output / f"td1-{label}-timed"
            argv = [compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
                    "-I", REPO / "include", REPO / "experiments/cpu-detector/paired_trained_native.c",
                    library, "-Wl,-rpath," + str(library.parent), "-lm", "-o", binary]
            command(argv, output / f"compile-{label}.log", recorder=evidence,
                    label=f"compile-{label}")
            binaries.append(binary)
            provenance.append({"source": BASE_SHA if label == "base" else head,
                               "library_sha256": sha256(library),
                               "library_path": str(library)})
        all_rows = []
        backends = {}
        for name, native_path in (("burn", burn_native), ("candle", candle_native)):
            model_dir = training / name
            report = json.loads((model_dir / "metrics.json").read_text())
            accepted = json.loads((native_path / "summary.json").read_text())
            model = model_dir / "model.fytd"
            model_hash = sha256(model)
            require(model.stat().st_size == 51284, f"{name}: invalid trained model byte count")
            require(report.get("passed") is True and report.get("source_commit") == head
                    and report.get("steps") == 2000 and report.get("test_seed") == 909
                    and report.get("test_samples") == 128
                    and report.get("test_fingerprint") == TEST_FINGERPRINT
                    and report.get("backend") == BACKEND_IDS[name],
                    f"{name}: stale or unaccepted full training/held-out report")
            require(accepted.get("passed") is True and accepted.get("source_commit") == head
                    and accepted.get("artifact_sha256") == model_hash
                    and accepted.get("training_backend") == BACKEND_IDS[name],
                    f"{name}: native C/.NET accepted artifact does not match")
            cases = native_path / "cases"
            sample_hashes = []
            for i in range(6):
                require(report["predictions"][i]["index"] == i, f"{name}: wrong sample order")
                image = cases / f"sample-{i}.rgb"
                require(image.stat().st_size == 12288
                        and (cases / f"expected-{i}.txt").is_file(),
                        f"{name}: incomplete previously verified native case")
                sample_hashes.append(sha256(image))
            for process in range(3):
                for repeat in range(7):
                    for sample in range(6):
                        args = [model, cases / f"sample-{sample}.rgb",
                                cases / f"expected-{sample}.txt"]
                        # Each mode runs in a fresh process (no shared-library collision).
                        # Alternate AB/BA order with the exact same sample and CPU.
                        order = (0, 1) if (process + repeat + sample) % 2 == 0 else (1, 0)
                        timings = [None, None]
                        evidence.begin_pair({"backend": name, "process": process+1,
                                             "pair": repeat+1, "sample": sample}, order)
                        for mode in order:
                            stdout = command([binaries[mode], *args,
                                              BASE_SHA if mode == 0 else head],
                                             recorder=evidence,
                                             label=f"{name}-g{process+1}-r{repeat+1}-s{sample}-m{mode}")
                            timings[mode] = json.loads(stdout.strip())
                            evidence.mode(mode, timings[mode])
                            require(timings[mode]["calls"] == 512
                                    and timings[mode]["wall_ns"] > 0
                                    and timings[mode]["cpu_ns"] > 0,
                                    f"{name}/{sample}: invalid native measurement")
                        require(timings[0]["count"] == timings[1]["count"]
                                and timings[0]["prediction_bits"] == timings[1]["prediction_bits"],
                                f"{name}/{sample}: native bitwise output parity regression")
                        row = {
                            "backend": name, "process": process+1, "pair": repeat+1,
                            "sample": sample, "calls_per_library": 512,
                            "base_wall_ns": timings[0]["wall_ns"],
                            "current_wall_ns": timings[1]["wall_ns"],
                            "base_cpu_ns": timings[0]["cpu_ns"],
                            "current_cpu_ns": timings[1]["cpu_ns"],
                            "base_process_maxrss_kib": timings[0]["process_maxrss_kib"],
                            "current_process_maxrss_kib": timings[1]["process_maxrss_kib"]}
                        all_rows.append(row)
                        evidence.complete_pair(row)
            rows = [row for row in all_rows if row["backend"] == name]
            backends[name] = {
                "artifact_sha256": model_hash,
                "heldout_sample_sha256": sample_hashes,
                "accuracy": report["test"],
                "matched_pairs": len(rows),
                "calls_per_library": sum(row["calls_per_library"] for row in rows),
                "median_base_wall_us_per_call": statistics.median(
                    row["base_wall_ns"] / 512000 for row in rows),
                "median_current_wall_us_per_call": statistics.median(
                    row["current_wall_ns"] / 512000 for row in rows),
                "median_paired_wall_saving_percent": statistics.median(
                    100 * (1 - row["current_wall_ns"] / row["base_wall_ns"]) for row in rows),
                "faster_current_pairs": sum(row["current_wall_ns"] < row["base_wall_ns"]
                                            for row in rows),
            }

        cpu_model = next((line.split(":", 1)[1].strip() for line in
                          Path("/proc/cpuinfo").read_text().splitlines()
                          if line.startswith("model name")), "unknown")
        summary = {
            "passed": True, "scope": "native TD1 C ABI detect/copy/release (no model open in timer)",
            "model": "fresh 2000-step synthetic Burn/Candle trained FYTDv001",
            "base_source": BASE_SHA, "current_source": head, "binary_provenance": provenance,
            "benchmark_source_sha256": sha256(REPO / "experiments/cpu-detector/paired_trained_native.c"),
            "machine": {"platform": platform.platform(), "cpu": cpu_model,
                        "pinned_logical_cpu": cpu,
                        "compiler": command([compiler, "--version"], recorder=evidence,
                                            label="compiler-version").splitlines()[0]},
            "backends": backends,
            "limitations": ("held-out synthetic images, not natural-image accuracy; "
                            "process-level RSS (not per-operation); heap allocation count not measured; "
                            "single hosted CPU and Linux x64 only; no production latency/throughput claim; "
                            "training, model loading, managed SDKs and image acquisition excluded"),
        }
        require_clean_source_tree(REPO, head)
        evidence.succeed(summary)
        print(json.dumps(summary, indent=2))
    except BaseException as error:
        evidence.fail(error)
        raise


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)

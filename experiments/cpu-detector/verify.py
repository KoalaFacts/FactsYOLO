#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the additive CPU experiment gates, preserving commands and actual results."""
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from inference_gate import verify_inference_benchmark
from verify_report import enforce_heldout_thresholds, verify_test_predictions

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
MANIFEST = str(ROOT / "Cargo.toml")


def checked_training_heldout(report, expected_sha):
    """Audit a completed candidate before recording it as accepted evidence."""
    if report.get("passed") is not True or report.get("source_commit") != expected_sha:
        raise RuntimeError("training report failed or stale")
    audited = verify_test_predictions(report)
    enforce_heldout_thresholds(audited, report.get("test"))
    # Row arithmetic alone cannot prove the runner used the preregistered final
    # holdout: a shortened, easier subset could have self-consistent metrics.
    # These constants pin TD1's accepted seed/count and existing data fingerprint,
    # not a new format or post-hoc quality threshold.
    if (type(report.get("test_seed")) is not int or report["test_seed"] != 909
            or report.get("test_samples") != 128
            or report.get("test_fingerprint") != "188366b67882eb3599300ac6b056c31b5b28e41733ab801fef1362bb05354c33"):
        raise ValueError("unregistered TD1 held-out split")
    return audited


def require_clean_source_tree(repository, expected_sha=None):
    """Reject dirty sources and, when supplied, a changed measurement revision."""
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repository, text=True)
    if status.strip():
        raise RuntimeError("source worktree is dirty; HEAD cannot attest to measured code")
    # Git status trusts index flags. A modified tracked source marked
    # assume-unchanged or skip-worktree is otherwise invisible even if HEAD
    # is unchanged. Reject those flags rather than certify false provenance.
    index_entries = subprocess.check_output(
        ["git", "ls-files", "-v", "-z"], cwd=repository)
    if any(not entry.startswith(b"H ") for entry in index_entries.split(bytes([0])) if entry):
        raise RuntimeError("source index hides tracked files; cannot attest to measured code")
    if expected_sha is not None:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
        if head != expected_sha:
            raise RuntimeError("source HEAD changed during verification")


def main():
    if len(sys.argv) != 2:
        raise ValueError("usage: python3 experiments/cpu-detector/verify.py <new-evidence-directory>")
    evidence = Path(sys.argv[1]).resolve()
    require_clean_source_tree(REPO)
    evidence.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "1"
    env["FACTSYOLO_BUILD_ID"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    commands = []

    def run(label, arguments, *, expected_success=True):
        print(f"{label}: {' '.join(map(str, arguments))}", flush=True)
        started = time.monotonic()
        with (evidence / f"{label}.log").open("w") as log:
            result = subprocess.run(list(map(str, arguments)), cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        commands.append({"label": label, "argv": list(map(str, arguments)), "exit": result.returncode, "seconds": time.monotonic()-started})
        (evidence / "commands.json").write_text(json.dumps(commands, indent=2)+"\n")
        if (result.returncode == 0) != expected_success:
            raise RuntimeError(f"unexpected exit {result.returncode}: {label}; see {evidence / (label+'.log')}")
        return (evidence / f"{label}.log").read_text()

    common = ["--manifest-path", MANIFEST, "--locked", "--offline"]
    run("format", ["cargo", "fmt", "--manifest-path", MANIFEST, "--all", "--", "--check"])
    run("clippy", ["cargo", "clippy", *common, "--workspace", "--all-targets", "--", "-D", "warnings"])
    run("tests-debug", ["cargo", "test", *common, "--workspace"])
    run("tests-release", ["cargo", "test", *common, "--workspace", "--release"])
    run("portable-msrv", ["cargo", "+1.85.0", "test", *common, "-p", "fy-tiny-model"])
    run("report-audit-tests", [sys.executable, "-m", "unittest", "discover", "-s", str(ROOT), "-p", "test_*.py"])
    run("build-release", ["cargo", "build", *common, "--release", "-p", "fy-cpu-evaluate"])
    binary = ROOT / "target" / "release" / ("fy-cpu-evaluate.exe" if os.name == "nt" else "fy-cpu-evaluate")
    # The benchmark compares only training batch preparation, not end-to-end training.
    bench_binary = ROOT / "target" / "release" / ("fy-batch-benchmark.exe" if os.name == "nt" else "fy-batch-benchmark")
    batch_report = json.loads(run("batch-benchmark", [bench_binary, "--measure"]))
    if (batch_report["source_commit"] != env["FACTSYOLO_BUILD_ID"]
            or batch_report["equivalent_steps"] != 2000
            or not batch_report["bitwise_batch_and_rng_parity"]
            or len(batch_report["historical"]) != 7 or len(batch_report["reused"]) != 7
            or any(h["allocation_requests"] <= r["allocation_requests"]
                   for h, r in zip(batch_report["historical"], batch_report["reused"]))):
        raise RuntimeError("batch preparation benchmark parity/provenance/allocation gate failed")
    (evidence / "batch-benchmark.json").write_text(json.dumps(batch_report, indent=2) + "\n")
    # Compare the exact pre-#33 compact inference call with the current path,
    # on the same seeded RGB frames, model, thread and host. Separate from the
    # full Burn/Candle training budget and native C/C# execution.
    inference_binary = ROOT / "target" / "release" / ("fy-inference-benchmark.exe" if os.name == "nt" else "fy-inference-benchmark")
    inference_report = json.loads(run("inference-benchmark", [inference_binary, "--measure"]))
    verify_inference_benchmark(inference_report, env["FACTSYOLO_BUILD_ID"])
    (evidence / "inference-benchmark.json").write_text(json.dumps(inference_report, indent=2) + "\n")
    reports = {}
    for backend in ["burn", "candle"]:
        output = evidence / backend
        run(f"train-{backend}", [binary, "train", backend, output])
        report = json.loads((output / "metrics.json").read_text())
        checked_metrics = checked_training_heldout(report, env["FACTSYOLO_BUILD_ID"])
        (evidence / f"test-metrics-audit-{backend}.json").write_text(json.dumps(checked_metrics, indent=2) + "\n")
        reports[backend] = report
        for i in range(6):
            result = json.loads(run(f"infer-{backend}-{i}", [binary, "infer", output / "model.fytd", output / f"sample-{i}.rgb"]))
            if result != report["predictions"][i]["prediction"]:
                raise RuntimeError(f"fresh process inference disagrees: {backend}/{i}")
        corrupted = bytearray((output / "model.fytd").read_bytes())
        corrupted[-1] ^= 1
        bad = output / "corrupt.fytd"
        bad.write_bytes(corrupted)
        run(f"reject-corrupt-{backend}", [binary, "infer", bad, output / "sample-1.rgb"], expected_success=False)
        bad.write_bytes(b"\x00" * 65537)
        run(f"reject-oversize-{backend}", [binary, "infer", bad, output / "sample-1.rgb"], expected_success=False)
        bad.unlink()
    require_clean_source_tree(REPO, env["FACTSYOLO_BUILD_ID"])
    summary = {"passed": True, "source_commit": env["FACTSYOLO_BUILD_ID"], "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"],cwd=REPO,text=True).strip()), "platform": platform.platform(),
               "rayon_threads": 1, "cpu_count": os.cpu_count(), "results": {k: {f:v for f,v in r.items() if f!="predictions"} for k,r in reports.items()}}
    (evidence / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)

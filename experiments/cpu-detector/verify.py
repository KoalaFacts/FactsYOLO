#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the additive CPU experiment gates, preserving commands and actual results."""
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import struct
import sys
import time
from inference_gate import verify_inference_benchmark
from verify_report import enforce_heldout_thresholds, verify_test_predictions

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
MANIFEST = str(ROOT / "Cargo.toml")


# Fixed by an independent untrained-generator replay before prospective inference.
PROSPECTIVE_EDGE_FINGERPRINT = "0239bac3429d484db479974c10e5a038cef8829762a7d5533f8cd915e3a6ede1"
PROSPECTIVE_GROUPS = ("small_edge", "small_interior", "larger_edge", "larger_interior", "background")


def checked_prospective_edge_report(report, expected_sha):
    """Fail closed on wrong split/provenance/arithmetic, not on a negative accuracy result."""
    if (report.get("source_commit") != expected_sha
            or report.get("prospective_seed") != 20261101
            or report.get("prospective_candidate_budget") != 4096
            or report.get("prospective_samples") != 240
            or report.get("prospective_fingerprint") != PROSPECTIVE_EDGE_FINGERPRINT
            or report.get("final_holdout_evaluated") is not False):
        raise ValueError("prospective model comparison provenance mismatch")

    arms = []
    for arm in ("baseline_prospective", "edge_sampling_prospective"):
        groups = report.get(arm)
        if not isinstance(groups, dict) or set(groups) != set(PROSPECTIVE_GROUPS):
            raise ValueError(f"missing prospective groups: {arm}")
        for name in PROSPECTIVE_GROUPS:
            row = groups[name]
            if not isinstance(row, dict) or type(row.get("samples")) is not int or row["samples"] != 48:
                raise ValueError(f"invalid prospective sample count: {arm}/{name}")
            if name == "background":
                n = row.get("false_positives")
                if type(n) is not int or not 0 <= n <= 48:
                    raise ValueError(f"invalid background errors: {arm}/{name}")
                continue
            hits = row.get("correct_class_iou50")
            missing = row.get("missing_detections")
            wrong = row.get("wrong_class_detections")
            if (any(type(n) is not int or n < 0 for n in (hits, missing, wrong))
                    or hits + missing + wrong > 48):
                raise ValueError(f"invalid detection counts: {arm}/{name}")
            mean_iou = row.get("mean_iou_including_misses")
            if (type(mean_iou) not in (int, float) or not math.isfinite(mean_iou)
                    or not 0 <= mean_iou <= 1):
                raise ValueError(f"invalid prospective IoU: {arm}/{name}")
            recall = row.get("recall_correct_class_iou50")
            if (type(recall) not in (int, float)
                    or not math.isfinite(recall)
                    or abs(recall - hits / 48) > 1e-12):
                raise ValueError(f"invalid prospective recall: {arm}/{name}")
            # Rust evaluate() sums IoU only for correct-class detections.
            # Each IoU50 hit contributes at least 0.5; non-hits contribute
            # less than 0.5, while missing/wrong-class detections contribute 0.
            # Allow for its f32 accumulation and JSON serialization rounding.
            correct_class_count = 48 - missing - wrong
            min_mean_iou = 0.5 * hits / 48
            max_mean_iou = (hits + 0.5 * (correct_class_count - hits)) / 48
            if mean_iou < min_mean_iou - 1e-6 or mean_iou > max_mean_iou + 1e-6:
                raise ValueError(f"inconsistent prospective IoU: {arm}/{name}")
        arms.append(groups)

    for name in ("small_edge", "small_interior"):
        key = f"prospective_edge_minus_baseline_{name}_iou50_recall"
        reported = report.get(key)
        actual = ((arms[1][name]["correct_class_iou50"]
                   - arms[0][name]["correct_class_iou50"]) / 48)
        if (type(reported) not in (float, int) or not math.isfinite(reported)
                or abs(reported - actual) > 1e-12):
            raise ValueError(f"prospective paired comparison mismatch: {name}")
    return True



def checked_mirrored_edge_report(report, expected_sha):
    """Validate the opt-in third arm without treating observed splits as new evidence."""
    if (report.get("source_commit") != expected_sha
            or report.get("status") != "mirrored-edge-training-experiment-only"
            or report.get("model") != "TD1-v1-192-64-7"
            or report.get("backend") not in ("burn-flex-0.22.0", "candle-cpu-0.11.0")
            or report.get("training_seed") != 101
            or report.get("validation_seed") != 202
            or report.get("initialization_seed") != 404
            or report.get("sampler_seed") != 707
            or report.get("steps_per_arm") != 2000
            or report.get("batch_size") != 64
            or report.get("train_samples") != 4096
            or report.get("edge_pool_count") != 48
            or report.get("edge_override_slots_per_batch") != 4
            or report.get("mirror_slots") != [0, 16, 32, 48]
            or report.get("mirror_orientations") !=
                ["identity", "horizontal", "vertical", "both"]
            or report.get("development_seed") != 20261030
            or report.get("development_samples") != 240
            or report.get("development_fingerprint") !=
                "04739d79ab725a8bef7cf0de7b8d1a8f6c929ed08e9e78b21f7392b81861b9c4"
            or report.get("prospective_seed") != 20261101
            or report.get("prospective_candidate_budget") != 4096
            or report.get("prospective_samples") != 240
            or report.get("prospective_fingerprint") != PROSPECTIVE_EDGE_FINGERPRINT
            or report.get("final_holdout_evaluated") is not False):
        raise ValueError("unverifiable mirrored edge training provenance")

    seconds = report.get("training_seconds")
    if type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("invalid mirrored training duration")

    # Reuse exact per-stratum count/IoU arithmetic checks for both observed
    # development splits; no criterion selects a winner from these splits.
    for section in ("development", "prospective"):
        probe = {
            "source_commit": expected_sha,
            "prospective_seed": 20261101,
            "prospective_candidate_budget": 4096,
            "prospective_samples": 240,
            "prospective_fingerprint": PROSPECTIVE_EDGE_FINGERPRINT,
            "final_holdout_evaluated": False,
            "baseline_prospective": report.get(section),
            "edge_sampling_prospective": report.get(section),
            "prospective_edge_minus_baseline_small_edge_iou50_recall": 0.0,
            "prospective_edge_minus_baseline_small_interior_iou50_recall": 0.0,
        }
        checked_prospective_edge_report(probe, expected_sha)

    validation = report.get("validation")
    if not isinstance(validation, dict) or report.get("validation_passes_original_thresholds") is not True:
        raise ValueError("mirrored arm failed original validation gate")
    if (type(validation.get("positives")) is not int or validation["positives"] != 102
            or type(validation.get("negatives")) is not int or validation["negatives"] != 26):
        raise ValueError("invalid mirrored validation population")
    for key, lower, upper in (
        ("class_accuracy", 0.95, 1.0),
        ("recall_iou50", 0.85, 1.0),
        ("mean_iou", 0.70, 1.0),
        ("false_positive_rate", 0.0, 0.05),
    ):
        value = validation.get(key)
        # Match Metrics::passes f32 constants; JSON Value promotes f32 to f64.
        lower = struct.unpack("<f", struct.pack("<f", lower))[0]
        upper = struct.unpack("<f", struct.pack("<f", upper))[0]
        if type(value) not in (float, int) or not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"invalid mirrored validation metric: {key}")
    return True


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


def require_expected_inference_rejection(returncode, output, expected_error):
    """A crashed or unrelated failed infer is not a successful negative control."""
    if returncode != 1 or output.strip() != f"error: {expected_error}":
        raise RuntimeError(
            f"expected infer rejection {expected_error!r}, got exit {returncode}"
        )


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

    def run(label, arguments, *, expected_error=None):
        print(f"{label}: {' '.join(map(str, arguments))}", flush=True)
        started = time.monotonic()
        with (evidence / f"{label}.log").open("w") as log:
            result = subprocess.run(list(map(str, arguments)), cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        commands.append({"label": label, "argv": list(map(str, arguments)), "exit": result.returncode, "seconds": time.monotonic()-started})
        (evidence / "commands.json").write_text(json.dumps(commands, indent=2)+"\n")
        output = (evidence / f"{label}.log").read_text()
        if expected_error is None:
            if result.returncode != 0:
                raise RuntimeError(f"unexpected exit {result.returncode}: {label}; see {evidence / (label+'.log')}")
        else:
            try:
                require_expected_inference_rejection(result.returncode, output, expected_error)
            except RuntimeError as error:
                raise RuntimeError(f"{label}: {error}; see {evidence / (label+'.log')}") from error
        return output

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
        # Prospective development-only slice validation runs on an independent
        # seed; never use its results to tune the registered final test gate.
        prospective = json.loads(run(f"prospective-validation-{backend}", [
            sys.executable, ROOT / "diagnostic_validation.py", binary,
            output / "model.fytd", output / "metrics.json",
        ]))
        if (prospective.get("source_commit") != env["FACTSYOLO_BUILD_ID"]
                or prospective.get("status") != "development-diagnostic-only"):
            raise RuntimeError("unverifiable prospective validation provenance")
        (evidence / f"prospective-validation-{backend}.json").write_text(
            json.dumps(prospective, indent=2) + "\n")
        # Frozen geometry-only prospective split, separate from the original
        # final holdout and the earlier unstratified development diagnostic.
        edge_report = json.loads(run(f"edge-validation-{backend}", [
            sys.executable, ROOT / "edge_validation.py", binary,
            output / "model.fytd", output / "metrics.json",
        ]))
        if (edge_report.get("source_commit") != env["FACTSYOLO_BUILD_ID"]
                or edge_report.get("status") != "development-diagnostic-only"
                or edge_report.get("selected_count") != 240
                or edge_report.get("selected_fingerprint") !=
                    "04739d79ab725a8bef7cf0de7b8d1a8f6c929ed08e9e78b21f7392b81861b9c4"):
            raise RuntimeError("unverifiable prospective edge diagnostics")
        (evidence / f"edge-validation-{backend}.json").write_text(
            json.dumps(edge_report, indent=2) + "\n")
        for i in range(6):
            result = json.loads(run(f"infer-{backend}-{i}", [binary, "infer", output / "model.fytd", output / f"sample-{i}.rgb"]))
            if result != report["predictions"][i]["prediction"]:
                raise RuntimeError(f"fresh process inference disagrees: {backend}/{i}")
        corrupted = bytearray((output / "model.fytd").read_bytes())
        corrupted[-1] ^= 1
        bad = output / "corrupt.fytd"
        bad.write_bytes(corrupted)
        run(f"reject-corrupt-{backend}", [binary, "infer", bad, output / "sample-1.rgb"], expected_error="artifact checksum mismatch")
        bad.write_bytes(b"\x00" * 65537)
        run(f"reject-oversize-{backend}", [binary, "infer", bad, output / "sample-1.rgb"], expected_error="input exceeds bounded file size")
        bad.unlink()
    # Controlled model-training intervention: compare the original sampler
    # against a preregistered small-edge sampling distribution. This does not
    # read the registered final holdout, produce weights, or replace the
    # original seed909 acceptance training above.
    edge_binary = ROOT / "target" / "release" / (
        "fy-edge-training.exe" if os.name == "nt" else "fy-edge-training")
    for backend_name in ("burn", "candle"):
        report = json.loads(run(f"edge-training-{backend_name}", [edge_binary, backend_name]))
        if (report.get("status") != "controlled-training-experiment-only"
                or report.get("source_commit") != env["FACTSYOLO_BUILD_ID"]
                or report.get("training_seed") != 101
                or report.get("validation_seed") != 202
                or report.get("steps_per_arm") != 2000
                or report.get("batch_size") != 64
                or report.get("edge_override_slots_per_batch") != 4
                or report.get("development_samples") != 240
                or report.get("development_fingerprint") !=
                    "04739d79ab725a8bef7cf0de7b8d1a8f6c929ed08e9e78b21f7392b81861b9c4"
                or report.get("final_holdout_evaluated") is not False
                or report.get("baseline", {}).get("validation_passes_original_thresholds") is not True
                or report.get("edge_pool_count") != 48):
            raise RuntimeError(f"unverifiable controlled edge-training comparison: {backend_name}")
        # Both baselines must reproduce the *existing* training command's
        # validation result on this same backend and exact source commit.
        checked_prospective_edge_report(report, env["FACTSYOLO_BUILD_ID"])
        reference = reports[backend_name]["validation"]
        paired = report["baseline"]["validation"]
        for key in ("positives", "negatives", "class_accuracy", "recall_iou50",
                    "mean_iou", "false_positive_rate"):
            if abs(reference[key] - paired[key]) > 1e-5:
                raise RuntimeError(
                    f"existing-training parity failure: {backend_name}/{key}: "
                    f"{reference[key]} != {paired[key]}")
        (evidence / f"edge-training-{backend_name}.json").write_text(
            json.dumps(report, indent=2) + "\n")
        # Real third-arm Burn/Candle training, with the same sampled edge
        # identities and four fixed slot reflections. Existing two-arm
        # prospective results remain unchanged; both observed splits here
        # are exploratory diagnostics, NOT a new independent quality gate.
        mirrored = json.loads(run(f"mirror-training-{backend_name}", [
            edge_binary, backend_name, "--mirror"]))
        checked_mirrored_edge_report(mirrored, env["FACTSYOLO_BUILD_ID"])
        if mirrored["backend"] != {"burn": "burn-flex-0.22.0", "candle": "candle-cpu-0.11.0"}[backend_name]:
            raise RuntimeError(f"mirrored backend mismatch: {backend_name}")
        (evidence / f"mirror-training-{backend_name}.json").write_text(
            json.dumps(mirrored, indent=2) + "\n")

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

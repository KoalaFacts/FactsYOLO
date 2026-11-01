#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the additive CPU experiment gates, preserving commands and actual results."""
import hashlib
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
from full_training_gate import (validate_full_training_benchmark,
                                validate_full_training_diagnostics,
                                validate_interrupted_full_training_diagnostics)
from verify_report import enforce_heldout_thresholds, verify_test_predictions
from verify_native import NATIVE_INDICES
from cross_generator_validation import check_report as check_cross_generator_report
from blind_center_comparison import check_report as check_blind_center_report

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
MANIFEST = str(ROOT / "Cargo.toml")


# Fixed by an independent untrained-generator replay before prospective inference.
PROSPECTIVE_EDGE_FINGERPRINT = "0239bac3429d484db479974c10e5a038cef8829762a7d5533f8cd915e3a6ede1"
PROSPECTIVE_GROUPS = ("small_edge", "small_interior", "larger_edge", "larger_interior", "background")


# Independently frozen from the untrained TD1 generator before any backend prediction.
FOUR_SIDE_FINGERPRINT = "59a4a885ff0ca2c5c8e700f8ccaaefbdaf454f2f2d0e6a72d280026b8398b3bd"
FOUR_SIDE_NAMES = ("left", "top", "right", "bottom")


def checked_four_side_report(report, expected_sha, arms):
    """Audit paired synthetic four-side evidence, never infer independent samples."""
    if (report.get("source_commit") != expected_sha
            or report.get("four_side_generator") != "TD1-four-side-paired-v1"
            or report.get("four_side_seed") != 20261113
            or report.get("four_side_candidate_budget") != 4096
            or report.get("four_side_source_rows") != 48
            or report.get("four_side_transformed_samples") != 192
            or report.get("four_side_fingerprint") != FOUR_SIDE_FINGERPRINT
            or report.get("final_holdout_evaluated") is not False):
        raise ValueError("four-side provenance mismatch")

    for arm in arms:
        groups = report.get(arm)
        if not isinstance(groups, dict) or set(groups) != set(FOUR_SIDE_NAMES):
            raise ValueError(f"missing four-side groups: {arm}")
        for name in FOUR_SIDE_NAMES:
            row = groups[name]
            if not isinstance(row, dict) or type(row.get("samples")) is not int or row["samples"] != 48:
                raise ValueError(f"invalid four-side group count: {arm}/{name}")
            hits = row.get("correct_class_iou50")
            missing = row.get("missing_detections")
            wrong = row.get("wrong_class_detections")
            if (any(type(n) is not int or n < 0 for n in (hits, missing, wrong))
                    or hits + missing + wrong > 48):
                raise ValueError(f"invalid four-side detection counts: {arm}/{name}")
            recall = row.get("recall_correct_class_iou50")
            if (type(recall) not in (int, float) or not math.isfinite(recall)
                    or abs(recall - hits / 48) > 1e-12):
                raise ValueError(f"inconsistent four-side recall: {arm}/{name}")
            iou = row.get("mean_iou_including_misses")
            correct_class = 48 - missing - wrong
            minimum = 0.5 * hits / 48
            maximum = (hits + 0.5 * (correct_class - hits)) / 48
            if (type(iou) not in (int, float) or not math.isfinite(iou)
                    or iou < minimum - 1e-6 or iou > maximum + 1e-6
                    or iou < 0 or iou > 1):
                raise ValueError(f"inconsistent four-side IoU: {arm}/{name}")
    return True


# Preregistered while the new seed's predictions were still unobserved.
DISJOINT_FINGERPRINT = "e156306394733b94c5d99daec4ae22647819cab6976bc3dd55938eb988500a56"
DISJOINT_NAMES = ("left", "top", "right", "bottom", "background")


def checked_disjoint_four_side_report(report, expected_sha, arms):
    """Check one-source-per-side synthetic evidence, including the background."""
    if (report.get("source_commit") != expected_sha
            or report.get("disjoint_generator") != "TD1-four-side-disjoint-v1"
            or report.get("disjoint_seed") != 20261123
            or report.get("disjoint_candidate_budget") != 8192
            or report.get("disjoint_source_rows") != 144
            or report.get("disjoint_last_source_index") != 6876
            or report.get("disjoint_fingerprint") != DISJOINT_FINGERPRINT
            or report.get("final_holdout_evaluated") is not False):
        raise ValueError("disjoint four-side provenance mismatch")
    for arm in arms:
        groups = report.get(arm)
        if not isinstance(groups, dict) or set(groups) != set(DISJOINT_NAMES):
            raise ValueError(f"missing disjoint four-side groups: {arm}")
        for name in DISJOINT_NAMES:
            row = groups[name]
            quota = 48 if name == "background" else 24
            if not isinstance(row, dict) or type(row.get("samples")) is not int or row["samples"] != quota:
                raise ValueError(f"invalid disjoint group population: {arm}/{name}")
            if name == "background":
                errors = row.get("false_positives")
                if type(errors) is not int or not 0 <= errors <= quota:
                    raise ValueError(f"invalid disjoint background errors: {arm}")
                continue
            hits = row.get("correct_class_iou50")
            missing = row.get("missing_detections")
            wrong = row.get("wrong_class_detections")
            if (any(type(n) is not int or n < 0 for n in (hits, missing, wrong))
                    or hits + missing + wrong > quota):
                raise ValueError(f"invalid disjoint detection counts: {arm}/{name}")
            recall = row.get("recall_correct_class_iou50")
            if (type(recall) not in (int, float) or not math.isfinite(recall)
                    or abs(recall - hits / quota) > 1e-12):
                raise ValueError(f"inconsistent disjoint recall: {arm}/{name}")
            mean = row.get("mean_iou_including_misses")
            correct = quota - missing - wrong
            minimum = 0.5 * hits / quota
            maximum = (hits + 0.5 * (correct - hits)) / quota
            if (type(mean) not in (int, float) or not math.isfinite(mean)
                    or not 0 <= mean <= 1 or mean < minimum - 1e-6
                    or mean > maximum + 1e-6):
                raise ValueError(f"inconsistent disjoint IoU: {arm}/{name}")
    return True


def assess_disjoint_mirror_candidate(baseline, mirrored):
    """Apply the fixed directional rule; a negative result is valid evidence."""
    if (baseline.get("source_commit") != mirrored.get("source_commit")
            or baseline.get("backend") != mirrored.get("backend")
            or baseline.get("disjoint_fingerprint") != mirrored.get("disjoint_fingerprint")):
        raise ValueError("disjoint candidate arms have incompatible provenance")
    base = baseline["baseline_disjoint_four_side"]
    mirror = mirrored["mirrored_disjoint_four_side"]
    deltas = {side: (mirror[side]["correct_class_iou50"] -
                     base[side]["correct_class_iou50"]) for side in DISJOINT_NAMES[:4]}
    baseline_fp = base["background"]["false_positives"]
    mirror_fp = mirror["background"]["false_positives"]
    original_ok = baseline["baseline"]["validation_passes_original_thresholds"] is True
    mirrored_ok = mirrored["validation_passes_original_thresholds"] is True
    qualifies = (deltas["right"] >= 8 and deltas["bottom"] >= 8
                 and deltas["left"] >= -2 and deltas["top"] >= -2
                 and mirror_fp <= baseline_fp and original_ok and mirrored_ok)
    return {
        "status": "synthetic-research-only",
        "qualified_for_further_synthetic_research": qualifies,
        "correct_class_iou50_hit_delta_mirror_minus_baseline": deltas,
        "background_false_positives": {"baseline": baseline_fp, "mirrored": mirror_fp},
        "baseline_validation_passes_original_thresholds": original_ok,
        "mirrored_validation_passes_original_thresholds": mirrored_ok,
        "source_commit": baseline["source_commit"],
        "backend": baseline["backend"],
        "seed": 20261123,
        "source_rows": 144,
        "fingerprint": DISJOINT_FINGERPRINT,
        "final_holdout_evaluated": False,
    }


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



def checked_mirrored_edge_report(report, expected_sha, *, textured=False,
                                 center_weighted=False):
    """Validate opt-in mirror, texture and center arms on observed diagnostics."""
    if textured and center_weighted:
        raise ValueError("training arms must change only one variable")
    expected_status = (
        "textured-mirror-training-experiment-only" if textured
        else ("center-weighted-mirror-training-experiment-only" if center_weighted
              else "mirrored-edge-training-experiment-only"))
    multiplier = report.get("center_gradient_multiplier")
    if (type(multiplier) not in (float, int)
            or multiplier != (2.0 if center_weighted else 1.0)):
        raise ValueError("invalid center gradient intervention metadata")
    if (report.get("source_commit") != expected_sha
            or report.get("status") != expected_status
            or report.get("feature_illumination") is not textured
            or report.get("illumination_phases") !=
                ["identity", "checkerboard", "horizontal-gradient", "vertical-gradient"]
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

    def run(label, arguments, *, expected_error=None, diagnostic_stderr=False):
        print(f"{label}: {' '.join(map(str, arguments))}", flush=True)
        started = time.monotonic()
        with (evidence / f"{label}.log").open("w") as log:
            if diagnostic_stderr:
                # Keep the successful stdout JSON intact; each stderr JSONL
                # progress line is flushed and retained even if a later arm fails.
                with (evidence / f"{label}.diagnostics.jsonl").open("w") as trace:
                    result = subprocess.run(
                        list(map(str, arguments)), cwd=REPO, env=env,
                        stdout=log, stderr=trace, timeout=1800)
            else:
                result = subprocess.run(
                    list(map(str, arguments)), cwd=REPO, env=env,
                    stdout=log, stderr=subprocess.STDOUT, timeout=1800)
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
    # An independent stdlib LCG skip-ahead replay reproduces the registered
    # disjoint sample identity before any candidate's trained predictions run.
    freeze = json.loads(run("disjoint-source-freeze", [
        sys.executable, ROOT / "tools" / "freeze_disjoint_four_sides.py"]))
    if (freeze.get("fingerprint") != DISJOINT_FINGERPRINT
            or freeze.get("source_rows") != 144
            or freeze.get("last_selected_index") != 6876
            or freeze.get("unique_pixel_arrays") != 144
            or freeze.get("selected") != {
                "left": 24, "top": 24, "right": 24,
                "bottom": 24, "background": 48}):
        raise RuntimeError("independent disjoint TD1 source replay failed")
    (evidence / "disjoint-source-freeze.json").write_text(
        json.dumps(freeze, indent=2) + "\n")
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
    # End-to-end counterpart of the #27 batch-only benchmark: all 2,000
    # actual Burn/Candle forward/loss/backward/Adam steps, same seed, initial
    # model and sampler. Alternate historical allocation and buffer reuse;
    # assert identical final weights/RNG/loss/validation. Timings and allocator
    # counts are evidence, never a hard-coded speedup acceptance threshold.
    full_training_binary = ROOT / "target" / "release" / (
        "fy-full-training-benchmark.exe" if os.name == "nt"
        else "fy-full-training-benchmark")
    for backend_name, expected_name in (
            ("burn", "burn-flex-0.22.0"),
            ("candle", "candle-cpu-0.11.0")):

        # Exercise a real Release Rust process failure *after* one full 2,000-step
        # arm. The test-only fault is outside the measured optimizer loop.
        # Separate stderr must preserve completed progress; stdout must contain
        # no apparently successful report, and the process must fail closed.
        interrupted = f"full-training-interrupted-{backend_name}"
        interrupted_args = [str(full_training_binary), backend_name, "--measure"]
        fault_env = env.copy()
        fault_env["FACTSYOLO_TEST_FAIL_SECOND_ARM"] = "1"
        interrupted_started = time.monotonic()
        with (evidence / f"{interrupted}.log").open("w") as stdout:
            with (evidence / f"{interrupted}.diagnostics.jsonl").open("w") as stderr:
                failure = subprocess.run(
                    interrupted_args, cwd=REPO, env=fault_env,
                    stdout=stdout, stderr=stderr, timeout=1800)
        commands.append({
            "label": interrupted, "argv": interrupted_args, "exit": failure.returncode,
            "seconds": time.monotonic() - interrupted_started,
            "test_only_env": {"FACTSYOLO_TEST_FAIL_SECOND_ARM": "1"},
        })
        (evidence / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if failure.returncode != 1 or (evidence / f"{interrupted}.log").read_text().strip():
            raise RuntimeError(f"controlled Rust failure did not fail closed: {interrupted}")
        validate_interrupted_full_training_diagnostics(
            (evidence / f"{interrupted}.diagnostics.jsonl").read_text(),
            env["FACTSYOLO_BUILD_ID"], expected_name)

        full_training = json.loads(run(f"full-training-benchmark-{backend_name}", [
            full_training_binary, backend_name, "--measure"], diagnostic_stderr=True))
        validate_full_training_benchmark(
            full_training, env["FACTSYOLO_BUILD_ID"], expected_name)
        validate_full_training_diagnostics(
            full_training,
            (evidence / f"full-training-benchmark-{backend_name}.diagnostics.jsonl").read_text())
        (evidence / f"full-training-benchmark-{backend_name}.json").write_text(
            json.dumps(full_training, indent=2) + "\n")

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
        # Independent xorshift/patterned-background source distribution; no
        # training change, final holdout reuse or post-hoc quality threshold.
        cross_report = json.loads(run(f"cross-generator-{backend}", [
            sys.executable, ROOT / "cross_generator_validation.py", binary,
            output / "model.fytd", output / "metrics.json",
        ]))
        check_cross_generator_report(
            cross_report, env["FACTSYOLO_BUILD_ID"], report["backend"])
        (evidence / f"cross-generator-{backend}.json").write_text(
            json.dumps(cross_report, indent=2) + "\n")
        for i in NATIVE_INDICES:
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
        checked_four_side_report(report, env["FACTSYOLO_BUILD_ID"],
                                 ("baseline_four_side", "edge_sampling_four_side"))
        checked_disjoint_four_side_report(report, env["FACTSYOLO_BUILD_ID"],
                                          ("baseline_disjoint_four_side",
                                           "edge_sampling_disjoint_four_side"))
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
        mirror_artifact = evidence / f"mirror-model-{backend_name}.fytd"
        mirrored = json.loads(run(f"mirror-training-{backend_name}", [
            edge_binary, backend_name, "--mirror", "--evidence-artifact", mirror_artifact]))
        checked_mirrored_edge_report(mirrored, env["FACTSYOLO_BUILD_ID"])
        if mirrored.get("artifact_sha256") != hashlib.sha256(mirror_artifact.read_bytes()).hexdigest():
            raise RuntimeError(f"mirrored model evidence digest mismatch: {backend_name}")
        checked_four_side_report(mirrored, env["FACTSYOLO_BUILD_ID"],
                                 ("mirrored_four_side",))
        checked_disjoint_four_side_report(mirrored, env["FACTSYOLO_BUILD_ID"],
                                          ("mirrored_disjoint_four_side",))
        if mirrored["backend"] != {"burn": "burn-flex-0.22.0", "candle": "candle-cpu-0.11.0"}[backend_name]:
            raise RuntimeError(f"mirrored backend mismatch: {backend_name}")
        (evidence / f"mirror-training-{backend_name}.json").write_text(
            json.dumps(mirrored, indent=2) + "\n")
        # Fourth, opt-in training arm: same mirrored edge draws, initialized
        # weights, Adam, budget and labels; only RGB block-mean illumination
        # differs. These already-observed geometry probes are development
        # diagnostics, NOT a fresh cross-generator generalization result.
        textured = json.loads(run(f"texture-training-{backend_name}", [
            edge_binary, backend_name, "--texture"]))
        checked_mirrored_edge_report(
            textured, env["FACTSYOLO_BUILD_ID"], textured=True)
        checked_four_side_report(textured, env["FACTSYOLO_BUILD_ID"],
                                 ("mirrored_four_side",))
        checked_disjoint_four_side_report(textured, env["FACTSYOLO_BUILD_ID"],
                                          ("mirrored_disjoint_four_side",))
        if (textured["backend"] != mirrored["backend"]
                or textured["mirror_slots"] != mirrored["mirror_slots"]
                or textured["sampler_seed"] != mirrored["sampler_seed"]
                or textured["steps_per_arm"] != mirrored["steps_per_arm"]
                or textured["train_samples"] != mirrored["train_samples"]):
            raise RuntimeError(f"texture/mirror training parity failure: {backend_name}")
        (evidence / f"texture-training-{backend_name}.json").write_text(
            json.dumps(textured, indent=2) + "\n")
        # Fifth, *opt-in* center-loss arm. Same mirror source identities, Adam
        # budget, initialization and unmodified image features; only center
        # x/y gradient on positives is doubled. All scores here are already-
        # observed diagnostics, not a new cross-generator holdout.
        center_artifact = evidence / f"center-model-{backend_name}.fytd"
        center = json.loads(run(f"center-training-{backend_name}", [
            edge_binary, backend_name, "--center", "--evidence-artifact", center_artifact]))
        checked_mirrored_edge_report(
            center, env["FACTSYOLO_BUILD_ID"], center_weighted=True)
        if center.get("artifact_sha256") != hashlib.sha256(center_artifact.read_bytes()).hexdigest():
            raise RuntimeError(f"center model evidence digest mismatch: {backend_name}")
        checked_four_side_report(center, env["FACTSYOLO_BUILD_ID"],
                                 ("mirrored_four_side",))
        checked_disjoint_four_side_report(center, env["FACTSYOLO_BUILD_ID"],
                                          ("mirrored_disjoint_four_side",))
        if any(center[key] != mirrored[key] for key in (
                "backend", "mirror_slots", "sampler_seed", "steps_per_arm",
                "train_samples", "edge_override_slots_per_batch",
                "development_fingerprint", "prospective_fingerprint",
                "four_side_fingerprint", "disjoint_fingerprint")):
            raise RuntimeError(f"center/mirror training parity failure: {backend_name}")
        (evidence / f"center-training-{backend_name}.json").write_text(
            json.dumps(center, indent=2) + "\n")
        # This cohort was source-frozen before reading either model's predictions.
        # Both arms must use independently reloaded real trained artifacts,
        # identical source/order/model architecture and the same test images.
        # The result is an experimental synthetic diagnostic, not a new
        # production accuracy claim or a training-treatment selection rule.
        blind = json.loads(run(f"blind-center-{backend_name}", [
            sys.executable, ROOT / "blind_center_comparison.py", binary,
            mirror_artifact, center_artifact,
            evidence / f"mirror-training-{backend_name}.json",
            evidence / f"center-training-{backend_name}.json",
        ]))
        check_blind_center_report(
            blind, env["FACTSYOLO_BUILD_ID"], mirrored["backend"],
            mirrored["artifact_sha256"], center["artifact_sha256"])
        (evidence / f"blind-center-{backend_name}.json").write_text(
            json.dumps(blind, indent=2) + "\n")
        # Record disjoint-source selection outcome even when the mirror loses.
        # Across-host qualification requires examining both OS artifact reports.
        assessment = assess_disjoint_mirror_candidate(report, mirrored)
        (evidence / f"disjoint-four-side-{backend_name}.json").write_text(
            json.dumps(assessment, indent=2) + "\n")

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

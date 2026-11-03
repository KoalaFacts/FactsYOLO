# SPDX-License-Identifier: Apache-2.0
"""Validate paired live allocator evidence without confusing it with RSS."""
import re

SOURCE_BYTES = 4096 * 64 * 64 * 3
SCOPE = "feature caching and source buffer residency only; excludes training"
ALLOCATOR_METRIC = "System allocator live requested payload bytes, not RSS"


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def validate_cache_report(report, source_commit):
    """Only claim the freed source payload after checking matched sample identity."""
    if (not isinstance(report, dict)
            or not re.fullmatch(r"[0-9a-f]{40}", source_commit)
            or report.get("schema") != "fy-td1-source-cache-allocation-v1"
            or report.get("source_commit") != source_commit
            or report.get("build") != "release"
            or report.get("scope") != SCOPE
            or report.get("allocator_metric") != ALLOCATOR_METRIC
            or report.get("model_training_executed") is not False
            or type(report.get("training_seed")) is not int
            or report["training_seed"] != 101
            or type(report.get("training_samples")) is not int
            or report["training_samples"] != 4096
            or type(report.get("features_per_sample")) is not int
            or report["features_per_sample"] != 192
            or type(report.get("repetitions")) is not int
            or report["repetitions"] != 3):
        raise ValueError("unverified source-cache measurement scope/provenance")

    runs = report.get("runs")
    if not isinstance(runs, list) or len(runs) != 3:
        raise ValueError("missing balanced source-cache comparisons")

    fingerprint = None
    for index, pair in enumerate(runs):
        if (not isinstance(pair, dict)
                or set(pair) != {"first", "retained", "released"}
                or pair["first"] != ("retained" if index % 2 == 0 else "released")):
            raise ValueError("source-cache comparison order drift")
        arms = []
        for name in ("retained", "released"):
            arm = pair[name]
            if (not isinstance(arm, dict)
                    or set(arm) != {
                        "elapsed_ns", "source_capacity_bytes",
                        "live_after_cache_bytes",
                        "peak_live_during_preparation_bytes",
                        "feature_label_sha256"}):
                raise ValueError("incomplete source-cache arm")
            if (not _integer(arm["elapsed_ns"], 1)
                    or not _integer(arm["source_capacity_bytes"], SOURCE_BYTES)
                    or arm["source_capacity_bytes"] != SOURCE_BYTES
                    or not _integer(arm["live_after_cache_bytes"], 1)
                    or not _integer(arm["peak_live_during_preparation_bytes"],
                                    arm["live_after_cache_bytes"])
                    or not isinstance(arm["feature_label_sha256"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", arm["feature_label_sha256"])):
                raise ValueError("impossible source-cache allocator measurement")
            if fingerprint is None:
                fingerprint = arm["feature_label_sha256"]
            elif arm["feature_label_sha256"] != fingerprint:
                raise ValueError("feature/label identity drift between cached source modes")
            arms.append(arm)

        retained, released = arms
        if (retained["live_after_cache_bytes"] - released["live_after_cache_bytes"]
                != SOURCE_BYTES):
            raise ValueError("live source allocation difference not confirmed")
        # A release *after* feature extraction cannot reduce the earlier peak
        # during feature preparation; do not infer peak RSS or full training speed.
    return True

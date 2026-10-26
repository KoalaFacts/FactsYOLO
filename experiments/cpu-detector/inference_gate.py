# SPDX-License-Identifier: Apache-2.0
"""Independently check the fixed-input TD1 portable inference evidence."""

# Registered fingerprints from the seven-repetition baseline on both CI hosts.
# A changed input or model must receive a consciously updated comparison gate.
_TD1_INPUTS_SHA256 = "3fc69efa48a1b4d9a7f596476c0c405295dc96355c9c587b4058bce20aebcca4"
_TD1_MODEL_SHA256 = "4d3ce2b2a284dd2644f91ae4b73a12c486d6a12d9fd1cecfe4c9ce8fd5b230bd"


def verify_inference_benchmark(report, expected_sha):
    """Reject stale, incomplete, or incomparable benchmark measurements.

    This gate checks provenance, identity, parity and allocator accounting only.
    It does not assert a latency improvement or production resource envelope.
    """
    if (not isinstance(report, dict)
            or report.get("source_commit") != expected_sha
            or report.get("build") != "release"
            or report.get("images") != 8
            or report.get("iterations_per_repeat") != 256
            or report.get("repetitions") != 7
            or report.get("inputs_sha256") != _TD1_INPUTS_SHA256
            or report.get("model_sha256") != _TD1_MODEL_SHA256
            or report.get("exact_positive_and_background_parity") is not True):
        raise ValueError("portable TD1 inference benchmark provenance/parity gate failed")

    for mode, expected_requests, expected_bytes in (
            ("historical", 2_048, 2_048 * 192 * 4), ("current", 0, 0)):
        samples = report.get(mode)
        if not isinstance(samples, list) or len(samples) != 7:
            raise ValueError(f"portable TD1 inference benchmark {mode} repetitions missing")
        for index, sample in enumerate(samples):
            if (not isinstance(sample, dict)
                    or type(sample.get("elapsed_ns")) is not int
                    or sample["elapsed_ns"] <= 0
                    or type(sample.get("allocation_requests")) is not int
                    or sample["allocation_requests"] != expected_requests
                    or type(sample.get("requested_bytes")) is not int
                    or sample["requested_bytes"] != expected_bytes):
                raise ValueError(f"portable TD1 inference benchmark {mode} sample {index} invalid")
    return report

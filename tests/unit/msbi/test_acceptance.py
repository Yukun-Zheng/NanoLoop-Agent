from __future__ import annotations

from app.msbi.acceptance import (
    build_superiority_policy,
    build_tolerance_policy,
    evaluate_candidate,
)


def _baselines() -> dict:
    return {
        "small": {
            "metrics_macro": {
                "instance_f1_50": 0.70,
                "count_absolute_error": 10.0,
                "pixel_dice": 0.80,
                "boundary_f1": 0.60,
                "panoptic_quality": 0.55,
            },
            "runtime_ms": {"mean": 900.0, "p95": 1000.0},
        },
        "large": {
            "metrics_macro": {
                "instance_f1_50": 0.60,
                "count_absolute_error": 8.0,
                "pixel_dice": 0.75,
                "boundary_f1": 0.55,
                "panoptic_quality": 0.50,
            },
            "runtime_ms": {"mean": 700.0, "p95": 800.0},
        },
    }


def test_acceptance_policy_is_pareto_and_fail_closed() -> None:
    policy = build_tolerance_policy(
        _baselines(),
        baseline_results_sha256="baseline",
        split_manifest_sha256="split",
    )
    assert policy["reference_model_id"] == "small"
    candidate = {
        "model_id": "candidate",
        "metrics_macro": {
            "instance_f1_50": 0.71,
            "count_absolute_error": 9.0,
            "pixel_dice": 0.79,
            "boundary_f1": 0.59,
            "runtime_ms": 900.0,
            "mean_small_gate": 0.4,
            "mean_large_gate": 0.6,
        },
    }
    result = evaluate_candidate(
        policy,
        candidate,
        policy_sha256="policy",
        candidate_results_sha256="candidate",
    )
    assert result["status"] == "FAILED"
    assert result["independent_test_status"] == "SEALED_NOT_ACCESSED"


def test_strict_superiority_policy_uses_best_baseline_per_metric() -> None:
    policy = build_superiority_policy(
        _baselines(),
        baseline_results_sha256="baseline",
        split_manifest_sha256="split",
    )
    candidate = {
        "model_id": "candidate",
        "metrics_macro": {
            "instance_f1_50": 0.71,
            "count_absolute_error": 7.9,
            "pixel_dice": 0.81,
            "boundary_f1": 0.61,
            "panoptic_quality": 0.56,
            "runtime_ms": 699.0,
            "runtime_p95_ms": 799.0,
            "mean_small_gate": 0.4,
            "mean_large_gate": 0.6,
        },
    }
    result = evaluate_candidate(
        policy,
        candidate,
        policy_sha256="policy",
        candidate_results_sha256="candidate",
    )
    assert result["status"] == "PASSED"
    assert result["independent_test_status"] == "AUTHORIZED_BY_VALIDATION_GATE"

    candidate["metrics_macro"]["boundary_f1"] = 0.60
    failed = evaluate_candidate(
        policy,
        candidate,
        policy_sha256="policy",
        candidate_results_sha256="candidate",
    )
    assert failed["status"] == "FAILED"
    assert failed["independent_test_status"] == "SEALED_NOT_ACCESSED"

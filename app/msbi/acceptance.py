"""Fail-closed validation acceptance policy for MSBI candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_tolerance_policy(
    baseline_results: dict[str, Any],
    *,
    baseline_results_sha256: str,
    split_manifest_sha256: str,
) -> dict[str, Any]:
    """Freeze a Pareto policy relative to the best instance-F1 baseline."""
    reference_id, reference = max(
        baseline_results.items(),
        key=lambda item: float(item[1]["metrics_macro"]["instance_f1_50"]),
    )
    metrics = reference["metrics_macro"]
    return {
        "schema_version": 1,
        "policy_id": "msbi-validation-pareto-v1",
        "scope": "validation_only_before_independent_test",
        "reference_model_id": reference_id,
        "baseline_results_sha256": baseline_results_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "primary_rules": [
            {
                "metric": "instance_f1_50",
                "operator": ">=",
                "threshold": float(metrics["instance_f1_50"]) + 0.02,
                "basis": "best baseline plus 0.02 absolute",
            },
            {
                "metric": "count_absolute_error",
                "operator": "<=",
                "threshold": float(metrics["count_absolute_error"]) * 0.95,
                "basis": "at least five percent below the reference baseline",
            },
        ],
        "guardrail_rules": [
            {
                "metric": "pixel_dice",
                "operator": ">=",
                "threshold": float(metrics["pixel_dice"]) - 0.02,
                "basis": "no more than 0.02 absolute regression",
            },
            {
                "metric": "boundary_f1",
                "operator": ">=",
                "threshold": float(metrics["boundary_f1"]) - 0.02,
                "basis": "no more than 0.02 absolute regression",
            },
            {
                "metric": "runtime_ms",
                "operator": "<=",
                "threshold": float(reference["runtime_ms"]["p95"]) * 3.0,
                "basis": "mean full-image runtime at most three times baseline p95",
            },
            {
                "metric": "mean_small_gate",
                "operator": ">=",
                "threshold": 0.10,
                "basis": "validation-average anti-collapse floor",
            },
            {
                "metric": "mean_large_gate",
                "operator": ">=",
                "threshold": 0.10,
                "basis": "validation-average anti-collapse floor",
            },
        ],
        "pass_rule": "all primary and guardrail rules must pass",
        "tie_break": [
            "higher instance_f1_50",
            "lower count_absolute_error",
            "lower diameter_wasserstein_px",
            "lower runtime_ms",
        ],
        "independent_test_action": (
            "independent test pixels may be accessed only when the validation "
            "acceptance gate status is PASSED"
        ),
        "immutable_after_independent_test_access": True,
    }


def build_superiority_policy(
    baseline_results: dict[str, Any],
    *,
    baseline_results_sha256: str,
    split_manifest_sha256: str,
) -> dict[str, Any]:
    """Freeze strict effect-and-runtime rules against the best baseline per metric."""

    def best_rule(
        metric: str,
        *,
        higher_is_better: bool,
    ) -> dict[str, Any]:
        reference_id, reference = (
            max if higher_is_better else min
        )(
            baseline_results.items(),
            key=lambda item: float(item[1]["metrics_macro"][metric]),
        )
        return {
            "metric": metric,
            "operator": ">" if higher_is_better else "<",
            "threshold": float(reference["metrics_macro"][metric]),
            "reference_model_id": reference_id,
            "basis": (
                "strictly higher than the best existing validation baseline"
                if higher_is_better
                else "strictly lower than the best existing validation baseline"
            ),
        }

    def fastest_runtime_rule(summary_key: str, candidate_metric: str) -> dict[str, Any]:
        reference_id, reference = min(
            baseline_results.items(),
            key=lambda item: float(item[1]["runtime_ms"][summary_key]),
        )
        return {
            "metric": candidate_metric,
            "operator": "<",
            "threshold": float(reference["runtime_ms"][summary_key]),
            "reference_model_id": reference_id,
            "basis": (
                f"strictly lower than the fastest existing CUDA {summary_key} runtime"
            ),
        }

    return {
        "schema_version": 1,
        "policy_id": "msbi-validation-strict-superiority-v1",
        "scope": "validation_only_before_independent_test",
        "baseline_results_sha256": baseline_results_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "effect_rules": [
            best_rule("pixel_dice", higher_is_better=True),
            best_rule("boundary_f1", higher_is_better=True),
            best_rule("instance_f1_50", higher_is_better=True),
            best_rule("panoptic_quality", higher_is_better=True),
            best_rule("count_absolute_error", higher_is_better=False),
        ],
        "performance_rules": [
            fastest_runtime_rule("mean", "runtime_ms"),
            fastest_runtime_rule("p95", "runtime_p95_ms"),
        ],
        "guardrail_rules": [
            {
                "metric": "mean_small_gate",
                "operator": ">=",
                "threshold": 0.10,
                "basis": "validation-average anti-collapse floor",
            },
            {
                "metric": "mean_large_gate",
                "operator": ">=",
                "threshold": 0.10,
                "basis": "validation-average anti-collapse floor",
            },
        ],
        "pass_rule": "all effect, performance, and guardrail rules must pass",
        "independent_test_action": (
            "independent test pixels may be accessed only when the strict validation "
            "superiority gate status is PASSED"
        ),
        "immutable_after_independent_test_access": True,
    }


def evaluate_candidate(
    policy: dict[str, Any],
    candidate_results: dict[str, Any],
    *,
    policy_sha256: str,
    candidate_results_sha256: str,
) -> dict[str, Any]:
    """Evaluate every frozen rule without reading an independent-test manifest."""
    metrics = candidate_results["metrics_macro"]
    checks: list[dict[str, Any]] = []
    rule_groups = [
        group
        for group in (
            "primary_rules",
            "effect_rules",
            "performance_rules",
            "guardrail_rules",
        )
        if group in policy
    ]
    for group in rule_groups:
        for rule in policy[group]:
            value = float(metrics[rule["metric"]])
            threshold = float(rule["threshold"])
            operator = rule["operator"]
            if operator == ">=":
                passed = value >= threshold
            elif operator == "<=":
                passed = value <= threshold
            elif operator == ">":
                passed = value > threshold
            elif operator == "<":
                passed = value < threshold
            else:
                raise ValueError(f"unsupported policy operator: {operator}")
            checks.append(
                {
                    "group": group,
                    "metric": rule["metric"],
                    "operator": operator,
                    "threshold": threshold,
                    "value": value,
                    "passed": passed,
                }
            )
    passed = all(check["passed"] for check in checks)
    return {
        "schema_version": 1,
        "model_id": candidate_results["model_id"],
        "status": "PASSED" if passed else "FAILED",
        "policy_sha256": policy_sha256,
        "candidate_results_sha256": candidate_results_sha256,
        "checks": checks,
        "independent_test_status": (
            "AUTHORIZED_BY_VALIDATION_GATE"
            if passed
            else "SEALED_NOT_ACCESSED"
        ),
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, Any], value)

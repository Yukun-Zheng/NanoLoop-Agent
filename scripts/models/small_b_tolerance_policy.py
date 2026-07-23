"""Small-B scientific tolerance-policy contract and per-image assessment."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

MODEL_ID = "unet-small-balanced-v1"
THRESHOLD_COMPARISON = "gt"
BOTTOM_CROP_PX = 130
SELECTED = "SELECTED"
FROZEN_PREDEFINED = "FROZEN_PREDEFINED"

_MINIMUM_TOLERANCES = (
    ("instance_precision", "minimum_instance_precision"),
    ("instance_recall", "minimum_instance_recall"),
    ("instance_f1", "minimum_instance_f1"),
)
_MAXIMUM_TOLERANCES = (
    ("count_absolute_error", "maximum_count_absolute_error"),
    ("count_relative_error", "maximum_count_relative_error"),
    ("mean_area_relative_error", "maximum_mean_area_relative_error"),
    (
        "mean_equivalent_diameter_relative_error",
        "maximum_mean_equivalent_diameter_relative_error",
    ),
    ("number_density_relative_error", "maximum_number_density_relative_error"),
    ("perimeter_density_relative_error", "maximum_perimeter_density_relative_error"),
)
_RELATIVE_TOLERANCE_FIELDS = {
    tolerance_field
    for _metric, tolerance_field in _MAXIMUM_TOLERANCES
    if tolerance_field != "maximum_count_absolute_error"
}


@dataclass(frozen=True, slots=True)
class FrozenScientificParameters:
    threshold: float
    threshold_comparison: str
    min_area_px: int
    bottom_crop_px: int


@dataclass(frozen=True, slots=True)
class InstanceMatchingPolicy:
    metric: str
    mask_iou_threshold: float


@dataclass(frozen=True, slots=True)
class PolicyApproval:
    frozen_before_independent_test: bool
    approved_by: str
    approved_at: str
    rationale: str


@dataclass(frozen=True, slots=True)
class SmallBTolerancePolicy:
    schema_version: str
    policy_id: str
    policy_version: str
    model_id: str
    threshold_evidence_sha256: str
    min_area_evidence_sha256: str
    frozen_scientific_parameters: FrozenScientificParameters
    instance_matching: InstanceMatchingPolicy
    per_image_tolerances: Mapping[str, int | float]
    not_evaluable_rule: str
    approval: PolicyApproval
    sha256: str


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of the exact file bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a file: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {resolved}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _mapping(payload: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = payload.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _sha256(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be 64 lowercase hexadecimal characters")
    return value


def _approval(payload: Mapping[str, Any]) -> PolicyApproval:
    approval = _mapping(payload, "approval")
    frozen = approval.get("frozen_before_independent_test")
    if frozen is not True:
        raise ValueError("approval.frozen_before_independent_test must be true")
    approved_by = _string(approval, "approved_by")
    approved_at = _string(approval, "approved_at")
    try:
        timestamp = datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("approval.approved_at must be an ISO-8601 timestamp") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("approval.approved_at must include a timezone")
    rationale = _string(approval, "rationale")
    return PolicyApproval(
        frozen_before_independent_test=True,
        approved_by=approved_by,
        approved_at=approved_at,
        rationale=rationale,
    )


def _frozen_parameters(payload: Mapping[str, Any]) -> FrozenScientificParameters:
    frozen = _mapping(payload, "frozen_scientific_parameters")
    threshold = _finite_number(
        frozen.get("threshold"),
        field="frozen_scientific_parameters.threshold",
    )
    if not 0.0 < threshold < 1.0:
        raise ValueError("frozen_scientific_parameters.threshold must be in (0, 1)")
    comparison = frozen.get("threshold_comparison")
    if comparison != THRESHOLD_COMPARISON:
        raise ValueError(
            f"frozen_scientific_parameters.threshold_comparison must be "
            f"'{THRESHOLD_COMPARISON}'"
        )
    min_area = frozen.get("min_area_px")
    if isinstance(min_area, bool) or not isinstance(min_area, int) or min_area < 0:
        raise ValueError("frozen_scientific_parameters.min_area_px must be a non-negative integer")
    bottom_crop = frozen.get("bottom_crop_px")
    if bottom_crop != BOTTOM_CROP_PX:
        raise ValueError(
            f"frozen_scientific_parameters.bottom_crop_px must be {BOTTOM_CROP_PX}"
        )
    return FrozenScientificParameters(
        threshold=threshold,
        threshold_comparison=comparison,
        min_area_px=min_area,
        bottom_crop_px=bottom_crop,
    )


def _instance_matching(payload: Mapping[str, Any]) -> InstanceMatchingPolicy:
    matching = _mapping(payload, "instance_matching")
    if matching.get("metric") != "mask_iou":
        raise ValueError("instance_matching.metric must be 'mask_iou'")
    threshold = _finite_number(
        matching.get("mask_iou_threshold"),
        field="instance_matching.mask_iou_threshold",
    )
    if not 0.0 < threshold <= 1.0:
        raise ValueError("instance_matching.mask_iou_threshold must be in (0, 1]")
    return InstanceMatchingPolicy(metric="mask_iou", mask_iou_threshold=threshold)


def _tolerances(payload: Mapping[str, Any]) -> dict[str, int | float]:
    tolerances = _mapping(payload, "per_image_tolerances")
    validated: dict[str, int | float] = {}
    for _metric, field in _MINIMUM_TOLERANCES:
        number = _finite_number(tolerances.get(field), field=f"per_image_tolerances.{field}")
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"per_image_tolerances.{field} must be in [0, 1]")
        validated[field] = number

    absolute_count = tolerances.get("maximum_count_absolute_error")
    if (
        isinstance(absolute_count, bool)
        or not isinstance(absolute_count, int)
        or absolute_count < 0
    ):
        raise ValueError(
            "per_image_tolerances.maximum_count_absolute_error "
            "must be a non-negative integer"
        )
    validated["maximum_count_absolute_error"] = absolute_count

    for field in sorted(_RELATIVE_TOLERANCE_FIELDS):
        number = _finite_number(tolerances.get(field), field=f"per_image_tolerances.{field}")
        if number < 0.0:
            raise ValueError(f"per_image_tolerances.{field} must be non-negative")
        validated[field] = number
    return validated


def _validate_evidence(
    *,
    threshold_payload: Mapping[str, Any],
    min_area_payload: Mapping[str, Any],
    threshold_sha256: str,
    min_area_sha256: str,
    policy_threshold_sha256: str,
    policy_min_area_sha256: str,
    frozen: FrozenScientificParameters,
) -> None:
    if threshold_sha256 != policy_threshold_sha256:
        raise ValueError("threshold calibration evidence SHA does not match the policy")
    if min_area_sha256 != policy_min_area_sha256:
        raise ValueError("min-area calibration evidence SHA does not match the policy")
    threshold_status = threshold_payload.get("selection_status")
    min_area_status = min_area_payload.get("selection_status")
    if threshold_status != min_area_status:
        raise ValueError("threshold and min-area evidence selection_status differs")
    if threshold_status == SELECTED:
        threshold_field = "selected_threshold"
        min_area_threshold_field = "selected_threshold"
        min_area_field = "selected_min_area_px"
    elif threshold_status == FROZEN_PREDEFINED:
        _validate_predefined_evidence_pair(
            threshold_payload=threshold_payload,
            min_area_payload=min_area_payload,
        )
        threshold_field = "threshold"
        min_area_threshold_field = "threshold"
        min_area_field = "min_area_px"
    else:
        raise ValueError(
            "evidence selection_status must be SELECTED or FROZEN_PREDEFINED"
        )

    threshold = _finite_number(
        threshold_payload.get(threshold_field),
        field=f"threshold evidence {threshold_field}",
    )
    min_area_threshold = _finite_number(
        min_area_payload.get(min_area_threshold_field),
        field=f"min-area evidence {min_area_threshold_field}",
    )
    if not math.isclose(threshold, frozen.threshold) or not math.isclose(
        min_area_threshold, frozen.threshold
    ):
        raise ValueError("frozen threshold differs from calibration evidence")

    selected_min_area = min_area_payload.get(min_area_field)
    if (
        isinstance(selected_min_area, bool)
        or not isinstance(selected_min_area, int)
        or selected_min_area != frozen.min_area_px
    ):
        raise ValueError("frozen min_area_px differs from calibration evidence")
    if min_area_payload.get("threshold_evidence_sha256") != threshold_sha256:
        raise ValueError("min-area evidence references a different threshold evidence SHA")
    if threshold_payload.get("comparison_rule") != "probability > threshold":
        raise ValueError("threshold evidence does not use strict probability > threshold")
    if min_area_payload.get("comparison_rule") != "probability > threshold":
        raise ValueError("min-area evidence does not use strict probability > threshold")
    if threshold_payload.get("bottom_crop_px") != BOTTOM_CROP_PX:
        raise ValueError("threshold evidence bottom_crop_px differs from the Small contract")
    if min_area_payload.get("bottom_crop_px") != BOTTOM_CROP_PX:
        raise ValueError("min-area evidence bottom_crop_px differs from the Small contract")


def _frozen_timestamp(payload: Mapping[str, Any]) -> str:
    frozen_at = _string(payload, "frozen_at")
    try:
        timestamp = datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("frozen_at must be an ISO-8601 timestamp") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("frozen_at must include a timezone")
    return frozen_at


def _validate_predefined_evidence_pair(
    *,
    threshold_payload: Mapping[str, Any],
    min_area_payload: Mapping[str, Any],
) -> None:
    identity_fields = (
        "frozen_at",
        "manifest_sha256",
        "model_sha256",
        "config_sha256",
        "weight_sha256",
    )
    for label, payload in (
        ("threshold", threshold_payload),
        ("min-area", min_area_payload),
    ):
        if payload.get("calibration_performed") is not False:
            raise ValueError(f"{label} calibration_performed must be false")
        if payload.get("frozen_before_test") is not True:
            raise ValueError(f"{label} frozen_before_test must be true")
        if payload.get("parameters_changed_after_test") is not False:
            raise ValueError(f"{label} parameters_changed_after_test must be false")
        if payload.get("independent_test_used_for_tuning") is not False:
            raise ValueError(f"{label} independent_test_used_for_tuning must be false")
        _string(payload, "parameter_source")
        _frozen_timestamp(payload)
        for field in ("manifest_sha256", "model_sha256", "config_sha256", "weight_sha256"):
            _sha256(payload, field)
    for field in identity_fields:
        if threshold_payload.get(field) != min_area_payload.get(field):
            raise ValueError(f"threshold and min-area evidence {field} differs")


def load_tolerance_policy(
    policy_path: Path,
    *,
    threshold_evidence_path: Path,
    min_area_evidence_path: Path,
) -> SmallBTolerancePolicy:
    """Load a frozen policy and bind it to the exact calibration evidence files."""

    policy_resolved = policy_path.expanduser().resolve(strict=True)
    payload = _load_json(policy_resolved, label="tolerance policy")
    threshold_resolved = threshold_evidence_path.expanduser().resolve(strict=True)
    min_area_resolved = min_area_evidence_path.expanduser().resolve(strict=True)
    threshold_payload = _load_json(
        threshold_resolved,
        label="threshold calibration evidence",
    )
    min_area_payload = _load_json(
        min_area_resolved,
        label="min-area calibration evidence",
    )

    if payload.get("schema_version") != "1":
        raise ValueError("schema_version must be '1'")
    policy_id = _string(payload, "policy_id")
    policy_version = _string(payload, "policy_version")
    if payload.get("model_id") != MODEL_ID:
        raise ValueError(f"model_id must be '{MODEL_ID}'")
    threshold_evidence_sha256 = _sha256(payload, "threshold_evidence_sha256")
    min_area_evidence_sha256 = _sha256(payload, "min_area_evidence_sha256")
    frozen = _frozen_parameters(payload)
    matching = _instance_matching(payload)
    tolerances = _tolerances(payload)
    if payload.get("not_evaluable_rule") != "fail":
        raise ValueError("not_evaluable_rule must be 'fail'")
    approval = _approval(payload)
    threshold_sha256 = sha256_file(threshold_resolved)
    min_area_sha256 = sha256_file(min_area_resolved)
    _validate_evidence(
        threshold_payload=threshold_payload,
        min_area_payload=min_area_payload,
        threshold_sha256=threshold_sha256,
        min_area_sha256=min_area_sha256,
        policy_threshold_sha256=threshold_evidence_sha256,
        policy_min_area_sha256=min_area_evidence_sha256,
        frozen=frozen,
    )
    return SmallBTolerancePolicy(
        schema_version="1",
        policy_id=policy_id,
        policy_version=policy_version,
        model_id=MODEL_ID,
        threshold_evidence_sha256=threshold_evidence_sha256,
        min_area_evidence_sha256=min_area_evidence_sha256,
        frozen_scientific_parameters=frozen,
        instance_matching=matching,
        per_image_tolerances=tolerances,
        not_evaluable_rule="fail",
        approval=approval,
        sha256=sha256_file(policy_resolved),
    )


def _observed_metric(metrics: Mapping[str, Any], metric: str) -> float | None:
    value = metrics.get(metric)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def evaluate_per_image_tolerances(
    sample_id: str,
    metrics: Mapping[str, Any],
    policy: SmallBTolerancePolicy,
) -> tuple[dict[str, Any], ...]:
    """Return one failure record for each missing, invalid, or out-of-tolerance metric."""

    failures: list[dict[str, Any]] = []
    rules = (
        *((metric, field, ">=") for metric, field in _MINIMUM_TOLERANCES),
        *((metric, field, "<=") for metric, field in _MAXIMUM_TOLERANCES),
    )
    for metric, tolerance_field, operator in rules:
        observed = _observed_metric(metrics, metric)
        tolerance = policy.per_image_tolerances[tolerance_field]
        if observed is None:
            failures.append(
                {
                    "sample_id": sample_id,
                    "metric": metric,
                    "operator": operator,
                    "observed": None,
                    "tolerance": tolerance,
                    "reason_code": "METRIC_NOT_EVALUABLE",
                }
            )
            continue
        passed = observed >= tolerance if operator == ">=" else observed <= tolerance
        if not passed:
            failures.append(
                {
                    "sample_id": sample_id,
                    "metric": metric,
                    "operator": operator,
                    "observed": observed,
                    "tolerance": tolerance,
                    "reason_code": "TOLERANCE_NOT_MET",
                }
            )
    return tuple(failures)

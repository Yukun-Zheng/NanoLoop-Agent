from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.models.small_b_tolerance_policy import (
    evaluate_per_image_tolerances,
    load_tolerance_policy,
    sha256_file,
)


def _threshold_evidence() -> dict[str, object]:
    return {
        "selection_status": "SELECTED",
        "selected_threshold": 0.4,
        "comparison_rule": "probability > threshold",
        "bottom_crop_px": 130,
        "independent_test_accessed": False,
    }


def _min_area_evidence(*, threshold_sha256: str) -> dict[str, object]:
    return {
        "selection_status": "SELECTED",
        "selected_threshold": 0.4,
        "selected_min_area_px": 16,
        "threshold_evidence_sha256": threshold_sha256,
        "comparison_rule": "probability > threshold",
        "bottom_crop_px": 130,
        "independent_test_accessed": False,
    }


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _evidence_files(tmp_path: Path) -> tuple[Path, Path]:
    threshold_path = _write_json(tmp_path / "threshold.json", _threshold_evidence())
    min_area_path = _write_json(
        tmp_path / "min-area.json",
        _min_area_evidence(threshold_sha256=sha256_file(threshold_path)),
    )
    return threshold_path, min_area_path


def _policy_payload(
    *,
    threshold_sha256: str,
    min_area_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": "1",
        "policy_id": "small-b-policy",
        "policy_version": "1",
        "model_id": "unet-small-balanced-v1",
        "threshold_evidence_sha256": threshold_sha256,
        "min_area_evidence_sha256": min_area_sha256,
        "frozen_scientific_parameters": {
            "threshold": 0.4,
            "threshold_comparison": "gt",
            "min_area_px": 16,
            "bottom_crop_px": 130,
        },
        "instance_matching": {
            "metric": "mask_iou",
            "mask_iou_threshold": 0.5,
        },
        "per_image_tolerances": {
            "minimum_instance_precision": 0.8,
            "minimum_instance_recall": 0.7,
            "minimum_instance_f1": 0.75,
            "maximum_count_absolute_error": 2,
            "maximum_count_relative_error": 0.2,
            "maximum_mean_area_relative_error": 0.25,
            "maximum_mean_equivalent_diameter_relative_error": 0.2,
            "maximum_number_density_relative_error": 0.25,
            "maximum_perimeter_density_relative_error": 0.25,
        },
        "not_evaluable_rule": "fail",
        "approval": {
            "frozen_before_independent_test": True,
            "approved_by": "Guo Jinghao",
            "approved_at": "2026-07-24T09:00:00+08:00",
            "rationale": "Prespecified Small-B scientific tolerances.",
        },
    }


def _policy_files(tmp_path: Path):
    threshold_path, min_area_path = _evidence_files(tmp_path)
    payload = _policy_payload(
        threshold_sha256=sha256_file(threshold_path),
        min_area_sha256=sha256_file(min_area_path),
    )
    policy_path = _write_json(tmp_path / "tolerance-policy.json", payload)
    return policy_path, threshold_path, min_area_path, payload


def _load(tmp_path: Path):
    policy_path, threshold_path, min_area_path, _payload = _policy_files(tmp_path)
    return load_tolerance_policy(
        policy_path,
        threshold_evidence_path=threshold_path,
        min_area_evidence_path=min_area_path,
    )


def _passing_metrics() -> dict[str, float | int]:
    return {
        "instance_precision": 0.9,
        "instance_recall": 0.8,
        "instance_f1": 0.84,
        "count_absolute_error": 1,
        "count_relative_error": 0.1,
        "mean_area_relative_error": 0.1,
        "mean_equivalent_diameter_relative_error": 0.1,
        "number_density_relative_error": 0.1,
        "perimeter_density_relative_error": 0.1,
    }


@pytest.mark.parametrize(
    ("field_path", "expected"),
    [
        (("policy_id",), "policy_id"),
        (("frozen_scientific_parameters", "threshold"), "threshold"),
        (("instance_matching", "mask_iou_threshold"), "mask_iou_threshold"),
        (("approval", "approved_by"), "approved_by"),
    ],
)
def test_rejects_missing_required_fields(
    tmp_path: Path,
    field_path: tuple[str, ...],
    expected: str,
) -> None:
    policy_path, threshold_path, min_area_path, payload = _policy_files(tmp_path)
    target = payload
    for part in field_path[:-1]:
        target = target[part]  # type: ignore[assignment]
    target.pop(field_path[-1])  # type: ignore[union-attr]
    _write_json(policy_path, payload)

    with pytest.raises(ValueError, match=expected):
        load_tolerance_policy(
            policy_path,
            threshold_evidence_path=threshold_path,
            min_area_evidence_path=min_area_path,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("minimum_instance_precision", -0.1),
        ("minimum_instance_recall", 1.1),
        ("minimum_instance_f1", float("inf")),
        ("maximum_count_absolute_error", 1.5),
        ("maximum_count_absolute_error", -1),
        ("maximum_count_relative_error", -0.1),
        ("maximum_mean_area_relative_error", float("nan")),
    ],
)
def test_rejects_invalid_tolerances(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    policy_path, threshold_path, min_area_path, payload = _policy_files(tmp_path)
    payload["per_image_tolerances"][field] = value  # type: ignore[index]
    _write_json(policy_path, payload)

    with pytest.raises(ValueError, match=field):
        load_tolerance_policy(
            policy_path,
            threshold_evidence_path=threshold_path,
            min_area_evidence_path=min_area_path,
        )


@pytest.mark.parametrize("iou", [0.0, -0.1, 1.1, float("nan")])
def test_rejects_invalid_instance_iou(tmp_path: Path, iou: float) -> None:
    policy_path, threshold_path, min_area_path, payload = _policy_files(tmp_path)
    payload["instance_matching"]["mask_iou_threshold"] = iou  # type: ignore[index]
    _write_json(policy_path, payload)

    with pytest.raises(ValueError, match="mask_iou_threshold"):
        load_tolerance_policy(
            policy_path,
            threshold_evidence_path=threshold_path,
            min_area_evidence_path=min_area_path,
        )


@pytest.mark.parametrize("evidence", ["threshold", "min-area"])
def test_rejects_evidence_sha_mismatch(tmp_path: Path, evidence: str) -> None:
    policy_path, threshold_path, min_area_path, payload = _policy_files(tmp_path)
    payload[f"{evidence.replace('-', '_')}_evidence_sha256"] = "0" * 64
    _write_json(policy_path, payload)

    with pytest.raises(ValueError, match="evidence SHA"):
        load_tolerance_policy(
            policy_path,
            threshold_evidence_path=threshold_path,
            min_area_evidence_path=min_area_path,
        )


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("threshold", 0.45, "threshold differs"),
        ("min_area_px", 32, "min_area_px differs"),
    ],
)
def test_rejects_frozen_parameter_drift(
    tmp_path: Path,
    field: str,
    value: object,
    expected: str,
) -> None:
    policy_path, threshold_path, min_area_path, payload = _policy_files(tmp_path)
    payload["frozen_scientific_parameters"][field] = value  # type: ignore[index]
    _write_json(policy_path, payload)

    with pytest.raises(ValueError, match=expected):
        load_tolerance_policy(
            policy_path,
            threshold_evidence_path=threshold_path,
            min_area_evidence_path=min_area_path,
        )


def test_not_evaluable_metric_creates_one_failure(tmp_path: Path) -> None:
    policy = _load(tmp_path)
    metrics = _passing_metrics()
    metrics["mean_area_relative_error"] = None  # type: ignore[assignment]

    failures = evaluate_per_image_tolerances("sample-1", metrics, policy)

    assert failures == (
        {
            "sample_id": "sample-1",
            "metric": "mean_area_relative_error",
            "operator": "<=",
            "observed": None,
            "tolerance": 0.25,
            "reason_code": "METRIC_NOT_EVALUABLE",
        },
    )


def test_single_and_multiple_tolerance_failures(tmp_path: Path) -> None:
    policy = _load(tmp_path)
    single = _passing_metrics()
    single["instance_recall"] = 0.6

    assert evaluate_per_image_tolerances("sample-1", single, policy) == (
        {
            "sample_id": "sample-1",
            "metric": "instance_recall",
            "operator": ">=",
            "observed": 0.6,
            "tolerance": 0.7,
            "reason_code": "TOLERANCE_NOT_MET",
        },
    )

    multiple = _passing_metrics()
    multiple["instance_precision"] = 0.7
    multiple["count_absolute_error"] = 3
    failures = evaluate_per_image_tolerances("sample-2", multiple, policy)
    assert [failure["metric"] for failure in failures] == [
        "instance_precision",
        "count_absolute_error",
    ]


def test_policy_sha_is_stable_for_exact_file_bytes(tmp_path: Path) -> None:
    policy_path, threshold_path, min_area_path, _payload = _policy_files(tmp_path)

    first = load_tolerance_policy(
        policy_path,
        threshold_evidence_path=threshold_path,
        min_area_evidence_path=min_area_path,
    )
    second = load_tolerance_policy(
        policy_path,
        threshold_evidence_path=threshold_path,
        min_area_evidence_path=min_area_path,
    )

    assert first.sha256 == second.sha256 == sha256_file(policy_path)

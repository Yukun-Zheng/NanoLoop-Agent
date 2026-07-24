from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.models.evaluate_unet_small_independent_test import (
    SmallBEvaluationContract,
    build_parser,
    evaluate_independent_test,
    load_evaluation_contract,
    main,
    validate_evaluation_contract,
    write_evaluation_outputs,
)
from scripts.models.small_b_contracts import SmallBSplitManifest
from scripts.models.small_b_tolerance_policy import sha256_file

MANIFEST_FIELDS = (
    "sample_id",
    "source_sample_id",
    "source_image_id",
    "field_of_view_id",
    "split",
    "image_path",
    "mask_path",
    "image_sha256",
    "mask_sha256",
    "included",
    "exclusion_reason",
)


def _row(sample_id: str, split: str, digit: str) -> dict[str, str]:
    has_mask = split in {"calibration", "independent_test"}
    return {
        "sample_id": sample_id,
        "source_sample_id": f"source-sample-{digit}",
        "source_image_id": f"source-image-{digit}",
        "field_of_view_id": f"field-{digit}",
        "split": split,
        "image_path": f"images/{sample_id}.tif",
        "mask_path": f"masks/{sample_id}.png" if has_mask else "",
        "image_sha256": digit * 64,
        "mask_sha256": ("f" * 63 + digit) if has_mask else "",
        "included": "true",
        "exclusion_reason": "",
    }


def _write_manifest(
    path: Path,
    *,
    include_calibration: bool = True,
    include_independent_test: bool = True,
) -> Path:
    rows = [_row("train-1", "train", "1")]
    if include_calibration:
        rows.append(_row("calibration-2", "calibration", "2"))
    if include_independent_test:
        rows.append(_row("test-3", "independent_test", "3"))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _policy_payload(
    *,
    threshold_sha: str,
    min_area_sha: str,
    strict: bool = False,
    threshold: float = 0.5,
    min_area_px: int = 0,
) -> dict[str, object]:
    maximum = 0.0 if strict else 1.0
    minimum = 1.0 if strict else 0.0
    return {
        "schema_version": "1",
        "policy_id": "small-b-policy",
        "policy_version": "1",
        "model_id": "unet-small-balanced-v1",
        "threshold_evidence_sha256": threshold_sha,
        "min_area_evidence_sha256": min_area_sha,
        "frozen_scientific_parameters": {
            "threshold": threshold,
            "threshold_comparison": "gt",
            "min_area_px": min_area_px,
            "bottom_crop_px": 130,
        },
        "instance_matching": {"metric": "mask_iou", "mask_iou_threshold": 0.5},
        "per_image_tolerances": {
            "minimum_instance_precision": minimum,
            "minimum_instance_recall": minimum,
            "minimum_instance_f1": minimum,
            "maximum_count_absolute_error": 0 if strict else 100,
            "maximum_count_relative_error": maximum,
            "maximum_mean_area_relative_error": maximum,
            "maximum_mean_equivalent_diameter_relative_error": maximum,
            "maximum_number_density_relative_error": maximum,
            "maximum_perimeter_density_relative_error": maximum,
        },
        "not_evaluable_rule": "fail",
        "approval": {
            "frozen_before_independent_test": True,
            "approved_by": "Guo Jinghao",
            "approved_at": "2026-07-24T10:00:00+08:00",
            "rationale": "Frozen before the independent test.",
        },
    }


def _contract_files(
    tmp_path: Path,
    *,
    strict: bool = False,
    evidence_status: str = "SELECTED",
    frozen_before_test: bool = True,
    parameters_changed_after_test: bool = False,
    remove_predefined_field: str | None = None,
    include_calibration: bool = True,
    include_independent_test: bool = True,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest_path = _write_manifest(
        tmp_path / "split-manifest.csv",
        include_calibration=include_calibration,
        include_independent_test=include_independent_test,
    )
    manifest_sha = sha256_file(manifest_path)
    if evidence_status == "FROZEN_PREDEFINED":
        common = {
            "selection_status": evidence_status,
            "calibration_performed": False,
            "frozen_before_test": frozen_before_test,
            "parameter_source": "predefined Small model operating point",
            "frozen_at": "2026-07-24T08:00:00+08:00",
            "manifest_sha256": manifest_sha,
            "model_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "weight_sha256": "c" * 64,
            "parameters_changed_after_test": parameters_changed_after_test,
            "independent_test_used_for_tuning": False,
            "comparison_rule": "probability > threshold",
            "bottom_crop_px": 130,
        }
        if remove_predefined_field is not None:
            common.pop(remove_predefined_field)
        threshold_payload = {
            **common,
            "threshold": 0.30,
        }
    else:
        threshold_payload = {
            "selection_status": evidence_status,
            "selected_threshold": 0.5,
            "comparison_rule": "probability > threshold",
            "prediction_min_area_px": 0,
            "ground_truth_min_area_px": 0,
            "bottom_crop_px": 130,
            "independent_test_accessed": False,
            "split_manifest_sha256": manifest_sha,
        }
    threshold_path = _write_json(
        tmp_path / "threshold.json",
        threshold_payload,
    )
    threshold_sha = sha256_file(threshold_path)
    if evidence_status == "FROZEN_PREDEFINED":
        min_area_payload = {
            **common,
            "threshold": 0.30,
            "min_area_px": 64,
            "threshold_evidence_sha256": threshold_sha,
        }
        policy_threshold = 0.30
        policy_min_area = 64
    else:
        min_area_payload = {
            "selection_status": evidence_status,
            "selected_threshold": 0.5,
            "selected_min_area_px": 0,
            "threshold_evidence_sha256": threshold_sha,
            "comparison_rule": "probability > threshold",
            "bottom_crop_px": 130,
            "independent_test_accessed": False,
            "split_manifest_sha256": manifest_sha,
        }
        policy_threshold = 0.5
        policy_min_area = 0
    min_area_path = _write_json(
        tmp_path / "min-area.json",
        min_area_payload,
    )
    policy_path = _write_json(
        tmp_path / "policy.json",
        _policy_payload(
            threshold_sha=threshold_sha,
            min_area_sha=sha256_file(min_area_path),
            strict=strict,
            threshold=policy_threshold,
            min_area_px=policy_min_area,
        ),
    )
    return manifest_path, threshold_path, min_area_path, policy_path


def _load_contract(
    tmp_path: Path,
    *,
    strict: bool = False,
    evidence_status: str = "SELECTED",
    frozen_before_test: bool = True,
    parameters_changed_after_test: bool = False,
    remove_predefined_field: str | None = None,
    include_calibration: bool = True,
    include_independent_test: bool = True,
) -> SmallBEvaluationContract:
    manifest_path, threshold_path, min_area_path, policy_path = _contract_files(
        tmp_path,
        strict=strict,
        evidence_status=evidence_status,
        frozen_before_test=frozen_before_test,
        parameters_changed_after_test=parameters_changed_after_test,
        remove_predefined_field=remove_predefined_field,
        include_calibration=include_calibration,
        include_independent_test=include_independent_test,
    )
    return load_evaluation_contract(
        split_manifest_path=manifest_path,
        threshold_evidence_path=threshold_path,
        min_area_evidence_path=min_area_path,
        tolerance_policy_path=policy_path,
        expected_policy_sha256=sha256_file(policy_path),
    )


def _arrays(*, matching: bool = True) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.zeros((140, 10), dtype=bool)
    truth = np.zeros((140, 10), dtype=bool)
    truth[2:4, 2:4] = True
    if matching:
        prediction[2:4, 2:4] = True
    return prediction, truth


def test_only_selects_included_independent_test(tmp_path: Path) -> None:
    contract = _load_contract(tmp_path)
    requested: list[tuple[str, str]] = []
    prediction, truth = _arrays()

    def prediction_provider(record):
        requested.append((record.sample_id, record.split.value))
        return prediction

    evaluate_independent_test(
        contract,
        expected_policy_sha256=contract.tolerance_policy.sha256,
        prediction_provider=prediction_provider,
        truth_provider=lambda _record: truth,
    )

    assert requested == [("test-3", "independent_test")]


def test_sha_and_parameter_drift_fail_before_data_access(tmp_path: Path) -> None:
    contract = _load_contract(tmp_path)

    with pytest.raises(ValueError, match="policy SHA"):
        validate_evaluation_contract(contract, expected_policy_sha256="0" * 64)

    drifted_min_area = {**contract.min_area_evidence, "selected_min_area_px": 4}
    drifted = SmallBEvaluationContract(
        manifest=contract.manifest,
        manifest_sha256=contract.manifest_sha256,
        threshold_evidence=contract.threshold_evidence,
        min_area_evidence=drifted_min_area,
        tolerance_policy=contract.tolerance_policy,
        threshold_evidence_sha256=contract.threshold_evidence_sha256,
        min_area_evidence_sha256=contract.min_area_evidence_sha256,
    )
    with pytest.raises(ValueError, match="selected min-area"):
        validate_evaluation_contract(
            drifted,
            expected_policy_sha256=contract.tolerance_policy.sha256,
        )


def test_missing_frozen_evidence_fails(tmp_path: Path) -> None:
    contract = _load_contract(tmp_path)
    incomplete = SmallBEvaluationContract(
        manifest=contract.manifest,
        manifest_sha256=contract.manifest_sha256,
        threshold_evidence={
            **contract.threshold_evidence,
            "selection_status": "NOT_EVALUATED",
        },
        min_area_evidence={
            **contract.min_area_evidence,
            "selection_status": "NOT_EVALUATED",
        },
        tolerance_policy=contract.tolerance_policy,
        threshold_evidence_sha256=contract.threshold_evidence_sha256,
        min_area_evidence_sha256=contract.min_area_evidence_sha256,
    )

    with pytest.raises(ValueError, match="SELECTED or FROZEN_PREDEFINED"):
        validate_evaluation_contract(
            incomplete,
            expected_policy_sha256=contract.tolerance_policy.sha256,
        )


def test_accepts_frozen_predefined_and_runs_formal_verdict(tmp_path: Path) -> None:
    contract = _load_contract(
        tmp_path,
        strict=True,
        evidence_status="FROZEN_PREDEFINED",
    )
    prediction = np.zeros((142, 12), dtype=bool)
    truth = np.zeros((142, 12), dtype=bool)
    prediction[2:10, 2:10] = True
    truth[2:10, 2:10] = True

    result = evaluate_independent_test(
        contract,
        expected_policy_sha256=contract.tolerance_policy.sha256,
        prediction_provider=lambda _record: prediction,
        truth_provider=lambda _record: truth,
    )

    assert result["verdict"] == "PASS"
    assert result["selection_status"] == "FROZEN_PREDEFINED"
    assert result["generalization_scope"] == {
        "physical_sample_count": 1,
        "independent_region_count": 1,
        "description": "same-sample, independent-region generalization",
        "cross_sample_generalization": False,
        "statement": "three independent regions from one physical sample",
    }


def test_frozen_predefined_accepts_empty_calibration(tmp_path: Path) -> None:
    contract = _load_contract(
        tmp_path,
        evidence_status="FROZEN_PREDEFINED",
        include_calibration=False,
    )

    assert contract.manifest.select("calibration") == ()
    assert [record.sample_id for record in contract.manifest.select("independent_test")] == [
        "test-3"
    ]


def test_selected_rejects_empty_calibration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SELECTED requires calibration"):
        _load_contract(
            tmp_path,
            evidence_status="SELECTED",
            include_calibration=False,
        )


def test_frozen_predefined_still_requires_independent_test(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="independent_test is required"):
        _load_contract(
            tmp_path,
            evidence_status="FROZEN_PREDEFINED",
            include_calibration=False,
            include_independent_test=False,
        )


def test_empty_calibration_does_not_weaken_frozen_approval(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frozen_before_test must be true"):
        _load_contract(
            tmp_path,
            evidence_status="FROZEN_PREDEFINED",
            frozen_before_test=False,
            include_calibration=False,
        )


def test_empty_calibration_does_not_weaken_array_shape_validation(
    tmp_path: Path,
) -> None:
    contract = _load_contract(
        tmp_path,
        evidence_status="FROZEN_PREDEFINED",
        include_calibration=False,
    )
    invalid = np.zeros((130, 10), dtype=bool)

    with pytest.raises(ValueError, match="invalid full-image shape"):
        evaluate_independent_test(
            contract,
            expected_policy_sha256=contract.tolerance_policy.sha256,
            prediction_provider=lambda _record: invalid,
            truth_provider=lambda _record: invalid,
        )


def test_not_evaluated_cli_validates_frozen_manifest_without_calibration(
    tmp_path: Path,
) -> None:
    manifest, threshold, min_area, policy = _contract_files(
        tmp_path / "inputs",
        evidence_status="FROZEN_PREDEFINED",
        include_calibration=False,
    )
    output = tmp_path / "output"

    exit_code = main(
        [
            "--split-manifest",
            str(manifest),
            "--threshold-evidence",
            str(threshold),
            "--min-area-evidence",
            str(min_area),
            "--tolerance-policy",
            str(policy),
            "--expected-policy-sha256",
            sha256_file(policy),
            "--output-root",
            str(output),
            "--not-evaluated",
        ]
    )

    assert exit_code == 0
    assert {path.name for path in output.iterdir()} == {
        "independent-test-metrics.json",
        "independent-test-per-image.csv",
        "failure-cases.csv",
        "scientific-verdict.json",
    }
    verdict = json.loads(
        (output / "scientific-verdict.json").read_text(encoding="utf-8")
    )
    assert verdict["verdict"] == "NOT_EVALUATED"
    assert verdict["selection_status"] == "FROZEN_PREDEFINED"


def test_rejects_engineering_preset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SELECTED or FROZEN_PREDEFINED"):
        _load_contract(tmp_path, evidence_status="ENGINEERING_PRESET")


def test_rejects_predefined_not_frozen_before_test(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frozen_before_test must be true"):
        _load_contract(
            tmp_path,
            evidence_status="FROZEN_PREDEFINED",
            frozen_before_test=False,
        )


def test_rejects_predefined_parameters_changed_after_test(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="parameters_changed_after_test must be false"):
        _load_contract(
            tmp_path,
            evidence_status="FROZEN_PREDEFINED",
            parameters_changed_after_test=True,
        )


@pytest.mark.parametrize(
    "field",
    [
        "parameter_source",
        "frozen_at",
        "manifest_sha256",
        "model_sha256",
        "config_sha256",
        "weight_sha256",
    ],
)
def test_rejects_predefined_missing_provenance_field(
    tmp_path: Path,
    field: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        _load_contract(
            tmp_path,
            evidence_status="FROZEN_PREDEFINED",
            remove_predefined_field=field,
        )


def test_pass_and_fail_results(tmp_path: Path) -> None:
    passing_contract = _load_contract(tmp_path / "passing", strict=True)
    prediction, truth = _arrays()
    passed = evaluate_independent_test(
        passing_contract,
        expected_policy_sha256=passing_contract.tolerance_policy.sha256,
        prediction_provider=lambda _record: prediction,
        truth_provider=lambda _record: truth,
    )
    assert passed["verdict"] == "PASS"
    assert passed["selection_status"] == "SELECTED"
    assert passed["failures"] == []

    failing_root = tmp_path / "failing"
    failing_root.mkdir()
    failing_contract = _load_contract(failing_root, strict=True)
    missing_prediction, truth = _arrays(matching=False)
    failed = evaluate_independent_test(
        failing_contract,
        expected_policy_sha256=failing_contract.tolerance_policy.sha256,
        prediction_provider=lambda _record: missing_prediction,
        truth_provider=lambda _record: truth,
    )
    assert failed["verdict"] == "FAIL"
    assert {failure["metric"] for failure in failed["failures"]} >= {
        "instance_recall",
        "instance_f1",
    }


def test_empty_dataset_is_not_evaluated(tmp_path: Path) -> None:
    contract = _load_contract(tmp_path)
    empty = SmallBEvaluationContract(
        manifest=SmallBSplitManifest(records=()),
        manifest_sha256=contract.manifest_sha256,
        threshold_evidence=contract.threshold_evidence,
        min_area_evidence=contract.min_area_evidence,
        tolerance_policy=contract.tolerance_policy,
        threshold_evidence_sha256=contract.threshold_evidence_sha256,
        min_area_evidence_sha256=contract.min_area_evidence_sha256,
    )

    result = evaluate_independent_test(
        empty,
        expected_policy_sha256=empty.tolerance_policy.sha256,
        prediction_provider=lambda _record: pytest.fail("prediction must not be read"),
        truth_provider=lambda _record: pytest.fail("GT must not be read"),
    )

    assert result["verdict"] == "NOT_EVALUATED"
    assert result["reason_codes"] == ["INDEPENDENT_TEST_EMPTY"]


def test_not_evaluable_failure_is_unique_and_has_required_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _load_contract(tmp_path)
    prediction, truth = _arrays()

    def fake_values(_metrics: Mapping[str, Any]) -> dict[str, float | None]:
        values = {
            "instance_precision": 1.0,
            "instance_recall": 1.0,
            "instance_f1": 1.0,
            "count_absolute_error": 0.0,
            "count_relative_error": 0.0,
            "mean_area_relative_error": None,
            "mean_equivalent_diameter_relative_error": 0.0,
            "number_density_relative_error": 0.0,
            "perimeter_density_relative_error": 0.0,
        }
        return values

    monkeypatch.setattr(
        "scripts.models.evaluate_unet_small_independent_test._scientific_metric_values",
        fake_values,
    )
    result = evaluate_independent_test(
        contract,
        expected_policy_sha256=contract.tolerance_policy.sha256,
        prediction_provider=lambda _record: prediction,
        truth_provider=lambda _record: truth,
    )

    assert result["failures"] == [
        {
            "evaluation_id": "small-b-independent-test-v1",
            "sample_id": "test-3",
            "metric": "mean_area_relative_error",
            "operator": "<=",
            "observed": None,
            "tolerance": 1.0,
            "reason_code": "METRIC_NOT_EVALUABLE",
        }
    ]


def test_four_output_files_have_consistent_references(tmp_path: Path) -> None:
    contract = _load_contract(tmp_path)
    prediction, truth = _arrays()
    result = evaluate_independent_test(
        contract,
        expected_policy_sha256=contract.tolerance_policy.sha256,
        prediction_provider=lambda _record: prediction,
        truth_provider=lambda _record: truth,
    )
    output_root = tmp_path / "output"

    write_evaluation_outputs(output_root, contract, result)

    assert {path.name for path in output_root.iterdir()} == {
        "independent-test-metrics.json",
        "independent-test-per-image.csv",
        "failure-cases.csv",
        "scientific-verdict.json",
    }
    metrics = json.loads(
        (output_root / "independent-test-metrics.json").read_text(encoding="utf-8")
    )
    verdict = json.loads(
        (output_root / "scientific-verdict.json").read_text(encoding="utf-8")
    )
    assert metrics["evaluation_id"] == verdict["evaluation_id"] == result["evaluation_id"]
    assert metrics["artifacts"]["scientific_verdict"]["path"] == "scientific-verdict.json"
    assert verdict["independent_test_metrics"]["sha256"] == sha256_file(
        output_root / "independent-test-metrics.json"
    )
    assert metrics["artifacts"]["per_image_csv"] == verdict["per_image_csv"]
    assert metrics["artifacts"]["failure_cases_csv"] == verdict["failure_cases_csv"]


def test_cli_has_no_scientific_parameter_overrides() -> None:
    parser = build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert "--threshold" not in option_strings
    assert "--min-area-px" not in option_strings
    assert "--mask-iou-threshold" not in option_strings
    assert "--tolerance" not in option_strings


def test_module_does_not_write_parameter_files() -> None:
    source = Path(__import__(
        "scripts.models.evaluate_unet_small_independent_test",
        fromlist=["__file__"],
    ).__file__).read_text(encoding="utf-8")

    assert "model_artifacts/configs" not in source
    assert "model_artifacts/registry.yaml" not in source
    assert "model_artifacts/model_cards" not in source

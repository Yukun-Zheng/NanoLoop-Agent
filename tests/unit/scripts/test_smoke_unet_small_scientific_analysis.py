from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.models.evaluate_unet_small_independent_test import load_evaluation_contract
from scripts.models.small_b_tolerance_policy import sha256_file
from scripts.models.smoke_unet_small_scientific_analysis import (
    build_parser,
    run_scientific_smoke,
)

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
WEIGHT_SHA = "a" * 64
CONFIG_SHA = "b" * 64
CARD_SHA = "c" * 64
ADAPTER_SHA = "d" * 64
IMAGE_SHA = "3" * 64


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_manifest(path: Path) -> Path:
    rows = [
        {
            "sample_id": "calibration-2",
            "source_sample_id": "source-sample-2",
            "source_image_id": "source-image-2",
            "field_of_view_id": "field-2",
            "split": "calibration",
            "image_path": "images/calibration-2.tif",
            "mask_path": "masks/calibration-2.png",
            "image_sha256": "2" * 64,
            "mask_sha256": "e" * 64,
            "included": "true",
            "exclusion_reason": "",
        },
        {
            "sample_id": "test-3",
            "source_sample_id": "source-sample-3",
            "source_image_id": "source-image-3",
            "field_of_view_id": "field-3",
            "split": "independent_test",
            "image_path": "images/test-3.tif",
            "mask_path": "masks/test-3.png",
            "image_sha256": IMAGE_SHA,
            "mask_sha256": "f" * 64,
            "included": "true",
            "exclusion_reason": "",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _contract(tmp_path: Path):
    manifest_path = _write_manifest(tmp_path / "split-manifest.csv")
    manifest_sha = sha256_file(manifest_path)
    threshold_path = _write_json(
        tmp_path / "threshold.json",
        {
            "selection_status": "SELECTED",
            "selected_threshold": 0.4,
            "comparison_rule": "probability > threshold",
            "prediction_min_area_px": 0,
            "ground_truth_min_area_px": 0,
            "bottom_crop_px": 130,
            "independent_test_accessed": False,
            "split_manifest_sha256": manifest_sha,
        },
    )
    threshold_sha = sha256_file(threshold_path)
    min_area_path = _write_json(
        tmp_path / "min-area.json",
        {
            "selection_status": "SELECTED",
            "selected_threshold": 0.4,
            "selected_min_area_px": 16,
            "threshold_evidence_sha256": threshold_sha,
            "comparison_rule": "probability > threshold",
            "bottom_crop_px": 130,
            "independent_test_accessed": False,
            "split_manifest_sha256": manifest_sha,
        },
    )
    policy_path = _write_json(
        tmp_path / "policy.json",
        {
            "schema_version": "1",
            "policy_id": "small-b-policy",
            "policy_version": "1",
            "model_id": "unet-small-balanced-v1",
            "threshold_evidence_sha256": threshold_sha,
            "min_area_evidence_sha256": sha256_file(min_area_path),
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
                "minimum_instance_precision": 0.0,
                "minimum_instance_recall": 0.0,
                "minimum_instance_f1": 0.0,
                "maximum_count_absolute_error": 100,
                "maximum_count_relative_error": 1.0,
                "maximum_mean_area_relative_error": 1.0,
                "maximum_mean_equivalent_diameter_relative_error": 1.0,
                "maximum_number_density_relative_error": 1.0,
                "maximum_perimeter_density_relative_error": 1.0,
            },
            "not_evaluable_rule": "fail",
            "approval": {
                "frozen_before_independent_test": True,
                "approved_by": "Guo Jinghao",
                "approved_at": "2026-07-24T11:00:00+08:00",
                "rationale": "Frozen Small-B scientific smoke contract.",
            },
        },
    )
    contract = load_evaluation_contract(
        split_manifest_path=manifest_path,
        threshold_evidence_path=threshold_path,
        min_area_evidence_path=min_area_path,
        tolerance_policy_path=policy_path,
        expected_policy_sha256=sha256_file(policy_path),
    )
    return contract


def _analysis_result(tmp_path: Path) -> dict[str, object]:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    mask_path = artifacts / "pred_mask.png"
    Image.new("L", (10, 140), color=0).save(mask_path)
    instances_path = _write_json(
        artifacts / "instances.json",
        {
            "coordinate_space": "original_px",
            "width": 10,
            "height": 140,
            "instance_count": 0,
            "instances": [],
        },
    )
    run_config_path = _write_json(
        artifacts / "run_config.json",
        {
            "contract_schema_version": 3,
            "model_id": "unet-small-balanced-v1",
            "model_version": "1",
            "adapter_path": "app.inference.adapters.unet:UNetAdapter",
            "weight_sha256": WEIGHT_SHA,
            "config_sha256": CONFIG_SHA,
            "model_card_sha256": CARD_SHA,
            "adapter_sha256": ADAPTER_SHA,
            "image_sha256": IMAGE_SHA,
            "inference": {
                "threshold": 0.4,
                "min_area_px": 16,
                "watershed_enabled": False,
                "exclude_border": True,
            },
            "resolved_postprocess": {
                "min_area_px": 16,
                "fill_holes": True,
                "watershed_enabled": False,
                "exclude_border": True,
                "connectivity": 2,
            },
            "analysis_roi": {
                "invalid_rects": [
                    {
                        "x1": 0,
                        "y1": 10,
                        "x2": 10,
                        "y2": 140,
                        "reason": "instrument_bar",
                    }
                ]
            },
        },
    )
    return {
        "run_id": "run-small-b-1",
        "final_status": "completed",
        "model": {
            "model_id": "unet-small-balanced-v1",
            "version": "1",
            "adapter_path": "app.inference.adapters.unet:UNetAdapter",
            "weight_sha256": WEIGHT_SHA,
            "config_sha256": CONFIG_SHA,
            "model_card_sha256": CARD_SHA,
            "adapter_sha256": ADAPTER_SHA,
        },
        "frozen_inference": {
            "threshold": 0.4,
            "min_area_px": 16,
        },
        "resolved_postprocess": {
            "min_area_px": 16,
            "fill_holes": True,
            "watershed_enabled": False,
            "exclude_border": True,
            "connectivity": 2,
        },
        "artifacts": {
            "mask": str(mask_path),
            "instances": str(instances_path),
            "run_configuration": str(run_config_path),
        },
    }


def test_cli_has_no_frozen_parameter_overrides() -> None:
    parser = build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert "--threshold" not in options
    assert "--min-area-px" not in options
    assert "--mask-iou-threshold" not in options
    assert "--watershed-enabled" not in options
    assert "--exclude-border" not in options


def test_missing_canonical_artifact_fails_closed(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    result = _analysis_result(tmp_path)
    Path(result["artifacts"]["instances"]).unlink()  # type: ignore[index]

    report = run_scientific_smoke(
        contract,
        expected_policy_sha256=contract.tolerance_policy.sha256,
        sample_ids=["test-3"],
        analysis_executor=lambda _record, _parameters: result,
    )

    assert report["status"] == "FAIL"
    assert report["failures"][0]["reason_code"] == "SCIENTIFIC_ANALYSIS_INVALID"
    assert "instances" in report["failures"][0]["message"]


@pytest.mark.parametrize("drift", ["threshold", "config_sha"])
def test_parameter_or_identity_sha_drift_fails_closed(
    tmp_path: Path,
    drift: str,
) -> None:
    contract = _contract(tmp_path)
    result = _analysis_result(tmp_path)
    if drift == "threshold":
        result["frozen_inference"]["threshold"] = 0.45  # type: ignore[index]
    else:
        result["model"]["config_sha256"] = "0" * 64  # type: ignore[index]

    report = run_scientific_smoke(
        contract,
        expected_policy_sha256=contract.tolerance_policy.sha256,
        sample_ids=["test-3"],
        analysis_executor=lambda _record, _parameters: result,
    )

    assert report["status"] == "FAIL"
    assert len(report["failures"]) == 1


def test_correct_formal_artifacts_pass(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    result = _analysis_result(tmp_path)
    requested: list[str] = []

    def executor(record, parameters):
        requested.append(record.sample_id)
        assert parameters.threshold == pytest.approx(0.4)
        assert parameters.min_area_px == 16
        assert parameters.bottom_crop_px == 130
        return result

    report = run_scientific_smoke(
        contract,
        expected_policy_sha256=contract.tolerance_policy.sha256,
        sample_ids=["test-3"],
        analysis_executor=executor,
    )

    assert requested == ["test-3"]
    assert report["status"] == "PASS"
    assert report["failures"] == []
    assert report["samples"][0]["artifact_sha256"]["mask"] == sha256_file(
        Path(result["artifacts"]["mask"])  # type: ignore[index]
    )


def test_no_selected_record_is_not_evaluated(tmp_path: Path) -> None:
    contract = _contract(tmp_path)

    report = run_scientific_smoke(
        contract,
        expected_policy_sha256=contract.tolerance_policy.sha256,
        sample_ids=[],
        analysis_executor=lambda _record, _parameters: pytest.fail("must not execute"),
    )

    assert report["status"] == "NOT_EVALUATED"


def test_module_does_not_write_frozen_contracts() -> None:
    source = Path(__import__(
        "scripts.models.smoke_unet_small_scientific_analysis",
        fromlist=["__file__"],
    ).__file__).read_text(encoding="utf-8")

    assert "model_artifacts/configs" not in source
    assert "model_artifacts/registry.yaml" not in source
    assert "model_artifacts/model_cards" not in source
    assert "tolerance-policy.json" not in source

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.models.calibrate_unet_small_min_area import (
    evaluate_min_area_calibration,
    load_min_area_plan,
    load_threshold_evidence,
    write_min_area_outputs,
)
from scripts.models.calibrate_unet_small_threshold import (
    evaluate_threshold_calibration,
    load_threshold_plan,
    write_calibration_outputs,
)
from scripts.models.evaluate_unet_small_independent_test import (
    evaluate_independent_test,
    load_evaluation_contract,
    write_evaluation_outputs,
)
from scripts.models.small_b_contracts import load_split_manifest
from scripts.models.small_b_tolerance_policy import sha256_file
from scripts.models.smoke_unet_small_scientific_analysis import run_scientific_smoke

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


@dataclass(frozen=True, slots=True)
class Workflow:
    manifest_path: Path
    threshold_path: Path
    min_area_path: Path
    policy_path: Path
    calibration_probability_calls: tuple[str, ...]
    min_area_cache_calls: tuple[str, ...]
    prediction: np.ndarray
    truth: np.ndarray


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _manifest_row(sample_id: str, split: str, digit: str) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "source_sample_id": f"source-sample-{digit}",
        "source_image_id": f"source-image-{digit}",
        "field_of_view_id": f"field-{digit}",
        "split": split,
        "image_path": f"images/{sample_id}.tif",
        "mask_path": f"masks/{sample_id}.png",
        "image_sha256": digit * 64,
        "mask_sha256": ("f" * 63 + digit),
        "included": "true",
        "exclusion_reason": "",
    }


def _write_manifest(path: Path) -> Path:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(
            [
                _manifest_row("calibration-1", "calibration", "1"),
                _manifest_row("independent-2", "independent_test", "2"),
            ]
        )
    return path


def _virtual_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability = np.zeros((140, 10), dtype=np.float32)
    truth = np.zeros((140, 10), dtype=bool)
    truth[2:4, 2:4] = True
    probability[truth] = 0.9
    prediction = probability > 0.5
    return probability, prediction, truth


def _policy_payload(
    *,
    threshold_sha256: str,
    min_area_sha256: str,
    threshold: float,
    min_area_px: int,
) -> dict[str, object]:
    return {
        "schema_version": "1",
        "policy_id": "small-b-virtual-policy",
        "policy_version": "1",
        "model_id": "unet-small-balanced-v1",
        "threshold_evidence_sha256": threshold_sha256,
        "min_area_evidence_sha256": min_area_sha256,
        "frozen_scientific_parameters": {
            "threshold": threshold,
            "threshold_comparison": "gt",
            "min_area_px": min_area_px,
            "bottom_crop_px": 130,
        },
        "instance_matching": {
            "metric": "mask_iou",
            "mask_iou_threshold": 0.5,
        },
        "per_image_tolerances": {
            "minimum_instance_precision": 1.0,
            "minimum_instance_recall": 1.0,
            "minimum_instance_f1": 1.0,
            "maximum_count_absolute_error": 0,
            "maximum_count_relative_error": 0.0,
            "maximum_mean_area_relative_error": 0.0,
            "maximum_mean_equivalent_diameter_relative_error": 0.0,
            "maximum_number_density_relative_error": 0.0,
            "maximum_perimeter_density_relative_error": 0.0,
        },
        "not_evaluable_rule": "fail",
        "approval": {
            "frozen_before_independent_test": True,
            "approved_by": "Guo Jinghao",
            "approved_at": "2026-07-24T12:00:00+08:00",
            "rationale": "Frozen virtual Small-B integration-test policy.",
        },
    }


def _build_calibration_chain(tmp_path: Path) -> Workflow:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest_path = _write_manifest(tmp_path / "split-manifest.csv")
    manifest = load_split_manifest(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    probability, prediction, truth = _virtual_arrays()

    threshold_plan_path = _write_json(
        tmp_path / "threshold-plan.json",
        {
            "candidate_thresholds": [0.5],
            "selection_metric": {
                "metric": "instance_f1",
                "direction": "maximize",
            },
            "ordered_tie_break": [],
            "bottom_crop_px": 130,
        },
    )
    threshold_plan = load_threshold_plan(threshold_plan_path)
    probability_calls: list[str] = []

    def calibration_probability(record):
        if record.split.value != "calibration":
            raise AssertionError("Independent Test data reached threshold calibration")
        probability_calls.append(record.sample_id)
        return probability

    def calibration_truth(record):
        if record.split.value != "calibration":
            raise AssertionError("Independent Test GT reached calibration")
        return truth

    threshold_payload = evaluate_threshold_calibration(
        manifest,
        threshold_plan,
        probability_provider=calibration_probability,
        truth_provider=calibration_truth,
        manifest_sha256=manifest_sha,
        plan_sha256=sha256_file(threshold_plan_path),
    )
    threshold_root = tmp_path / "threshold-output"
    write_calibration_outputs(threshold_root, threshold_payload)
    threshold_path = threshold_root / "threshold-calibration.json"
    written_threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
    assert written_threshold["selection_status"] == "SELECTED"
    assert written_threshold["selected_threshold"] == pytest.approx(0.5)
    threshold_sha = sha256_file(threshold_path)

    min_area_plan_path = _write_json(
        tmp_path / "min-area-plan.json",
        {
            "candidate_min_area_px": [0],
            "selection_metric": {
                "metric": "instance_f1",
                "direction": "maximize",
            },
            "ordered_tie_break": [],
            "minimum_gt_retention": 1.0,
            "threshold_evidence_sha256": threshold_sha,
        },
    )
    min_area_plan = load_min_area_plan(min_area_plan_path)
    threshold_evidence = load_threshold_evidence(
        threshold_path,
        expected_sha256=min_area_plan.threshold_evidence_sha256,
    )
    cache_calls: list[str] = []

    def probability_cache(record):
        if record.split.value != "calibration":
            raise AssertionError("Independent Test data reached min-area calibration")
        cache_calls.append(record.sample_id)
        return probability

    min_area_payload = evaluate_min_area_calibration(
        manifest,
        min_area_plan,
        threshold_evidence,
        probability_cache_provider=probability_cache,
        truth_provider=calibration_truth,
        manifest_sha256=manifest_sha,
        plan_sha256=sha256_file(min_area_plan_path),
    )
    min_area_root = tmp_path / "min-area-output"
    write_min_area_outputs(min_area_root, min_area_payload)
    min_area_path = min_area_root / "min-area-calibration.json"
    written_min_area = json.loads(min_area_path.read_text(encoding="utf-8"))
    assert written_min_area["selection_status"] == "SELECTED"
    assert written_min_area["selected_min_area_px"] == 0
    assert written_min_area["threshold_evidence_sha256"] == threshold_sha

    policy_path = _write_json(
        tmp_path / "tolerance-policy.json",
        _policy_payload(
            threshold_sha256=threshold_sha,
            min_area_sha256=sha256_file(min_area_path),
            threshold=float(written_threshold["selected_threshold"]),
            min_area_px=int(written_min_area["selected_min_area_px"]),
        ),
    )
    return Workflow(
        manifest_path=manifest_path,
        threshold_path=threshold_path,
        min_area_path=min_area_path,
        policy_path=policy_path,
        calibration_probability_calls=tuple(probability_calls),
        min_area_cache_calls=tuple(cache_calls),
        prediction=prediction,
        truth=truth,
    )


def _load_contract(workflow: Workflow):
    return load_evaluation_contract(
        split_manifest_path=workflow.manifest_path,
        threshold_evidence_path=workflow.threshold_path,
        min_area_evidence_path=workflow.min_area_path,
        tolerance_policy_path=workflow.policy_path,
        expected_policy_sha256=sha256_file(workflow.policy_path),
    )


def _virtual_analysis_result(tmp_path: Path) -> dict[str, object]:
    root = tmp_path / "virtual-analysis"
    root.mkdir()
    mask_path = root / "pred_mask.png"
    Image.fromarray(np.zeros((140, 10), dtype=np.uint8)).save(mask_path)
    instances_path = _write_json(
        root / "instances.json",
        {
            "coordinate_space": "original_px",
            "width": 10,
            "height": 140,
            "instance_count": 0,
            "instances": [],
        },
    )
    run_config_path = _write_json(
        root / "run-config.json",
        {
            "contract_schema_version": 3,
            "model_id": "unet-small-balanced-v1",
            "model_version": "1",
            "adapter_path": "app.inference.adapters.unet:UNetAdapter",
            "weight_sha256": WEIGHT_SHA,
            "config_sha256": CONFIG_SHA,
            "model_card_sha256": CARD_SHA,
            "adapter_sha256": ADAPTER_SHA,
            "image_sha256": "2" * 64,
            "inference": {
                "threshold": 0.5,
                "min_area_px": 0,
                "watershed_enabled": False,
                "exclude_border": True,
            },
            "resolved_postprocess": {
                "min_area_px": 0,
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
        "run_id": "virtual-small-b-run",
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
        "frozen_inference": {"threshold": 0.5, "min_area_px": 0},
        "resolved_postprocess": {
            "min_area_px": 0,
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


@pytest.mark.integration
def test_virtual_small_b_scientific_workflow_passes_end_to_end(tmp_path: Path) -> None:
    workflow = _build_calibration_chain(tmp_path)
    contract = _load_contract(workflow)
    analysis_result = _virtual_analysis_result(tmp_path)

    smoke = run_scientific_smoke(
        contract,
        expected_policy_sha256=contract.tolerance_policy.sha256,
        sample_ids=["independent-2"],
        analysis_executor=lambda _record, _parameters: analysis_result,
    )
    assert smoke["status"] == "PASS"
    assert smoke["threshold_evidence_sha256"] == sha256_file(workflow.threshold_path)
    assert smoke["min_area_evidence_sha256"] == sha256_file(workflow.min_area_path)

    evaluation = evaluate_independent_test(
        contract,
        expected_policy_sha256=contract.tolerance_policy.sha256,
        prediction_provider=lambda record: (
            workflow.prediction
            if record.sample_id == "independent-2"
            else pytest.fail("non-test prediction requested")
        ),
        truth_provider=lambda record: (
            workflow.truth
            if record.sample_id == "independent-2"
            else pytest.fail("non-test GT requested")
        ),
        evaluation_id="small-b-virtual-evaluation",
    )
    assert evaluation["verdict"] == "PASS"
    assert evaluation["failures"] == []

    output_root = tmp_path / "independent-test-output"
    write_evaluation_outputs(output_root, contract, evaluation)
    metrics_path = output_root / "independent-test-metrics.json"
    verdict_path = output_root / "scientific-verdict.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    assert metrics["verdict"] == verdict["verdict"] == "PASS"
    assert verdict["independent_test_metrics"]["sha256"] == sha256_file(metrics_path)
    assert metrics["artifacts"]["per_image_csv"] == verdict["per_image_csv"]
    assert metrics["artifacts"]["failure_cases_csv"] == verdict["failure_cases_csv"]
    assert metrics["tolerance_policy_sha256"] == sha256_file(workflow.policy_path)


@pytest.mark.integration
def test_evidence_sha_drift_is_rejected(tmp_path: Path) -> None:
    workflow = _build_calibration_chain(tmp_path)
    workflow.threshold_path.write_text(
        workflow.threshold_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="evidence SHA"):
        _load_contract(workflow)


@pytest.mark.integration
def test_frozen_parameter_drift_is_rejected(tmp_path: Path) -> None:
    workflow = _build_calibration_chain(tmp_path)
    min_area = json.loads(workflow.min_area_path.read_text(encoding="utf-8"))
    min_area["selected_min_area_px"] = 4
    _write_json(workflow.min_area_path, min_area)
    policy = json.loads(workflow.policy_path.read_text(encoding="utf-8"))
    policy["min_area_evidence_sha256"] = sha256_file(workflow.min_area_path)
    _write_json(workflow.policy_path, policy)

    with pytest.raises(ValueError, match="min_area_px differs"):
        _load_contract(workflow)


@pytest.mark.integration
def test_independent_test_data_never_reaches_calibration_providers(tmp_path: Path) -> None:
    workflow = _build_calibration_chain(tmp_path)

    assert workflow.calibration_probability_calls == ("calibration-1",)
    assert workflow.min_area_cache_calls == ("calibration-1",)

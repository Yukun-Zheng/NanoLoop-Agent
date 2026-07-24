from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.analysis.instance_artifacts import encode_binary_mask
from scripts.models.evaluate_unet_small_independent_test import evaluate_independent_test
from scripts.models.run_unet_small_independent_test import (
    AnalysisResultBinding,
    ApprovedAnalysisPredictionProvider,
    run_formal_evaluation,
)
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
WEIGHT_SHA = "a" * 64
CONFIG_SHA = "b" * 64
CARD_SHA = "c" * 64
ADAPTER_SHA = "d" * 64
IMAGE_SHAS = ("1" * 64, "2" * 64, "3" * 64)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _mask(index: int) -> np.ndarray:
    mask = np.zeros((150, 20), dtype=np.uint8)
    mask[2 + index : 11 + index, 2:11] = 255
    return mask


def _instances_payload(mask: np.ndarray) -> dict[str, object]:
    boolean = mask != 0
    starts, lengths = encode_binary_mask(boolean)
    return {
        "coordinate_space": "original_px",
        "width": mask.shape[1],
        "height": mask.shape[0],
        "instance_count": 1,
        "instances": [
            {
                "instance_index": 1,
                "bbox_xyxy": [2, 2, 11, 13],
                "area_px": int(boolean.sum()),
                "confidence": 1.0,
                "touches_roi_boundary": False,
                "mask": {
                    "encoding": "flat_rle_v1",
                    "order": "row_major",
                    "starts": starts,
                    "lengths": lengths,
                    "sha256": hashlib.sha256(
                        np.packbits(boolean, bitorder="little").tobytes()
                    ).hexdigest(),
                },
            }
        ],
    }


def _manifest(private_root: Path) -> Path:
    rows: list[dict[str, str]] = []
    for index, sample_id in enumerate(("SrNi-1", "SrNi-2", "SrNi-3")):
        gt_path = private_root / "masks" / f"{sample_id}.png"
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(_mask(index), mode="L").save(gt_path)
        rows.append(
            {
                "sample_id": sample_id,
                "source_sample_id": "SrNi",
                "source_image_id": sample_id,
                "field_of_view_id": f"SrNi-FOV-{index + 1}",
                "split": "independent_test",
                "image_path": f"images/{sample_id}.tif",
                "mask_path": f"masks/{sample_id}.png",
                "image_sha256": IMAGE_SHAS[index],
                "mask_sha256": sha256_file(gt_path),
                "included": "true",
                "exclusion_reason": "",
            }
        )
    path = private_root / "split-manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _contract_files(tmp_path: Path) -> dict[str, Path | str]:
    private_root = tmp_path / "private"
    private_root.mkdir()
    manifest = _manifest(private_root)
    manifest_sha = sha256_file(manifest)
    common = {
        "selection_status": "FROZEN_PREDEFINED",
        "calibration_performed": False,
        "frozen_before_test": True,
        "parameters_changed_after_test": False,
        "independent_test_used_for_tuning": False,
        "frozen_at": "2026-07-24T08:00:00+08:00",
        "manifest_sha256": manifest_sha,
        "model_sha256": "e" * 64,
        "config_sha256": CONFIG_SHA,
        "weight_sha256": WEIGHT_SHA,
        "comparison_rule": "probability > threshold",
        "bottom_crop_px": 130,
    }
    threshold = _write_json(
        private_root / "threshold.json",
        {
            **common,
            "parameter_source": "Frozen before Independent Test.",
            "threshold": 0.30,
        },
    )
    min_area = _write_json(
        private_root / "min-area.json",
        {
            **common,
            "parameter_source": "Frozen before Independent Test.",
            "threshold": 0.30,
            "min_area_px": 64,
            "threshold_evidence_sha256": sha256_file(threshold),
        },
    )
    policy = _write_json(
        private_root / "policy.json",
        {
            "schema_version": "1",
            "policy_id": "small-b-policy",
            "policy_version": "1",
            "model_id": "unet-small-balanced-v1",
            "threshold_evidence_sha256": sha256_file(threshold),
            "min_area_evidence_sha256": sha256_file(min_area),
            "frozen_scientific_parameters": {
                "threshold": 0.30,
                "threshold_comparison": "gt",
                "min_area_px": 64,
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
                "rationale": "Frozen before Independent Test.",
            },
        },
    )
    return {
        "private_root": private_root,
        "manifest": manifest,
        "threshold": threshold,
        "min_area": min_area,
        "policy": policy,
        "policy_sha": sha256_file(policy),
    }


def _analysis_results(tmp_path: Path) -> tuple[Path, list[AnalysisResultBinding]]:
    root = tmp_path / "approved-analysis"
    root.mkdir()
    bindings: list[AnalysisResultBinding] = []
    for index, sample_id in enumerate(("SrNi-1", "SrNi-2", "SrNi-3")):
        run_root = root / sample_id
        run_root.mkdir()
        mask_path = run_root / "pred_mask.png"
        mask = _mask(index)
        Image.fromarray(mask, mode="L").save(mask_path)
        instances_path = _write_json(
            run_root / "instances.json",
            _instances_payload(mask),
        )
        run_config_path = _write_json(
            run_root / "run_config.json",
            {
                "contract_schema_version": 3,
                "provenance_status": "complete",
                "model_id": "unet-small-balanced-v1",
                "model_version": "1",
                "adapter_path": "app.inference.adapters.unet:UNetAdapter",
                "weight_sha256": WEIGHT_SHA,
                "config_sha256": CONFIG_SHA,
                "model_card_sha256": CARD_SHA,
                "adapter_sha256": ADAPTER_SHA,
                "image_sha256": IMAGE_SHAS[index],
                "roi_mode": "full_image",
                "inference": {
                    "threshold": 0.30,
                    "min_area_px": 64,
                    "watershed_enabled": False,
                    "exclude_border": True,
                },
                "resolved_postprocess": {
                    "min_area_px": 64,
                    "fill_holes": True,
                    "watershed_enabled": False,
                    "exclude_border": True,
                    "connectivity": 2,
                },
                "analysis_roi": {
                    "invalid_rects": [
                        {
                            "x1": 0,
                            "y1": 20,
                            "x2": 20,
                            "y2": 150,
                            "reason": "model_bottom_information_bar",
                        }
                    ]
                },
            },
        )
        result_path = _write_json(
            root / f"{sample_id}-analysis-result.json",
            {
                "sample_id": sample_id,
                "source_image_sha256": IMAGE_SHAS[index],
                "run_id": f"run-{index + 1}",
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
                "frozen_inference": {"threshold": 0.30, "min_area_px": 64},
                "resolved_postprocess": {
                    "min_area_px": 64,
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
            },
        )
        bindings.append(AnalysisResultBinding(sample_id, result_path))
    return root, bindings


def _arguments(tmp_path: Path) -> dict[str, object]:
    files = _contract_files(tmp_path)
    analysis_root, bindings = _analysis_results(tmp_path)
    return {
        "split_manifest": files["manifest"],
        "threshold_evidence": files["threshold"],
        "min_area_evidence": files["min_area"],
        "tolerance_policy": files["policy"],
        "expected_policy_sha256": files["policy_sha"],
        "analysis_root": analysis_root,
        "analysis_results": bindings,
        "private_data_root": files["private_root"],
        "output_root": tmp_path / "evaluation",
    }


def _load_contract_for_provider(arguments: dict[str, object]):
    from scripts.models.evaluate_unet_small_independent_test import (
        load_evaluation_contract,
    )

    return load_evaluation_contract(
        split_manifest_path=arguments["split_manifest"],
        threshold_evidence_path=arguments["threshold_evidence"],
        min_area_evidence_path=arguments["min_area_evidence"],
        tolerance_policy_path=arguments["tolerance_policy"],
        expected_policy_sha256=arguments["expected_policy_sha256"],
    )


def test_three_predictions_and_gt_run_frozen_predefined_without_calibration(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)

    result = run_formal_evaluation(**arguments)

    assert result["verdict"] == "PASS"
    assert result["selection_status"] == "FROZEN_PREDEFINED"
    assert len(result["per_image"]) == 3
    assert result["independent_test_used_for_tuning"] is False
    output_root = arguments["output_root"]
    assert (output_root / "prediction-artifact-provenance.json").is_file()
    provenance = json.loads(
        (output_root / "prediction-artifact-provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(provenance["artifacts"]) == {"SrNi-1", "SrNi-2", "SrNi-3"}
    assert all(
        len(item["prediction_mask_sha256"]) == 64
        for item in provenance["artifacts"].values()
    )


def test_prediction_provider_has_no_inference_dependency() -> None:
    module_path = Path(
        __import__(
            "scripts.models.run_unet_small_independent_test",
            fromlist=["__file__"],
        ).__file__
    )
    source = module_path.read_text(encoding="utf-8")

    assert "InferenceGateway" not in source
    assert "TorchScript" not in source
    assert ".predict(" not in source
    assert "probability_path" not in source


def test_missing_prediction_lists_sample_id(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    arguments["analysis_results"] = arguments["analysis_results"][:-1]

    with pytest.raises(ValueError, match="SrNi-3"):
        run_formal_evaluation(**arguments)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sample_id", "different-sample", "sample_id mismatch"),
        ("source_image_sha256", "9" * 64, "source image SHA mismatch"),
    ],
)
def test_prediction_sample_or_source_identity_mismatch(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    arguments = _arguments(tmp_path)
    result_path = arguments["analysis_results"][0].result_path
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload[field] = value
    _write_json(result_path, payload)

    with pytest.raises(ValueError, match=message):
        run_formal_evaluation(**arguments)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_id", "other-model", "model_id mismatch"),
        ("weight_sha256", "9" * 64, "weight_sha256"),
    ],
)
def test_prediction_model_or_weight_identity_mismatch(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    arguments = _arguments(tmp_path)
    result_path = arguments["analysis_results"][0].result_path
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["model"][field] = value
    _write_json(result_path, payload)

    with pytest.raises(ValueError, match=message):
        run_formal_evaluation(**arguments)


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    [
        ("inference", "threshold", 0.31, "threshold"),
        ("inference", "min_area_px", 63, "min_area"),
        ("analysis_roi", "invalid_rects", [], "bottom 130"),
    ],
)
def test_prediction_frozen_parameter_drift(
    tmp_path: Path,
    target: str,
    field: str,
    value: object,
    message: str,
) -> None:
    arguments = _arguments(tmp_path)
    result_path = arguments["analysis_results"][0].result_path
    result = json.loads(result_path.read_text(encoding="utf-8"))
    run_config_path = Path(result["artifacts"]["run_configuration"])
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    run_config[target][field] = value
    _write_json(run_config_path, run_config)

    with pytest.raises(ValueError, match=message):
        run_formal_evaluation(**arguments)


def test_prediction_non_binary_is_rejected(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    result_path = arguments["analysis_results"][0].result_path
    result = json.loads(result_path.read_text(encoding="utf-8"))
    mask_path = Path(result["artifacts"]["mask"])
    mask = _mask(0)
    mask[0, 0] = 2
    Image.fromarray(mask, mode="L").save(mask_path)

    with pytest.raises(ValueError, match="binary"):
        run_formal_evaluation(**arguments)


def test_prediction_and_gt_shape_mismatch_is_rejected(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    contract = _load_contract_for_provider(arguments)
    prediction_provider = ApprovedAnalysisPredictionProvider(
        contract,
        analysis_root=arguments["analysis_root"],
        bindings=arguments["analysis_results"],
    )

    with pytest.raises(ValueError, match="shape mismatch"):
        evaluate_independent_test(
            contract,
            expected_policy_sha256=arguments["expected_policy_sha256"],
            prediction_provider=prediction_provider,
            truth_provider=lambda _record: np.zeros((151, 20), dtype=bool),
        )


def test_gt_sha_mismatch_is_rejected(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    private_root = arguments["private_data_root"]
    gt_path = private_root / "masks" / "SrNi-1.png"
    gt_path.write_bytes(gt_path.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="GT SHA mismatch"):
        run_formal_evaluation(**arguments)


def test_duplicate_prediction_binding_is_rejected(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    bindings = arguments["analysis_results"]
    arguments["analysis_results"] = [*bindings, bindings[0]]

    with pytest.raises(ValueError, match="duplicate prediction"):
        run_formal_evaluation(**arguments)


def test_output_root_must_not_exist(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    arguments["output_root"].mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        run_formal_evaluation(**arguments)


def test_expected_policy_sha_is_still_enforced(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    arguments["expected_policy_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="policy SHA"):
        run_formal_evaluation(**arguments)


def test_provider_rejects_duplicate_canonical_mask(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    second_result_path = arguments["analysis_results"][1].result_path
    second = json.loads(second_result_path.read_text(encoding="utf-8"))
    first_result_path = arguments["analysis_results"][0].result_path
    first = json.loads(first_result_path.read_text(encoding="utf-8"))
    second["artifacts"]["mask"] = first["artifacts"]["mask"]
    _write_json(second_result_path, second)

    with pytest.raises(ValueError, match="duplicate canonical prediction"):
        run_formal_evaluation(**arguments)


def test_runner_does_not_expose_tuning_or_frozen_parameter_overrides() -> None:
    from scripts.models.run_unet_small_independent_test import build_parser

    options = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
    }
    assert "--threshold" not in options
    assert "--min-area-px" not in options
    assert "--tolerance" not in options
    assert "--calibration" not in options

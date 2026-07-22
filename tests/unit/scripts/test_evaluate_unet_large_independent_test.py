from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from app.contracts.analysis_config import MorphometryConfig, PostprocessProfile
from scripts.models.evaluate_unet_large_independent_test import (
    ADAPTER_PATH,
    ADAPTER_SHA256,
    BOTTOM_CROP_PX,
    CONFIG_SHA256,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MIN_AREA_PX,
    MODEL_CARD_SHA256,
    MODEL_ID,
    MODEL_VERSION,
    SCALE_NM_PER_PIXEL,
    SEED,
    TEST_FILENAMES,
    THRESHOLD,
    TORCHSCRIPT_SHA256,
    VALID_HEIGHT,
    EvaluationParameters,
    _load_foreground,
    _validate_output_root,
    compute_metrics,
    compute_scientific_metrics,
    run_evaluation,
)

BUNDLE_ID = "e" * 64
BUILD = {
    "application_version": "test",
    "git_commit": "test",
    "docker_image_tag": "test",
    "python_version": "3.12.0",
    "dependency_contract_sha256": "1" * 64,
    "installed_dependencies_sha256": "2" * 64,
    "application_source_sha256": "3" * 64,
}


def _run_config(image_sha256: str) -> dict[str, Any]:
    return {
        "contract_schema_version": 3,
        "provenance_status": "complete",
        "provenance_warnings": [],
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "adapter_path": ADAPTER_PATH,
        "weight_sha256": TORCHSCRIPT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "model_card_sha256": MODEL_CARD_SHA256,
        "adapter_sha256": ADAPTER_SHA256,
        "image_sha256": image_sha256,
        "roi_mode": "full_image",
        "review_source": "model_inference",
        "model_bundle": {
            "schema_version": 1,
            "bundle_id": BUNDLE_ID,
            "manifest_ref": f"bundles/{BUNDLE_ID}/manifest.json",
            "weight_ref": f"{TORCHSCRIPT_SHA256}/weights.pt",
            "config_ref": f"{CONFIG_SHA256}/config.yaml",
            "model_card_ref": f"{MODEL_CARD_SHA256}/model-card.md",
            "adapter_ref": f"{ADAPTER_SHA256}/adapter.py",
            "adapter_sha256": ADAPTER_SHA256,
        },
        "execution_build": BUILD,
        "scale_nm_per_pixel": SCALE_NM_PER_PIXEL,
        "inference": {
            "threshold": THRESHOLD,
            "min_area_px": MIN_AREA_PX,
            "watershed_enabled": False,
            "exclude_border": True,
            "device": "cpu",
            "seed": 2026,
        },
        "resolved_postprocess": {
            "profile_id": "semantic-mask-v1",
            "min_area_px": MIN_AREA_PX,
            "fill_holes": True,
            "watershed_enabled": False,
            "exclude_border": True,
            "connectivity": 2,
            "instance_iou_threshold": 0.7,
        },
        "resolved_morphometry": {"perimeter_neighborhood": 8},
        "analysis_roi": {
            "valid_rect": {"x1": 0, "y1": 0, "x2": IMAGE_WIDTH, "y2": IMAGE_HEIGHT},
            "invalid_rects": [
                {
                    "x1": 0,
                    "y1": VALID_HEIGHT,
                    "x2": IMAGE_WIDTH,
                    "y2": IMAGE_HEIGHT,
                    "reason": "model_bottom_information_bar",
                }
            ],
        },
    }


def _execution_provenance() -> dict[str, Any]:
    return {
        "contract_schema_version": 1,
        "executor_build": BUILD,
        "build_identity_matches_contract": True,
        "requested_device": "cpu",
        "actual_device": "cpu",
        "seed": SEED,
        "python_random_seeded": True,
        "numpy_random_seeded": True,
        "torch_deterministic_algorithms": True,
        "global_inference_serialized": True,
        "backend": "app.inference.adapters.unet.UNetAdapter",
        "model_bundle_id": BUNDLE_ID,
        "adapter_sha256": ADAPTER_SHA256,
        "warnings": [],
        "executed_at": "2026-07-20T10:00:00+00:00",
    }


def _write_mask(path: Path, foreground: list[tuple[int, int]] = ()) -> None:
    pixels = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    for x, y in foreground:
        pixels[y, x] = 255
    Image.fromarray(pixels).save(path)


def _write_tolerance_policy(
    root: Path,
    *,
    overrides: dict[str, float] | None = None,
    iou_threshold: float = 0.5,
) -> Path:
    tolerances = {
        "minimum_instance_precision": 0.0,
        "minimum_instance_recall": 0.0,
        "minimum_instance_f1": 0.0,
        "maximum_count_absolute_error": 1_000_000.0,
        "maximum_count_relative_error": 1_000_000.0,
        "maximum_mean_area_relative_error": 1_000_000.0,
        "maximum_mean_equivalent_diameter_relative_error": 1_000_000.0,
        "maximum_number_density_relative_error": 1_000_000.0,
        "maximum_perimeter_density_relative_error": 1_000_000.0,
    }
    if overrides:
        tolerances.update(overrides)
    path = root / "tolerance-policy.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "policy_id": "test-large-policy",
                "policy_version": "test",
                "instance_matching": {
                    "metric": "mask_iou",
                    "mask_iou_threshold": iou_threshold,
                },
                "per_image_tolerances": tolerances,
                "ground_truth_count_zero_rule": (
                    "relative_error_denominator_is_maximum_of_gt_count_and_one"
                ),
            }
        ),
        encoding="utf-8",
    )
    return path


def _scientific_arrays(
    prediction_regions: list[tuple[slice, slice]],
    truth_regions: list[tuple[slice, slice]],
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.zeros((40, 40), dtype=bool)
    truth = np.zeros_like(prediction)
    for rows, columns in prediction_regions:
        prediction[rows, columns] = True
    for rows, columns in truth_regions:
        truth[rows, columns] = True
    return prediction, truth


def _scientific_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    return compute_scientific_metrics(
        prediction,
        truth,
        profile=PostprocessProfile(
            profile_id="test",
            min_area_px=0,
            fill_holes=False,
            watershed_enabled=False,
            exclude_border=False,
            connectivity=2,
        ),
        morphometry=MorphometryConfig(perimeter_neighborhood=8),
        scale_nm_per_pixel=SCALE_NM_PER_PIXEL,
        iou_threshold=iou_threshold,
        sample_id="test",
    )


def _write_fixture(
    root: Path,
    *,
    prediction_points: dict[str, list[tuple[int, int]]] | None = None,
    truth_points: dict[str, list[tuple[int, int]]] | None = None,
) -> tuple[Path, Path]:
    analysis_root = root / "analysis"
    mask_dir = root / "truth"
    mask_dir.mkdir()
    _write_tolerance_policy(root)
    prediction_points = prediction_points or {}
    truth_points = truth_points or {}
    for index, filename in enumerate(TEST_FILENAMES):
        sample_id = Path(filename).stem
        image_sha256 = f"{index + 4}" * 64
        image_dir = analysis_root / "artifacts" / "job_test" / "images" / f"img_{index}"
        run_dir = image_dir / "runs" / f"run_{index}"
        run_dir.mkdir(parents=True)
        (image_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": "1",
                    "image_id": f"img_{index}",
                    "job_id": "job_test",
                    "filename": filename,
                    "sample_id": sample_id,
                    "width": IMAGE_WIDTH,
                    "height": IMAGE_HEIGHT,
                    "sha256": image_sha256,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "run_config.json").write_text(
            json.dumps(_run_config(image_sha256)), encoding="utf-8"
        )
        (run_dir / "execution_provenance.json").write_text(
            json.dumps(_execution_provenance()), encoding="utf-8"
        )
        _write_mask(run_dir / "pred_mask.png", prediction_points.get(sample_id, []))
        _write_mask(mask_dir / filename, truth_points.get(sample_id, []))
    return analysis_root, mask_dir


def _evaluation_parameters(
    analysis_root: Path,
    mask_dir: Path,
    output_root: Path,
) -> EvaluationParameters:
    return EvaluationParameters(
        analysis_root,
        mask_dir,
        output_root,
        output_root.parent / "tolerance-policy.json",
    )


def test_bottom_180_pixels_are_excluded_and_metrics_are_correct(tmp_path: Path) -> None:
    path = tmp_path / "mask.png"
    _write_mask(path, [(1, 1), (2, VALID_HEIGHT - 1), (3, VALID_HEIGHT)])

    foreground = _load_foreground(path, sample_id="sample", kind="test")

    assert foreground.shape == (VALID_HEIGHT, IMAGE_WIDTH)
    assert foreground[1, 1]
    assert foreground[VALID_HEIGHT - 1, 2]
    assert foreground.sum() == 2
    assert IMAGE_HEIGHT - VALID_HEIGHT == BOTTOM_CROP_PX

    prediction = np.zeros((VALID_HEIGHT, IMAGE_WIDTH), dtype=bool)
    truth = np.zeros_like(prediction)
    prediction[1, 1] = True
    prediction[2, 2] = True
    truth[1, 1] = True
    truth[3, 3] = True
    metrics = compute_metrics(prediction, truth)
    assert metrics.tp == 1
    assert metrics.fp == 1
    assert metrics.fn == 1
    assert metrics.tn == IMAGE_WIDTH * VALID_HEIGHT - 3
    assert metrics.dice == pytest.approx(0.5)
    assert metrics.iou == pytest.approx(1 / 3)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)


def test_rejects_mask_dimension_mismatch(tmp_path: Path) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    Image.new("L", (IMAGE_WIDTH - 1, IMAGE_HEIGHT)).save(mask_dir / TEST_FILENAMES[0])

    with pytest.raises(ValueError, match="dimensions are not 2048x1536"):
        run_evaluation(
            _evaluation_parameters(analysis_root, mask_dir, tmp_path / "evaluation-output")
        )


def test_rejects_existing_and_repository_output_roots(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="must not already exist"):
        _validate_output_root(existing)
    repository_output = Path(__file__).resolve().parents[3] / "forbidden-evaluation"
    with pytest.raises(ValueError, match="outside the Git repository"):
        _validate_output_root(repository_output)


def test_three_sample_summary_and_artifacts(tmp_path: Path) -> None:
    points = {
        "SrZr-3": [(10, 10)],
        "BaCu-2": [(20, 20), (21, 21)],
        "PrCu-3": [(30, 30), (31, 31), (32, 32)],
    }
    analysis_root, mask_dir = _write_fixture(
        tmp_path, prediction_points=points, truth_points=points
    )
    output_root = tmp_path / "evaluation-output"

    result = run_evaluation(_evaluation_parameters(analysis_root, mask_dir, output_root))

    assert [item["sample_id"] for item in result["per_image"]] == [
        "SrZr-3",
        "BaCu-2",
        "PrCu-3",
    ]
    assert result["macro_average"] == {
        "dice": 1.0,
        "iou": 1.0,
        "precision": 1.0,
        "recall": 1.0,
    }
    assert result["micro_average"]["tp"] == 6
    assert result["micro_average"]["fp"] == 0
    assert result["micro_average"]["fn"] == 0
    assert result["evaluation_region"]["evaluated_pixels_per_image"] == (
        IMAGE_WIDTH * VALID_HEIGHT
    )
    assert (output_root / "metrics.json").is_file()
    assert result["overall_status"] == "PASS"
    assert result["failures"] == []
    assert (output_root / "scientific-metrics.csv").is_file()
    assert (output_root / "failure-cases.csv").is_file()
    with (output_root / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["scope"] for row in rows] == [
        "SrZr-3",
        "BaCu-2",
        "PrCu-3",
        "macro_average",
        "micro_average",
    ]
    for filename in TEST_FILENAMES:
        sample_id = Path(filename).stem
        review = output_root / f"{sample_id}_gt_pred_error.png"
        assert review.is_file()
        with Image.open(review) as image:
            assert image.size == (IMAGE_WIDTH * 3, VALID_HEIGHT)


def test_scientific_metrics_fully_match_instances_and_morphometry() -> None:
    prediction, truth = _scientific_arrays(
        [(slice(5, 10), slice(5, 10))],
        [(slice(5, 10), slice(5, 10))],
    )

    metrics = _scientific_metrics(prediction, truth)

    assert metrics["instance_matching"]["matched_count"] == 1
    assert metrics["instance_matching"]["precision"] == 1.0
    assert metrics["instance_matching"]["recall"] == 1.0
    assert metrics["instance_matching"]["f1"] == 1.0
    assert metrics["count"]["prediction_count"] == 1
    assert metrics["count"]["ground_truth_count"] == 1
    assert metrics["count"]["absolute_error"] == 0
    assert metrics["count"]["relative_error"] == 0.0
    assert metrics["morphometry"]["mean_area_relative_error"] == 0.0
    assert metrics["morphometry"]["mean_equivalent_diameter_absolute_error_px"] == 0.0
    assert metrics["morphometry"]["mean_equivalent_diameter_relative_error"] == 0.0
    assert metrics["morphometry"]["number_density_relative_error"] == 0.0
    assert metrics["morphometry"]["perimeter_density_relative_error"] == 0.0


def test_scientific_metrics_reports_missed_and_false_positive_instances() -> None:
    missed_prediction, missed_truth = _scientific_arrays(
        [], [(slice(5, 10), slice(5, 10))]
    )
    missed = _scientific_metrics(missed_prediction, missed_truth)
    assert missed["instance_matching"]["false_negative_count"] == 1
    assert missed["instance_matching"]["recall"] == 0.0
    assert missed["count"]["absolute_error"] == 1

    false_positive_prediction, false_positive_truth = _scientific_arrays(
        [(slice(5, 10), slice(5, 10))], []
    )
    false_positive = _scientific_metrics(false_positive_prediction, false_positive_truth)
    assert false_positive["instance_matching"]["false_positive_count"] == 1
    assert false_positive["instance_matching"]["precision"] == 0.0
    assert false_positive["count"]["ground_truth_count"] == 0
    assert false_positive["count"]["relative_error"] == 1.0
    assert false_positive["morphometry"]["mean_area_relative_error"] is None
    assert false_positive["morphometry"]["not_evaluable_reason"] == "ground_truth_count_zero"


def test_instance_matching_includes_iou_threshold_boundary() -> None:
    prediction, truth = _scientific_arrays(
        [(slice(5, 7), slice(5, 7))],
        [(slice(5, 7), slice(5, 9))],
    )

    at_threshold = _scientific_metrics(prediction, truth, iou_threshold=0.5)
    above_threshold = _scientific_metrics(prediction, truth, iou_threshold=0.500001)

    assert at_threshold["instance_matching"]["matched_count"] == 1
    assert above_threshold["instance_matching"]["matched_count"] == 0


def _write_large_component(path: Path, *, x: int, y: int) -> None:
    pixels = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    pixels[y : y + 30, x : x + 30] = 255
    Image.fromarray(pixels).save(path)


def test_policy_failures_and_failure_cases_csv(tmp_path: Path) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    prediction = next(analysis_root.glob("artifacts/job_*/images/img_0/runs/run_0/pred_mask.png"))
    _write_large_component(prediction, x=100, y=100)
    _write_tolerance_policy(
        tmp_path,
        overrides={
            "minimum_instance_precision": 1.0,
            "minimum_instance_recall": 1.0,
            "minimum_instance_f1": 1.0,
            "maximum_count_absolute_error": 0.0,
            "maximum_count_relative_error": 0.0,
        },
    )
    output_root = tmp_path / "evaluation-output"

    result = run_evaluation(_evaluation_parameters(analysis_root, mask_dir, output_root))

    assert result["overall_status"] == "FAIL"
    sample = result["per_image"][0]
    assert sample["tolerance_assessment"]["status"] == "FAIL"
    assert {failure["metric"] for failure in sample["tolerance_assessment"]["failures"]} == {
        "instance_precision",
        "instance_f1",
        "count_absolute_error",
        "count_relative_error",
    }
    with (output_root / "failure-cases.csv").open(encoding="utf-8", newline="") as stream:
        failures = list(csv.DictReader(stream))
    assert len(failures) == 4
    assert {row["sample_id"] for row in failures} == {"SrZr-3"}
    assert {row["reason_code"] for row in failures} == {"TOLERANCE_NOT_MET"}


def test_single_tolerance_failure_is_reported(tmp_path: Path) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    prediction = next(analysis_root.glob("artifacts/job_*/images/img_0/runs/run_0/pred_mask.png"))
    _write_large_component(prediction, x=100, y=100)
    _write_tolerance_policy(tmp_path, overrides={"maximum_count_absolute_error": 0.0})

    result = run_evaluation(
        _evaluation_parameters(analysis_root, mask_dir, tmp_path / "evaluation-output")
    )

    assert result["overall_status"] == "FAIL"
    assert result["per_image"][0]["tolerance_assessment"]["failures"] == [
        {
            "metric": "count_absolute_error",
            "operator": "<=",
            "observed": 1.0,
            "tolerance": 0.0,
            "reason_code": "TOLERANCE_NOT_MET",
        }
    ]


def test_missing_tolerance_policy_fails_before_output_creation(tmp_path: Path) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    output_root = tmp_path / "evaluation-output"

    with pytest.raises(ValueError, match="tolerance-policy is required"):
        run_evaluation(EvaluationParameters(analysis_root, mask_dir, output_root))

    assert not output_root.exists()


def test_rejects_duplicate_prediction_for_a_fixed_sample(tmp_path: Path) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    first_image = analysis_root / "artifacts" / "job_test" / "images" / "img_0"
    duplicate = first_image / "runs" / "run_duplicate"
    duplicate.mkdir()
    _write_mask(duplicate / "pred_mask.png")
    (duplicate / "run_config.json").write_text(
        json.dumps(_run_config("4" * 64)), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=r"multiple pred_mask\.png"):
        run_evaluation(
            _evaluation_parameters(analysis_root, mask_dir, tmp_path / "evaluation-output")
        )


def test_rejects_prediction_outside_formal_analysis_layout(tmp_path: Path) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    informal = analysis_root / "legacy" / "pred_mask.png"
    informal.parent.mkdir()
    _write_mask(informal)

    with pytest.raises(ValueError, match="outside the formal Analysis layout"):
        run_evaluation(
            _evaluation_parameters(analysis_root, mask_dir, tmp_path / "evaluation-output")
        )


def test_rejects_nonzero_prediction_in_bottom_exclusion(tmp_path: Path) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    prediction = next(analysis_root.glob("artifacts/job_*/images/img_*/runs/run_*/pred_mask.png"))
    _write_mask(prediction, [(10, VALID_HEIGHT)])

    with pytest.raises(ValueError, match="bottom 180 px is not zero"):
        run_evaluation(
            _evaluation_parameters(analysis_root, mask_dir, tmp_path / "evaluation-output")
        )


@pytest.mark.parametrize(
    "field",
    ["weight_sha256", "config_sha256", "model_card_sha256", "adapter_sha256"],
)
def test_rejects_wrong_frozen_asset_identity(tmp_path: Path, field: str) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    run_config_path = next(analysis_root.rglob("run_config.json"))
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    run_config[field] = "0" * 64
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen Large asset"):
        run_evaluation(
            _evaluation_parameters(analysis_root, mask_dir, tmp_path / "evaluation-output")
        )


def test_rejects_missing_or_mismatched_execution_provenance(tmp_path: Path) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    execution_path = next(analysis_root.rglob("execution_provenance.json"))
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["model_bundle_id"] = "0" * 64
    execution_path.write_text(json.dumps(execution), encoding="utf-8")

    with pytest.raises(ValueError, match="model_bundle_id differs"):
        run_evaluation(
            _evaluation_parameters(analysis_root, mask_dir, tmp_path / "evaluation-output")
        )

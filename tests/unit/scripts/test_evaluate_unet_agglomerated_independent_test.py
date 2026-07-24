from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from app.analysis.config import MorphometryConfig, PostprocessProfile
from app.analysis.postprocessing import NormalizedInstance
from scripts.models import (
    evaluate_unet_agglomerated_independent_test as agglomerated_evaluation_module,
)
from scripts.models import evaluate_unet_large_independent_test as large_evaluation_module
from scripts.models.evaluate_unet_agglomerated_independent_test import (
    ADAPTER_PATH,
    ADAPTER_SHA256,
    BOTTOM_CROP_PX,
    CHECKPOINT_SHA256,
    CONFIG_SHA256,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MIN_AREA_PX,
    MODEL_CARD_SHA256,
    MODEL_ID,
    MODEL_VERSION,
    SCALE_NM_PER_PIXEL,
    SEED,
    THRESHOLD,
    TORCHSCRIPT_SHA256,
    VALID_HEIGHT,
    _load_foreground,
    _match_instances,
    _statistical_errors,
    _validate_output_root,
    compute_metrics,
    compute_scientific_metrics,
    run_evaluation,
)
from scripts.models.evaluate_unet_agglomerated_independent_test import (
    EvaluationParameters as AgglomeratedEvaluationParameters,
)

BUNDLE_ID = "e" * 64
TEST_FILENAMES = ("YCu-1.tif", "YCu-2.tif", "YCu-3.tif")
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
            "threshold_comparison": "gte",
            "min_area_px": MIN_AREA_PX,
            "watershed_enabled": False,
            "exclude_border": True,
            "device": "cpu",
            "seed": SEED,
        },
        "resolved_postprocess": {
            "profile_id": "semantic-agglomerate-mask-v1",
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


def _write_mask(
    path: Path,
    foreground: Sequence[tuple[int, int]] = (),
) -> None:
    pixels = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    for x, y in foreground:
        pixels[y, x] = 255
    Image.fromarray(pixels).save(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def EvaluationParameters(
    analysis_output_root: Path,
    mask_dir: Path,
    output_root: Path,
    tolerance_policy_path: Path | None = None,
    instance_iou_threshold: float | None = None,
    checkpoint_sha256: str = CHECKPOINT_SHA256,
) -> AgglomeratedEvaluationParameters:
    return AgglomeratedEvaluationParameters(
        analysis_output_root=analysis_output_root,
        mask_dir=mask_dir,
        output_root=output_root,
        independent_test_manifest_path=mask_dir.parent / "split-manifest.csv",
        checkpoint_sha256=checkpoint_sha256,
        tolerance_policy_path=tolerance_policy_path,
        instance_iou_threshold=instance_iou_threshold,
    )


def _write_tolerance_policy(
    root: Path,
    *,
    overrides: dict[str, float] | None = None,
    iou_threshold: float = 0.5,
    approval_status: str = "APPROVED",
    requirement_overrides: dict[str, str] | None = None,
    identity_overrides: dict[str, str] | None = None,
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
        "maximum_coverage_relative_error": 1_000_000.0,
    }
    if overrides:
        tolerances.update(overrides)
    requirements = requirement_overrides or {}
    rules = (
        ("instance_precision", "minimum_instance_precision", "gte"),
        ("instance_recall", "minimum_instance_recall", "gte"),
        ("instance_f1", "minimum_instance_f1", "gte"),
        ("count_absolute_error", "maximum_count_absolute_error", "lte"),
        ("count_relative_error", "maximum_count_relative_error", "lte"),
        ("mean_area_relative_error", "maximum_mean_area_relative_error", "lte"),
        (
            "mean_equivalent_diameter_relative_error",
            "maximum_mean_equivalent_diameter_relative_error",
            "lte",
        ),
        ("number_density_relative_error", "maximum_number_density_relative_error", "lte"),
        (
            "perimeter_density_relative_error",
            "maximum_perimeter_density_relative_error",
            "lte",
        ),
        ("coverage_relative_error", "maximum_coverage_relative_error", "lte"),
    )
    identities = {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "torchscript_sha256": TORCHSCRIPT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "model_card_sha256": MODEL_CARD_SHA256,
        "adapter_sha256": ADAPTER_SHA256,
        "independent_test_manifest_sha256": _sha256(root / "split-manifest.csv"),
    }
    identities.update(identity_overrides or {})
    path = root / "tolerance-policy.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "policy_id": "test-agglomerated-policy",
                "policy_version": "test",
                "model_id": MODEL_ID,
                **identities,
                "frozen_scientific_parameters": {
                    "threshold": THRESHOLD,
                    "threshold_comparison": "gte",
                    "min_area_px": MIN_AREA_PX,
                    "bottom_crop_px": BOTTOM_CROP_PX,
                },
                "instance_matching": {
                    "metric": "mask_iou",
                    "mask_iou_threshold": iou_threshold,
                },
                "per_image_tolerances": tolerances,
                "metric_contracts": [
                    {
                        "metric_name": metric,
                        "aggregation_scope": "per-sample",
                        "operator": operator,
                        "tolerance_field": tolerance_field,
                        "tolerance": tolerances[tolerance_field],
                        "requirement": requirements.get(metric, "required"),
                        "applicability": (
                            "always"
                            if metric.startswith("instance_") or metric.startswith("count_")
                            else "ground_truth_and_prediction_instances_present"
                        ),
                        "undefined_rule": "fail_if_required_otherwise_not_evaluated",
                        "zero_denominator_rule": (
                            {
                                "instance_precision": ("value_is_one_when_no_prediction_instances"),
                                "instance_recall": ("value_is_one_when_no_ground_truth_instances"),
                                "instance_f1": (
                                    "value_is_one_when_prediction_and_ground_truth_are_both_empty"
                                ),
                            }[metric]
                            if metric.startswith("instance_")
                            else (
                                "denominator_is_maximum_of_ground_truth_count_and_one"
                                if metric == "count_relative_error"
                                else (
                                    "not_applicable_metric_has_no_denominator"
                                    if metric == "count_absolute_error"
                                    else (
                                        "not_evaluable_when_ground_truth_value_is_zero_or_"
                                        "either_instance_set_is_empty"
                                    )
                                )
                            )
                        ),
                        "reason_codes": {
                            "tolerance_met": "TOLERANCE_MET",
                            "tolerance_not_met": "TOLERANCE_NOT_MET",
                            "not_evaluable": "METRIC_NOT_EVALUABLE",
                            "not_applicable": "METRIC_NOT_APPLICABLE",
                        },
                    }
                    for metric, tolerance_field, operator in rules
                ],
                "ground_truth_count_zero_rule": (
                    "relative_error_denominator_is_maximum_of_gt_count_and_one"
                ),
                "approval": {
                    "status": approval_status,
                    "frozen_before_independent_test": approval_status == "APPROVED",
                    "approved_by": "unit-test",
                    "approved_at": "2026-07-20T00:00:00+00:00",
                    "rationale": "unit-test policy",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _scientific_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    iou_threshold: float = 0.5,
    min_area_px: int = 0,
) -> dict[str, Any]:
    return compute_scientific_metrics(
        prediction,
        truth,
        profile=PostprocessProfile(
            profile_id="test",
            min_area_px=min_area_px,
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
    filenames: tuple[str, ...] = TEST_FILENAMES,
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    analysis_root = root / "analysis"
    mask_dir = root / "truth"
    image_dir_root = root / "images"
    mask_dir.mkdir()
    image_dir_root.mkdir()
    prediction_points = prediction_points or {}
    truth_points = truth_points or {}
    manifest_rows: list[dict[str, str]] = []
    for index, filename in enumerate(filenames):
        sample_id = Path(filename).stem
        image_path = image_dir_root / filename
        image_path.write_bytes(f"unit-test-image-{sample_id}".encode())
        image_sha256 = _sha256(image_path)
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
        truth_path = mask_dir / f"{sample_id}_mask.tif"
        _write_mask(truth_path, truth_points.get(sample_id, []))
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "source_sample_id": f"source-sample-{index}",
                "source_image_id": f"source-image-{index}",
                "field_of_view_id": sample_id,
                "split": "independent_test",
                "image_path": f"images/{filename}",
                "mask_path": f"truth/{truth_path.name}",
                "image_sha256": image_sha256,
                "mask_sha256": _sha256(truth_path),
                "included": "true",
                "exclusion_reason": "",
            }
        )
    with (root / "split-manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return analysis_root, mask_dir


def _manifest_rows(root: Path) -> list[dict[str, str]]:
    with (root / "split-manifest.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_manifest_rows(root: Path, rows: Sequence[dict[str, str]]) -> None:
    with (root / "split-manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_bottom_130_pixels_are_excluded_and_metrics_are_correct(tmp_path: Path) -> None:
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
    assert metrics.f1 == pytest.approx(0.5)
    assert metrics.f1 == metrics.dice


def test_statistical_error_formula() -> None:
    prediction = {
        "agglomerate_count": 12,
        "mean_equivalent_diameter_nm": 8.0,
        "number_density_um2": 4.0,
        "perimeter_density_um": 2.0,
        "coverage_ratio": 0.3,
    }
    truth = {
        "agglomerate_count": 10,
        "mean_equivalent_diameter_nm": 10.0,
        "number_density_um2": 4.0,
        "perimeter_density_um": 1.0,
        "coverage_ratio": 0.2,
    }

    errors = _statistical_errors(prediction, truth)

    assert errors["agglomerate_count"] == {
        "prediction": 12,
        "ground_truth": 10,
        "signed_error": 2.0,
        "absolute_error": 2.0,
        "relative_error": pytest.approx(0.2),
        "absolute_percentage_error": pytest.approx(20.0),
    }
    assert errors["mean_equivalent_diameter_nm"]["relative_error"] == pytest.approx(-0.2)
    assert errors["number_density_um2"]["absolute_percentage_error"] == 0


def test_rejects_mask_dimension_mismatch(tmp_path: Path) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    Image.new("L", (IMAGE_WIDTH - 1, IMAGE_HEIGHT)).save(mask_dir / "YCu-1_mask.tif")
    rows = _manifest_rows(tmp_path)
    rows[0]["mask_sha256"] = _sha256(mask_dir / "YCu-1_mask.tif")
    _write_manifest_rows(tmp_path, rows)

    with pytest.raises(ValueError, match="dimensions are not 2048x1536"):
        run_evaluation(
            EvaluationParameters(analysis_root, mask_dir, tmp_path / "evaluation-output")
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
        "YCu-1": [(10, 10)],
        "YCu-2": [(20, 20), (21, 21)],
        "YCu-3": [(30, 30), (31, 31), (32, 32)],
    }
    analysis_root, mask_dir = _write_fixture(
        tmp_path, prediction_points=points, truth_points=points
    )
    output_root = tmp_path / "evaluation-output"

    result = run_evaluation(EvaluationParameters(analysis_root, mask_dir, output_root))

    assert [item["sample_id"] for item in result["per_image"]] == [
        "YCu-1",
        "YCu-2",
        "YCu-3",
    ]
    assert result["macro_average"] == {
        "dice": 1.0,
        "iou": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    assert result["micro_average"]["tp"] == 6
    assert result["micro_average"]["fp"] == 0
    assert result["micro_average"]["fn"] == 0
    assert result["evaluation_region"]["evaluated_pixels_per_image"] == (IMAGE_WIDTH * VALID_HEIGHT)
    assert (output_root / "metrics.json").is_file()
    assert (output_root / "statistics.csv").is_file()
    assert (output_root / "failure-cases.csv").is_file()
    assert (output_root / "evidence-manifest.json").is_file()
    assert result["overall_status"] == "NOT_EVALUATED"
    with (output_root / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["scope"] for row in rows] == [
        "YCu-1",
        "YCu-2",
        "YCu-3",
        "macro_average",
        "micro_average",
    ]
    for filename in TEST_FILENAMES:
        sample_id = Path(filename).stem
        review = output_root / f"{sample_id}_gt_pred_error.png"
        assert review.is_file()
        with Image.open(review) as image:
            assert image.size == (IMAGE_WIDTH * 3, VALID_HEIGHT)
    for item in result["per_image"]:
        assert item["prediction_statistics"] == item["ground_truth_statistics"]
        assert all(error["absolute_error"] == 0 for error in item["statistical_errors"].values())
    assert all(
        summary["mean_absolute_percentage_error"] == 0
        for summary in result["macro_statistical_errors"].values()
    )
    assert [Path(item["input_provenance"]["truth_path"]).name for item in result["per_image"]] == [
        "YCu-1_mask.tif",
        "YCu-2_mask.tif",
        "YCu-3_mask.tif",
    ]
    assert result["metric_definitions"]["pixel_f1_equals_dice"] is True
    assert all(row["f1"] == "1.0" for row in rows)


def test_ground_truth_small_target_is_not_filtered_by_model_min_area() -> None:
    prediction = np.zeros((40, 40), dtype=bool)
    truth = np.zeros_like(prediction)
    truth[10:12, 10:12] = True

    metrics = _scientific_metrics(
        prediction,
        truth,
        min_area_px=MIN_AREA_PX,
    )

    matching = metrics["instance_matching"]
    assert matching["prediction_min_area_px"] == MIN_AREA_PX
    assert matching["ground_truth_min_area_px"] == 0
    assert matching["ground_truth_count"] == 1
    assert matching["false_negative_count"] == 1
    assert matching["recall"] == 0.0


def test_instance_iou_threshold_boundary_and_instance_metrics() -> None:
    prediction = np.zeros((40, 40), dtype=bool)
    truth = np.zeros_like(prediction)
    prediction[5:7, 5:7] = True
    truth[5:7, 5:9] = True

    at_threshold = _scientific_metrics(prediction, truth, iou_threshold=0.5)
    above_threshold = _scientific_metrics(prediction, truth, iou_threshold=0.500001)

    matching = at_threshold["instance_matching"]
    assert matching["matched_count"] == 1
    assert matching["precision"] == 1.0
    assert matching["recall"] == 1.0
    assert matching["f1"] == 1.0
    assert above_threshold["instance_matching"]["matched_count"] == 0


def test_instance_matching_maximizes_cardinality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    masks = [np.zeros((1, 1), dtype=bool) for _ in range(4)]
    prediction_instances = [
        NormalizedInstance(0, masks[0], (0, 0, 1, 1), 1, None, False),
        NormalizedInstance(1, masks[1], (0, 0, 1, 1), 1, None, False),
    ]
    truth_instances = [
        NormalizedInstance(0, masks[2], (0, 0, 1, 1), 1, None, False),
        NormalizedInstance(1, masks[3], (0, 0, 1, 1), 1, None, False),
    ]
    scores = {
        (id(masks[0]), id(masks[2])): 0.90,
        (id(masks[0]), id(masks[3])): 0.80,
        (id(masks[1]), id(masks[2])): 0.85,
        (id(masks[1]), id(masks[3])): 0.00,
    }

    monkeypatch.setattr(
        large_evaluation_module,
        "_mask_iou",
        lambda prediction, truth: scores[(id(prediction), id(truth))],
    )

    matches = _match_instances(
        prediction_instances,
        truth_instances,
        iou_threshold=0.5,
    )

    assert len(matches) == 2
    assert {
        (item["prediction_instance_index"], item["ground_truth_instance_index"]) for item in matches
    } == {(0, 1), (1, 0)}


def test_tolerance_failure_writes_failure_cases_csv(tmp_path: Path) -> None:
    square = [(x, y) for y in range(10, 43) for x in range(10, 43)]
    prediction_points = {
        "YCu-2": square,
        "YCu-3": square,
    }
    truth_points = {sample_id: square for sample_id in ("YCu-1", "YCu-2", "YCu-3")}
    analysis_root, mask_dir = _write_fixture(
        tmp_path,
        prediction_points=prediction_points,
        truth_points=truth_points,
    )
    policy = _write_tolerance_policy(
        tmp_path,
        overrides={"minimum_instance_recall": 1.0},
    )
    output_root = tmp_path / "evaluation-output"

    result = run_evaluation(
        EvaluationParameters(
            analysis_root,
            mask_dir,
            output_root,
            tolerance_policy_path=policy,
        )
    )

    assert result["overall_status"] == "FAIL"
    assert result["tolerance_policy"]["status"] == "APPROVED"
    assert result["tolerance_policy"]["sha256"]
    contracts = result["tolerance_policy"]["metric_contracts"]
    assert {contract["metric_name"] for contract in contracts} == {
        "instance_precision",
        "instance_recall",
        "instance_f1",
        "count_absolute_error",
        "count_relative_error",
        "mean_area_relative_error",
        "mean_equivalent_diameter_relative_error",
        "number_density_relative_error",
        "perimeter_density_relative_error",
        "coverage_relative_error",
    }
    assert all(
        {
            "aggregation_scope",
            "operator",
            "tolerance",
            "requirement",
            "applicability",
            "undefined_rule",
            "zero_denominator_rule",
            "reason_codes",
        }
        <= set(contract)
        for contract in contracts
    )
    evidence = json.loads((output_root / "evidence-manifest.json").read_text(encoding="utf-8"))
    assert (
        evidence["evaluation_identity"]["tolerance_policy_sha256"]
        == result["tolerance_policy"]["sha256"]
    )
    assert evidence["evaluation_identity"]["instance_iou_threshold_source"] == "tolerance_policy"
    with (output_root / "failure-cases.csv").open(encoding="utf-8", newline="") as stream:
        failures = list(csv.DictReader(stream))
    assert any(
        row["sample_id"] == "YCu-1"
        and row["metric"] == "instance_recall"
        and row["status"] == "FAIL"
        and row["reason_code"] == "TOLERANCE_NOT_MET"
        for row in failures
    )


def test_not_evaluable_metrics_fail_only_with_approved_policy(tmp_path: Path) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    policy = _write_tolerance_policy(tmp_path)

    result = run_evaluation(
        EvaluationParameters(
            analysis_root,
            mask_dir,
            tmp_path / "evaluation-output",
            tolerance_policy_path=policy,
        )
    )

    assert result["overall_status"] == "FAIL"
    assert result["tolerance_policy"]["status"] == "APPROVED"
    assert {failure["reason_code"] for failure in result["failures"]} == {"METRIC_NOT_EVALUABLE"}


def test_unapproved_policy_is_not_evaluated(tmp_path: Path) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    policy = _write_tolerance_policy(tmp_path, approval_status="DRAFT")

    result = run_evaluation(
        EvaluationParameters(
            analysis_root,
            mask_dir,
            tmp_path / "evaluation-output",
            tolerance_policy_path=policy,
        )
    )

    assert result["overall_status"] == "NOT_EVALUATED"
    assert result["tolerance_policy"]["status"] == "NOT_APPROVED"
    assert {
        assessment["reason_code"]
        for item in result["per_image"]
        for assessment in item["tolerance_assessment"]["assessments"]
    } == {"TOLERANCE_POLICY_NOT_APPROVED"}


def test_optional_metric_failure_does_not_change_overall_status(tmp_path: Path) -> None:
    square = [(x, y) for y in range(10, 43) for x in range(10, 43)]
    analysis_root, mask_dir = _write_fixture(
        tmp_path,
        truth_points={"YCu-1": square, "YCu-2": square, "YCu-3": square},
    )
    informational = {
        "instance_recall": "informational",
        "mean_area_relative_error": "informational",
        "mean_equivalent_diameter_relative_error": "informational",
        "number_density_relative_error": "informational",
        "perimeter_density_relative_error": "informational",
        "coverage_relative_error": "informational",
    }
    policy = _write_tolerance_policy(
        tmp_path,
        overrides={"minimum_instance_recall": 1.0},
        requirement_overrides=informational,
    )

    result = run_evaluation(
        EvaluationParameters(
            analysis_root,
            mask_dir,
            tmp_path / "evaluation-output",
            tolerance_policy_path=policy,
        )
    )

    recall = next(
        assessment
        for assessment in result["per_image"][0]["tolerance_assessment"]["assessments"]
        if assessment["metric"] == "instance_recall"
    )
    assert recall["status"] == "INFORMATIONAL"
    assert recall["reason_code"] == "TOLERANCE_NOT_MET"
    assert result["overall_status"] == "PASS"
    assert result["failures"] == []


def test_optional_metric_not_applicable_does_not_fail(tmp_path: Path) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    informational = {
        "mean_area_relative_error": "informational",
        "mean_equivalent_diameter_relative_error": "informational",
        "number_density_relative_error": "informational",
        "perimeter_density_relative_error": "informational",
        "coverage_relative_error": "informational",
    }
    policy = _write_tolerance_policy(
        tmp_path,
        requirement_overrides=informational,
    )

    result = run_evaluation(
        EvaluationParameters(
            analysis_root,
            mask_dir,
            tmp_path / "evaluation-output",
            tolerance_policy_path=policy,
        )
    )

    optional = [
        assessment
        for assessment in result["per_image"][0]["tolerance_assessment"]["assessments"]
        if assessment["requirement"] == "informational"
    ]
    assert optional
    assert {item["status"] for item in optional} == {"NOT_EVALUATED"}
    assert {item["reason_code"] for item in optional} == {"METRIC_NOT_APPLICABLE"}
    assert result["overall_status"] == "PASS"


def test_empty_instance_set_conventions() -> None:
    square = np.zeros((40, 40), dtype=bool)
    square[5:10, 5:10] = True
    empty = np.zeros_like(square)

    no_gt = _scientific_metrics(square, empty)
    no_prediction = _scientific_metrics(empty, square)
    both_empty = _scientific_metrics(empty, empty)

    assert no_gt["instance_matching"]["precision"] == 0.0
    assert no_gt["instance_matching"]["recall"] == 1.0
    assert no_gt["instance_matching"]["f1"] == 0.0
    assert no_prediction["instance_matching"]["precision"] == 1.0
    assert no_prediction["instance_matching"]["recall"] == 0.0
    assert no_prediction["instance_matching"]["f1"] == 0.0
    assert both_empty["instance_matching"]["precision"] == 1.0
    assert both_empty["instance_matching"]["recall"] == 1.0
    assert both_empty["instance_matching"]["f1"] == 1.0
    assert both_empty["count"]["relative_error"] == 0.0


def test_outputs_do_not_contain_absolute_private_paths(tmp_path: Path) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    output_root = tmp_path / "evaluation-output"

    result = run_evaluation(
        EvaluationParameters(
            analysis_root,
            mask_dir,
            output_root,
            instance_iou_threshold=0.5,
        )
    )

    serialized = json.dumps(result, ensure_ascii=False)
    assert str(tmp_path) not in serialized
    assert result["overall_status"] == "NOT_EVALUATED"
    assert all(
        item["tolerance_assessment"]["status"] == "NOT_EVALUATED" for item in result["per_image"]
    )
    for filename in (
        "metrics.json",
        "metrics.csv",
        "statistics.csv",
        "failure-cases.csv",
        "evidence-manifest.json",
    ):
        assert str(tmp_path) not in (output_root / filename).read_text(encoding="utf-8")


def test_cli_error_does_not_disclose_absolute_private_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    policy = _write_tolerance_policy(
        tmp_path,
        identity_overrides={"adapter_sha256": "0" * 64},
    )

    exit_code = agglomerated_evaluation_module.main(
        [
            "--analysis-output-root",
            str(analysis_root),
            "--mask-dir",
            str(mask_dir),
            "--independent-test-manifest",
            str(tmp_path / "split-manifest.csv"),
            "--checkpoint-sha256",
            CHECKPOINT_SHA256,
            "--output-root",
            str(tmp_path / "evaluation-output"),
            "--tolerance-policy",
            str(policy),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert str(tmp_path) not in captured.err
    assert "<tolerance-policy>" in captured.err
    assert captured.out == ""


def test_rejects_duplicate_manifest_sample_id(tmp_path: Path) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    rows = _manifest_rows(tmp_path)
    rows.append({**rows[0]})
    _write_manifest_rows(tmp_path, rows)

    with pytest.raises(ValueError, match="duplicates line"):
        run_evaluation(
            EvaluationParameters(analysis_root, mask_dir, tmp_path / "evaluation-output")
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("image_sha256", "image_sha256 mismatch"),
        ("mask_sha256", "gt_sha256 mismatch"),
    ],
)
def test_rejects_manifest_file_sha_mismatch(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    rows = _manifest_rows(tmp_path)
    rows[0][field] = "0" * 64
    _write_manifest_rows(tmp_path, rows)

    with pytest.raises(ValueError, match=message):
        run_evaluation(
            EvaluationParameters(analysis_root, mask_dir, tmp_path / "evaluation-output")
        )


def test_rejects_wrong_manifest_split(tmp_path: Path) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    rows = _manifest_rows(tmp_path)
    rows[0]["split"] = "train"
    _write_manifest_rows(tmp_path, rows)

    with pytest.raises(ValueError, match="sample_id is absent from the manifest"):
        run_evaluation(
            EvaluationParameters(analysis_root, mask_dir, tmp_path / "evaluation-output")
        )


def test_rejects_source_group_cross_split_leakage(tmp_path: Path) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    rows = _manifest_rows(tmp_path)
    rows.append(
        {
            **rows[0],
            "sample_id": "training-leak",
            "field_of_view_id": "training-leak",
            "split": "train",
            "mask_path": "",
            "mask_sha256": "",
        }
    )
    _write_manifest_rows(tmp_path, rows)

    with pytest.raises(ValueError, match=r"source_image_id.*crosses split"):
        run_evaluation(
            EvaluationParameters(analysis_root, mask_dir, tmp_path / "evaluation-output")
        )


@pytest.mark.parametrize(
    ("identity_field", "message"),
    [
        ("independent_test_manifest_sha256", "independent_test_manifest_sha256"),
        ("adapter_sha256", "adapter_sha256"),
    ],
)
def test_policy_identity_mismatch_fails_before_loading_masks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_field: str,
    message: str,
) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    policy = _write_tolerance_policy(
        tmp_path,
        identity_overrides={identity_field: "0" * 64},
    )
    opened: list[Path] = []
    hashed: list[Path] = []
    original_sha256 = agglomerated_evaluation_module._sha256

    def forbidden_load(path: Path, **_: Any) -> np.ndarray:
        opened.append(path)
        raise AssertionError("mask loading must not start before policy validation")

    def tracking_sha256(path: Path) -> str:
        hashed.append(path)
        return original_sha256(path)

    monkeypatch.setattr(agglomerated_evaluation_module, "_load_foreground", forbidden_load)
    monkeypatch.setattr(agglomerated_evaluation_module, "_sha256", tracking_sha256)
    with pytest.raises(ValueError, match=message):
        run_evaluation(
            EvaluationParameters(
                analysis_root,
                mask_dir,
                tmp_path / "evaluation-output",
                tolerance_policy_path=policy,
            )
        )
    assert opened == []
    assert not any(path.suffix.lower() in {".tif", ".png"} for path in hashed)


def test_policy_file_modification_changes_sha(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    policy = _write_tolerance_policy(tmp_path)
    original = _sha256(policy)
    payload = json.loads(policy.read_text(encoding="utf-8"))
    payload["approval"]["rationale"] = "modified after initial policy bytes"
    policy.write_text(json.dumps(payload), encoding="utf-8")

    assert _sha256(policy) != original


def test_manifest_sha_freezes_dynamic_sample_set(tmp_path: Path) -> None:
    filenames = ("heldout-alpha.tif", "heldout-beta.tif")
    analysis_root, mask_dir = _write_fixture(tmp_path, filenames=filenames)
    policy = _write_tolerance_policy(
        tmp_path,
        approval_status="DRAFT",
    )

    result = run_evaluation(
        EvaluationParameters(
            analysis_root,
            mask_dir,
            tmp_path / "evaluation-output",
            tolerance_policy_path=policy,
        )
    )

    assert [item["sample_id"] for item in result["per_image"]] == [
        "heldout-alpha",
        "heldout-beta",
    ]
    assert result["input_manifest"]["sample_count"] == 2
    assert result["overall_status"] == "NOT_EVALUATED"


def test_identity_failure_precedes_prediction_or_gt_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    run_config_path = sorted(analysis_root.rglob("run_config.json"))[-1]
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    run_config["inference"]["threshold"] = 0.26
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")
    opened: list[Path] = []

    def forbidden_load(path: Path, **_: Any) -> np.ndarray:
        opened.append(path)
        raise AssertionError("mask loading must not start before preflight succeeds")

    monkeypatch.setattr(agglomerated_evaluation_module, "_load_foreground", forbidden_load)

    with pytest.raises(ValueError, match="threshold is not frozen"):
        run_evaluation(
            EvaluationParameters(analysis_root, mask_dir, tmp_path / "evaluation-output")
        )
    assert opened == []


def test_rejects_checkpoint_identity_before_loading_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    opened: list[Path] = []

    def forbidden_load(path: Path, **_: Any) -> np.ndarray:
        opened.append(path)
        raise AssertionError("input loading must not start before identity validation")

    monkeypatch.setattr(agglomerated_evaluation_module, "_load_foreground", forbidden_load)
    with pytest.raises(ValueError, match="frozen Agglomerated asset"):
        run_evaluation(
            EvaluationParameters(
                analysis_root,
                mask_dir,
                tmp_path / "evaluation-output",
                checkpoint_sha256="0" * 64,
            )
        )
    assert opened == []


def test_evidence_manifest_is_deterministic_and_hashes_outputs(tmp_path: Path) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    output_root = tmp_path / "evaluation-output"
    run_evaluation(
        EvaluationParameters(
            analysis_root,
            mask_dir,
            output_root,
            instance_iou_threshold=0.5,
        )
    )
    manifest_path = output_root / "evidence-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    evidence = json.loads(manifest_bytes)
    assert (
        manifest_bytes
        == (json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    )
    assert evidence["inputs"] == sorted(
        evidence["inputs"],
        key=lambda item: (item["role"], item.get("sample_id", ""), item["path"]),
    )
    assert evidence["outputs"] == sorted(evidence["outputs"], key=lambda item: item["path"])
    assert evidence["input_manifest"]["split"] == "independent-test"
    assert evidence["input_manifest"]["sample_ids"] == ["YCu-1", "YCu-2", "YCu-3"]
    assert evidence["evaluation_identity"]["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert evidence["evaluation_identity"]["threshold"] == THRESHOLD
    assert evidence["evaluation_identity"]["threshold_comparison"] == "gte"
    assert evidence["evaluation_identity"]["min_area_px"] == MIN_AREA_PX
    assert evidence["evaluation_identity"]["bottom_crop_px"] == BOTTOM_CROP_PX
    assert evidence["evaluation_identity"]["instance_iou_threshold_source"] == "explicit_cli"
    for output in evidence["outputs"]:
        artifact = output_root / output["path"]
        assert output["size_bytes"] == artifact.stat().st_size
        assert output["sha256"] == _sha256(artifact)
    assert str(tmp_path) not in json.dumps(evidence)


@pytest.mark.parametrize("legacy_directory", ["prediction_results", "agglomerated_test_results"])
def test_rejects_prediction_outside_formal_analysis_layout(
    tmp_path: Path,
    legacy_directory: str,
) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    legacy = analysis_root / legacy_directory / "old" / "pred_mask.png"
    legacy.parent.mkdir(parents=True)
    _write_mask(legacy)

    with pytest.raises(ValueError, match="outside the formal Analysis layout"):
        run_evaluation(
            EvaluationParameters(analysis_root, mask_dir, tmp_path / "evaluation-output")
        )


def test_rejects_duplicate_prediction_for_a_fixed_sample(tmp_path: Path) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    first_image = analysis_root / "artifacts" / "job_test" / "images" / "img_0"
    duplicate = first_image / "runs" / "run_duplicate"
    duplicate.mkdir()
    _write_mask(duplicate / "pred_mask.png")
    (duplicate / "run_config.json").write_text(json.dumps(_run_config("4" * 64)), encoding="utf-8")

    with pytest.raises(ValueError, match=r"multiple pred_mask\.png"):
        run_evaluation(
            EvaluationParameters(analysis_root, mask_dir, tmp_path / "evaluation-output")
        )


def test_rejects_nonzero_prediction_in_bottom_exclusion(tmp_path: Path) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    prediction = next(analysis_root.rglob("pred_mask.png"))
    _write_mask(prediction, [(10, VALID_HEIGHT)])

    with pytest.raises(ValueError, match="bottom 130 px is not zero"):
        run_evaluation(
            EvaluationParameters(analysis_root, mask_dir, tmp_path / "evaluation-output")
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

    with pytest.raises(ValueError, match="frozen Agglomerated asset"):
        run_evaluation(
            EvaluationParameters(analysis_root, mask_dir, tmp_path / "evaluation-output")
        )


def test_rejects_metadata_identity_that_does_not_match_formal_path(tmp_path: Path) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    metadata_path = next(analysis_root.rglob("metadata.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["job_id"] = "job_other"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata identity does not match"):
        run_evaluation(
            EvaluationParameters(analysis_root, mask_dir, tmp_path / "evaluation-output")
        )


def test_rejects_incomplete_schema_v3_provenance(tmp_path: Path) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    run_config_path = next(analysis_root.rglob("run_config.json"))
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    run_config["provenance_status"] = "partial"
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    with pytest.raises(ValueError, match="provenance is not complete"):
        run_evaluation(
            EvaluationParameters(analysis_root, mask_dir, tmp_path / "evaluation-output")
        )


def test_rejects_non_normalized_model_bundle_reference(tmp_path: Path) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    run_config_path = next(analysis_root.rglob("run_config.json"))
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    run_config["model_bundle"]["weight_ref"] = f"../{TORCHSCRIPT_SHA256}/weights.pt"
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    with pytest.raises(ValueError, match="not a normalized relative path"):
        run_evaluation(
            EvaluationParameters(analysis_root, mask_dir, tmp_path / "evaluation-output")
        )


def test_rejects_missing_or_mismatched_execution_provenance(tmp_path: Path) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    execution_path = next(analysis_root.rglob("execution_provenance.json"))
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["model_bundle_id"] = "0" * 64
    execution_path.write_text(json.dumps(execution), encoding="utf-8")

    with pytest.raises(ValueError, match="model_bundle_id differs"):
        run_evaluation(
            EvaluationParameters(analysis_root, mask_dir, tmp_path / "evaluation-output")
        )

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from scripts.models.evaluate_unet_large_independent_test import (
    BOTTOM_CROP_PX,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MIN_AREA_PX,
    MODEL_ID,
    SCALE_NM_PER_PIXEL,
    TEST_FILENAMES,
    THRESHOLD,
    VALID_HEIGHT,
    EvaluationParameters,
    _load_foreground,
    _validate_output_root,
    compute_metrics,
    run_evaluation,
)


def _run_config() -> dict[str, Any]:
    return {
        "schema_version": "1",
        "contract_schema_version": 3,
        "model_id": MODEL_ID,
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


def _write_mask(path: Path, foreground: list[tuple[int, int]] = ()) -> None:
    pixels = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    for x, y in foreground:
        pixels[y, x] = 255
    Image.fromarray(pixels).save(path)


def _write_fixture(
    root: Path,
    *,
    prediction_points: dict[str, list[tuple[int, int]]] | None = None,
    truth_points: dict[str, list[tuple[int, int]]] | None = None,
) -> tuple[Path, Path]:
    analysis_root = root / "analysis"
    mask_dir = root / "truth"
    mask_dir.mkdir()
    prediction_points = prediction_points or {}
    truth_points = truth_points or {}
    for index, filename in enumerate(TEST_FILENAMES):
        sample_id = Path(filename).stem
        image_dir = analysis_root / "artifacts" / "job_test" / "images" / f"img_{index}"
        run_dir = image_dir / "runs" / f"run_{index}"
        run_dir.mkdir(parents=True)
        (image_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": "1",
                    "image_id": f"img_{index}",
                    "filename": filename,
                    "sample_id": sample_id,
                    "width": IMAGE_WIDTH,
                    "height": IMAGE_HEIGHT,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "run_config.json").write_text(
            json.dumps(_run_config()), encoding="utf-8"
        )
        _write_mask(run_dir / "pred_mask.png", prediction_points.get(sample_id, []))
        _write_mask(mask_dir / filename, truth_points.get(sample_id, []))
    return analysis_root, mask_dir


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
        "SrZr-3": [(10, 10)],
        "BaCu-2": [(20, 20), (21, 21)],
        "PrCu-3": [(30, 30), (31, 31), (32, 32)],
    }
    analysis_root, mask_dir = _write_fixture(
        tmp_path, prediction_points=points, truth_points=points
    )
    output_root = tmp_path / "evaluation-output"

    result = run_evaluation(EvaluationParameters(analysis_root, mask_dir, output_root))

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


def test_rejects_duplicate_prediction_for_a_fixed_sample(tmp_path: Path) -> None:
    analysis_root, mask_dir = _write_fixture(tmp_path)
    first_image = analysis_root / "artifacts" / "job_test" / "images" / "img_0"
    duplicate = first_image / "runs" / "run_duplicate"
    duplicate.mkdir()
    _write_mask(duplicate / "pred_mask.png")
    (duplicate / "run_config.json").write_text(
        json.dumps(_run_config()), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=r"multiple pred_mask\.png"):
        run_evaluation(
            EvaluationParameters(analysis_root, mask_dir, tmp_path / "evaluation-output")
        )

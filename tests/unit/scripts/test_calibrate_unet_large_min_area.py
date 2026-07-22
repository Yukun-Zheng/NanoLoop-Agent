from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.models.calibrate_unet_large_min_area import (
    BOTTOM_CROP_PX,
    CANDIDATE_MIN_AREAS,
    MODEL_ID,
    SCALE_NM_PER_PIXEL,
    THRESHOLD,
    TORCHSCRIPT_SHA256,
    VALIDATION_FILENAMES,
    CalibrationPaths,
    _validate_output_root,
    evaluate_min_areas,
    run,
    select_min_area,
)


def _arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    probabilities: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    for filename in VALIDATION_FILENAMES:
        probability = np.zeros((300, 200), dtype=np.float32)
        target = np.zeros((300, 200), dtype=bool)
        probability[20:60, 20:60] = 0.9
        target[20:60, 20:60] = True
        probability[70:90, 70:90] = 0.9
        target[70:90, 70:90] = True
        probabilities[filename] = probability
        targets[filename] = target
    return probabilities, targets


def test_fixed_large_min_area_contract() -> None:
    assert MODEL_ID == "unet-large-optimized-v1"
    assert THRESHOLD == 0.50
    assert BOTTOM_CROP_PX == 180
    assert pytest.approx(100 / 184) == SCALE_NM_PER_PIXEL
    assert CANDIDATE_MIN_AREAS == (0, 16, 32, 64, 128, 256, 512, 1024)
    assert VALIDATION_FILENAMES == (
        "NdZn-2.tif",
        "LaMn-3.tif",
        "LaMn-1.tif",
        "BaCo-3.tif",
        "BaCu-1.tif",
        "BaCr-3.tif",
    )


def test_evaluation_uses_strict_threshold_and_excludes_bottom_180() -> None:
    probabilities, targets = _arrays()
    for filename in VALIDATION_FILENAMES:
        probabilities[filename][70:90, 70:90] = THRESHOLD
        probabilities[filename][-10:, 100:120] = 1.0
        targets[filename][-10:, 100:120] = True

    result = evaluate_min_areas(probabilities, targets, candidates=[0])[0]

    for image in result["images"]:
        assert image["particle_count"] == 1
        assert image["gt_baseline"]["particle_count"] == 2
        assert image["count_mape"] == pytest.approx(0.5)
        assert image["coverage_ratio"] == pytest.approx(1600 / (120 * 200))


def test_evaluation_reuses_canonical_postprocess_and_morphometry() -> None:
    probabilities, targets = _arrays()

    results = evaluate_min_areas(probabilities, targets, candidates=[0, 512])

    unfiltered, filtered = results
    image = unfiltered["images"][0]
    assert image["particle_count"] == 2
    assert image["mean_equivalent_diameter_nm"] is not None
    assert image["perimeter_density_um"] > 0
    assert image["number_density_um2"] > 0
    assert image["coverage_ratio"] > 0
    assert image["excluded_border_count"] == 0
    assert image["count_mape"] == pytest.approx(0)
    assert image["mean_diameter_mape"] == pytest.approx(0)
    assert image["perimeter_density_mape"] == pytest.approx(0)
    assert unfiltered["gt_retention"] == pytest.approx(1)
    assert filtered["gt_retention"] == pytest.approx(0.5)
    assert filtered["physical_area_nm2"] == pytest.approx(512 * (100 / 184) ** 2)


def test_selection_rule_uses_composite_then_dice_then_smaller_area() -> None:
    results = [
        {"min_area_px": 16, "macro": {"composite_mape": 0.2, "dice": 0.8}},
        {"min_area_px": 32, "macro": {"composite_mape": 0.1, "dice": 0.7}},
        {"min_area_px": 64, "macro": {"composite_mape": 0.1, "dice": 0.8}},
        {"min_area_px": 128, "macro": {"composite_mape": 0.1, "dice": 0.8}},
    ]

    assert select_min_area(results)["min_area_px"] == 64

    results.append(
        {"min_area_px": 64, "macro": {"composite_mape": 0.1, "dice": 0.8}}
    )
    with pytest.raises(ValueError, match="did not resolve the tie"):
        select_min_area(results)


@pytest.mark.parametrize("kind", ["existing", "repository"])
def test_output_root_protection(tmp_path: Path, kind: str) -> None:
    output_root = (
        tmp_path / "already-exists"
        if kind == "existing"
        else Path(__file__).resolve().parents[3] / "forbidden-large-min-area"
    )
    if kind == "existing":
        output_root.mkdir()

    with pytest.raises(ValueError, match="output-root"):
        _validate_output_root(output_root)


def test_run_reads_existing_probability_cache_without_model(tmp_path: Path) -> None:
    threshold_root = tmp_path / "threshold-calibration-v1"
    image_dir = tmp_path / "train_images" / "images"
    mask_dir = tmp_path / "train_masks" / "masks"
    output_root = tmp_path / "min-area-output"
    threshold_root.mkdir()
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    probabilities, targets = _arrays()
    probability_paths: dict[str, str] = {}
    for filename in VALIDATION_FILENAMES:
        stem_dir = threshold_root / "probability-cache" / Path(filename).stem
        stem_dir.mkdir(parents=True)
        probability_path = stem_dir / "probability.npy"
        np.save(probability_path, probabilities[filename], allow_pickle=False)
        probability_paths[filename] = str(probability_path)
        Image.fromarray((targets[filename] * 255).astype(np.uint8)).save(mask_dir / filename)
        Image.fromarray(np.zeros((300, 200), dtype=np.uint8)).save(image_dir / filename)
    evidence = {
        "model_id": MODEL_ID,
        "torchscript_sha256": TORCHSCRIPT_SHA256,
        "validation_images": list(VALIDATION_FILENAMES),
        "bottom_crop_px": BOTTOM_CROP_PX,
        "comparison_rule": "probability > threshold",
        "selected": {"threshold": THRESHOLD},
        "artifacts": {"probabilities": probability_paths},
    }
    (threshold_root / "threshold-calibration.json").write_text(
        json.dumps(evidence),
        encoding="utf-8",
    )

    payload = run(CalibrationPaths(threshold_root, image_dir, mask_dir, output_root))

    assert payload["probability_source"].endswith("no repeated inference")
    assert payload["selected"]["min_area_px"] == 0
    assert (output_root / "min-area-calibration.json").is_file()
    assert (output_root / "min-area-calibration.csv").is_file()
    for filename in VALIDATION_FILENAMES:
        paths = payload["artifacts"]["selected_visualizations"][filename]
        assert Path(paths["overlay"]).is_file()
        assert Path(paths["labeled_particles"]).is_file()

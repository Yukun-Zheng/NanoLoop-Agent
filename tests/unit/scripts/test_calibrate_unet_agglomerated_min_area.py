from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.models.calibrate_unet_agglomerated_min_area import (
    BOTTOM_CROP_PX,
    CANDIDATE_MIN_AREAS,
    CHECKPOINT_SHA256,
    MODEL_ID,
    THRESHOLD,
    THRESHOLD_EVIDENCE_SHA256,
    TORCHSCRIPT_SHA256,
    VALIDATION_FILENAMES,
    CalibrationPaths,
    _load_threshold_evidence,
    _reject_test_path,
    _validate_output_root,
    build_parser,
    evaluate_min_areas,
    run,
    select_min_area,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arrays(
    shape: tuple[int, int] = (160, 80),
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    probabilities: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    for index, filename in enumerate(VALIDATION_FILENAMES):
        probability = np.zeros(shape, dtype=np.float32)
        target = np.zeros(shape, dtype=bool)
        target[2:28, 4:54] = True
        probability[2:28, 4:54] = THRESHOLD
        probability[10:12, 64:66] = 0.9
        probability[-BOTTOM_CROP_PX:, :] = 1.0
        target[-BOTTOM_CROP_PX:, :] = index % 2 == 0
        probabilities[filename] = probability
        targets[filename] = target
    return probabilities, targets


def _write_fixture(tmp_path: Path) -> tuple[CalibrationPaths, dict[str, str]]:
    threshold_root = tmp_path / "threshold-calibration"
    mask_dir = tmp_path / "train_masks" / "masks"
    output_root = tmp_path / "min-area-output"
    probability_dir = threshold_root / "probabilities"
    probability_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    probabilities, targets = _arrays()
    artifacts: dict[str, dict[str, str]] = {}
    validation_inputs: dict[str, dict[str, str]] = {}
    original_hashes: dict[str, str] = {}
    for filename in VALIDATION_FILENAMES:
        probability_path = probability_dir / f"{Path(filename).stem}.npy"
        np.save(probability_path, probabilities[filename], allow_pickle=False)
        mask_path = mask_dir / filename
        Image.fromarray(targets[filename].astype(np.uint8) * 255).save(mask_path)
        artifacts[filename] = {
            "path": str(probability_path),
            "sha256": _sha256(probability_path),
        }
        validation_inputs[filename] = {
            "image_path": f"/not-read/{filename}",
            "image_sha256": "not-used",
            "mask_path": str(mask_path),
            "mask_sha256": _sha256(mask_path),
        }
        original_hashes[filename] = _sha256(probability_path)
    evidence = {
        "schema_version": "1",
        "model_id": MODEL_ID,
        "torchscript_sha256": TORCHSCRIPT_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "validation_images": list(VALIDATION_FILENAMES),
        "validation_inputs": validation_inputs,
        "comparison_rule": "probability >= threshold",
        "selected_threshold": THRESHOLD,
        "selected": {"threshold": THRESHOLD},
        "fixed_inference_contract": {
            "bottom_crop_px": BOTTOM_CROP_PX,
            "threshold_comparison": "gte",
            "min_area_px_during_threshold_calibration": 0,
            "default_watershed_enabled": False,
        },
        "artifacts": {"probability_arrays": artifacts},
    }
    (threshold_root / "threshold-calibration.json").write_text(
        json.dumps(evidence), encoding="utf-8"
    )
    return CalibrationPaths(threshold_root, mask_dir, output_root), original_hashes


def test_fixed_read_only_contract_and_cli() -> None:
    assert THRESHOLD == 0.25
    assert THRESHOLD_EVIDENCE_SHA256 == (
        "9c76289a61ab870b59cda079eb732222a1267d3ecf47636244967872c4130a02"
    )
    assert CANDIDATE_MIN_AREAS == (0, 16, 32, 64, 128, 256, 512, 1024)
    assert VALIDATION_FILENAMES == (
        "BiCu-3.tif",
        "BaNi-3.tif",
        "BaNi-1.tif",
        "BaNi-2.tif",
    )
    destinations = {action.dest for action in build_parser()._actions}
    assert {"threshold_calibration_root", "mask_dir", "output_root"} <= destinations
    assert "image_dir" not in destinations
    assert "torchscript" not in destinations


def test_threshold_gte_and_bottom_130_are_frozen() -> None:
    probabilities, targets = _arrays()
    result = evaluate_min_areas(probabilities, targets, candidates=[0])[0]

    assert result["gt_baseline_count"] == 4
    assert result["macro"]["dice"] > 0
    for image in result["images"]:
        assert image["agglomerate_count"] == 2
        assert image["gt_baseline"]["agglomerate_count"] == 1
    probabilities["BiCu-3.tif"][-BOTTOM_CROP_PX:] = np.nan
    with pytest.raises(ValueError, match="invalid probability"):
        evaluate_min_areas(probabilities, targets, candidates=[0])


def test_min_area_filter_and_gt_baseline_policy() -> None:
    probabilities, targets = _arrays()
    results = evaluate_min_areas(probabilities, targets, candidates=[0, 8])

    assert all(image["agglomerate_count"] == 2 for image in results[0]["images"])
    assert all(image["agglomerate_count"] == 1 for image in results[1]["images"])
    assert results[0]["gt_baseline_count"] == results[1]["gt_baseline_count"] == 4
    assert results[1]["gt_retention"] == 1.0


def test_selection_rule_composite_then_dice_then_smaller_area() -> None:
    results = [
        {"min_area_px": 16, "macro": {"composite_mape": 0.2, "dice": 0.9}},
        {"min_area_px": 32, "macro": {"composite_mape": 0.1, "dice": 0.7}},
        {"min_area_px": 64, "macro": {"composite_mape": 0.1, "dice": 0.8}},
        {"min_area_px": 128, "macro": {"composite_mape": 0.1, "dice": 0.8}},
    ]
    assert select_min_area(results)["min_area_px"] == 64
    results.append({"min_area_px": 64, "macro": {"composite_mape": 0.1, "dice": 0.8}})
    with pytest.raises(ValueError, match="did not resolve the tie"):
        select_min_area(results)


@pytest.mark.parametrize("kind", ["existing", "repository"])
def test_output_root_refuses_overwrite_and_repository(tmp_path: Path, kind: str) -> None:
    output = (
        tmp_path / "existing"
        if kind == "existing"
        else Path(__file__).resolve().parents[3] / "forbidden-agglomerated-min-area"
    )
    if kind == "existing":
        output.mkdir()
    with pytest.raises(ValueError, match="output-root"):
        _validate_output_root(output)


def test_rejects_ycu_and_test_directories(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="independent test"):
        _reject_test_path(tmp_path / "test_mask_human", field="mask-dir")
    with pytest.raises(ValueError, match="independent test"):
        _reject_test_path(tmp_path / "YCu-1.tif", field="mask")


def test_rejects_unfrozen_or_wrong_validation_evidence(tmp_path: Path) -> None:
    paths, _ = _write_fixture(tmp_path)
    evidence_path = paths.threshold_calibration_root / "threshold-calibration.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["selected_threshold"] = 0.30
    evidence["validation_images"][-1] = "YCu-1.tif"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen agglomerated contract"):
        _load_threshold_evidence(paths.threshold_calibration_root, expected_sha256=None)


def test_four_field_run_reuses_and_does_not_modify_probability_cache(tmp_path: Path) -> None:
    paths, original_hashes = _write_fixture(tmp_path)
    payload = run(paths, expected_size=(80, 160), expected_evidence_sha256=None)

    assert payload["validation_images"] == list(VALIDATION_FILENAMES)
    assert payload["fixed_parameters"]["threshold"] == 0.25
    assert payload["fixed_parameters"]["threshold_comparison"] == "gte"
    assert payload["probability_source"].endswith("no model loading or inference")
    assert len(payload["candidate_results"]) == len(CANDIDATE_MIN_AREAS)
    assert (paths.output_root / "min-area-calibration.json").is_file()
    assert (paths.output_root / "min-area-calibration.csv").is_file()
    for filename in VALIDATION_FILENAMES:
        probability_path = paths.threshold_calibration_root / "probabilities" / (
            f"{Path(filename).stem}.npy"
        )
        assert _sha256(probability_path) == original_hashes[filename]
        assert Path(payload["artifacts"]["selected_prediction_masks"][filename]).is_file()
        assert Path(payload["artifacts"]["reviews"][filename]).is_file()
    assert any("YCu-1.tif" in item and "not read" in item for item in payload["limitations"])


def test_mask_or_probability_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    paths, _ = _write_fixture(tmp_path)
    mask_path = paths.mask_dir / VALIDATION_FILENAMES[0]
    mask_path.write_bytes(mask_path.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="mask identity mismatch"):
        run(paths, expected_size=(80, 160), expected_evidence_sha256=None)


def test_parser_requires_only_read_only_sources() -> None:
    namespace = build_parser().parse_args(
        [
            "--threshold-calibration-root", "threshold",
            "--mask-dir", "masks",
            "--output-root", "output",
        ]
    )
    assert isinstance(namespace, argparse.Namespace)

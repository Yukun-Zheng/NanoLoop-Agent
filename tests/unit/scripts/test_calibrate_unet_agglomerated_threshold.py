from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.models.calibrate_unet_agglomerated_threshold import (
    BOTTOM_CROP_PX,
    CANDIDATE_THRESHOLDS,
    EXPECTED_CONFIG,
    INDEPENDENT_TEST_FILENAMES,
    TORCHSCRIPT_SHA256,
    VALIDATION_FILENAMES,
    CalibrationPaths,
    _load_probabilities,
    _metrics,
    _validate_output_root,
    _verify_torchscript,
    cache_probabilities,
    evaluate_thresholds,
    run_with_inferencer,
    select_threshold,
    validate_validation_filenames,
)

TEST_SIZE = (4, 132)


def _write_mask(path: Path, foreground: list[tuple[int, int]]) -> None:
    pixels = np.zeros((TEST_SIZE[1], TEST_SIZE[0]), dtype=np.uint8)
    for x, y in foreground:
        pixels[y, x] = 255
    Image.fromarray(pixels).save(path)


def _write_fixture(tmp_path: Path) -> CalibrationPaths:
    image_dir = tmp_path / "train_images" / "images"
    mask_dir = tmp_path / "train_masks" / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for index, filename in enumerate(VALIDATION_FILENAMES):
        image = np.full((TEST_SIZE[1], TEST_SIZE[0]), index + 1, dtype=np.uint8)
        Image.fromarray(image).save(image_dir / filename)
        _write_mask(mask_dir / filename, [(0, 0)])
    torchscript = tmp_path / "external.pt"
    torchscript.write_bytes(b"test-only-not-a-model")
    return CalibrationPaths(
        image_dir=image_dir,
        mask_dir=mask_dir,
        torchscript=torchscript,
        output_root=tmp_path / "calibration-output",
    )


class FakeInferencer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, image_path: Path) -> np.ndarray:
        self.calls.append(image_path.name)
        probability = np.zeros((TEST_SIZE[1], TEST_SIZE[0]), dtype=np.float32)
        probability[0, 0] = 0.50
        probability[-BOTTOM_CROP_PX:] = 1.0
        return probability


def _targets() -> dict[str, np.ndarray]:
    targets: dict[str, np.ndarray] = {}
    for filename in VALIDATION_FILENAMES:
        target = np.zeros((131, 2), dtype=bool)
        target[0, 0] = True
        targets[filename] = target
    return targets


def _probabilities() -> dict[str, np.ndarray]:
    probabilities: dict[str, np.ndarray] = {}
    for filename in VALIDATION_FILENAMES:
        probability = np.zeros((131, 2), dtype=np.float32)
        probability[0] = [0.50, 0.49]
        probability[-BOTTOM_CROP_PX:] = 1.0
        probabilities[filename] = probability
    return probabilities


def test_fixed_validation_candidates_and_test_boundary() -> None:
    assert EXPECTED_CONFIG["default_threshold"] == 0.25
    assert VALIDATION_FILENAMES == (
        "BiCu-3.tif",
        "BaNi-3.tif",
        "BaNi-1.tif",
        "BaNi-2.tif",
    )
    assert CANDIDATE_THRESHOLDS == (
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
    )
    assert set(VALIDATION_FILENAMES).isdisjoint(INDEPENDENT_TEST_FILENAMES)
    with pytest.raises(ValueError, match="independent test samples are forbidden"):
        validate_validation_filenames((*VALIDATION_FILENAMES, "YCu-1.tif"))


def test_each_image_is_inferred_once_and_thresholds_reuse_probability_cache(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)
    paths.output_root.mkdir()
    inferencer = FakeInferencer()
    image_paths = [paths.image_dir / filename for filename in VALIDATION_FILENAMES]

    probability_paths = cache_probabilities(
        image_paths=image_paths,
        output_root=paths.output_root,
        infer_probability=inferencer,
        expected_size=TEST_SIZE,
    )
    probabilities = _load_probabilities(probability_paths)
    targets = {
        filename: np.zeros((TEST_SIZE[1], TEST_SIZE[0]), dtype=bool)
        for filename in VALIDATION_FILENAMES
    }
    for target in targets.values():
        target[0, 0] = True
    evaluate_thresholds(probabilities, targets, thresholds=[0.20, 0.50, 0.65])

    assert inferencer.calls == list(VALIDATION_FILENAMES)
    assert len(probability_paths) == 4
    assert all(path.suffix == ".npy" for path in probability_paths.values())


def test_gte_metrics_component_recall_and_bottom_130_exclusion() -> None:
    result = evaluate_thresholds(
        _probabilities(),
        _targets(),
        thresholds=[0.50],
    )[0]

    for image in result["images"]:
        assert image["tp"] == 1
        assert image["fp"] == 0
        assert image["fn"] == 0
        assert image["tn"] == 1
        assert image["dice"] == pytest.approx(1.0)
        detection = image["gt_agglomerate_detection"]
        assert detection["gt_count"] == 1
        assert detection["detected_count"] == 1
        assert detection["by_size_bucket"]["tiny"]["recall"] == pytest.approx(1.0)
    assert result["micro"]["tp"] == 4
    assert result["micro"]["fp"] == 0
    assert result["micro"]["fn"] == 0
    assert result["micro"]["tn"] == 4
    assert result["gt_agglomerate_detection"][
        "mean_available_size_bucket_recall"
    ] == pytest.approx(1.0)
    assert result["selection_score"] == pytest.approx(1.0)

    metrics = _metrics(tp=2, fp=1, fn=2, tn=5)
    assert metrics["dice"] == pytest.approx(4 / 7)
    assert metrics["iou"] == pytest.approx(2 / 5)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1 / 2)


def test_selection_rule_uses_score_then_micro_dice_then_lower_threshold() -> None:
    results = [
        {"threshold": 0.30, "selection_score": 0.80, "micro": {"dice": 0.70}},
        {"threshold": 0.40, "selection_score": 0.80, "micro": {"dice": 0.75}},
        {"threshold": 0.20, "selection_score": 0.80, "micro": {"dice": 0.75}},
        {"threshold": 0.50, "selection_score": 0.79, "micro": {"dice": 0.90}},
    ]

    assert select_threshold(results)["threshold"] == 0.20

    results.append(
        {"threshold": 0.20, "selection_score": 0.80, "micro": {"dice": 0.75}}
    )
    with pytest.raises(ValueError, match="did not resolve the tie"):
        select_threshold(results)


@pytest.mark.parametrize("kind", ["existing", "repository"])
def test_output_root_rejects_overwrite_and_repository_path(tmp_path: Path, kind: str) -> None:
    output_root = (
        tmp_path / "existing"
        if kind == "existing"
        else Path(__file__).resolve().parents[3] / "forbidden-agglomerated-calibration"
    )
    if kind == "existing":
        output_root.mkdir()

    with pytest.raises(ValueError, match="output-root"):
        _validate_output_root(output_root)


def test_four_image_run_writes_traceable_summary_and_review_artifacts(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    inferencer = FakeInferencer()

    payload = run_with_inferencer(
        paths,
        infer_probability=inferencer,
        torchscript_sha256=TORCHSCRIPT_SHA256,
        expected_size=TEST_SIZE,
    )

    assert inferencer.calls == list(VALIDATION_FILENAMES)
    assert payload["validation_images"] == list(VALIDATION_FILENAMES)
    assert len(payload["threshold_results"]) == len(CANDIDATE_THRESHOLDS)
    assert payload["selected_threshold"] == pytest.approx(0.20)
    assert "new traceable validation calibration evidence" in payload["evidence_statement"]
    assert any("YCu-1.tif" in limitation for limitation in payload["limitations"])
    assert any("min_area_px remains uncalibrated" in item for item in payload["limitations"])
    assert str(tmp_path) not in json.dumps(payload, ensure_ascii=False)
    assert (paths.output_root / "threshold-calibration.json").is_file()
    with (paths.output_root / "threshold-calibration.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(CANDIDATE_THRESHOLDS) * 6
    for filename in VALIDATION_FILENAMES:
        stem = Path(filename).stem
        assert (paths.output_root / "probabilities" / f"{stem}.npy").is_file()
        assert (paths.output_root / "probability-images" / f"{stem}-probability.png").is_file()
        assert (paths.output_root / "final-predictions" / f"{stem}-pred-mask.png").is_file()
        assert (paths.output_root / "reviews" / f"{stem}-gt-pred-error.png").is_file()


def test_torchscript_sha_mismatch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "wrong.pt"
    path.write_bytes(b"not-the-frozen-torchscript")

    with pytest.raises(ValueError, match="TorchScript SHA-256 differs"):
        _verify_torchscript(path)

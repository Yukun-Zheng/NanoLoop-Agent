"""Calibrate agglomerated U-Net min_area_px from frozen validation caches only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.analysis.config import MorphometryConfig, PostprocessProfile
from app.analysis.morphometry import measure
from app.analysis.postprocessing import (
    NormalizedInstance,
    PostprocessResult,
    normalize_semantic_mask_detailed,
)

MODEL_ID = "unet-agglomerated-specialized-v1"
TORCHSCRIPT_SHA256 = "d36cf627dc6d1eea83b769743b657e6bb3d9a1af39ddc6a00ae95acc3b20ffc9"
CHECKPOINT_SHA256 = "e2be19c6fe1e843856fb339d13de8baed8d748f88558ba7bd3eaaa20b90ede21"
THRESHOLD_EVIDENCE_SHA256 = (
    "9c76289a61ab870b59cda079eb732222a1267d3ecf47636244967872c4130a02"
)
THRESHOLD = 0.25
VALIDATION_FILENAMES = ("BiCu-3.tif", "BaNi-3.tif", "BaNi-1.tif", "BaNi-2.tif")
INDEPENDENT_TEST_FILENAMES = ("YCu-1.tif", "YCu-2.tif", "YCu-3.tif")
CANDIDATE_MIN_AREAS = (0, 16, 32, 64, 128, 256, 512, 1024)
IMAGE_WIDTH = 2048
IMAGE_HEIGHT = 1536
BOTTOM_CROP_PX = 130
SCALE_NM_PER_PIXEL = 100 / 184
PERIMETER_NEIGHBORHOOD = 8
POSTPROCESS_FIXED = {
    "fill_holes": True,
    "watershed_enabled": False,
    "exclude_border": True,
    "connectivity": 2,
}


@dataclass(frozen=True, slots=True)
class CalibrationPaths:
    threshold_calibration_root: Path
    mask_dir: Path
    output_root: Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_output_root(output_root: Path, *, repository: Path | None = None) -> Path:
    resolved = output_root.expanduser().resolve(strict=False)
    repository = (repository or _repository_root()).resolve(strict=True)
    if resolved.exists():
        raise ValueError(f"output-root already exists: {resolved}")
    if resolved == repository or resolved.is_relative_to(repository):
        raise ValueError("output-root must be outside the repository")
    return resolved


def _reject_test_path(path: Path, *, field: str) -> None:
    forbidden = {"test_images", "test_mask_human"}
    if any(part.casefold() in forbidden for part in path.parts):
        raise ValueError(f"{field} must not point to an independent test directory")
    if path.name in INDEPENDENT_TEST_FILENAMES:
        raise ValueError(f"{field} must not reference an independent test sample")


def _validated_paths(namespace: argparse.Namespace) -> CalibrationPaths:
    threshold_root = namespace.threshold_calibration_root.expanduser().resolve(strict=True)
    mask_dir = namespace.mask_dir.expanduser().resolve(strict=True)
    output_root = _validate_output_root(namespace.output_root)
    if not threshold_root.is_dir():
        raise ValueError(f"threshold-calibration-root is not a directory: {threshold_root}")
    if not mask_dir.is_dir():
        raise ValueError(f"mask-dir is not a directory: {mask_dir}")
    _reject_test_path(threshold_root, field="threshold-calibration-root")
    _reject_test_path(mask_dir, field="mask-dir")
    for filename in VALIDATION_FILENAMES:
        if not (mask_dir / filename).is_file():
            raise ValueError(f"validation mask is missing: {filename}")
    return CalibrationPaths(threshold_root, mask_dir, output_root)


def _load_threshold_evidence(
    root: Path,
    *,
    expected_sha256: str | None = THRESHOLD_EVIDENCE_SHA256,
) -> dict[str, Any]:
    path = root / "threshold-calibration.json"
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        raise ValueError("threshold calibration evidence SHA-256 mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid threshold calibration evidence: {path}") from error
    fixed = payload.get("fixed_inference_contract")
    selected = payload.get("selected")
    expected = {
        "schema_version": "1",
        "model_id": MODEL_ID,
        "torchscript_sha256": TORCHSCRIPT_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "validation_images": list(VALIDATION_FILENAMES),
        "comparison_rule": "probability >= threshold",
        "selected_threshold": THRESHOLD,
    }
    mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    fixed_expected = {
        "bottom_crop_px": BOTTOM_CROP_PX,
        "threshold_comparison": "gte",
        "min_area_px_during_threshold_calibration": 0,
        "default_watershed_enabled": False,
    }
    if not isinstance(selected, Mapping) or selected.get("threshold") != THRESHOLD:
        mismatches["selected.threshold"] = {
            "expected": THRESHOLD,
            "observed": selected.get("threshold") if isinstance(selected, Mapping) else None,
        }
    if not isinstance(fixed, Mapping):
        mismatches["fixed_inference_contract"] = {"expected": "mapping", "observed": fixed}
    else:
        for key, value in fixed_expected.items():
            if fixed.get(key) != value:
                mismatches[f"fixed_inference_contract.{key}"] = {
                    "expected": value,
                    "observed": fixed.get(key),
                }
    if mismatches:
        raise ValueError(
            "threshold evidence does not match the frozen agglomerated contract: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    return payload


def _probability_paths(root: Path, evidence: Mapping[str, Any]) -> dict[str, Path]:
    artifacts = evidence.get("artifacts")
    declared = artifacts.get("probability_arrays") if isinstance(artifacts, Mapping) else None
    if not isinstance(declared, Mapping):
        raise ValueError("threshold evidence is missing probability array artifacts")
    resolved_root = root.resolve(strict=True)
    result: dict[str, Path] = {}
    for filename in VALIDATION_FILENAMES:
        item = declared.get(filename)
        if not isinstance(item, Mapping):
            raise ValueError(f"threshold evidence is missing cached probability: {filename}")
        raw_path = item.get("path")
        declared_sha = item.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(declared_sha, str):
            raise ValueError(f"invalid cached probability declaration: {filename}")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = resolved_root / path
        path = path.resolve(strict=True)
        try:
            path.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(f"cached probability is outside threshold root: {filename}") from error
        if path.suffix.lower() != ".npy" or _sha256(path) != declared_sha:
            raise ValueError(f"cached probability identity mismatch: {filename}")
        result[filename] = path
    return result


def _load_arrays(
    paths: CalibrationPaths,
    *,
    expected_size: tuple[int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT),
    expected_evidence_sha256: str | None = THRESHOLD_EVIDENCE_SHA256,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    evidence = _load_threshold_evidence(
        paths.threshold_calibration_root, expected_sha256=expected_evidence_sha256
    )
    probability_paths = _probability_paths(paths.threshold_calibration_root, evidence)
    validation_inputs = evidence.get("validation_inputs")
    if not isinstance(validation_inputs, Mapping):
        raise ValueError("threshold evidence is missing validation input identities")
    expected_shape = (expected_size[1], expected_size[0])
    probabilities: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    for filename in VALIDATION_FILENAMES:
        probability = np.load(probability_paths[filename], allow_pickle=False)
        mask_path = paths.mask_dir / filename
        input_identity = validation_inputs.get(filename)
        if not isinstance(input_identity, Mapping) or input_identity.get("mask_sha256") != _sha256(
            mask_path
        ):
            raise ValueError(f"validation mask identity mismatch: {filename}")
        with Image.open(mask_path) as image:
            pixels = np.asarray(image)
        if pixels.ndim == 2:
            target = pixels != 0
        elif pixels.ndim == 3:
            target = np.any(pixels != 0, axis=2)
        else:
            raise ValueError(f"validation mask has unsupported dimensions: {filename}")
        if probability.shape != expected_shape or target.shape != expected_shape:
            raise ValueError(f"probability/GT shape mismatch for {filename}")
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError(f"probability must be finite and in [0, 1]: {filename}")
        probabilities[filename] = np.asarray(probability, dtype=np.float32)
        targets[filename] = np.asarray(target, dtype=np.bool_)
    return probabilities, targets, evidence


def _profile(min_area_px: int) -> PostprocessProfile:
    return PostprocessProfile(
        profile_id="unet_agglomerated_min_area_calibration_v1",
        min_area_px=min_area_px,
        **POSTPROCESS_FIXED,
    )


def _roi(shape: tuple[int, ...], *, bottom_crop_px: int = BOTTOM_CROP_PX) -> np.ndarray:
    if len(shape) != 2 or shape[0] <= bottom_crop_px:
        raise ValueError(f"invalid full-image shape: {shape}")
    roi = np.ones(shape, dtype=np.bool_)
    roi[-bottom_crop_px:] = False
    return roi


def _union(instances: Sequence[NormalizedInstance], shape: tuple[int, ...]) -> np.ndarray:
    result = np.zeros(shape, dtype=np.bool_)
    for instance in instances:
        result |= instance.mask
    return result


def _dice(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = int(np.count_nonzero(prediction & target))
    denominator = int(np.count_nonzero(prediction)) + int(np.count_nonzero(target))
    return 2 * intersection / denominator if denominator else 1.0


def _mape(observed: float | int | None, baseline: float | int | None, *, name: str) -> float:
    if observed is None or baseline is None or not math.isfinite(float(baseline)):
        raise ValueError(f"{name} baseline is unavailable")
    if float(baseline) == 0:
        raise ValueError(f"{name} baseline is zero; MAPE is undefined")
    return abs(float(observed) - float(baseline)) / abs(float(baseline))


def _summarize(
    result: PostprocessResult,
    *,
    filename: str,
    roi_mask: np.ndarray,
    suffix: str,
) -> dict[str, int | float]:
    summary = measure(
        run_id=f"agglomerated-min-area-{Path(filename).stem}-{suffix}",
        instances=result.instances,
        roi_mask=roi_mask,
        scale_nm_per_pixel=SCALE_NM_PER_PIXEL,
        config=MorphometryConfig(perimeter_neighborhood=PERIMETER_NEIGHBORHOOD),
    ).image_summary
    if (
        summary.mean_equivalent_diameter_nm is None
        or summary.number_density_um2 is None
        or summary.perimeter_density_um is None
    ):
        raise ValueError(f"physical morphometry is unavailable for {filename}")
    return {
        "agglomerate_count": summary.particle_count,
        "mean_equivalent_diameter_nm": summary.mean_equivalent_diameter_nm,
        "perimeter_density_um": summary.perimeter_density_um,
        "number_density_um2": summary.number_density_um2,
        "coverage_ratio": summary.coverage_ratio,
        "excluded_border_count": result.excluded_border_count,
    }


def evaluate_min_areas(
    probabilities: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    *,
    candidates: Sequence[int] = CANDIDATE_MIN_AREAS,
    bottom_crop_px: int = BOTTOM_CROP_PX,
) -> list[dict[str, object]]:
    if tuple(probabilities) != VALIDATION_FILENAMES or tuple(targets) != VALIDATION_FILENAMES:
        raise ValueError(
            "probability and target inputs must match the four fixed validation fields"
        )
    if not candidates or any(isinstance(value, bool) or value < 0 for value in candidates):
        raise ValueError("min_area candidates must be non-negative integers")
    if len(set(candidates)) != len(candidates):
        raise ValueError("min_area candidates must be unique")

    baselines: dict[str, tuple[PostprocessResult, dict[str, int | float], np.ndarray]] = {}
    for filename in VALIDATION_FILENAMES:
        probability = np.asarray(probabilities[filename], dtype=np.float32)
        target = np.asarray(targets[filename], dtype=np.bool_)
        if probability.shape != target.shape or not np.isfinite(probability).all():
            raise ValueError(f"invalid probability/GT arrays for {filename}")
        roi_mask = _roi(probability.shape, bottom_crop_px=bottom_crop_px)
        baseline = normalize_semantic_mask_detailed(target, roi_mask=roi_mask, profile=_profile(0))
        baselines[filename] = (
            baseline,
            _summarize(baseline, filename=filename, roi_mask=roi_mask, suffix="gt"),
            roi_mask,
        )

    results: list[dict[str, object]] = []
    for min_area_px in candidates:
        images: list[dict[str, object]] = []
        retained_gt = baseline_gt = 0
        for filename in VALIDATION_FILENAMES:
            probability = np.asarray(probabilities[filename], dtype=np.float32)
            target = np.asarray(targets[filename], dtype=np.bool_)
            baseline_result, baseline_summary, roi_mask = baselines[filename]
            prediction = probability >= THRESHOLD
            predicted_result = normalize_semantic_mask_detailed(
                prediction,
                roi_mask=roi_mask,
                profile=_profile(int(min_area_px)),
                probability=probability,
            )
            predicted_summary = _summarize(
                predicted_result,
                filename=filename,
                roi_mask=roi_mask,
                suffix=f"pred-{min_area_px}",
            )
            retained_result = normalize_semantic_mask_detailed(
                target, roi_mask=roi_mask, profile=_profile(int(min_area_px))
            )
            retained_gt += len(retained_result.instances)
            baseline_gt += len(baseline_result.instances)
            mapes = {
                "count_mape": _mape(
                    predicted_summary["agglomerate_count"],
                    baseline_summary["agglomerate_count"],
                    name=f"{filename} agglomerate count",
                ),
                "mean_diameter_mape": _mape(
                    predicted_summary["mean_equivalent_diameter_nm"],
                    baseline_summary["mean_equivalent_diameter_nm"],
                    name=f"{filename} mean equivalent diameter",
                ),
                "perimeter_density_mape": _mape(
                    predicted_summary["perimeter_density_um"],
                    baseline_summary["perimeter_density_um"],
                    name=f"{filename} perimeter density",
                ),
            }
            images.append(
                {
                    "filename": filename,
                    **predicted_summary,
                    "gt_baseline": baseline_summary,
                    "gt_retained_count": len(retained_result.instances),
                    "gt_retention": (
                        len(retained_result.instances) / len(baseline_result.instances)
                        if baseline_result.instances
                        else 1.0
                    ),
                    **mapes,
                    "composite_mape": float(np.mean(list(mapes.values()))),
                    "dice": _dice(
                        _union(predicted_result.instances, probability.shape),
                        _union(baseline_result.instances, target.shape),
                    ),
                }
            )
        macro = {
            metric: float(np.mean([float(item[metric]) for item in images]))
            for metric in (
                "count_mape",
                "mean_diameter_mape",
                "perimeter_density_mape",
                "composite_mape",
                "dice",
            )
        }
        results.append(
            {
                "min_area_px": int(min_area_px),
                "physical_area_nm2": min_area_px * SCALE_NM_PER_PIXEL**2,
                "equivalent_diameter_nm": 2 * math.sqrt(min_area_px / math.pi) * SCALE_NM_PER_PIXEL,
                "images": images,
                "macro": macro,
                "gt_baseline_count": baseline_gt,
                "gt_retained_count": retained_gt,
                "gt_retention": retained_gt / baseline_gt if baseline_gt else 1.0,
            }
        )
    return results


def _macro_metric(result: Mapping[str, object], name: str) -> float:
    macro = result.get("macro")
    if not isinstance(macro, Mapping) or not isinstance(macro.get(name), int | float):
        raise ValueError(f"min_area result is missing macro {name}")
    return float(macro[name])


def select_min_area(results: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if not results:
        raise ValueError("min_area results are empty")
    best_composite = min(_macro_metric(item, "composite_mape") for item in results)
    winners = [
        item
        for item in results
        if math.isclose(_macro_metric(item, "composite_mape"), best_composite, abs_tol=1e-12)
    ]
    best_dice = max(_macro_metric(item, "dice") for item in winners)
    winners = [
        item
        for item in winners
        if math.isclose(_macro_metric(item, "dice"), best_dice, abs_tol=1e-12)
    ]
    smallest = min(int(item["min_area_px"]) for item in winners)
    winners = [item for item in winners if int(item["min_area_px"]) == smallest]
    if len(winners) != 1:
        raise ValueError("the prespecified min_area selection rule did not resolve the tie")
    return winners[0]


def _write_csv(path: Path, results: Sequence[Mapping[str, object]]) -> None:
    fields = [
        "min_area_px", "scope", "filename", "agglomerate_count",
        "mean_equivalent_diameter_nm", "perimeter_density_um", "number_density_um2",
        "coverage_ratio", "excluded_border_count", "count_mape", "mean_diameter_mape",
        "perimeter_density_mape", "composite_mape", "dice", "gt_retained_count",
        "gt_retention",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            images = result.get("images")
            macro = result.get("macro")
            if not isinstance(images, list) or not isinstance(macro, Mapping):
                raise ValueError("min_area result has an invalid output structure")
            for image in images:
                if not isinstance(image, Mapping):
                    raise ValueError("min_area image result is invalid")
                row = {key: image.get(key) for key in fields}
                row.update(min_area_px=result["min_area_px"], scope="image")
                writer.writerow(row)
            row = {key: macro.get(key) for key in fields}
            row.update(
                min_area_px=result["min_area_px"], scope="macro",
                gt_retained_count=result["gt_retained_count"], gt_retention=result["gt_retention"],
            )
            writer.writerow(row)


def _write_selected_reviews(
    output_root: Path,
    probabilities: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    selected_min_area: int,
    *,
    bottom_crop_px: int = BOTTOM_CROP_PX,
) -> tuple[dict[str, str], dict[str, str]]:
    prediction_dir = output_root / "selected-predictions"
    review_dir = output_root / "reviews"
    prediction_dir.mkdir(parents=True, exist_ok=False)
    review_dir.mkdir(parents=True, exist_ok=False)
    predictions: dict[str, str] = {}
    reviews: dict[str, str] = {}
    for filename in VALIDATION_FILENAMES:
        probability = np.asarray(probabilities[filename], dtype=np.float32)
        roi_mask = _roi(probability.shape, bottom_crop_px=bottom_crop_px)
        result = normalize_semantic_mask_detailed(
            probability >= THRESHOLD,
            roi_mask=roi_mask,
            profile=_profile(selected_min_area),
            probability=probability,
        )
        prediction = _union(result.instances, probability.shape)
        destination = prediction_dir / f"{Path(filename).stem}-pred-mask.png"
        Image.fromarray(prediction.astype(np.uint8) * 255).save(destination)
        valid_prediction = prediction[:-bottom_crop_px]
        valid_target = np.asarray(targets[filename], dtype=bool)[:-bottom_crop_px]
        gt_panel = np.repeat((valid_target.astype(np.uint8) * 255)[..., None], 3, axis=2)
        pred_panel = np.repeat((valid_prediction.astype(np.uint8) * 255)[..., None], 3, axis=2)
        error_panel = np.zeros((*valid_target.shape, 3), dtype=np.uint8)
        error_panel[valid_prediction & valid_target] = (0, 180, 0)
        error_panel[valid_prediction & ~valid_target] = (255, 0, 0)
        error_panel[~valid_prediction & valid_target] = (0, 80, 255)
        review = np.concatenate((gt_panel, pred_panel, error_panel), axis=1)
        review_path = review_dir / f"{Path(filename).stem}-gt-pred-error.png"
        Image.fromarray(review, mode="RGB").save(review_path)
        predictions[filename] = str(destination)
        reviews[filename] = str(review_path)
    return predictions, reviews


def run(
    paths: CalibrationPaths,
    *,
    expected_size: tuple[int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT),
    bottom_crop_px: int = BOTTOM_CROP_PX,
    expected_evidence_sha256: str | None = THRESHOLD_EVIDENCE_SHA256,
) -> dict[str, object]:
    output_root = _validate_output_root(paths.output_root)
    probabilities, targets, threshold_evidence = _load_arrays(
        paths,
        expected_size=expected_size,
        expected_evidence_sha256=expected_evidence_sha256,
    )
    results = evaluate_min_areas(probabilities, targets, bottom_crop_px=bottom_crop_px)
    selected = select_min_area(results)
    output_root.mkdir(parents=True, exist_ok=False)
    csv_path = output_root / "min-area-calibration.csv"
    json_path = output_root / "min-area-calibration.json"
    predictions, reviews = _write_selected_reviews(
        output_root,
        probabilities,
        targets,
        int(selected["min_area_px"]),
        bottom_crop_px=bottom_crop_px,
    )
    _write_csv(csv_path, results)
    payload: dict[str, object] = {
        "schema_version": "1",
        "created_at": datetime.now(UTC).isoformat(),
        "model_id": MODEL_ID,
        "torchscript_sha256": TORCHSCRIPT_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "threshold_calibration_evidence": {
            "path": str(paths.threshold_calibration_root / "threshold-calibration.json"),
            "sha256": _sha256(paths.threshold_calibration_root / "threshold-calibration.json"),
            "selected_threshold": threshold_evidence["selected_threshold"],
        },
        "probability_source": (
            "threshold calibration cached .npy arrays; no model loading or inference"
        ),
        "validation_images": list(VALIDATION_FILENAMES),
        "validation_scope": "four field-of-view validation images; not sample-level independent",
        "fixed_parameters": {
            "threshold": THRESHOLD,
            "threshold_comparison": "gte",
            "bottom_crop_px": bottom_crop_px,
            "effective_image_size_px": [expected_size[0], expected_size[1] - bottom_crop_px],
            "scale_nm_per_pixel": SCALE_NM_PER_PIXEL,
            **POSTPROCESS_FIXED,
            "perimeter_neighborhood": PERIMETER_NEIGHBORHOOD,
        },
        "candidate_min_area_px": list(CANDIDATE_MIN_AREAS),
        "unit_conversion": {
            "area_nm2": "min_area_px * scale_nm_per_pixel^2",
            "equivalent_diameter_nm": "2 * sqrt(min_area_px / pi) * scale_nm_per_pixel",
        },
        "gt_policy": "GT uses min_area_px=0; candidate-filtered GT is diagnostic retention only",
        "selection_rule": [
            "minimize four-image macro composite MAPE",
            (
                "composite MAPE equally weights agglomerate count, mean equivalent diameter, "
                "and perimeter density"
            ),
            "then maximize macro Dice",
            "then choose the smaller min_area_px",
            "fail if the prespecified rules still leave a tie",
        ],
        "selected": selected,
        "candidate_results": results,
        "artifacts": {
            "json": str(json_path),
            "csv": str(csv_path),
            "selected_prediction_masks": predictions,
            "reviews": reviews,
        },
        "limitations": [
            "The scientific object is each whole agglomerate, not its internal primary particles.",
            (
                "Validation contains only four field-of-view images and is not "
                "sample-level independent."
            ),
            (
                "YCu-1.tif, YCu-2.tif, and YCu-3.tif were not read and remain reserved "
                "for independent test."
            ),
            "Threshold 0.25 was frozen before this min_area_px scan and was not reselected.",
            "This evidence selects min_area_px only and does not establish registry readiness.",
        ],
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-calibration-root", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        paths = _validated_paths(build_parser().parse_args(argv))
        payload = run(paths)
    except Exception as error:
        print(
            json.dumps(
                {"status": "error", "error_type": type(error).__name__, "message": str(error)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

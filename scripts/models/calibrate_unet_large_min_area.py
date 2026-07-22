"""Calibrate large U-Net min_area_px from cached validation probabilities."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
from app.analysis.visualization import write_review_visualizations

MODEL_ID = "unet-large-optimized-v1"
TORCHSCRIPT_SHA256 = "007d9a16bf31e5f960160c52eefa938b83feeac2e6c0d7dec9c8670a38626e05"
THRESHOLD = 0.50
BOTTOM_CROP_PX = 180
SCALE_NM_PER_PIXEL = 100.0 / 184.0
SEED = 2026
VALIDATION_FILENAMES = (
    "NdZn-2.tif",
    "LaMn-3.tif",
    "LaMn-1.tif",
    "BaCo-3.tif",
    "BaCu-1.tif",
    "BaCr-3.tif",
)
CANDIDATE_MIN_AREAS = (0, 16, 32, 64, 128, 256, 512, 1024)
POSTPROCESS_FIXED = {
    "fill_holes": True,
    "watershed_enabled": False,
    "exclude_border": True,
    "connectivity": 2,
}
PERIMETER_NEIGHBORHOOD = 8


@dataclass(frozen=True, slots=True)
class CalibrationPaths:
    threshold_calibration_root: Path
    image_dir: Path
    mask_dir: Path
    output_root: Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _validate_output_root(output_root: Path, *, repository: Path | None = None) -> Path:
    resolved = output_root.expanduser().resolve(strict=False)
    repository = (repository or _repository_root()).resolve(strict=True)
    if resolved.exists():
        raise ValueError(f"output-root already exists: {resolved}")
    if resolved == repository or resolved.is_relative_to(repository):
        raise ValueError("output-root must be outside the repository")
    return resolved


def _validated_paths(namespace: argparse.Namespace) -> CalibrationPaths:
    threshold_root = namespace.threshold_calibration_root.expanduser().resolve(strict=True)
    image_dir = namespace.image_dir.expanduser().resolve(strict=True)
    mask_dir = namespace.mask_dir.expanduser().resolve(strict=True)
    output_root = _validate_output_root(namespace.output_root)
    for label, directory in (
        ("threshold-calibration-root", threshold_root),
        ("image-dir", image_dir),
        ("mask-dir", mask_dir),
    ):
        if not directory.is_dir():
            raise ValueError(f"{label} is not a directory: {directory}")
    for filename in VALIDATION_FILENAMES:
        if not (image_dir / filename).is_file():
            raise ValueError(f"validation image is missing: {filename}")
        if not (mask_dir / filename).is_file():
            raise ValueError(f"validation mask is missing: {filename}")
    return CalibrationPaths(threshold_root, image_dir, mask_dir, output_root)


def _load_threshold_evidence(root: Path) -> dict[str, Any]:
    evidence_path = root / "threshold-calibration.json"
    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid threshold calibration evidence: {evidence_path}") from error
    expected = {
        "model_id": MODEL_ID,
        "torchscript_sha256": TORCHSCRIPT_SHA256,
        "validation_images": list(VALIDATION_FILENAMES),
        "bottom_crop_px": BOTTOM_CROP_PX,
        "comparison_rule": "probability > threshold",
    }
    mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    selected = payload.get("selected")
    selected_threshold = selected.get("threshold") if isinstance(selected, Mapping) else None
    if selected_threshold != THRESHOLD:
        mismatches["selected.threshold"] = {
            "expected": THRESHOLD,
            "observed": selected_threshold,
        }
    if mismatches:
        raise ValueError(
            "threshold calibration evidence does not match the frozen large contract: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    return payload


def _probability_paths(root: Path, evidence: Mapping[str, Any]) -> dict[str, Path]:
    artifacts = evidence.get("artifacts")
    declared = artifacts.get("probabilities") if isinstance(artifacts, Mapping) else None
    if not isinstance(declared, Mapping):
        raise ValueError("threshold evidence is missing probability artifacts")
    resolved_root = root.resolve(strict=True)
    result: dict[str, Path] = {}
    for filename in VALIDATION_FILENAMES:
        raw_path = declared.get(filename)
        if not isinstance(raw_path, str):
            raise ValueError(f"threshold evidence is missing cached probability: {filename}")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = resolved_root / path
        path = path.resolve(strict=True)
        try:
            path.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(f"cached probability is outside threshold root: {filename}") from error
        if path.suffix.lower() != ".npy":
            raise ValueError(f"cached probability is not .npy: {filename}")
        result[filename] = path
    return result


def _load_arrays(paths: CalibrationPaths) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    evidence = _load_threshold_evidence(paths.threshold_calibration_root)
    probability_paths = _probability_paths(paths.threshold_calibration_root, evidence)
    probabilities: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    for filename in VALIDATION_FILENAMES:
        probability = np.load(probability_paths[filename], allow_pickle=False)
        with Image.open(paths.mask_dir / filename) as image:
            target = np.asarray(image.convert("L")) > 0
        if probability.shape != target.shape:
            raise ValueError(f"probability/GT shape mismatch for {filename}")
        if probability.ndim != 2 or probability.shape[0] <= BOTTOM_CROP_PX:
            raise ValueError(f"invalid full-image shape for {filename}: {probability.shape}")
        if not np.isfinite(probability).all():
            raise ValueError(f"probability contains non-finite values for {filename}")
        probabilities[filename] = np.asarray(probability, dtype=np.float32)
        targets[filename] = np.asarray(target, dtype=np.bool_)
    return probabilities, targets


def _profile(min_area_px: int) -> PostprocessProfile:
    return PostprocessProfile(
        profile_id="unet_large_min_area_calibration_v1",
        min_area_px=min_area_px,
        **POSTPROCESS_FIXED,
    )


def _roi(shape: tuple[int, ...]) -> np.ndarray:
    if len(shape) != 2 or shape[0] <= BOTTOM_CROP_PX:
        raise ValueError(f"invalid full-image shape: {shape}")
    roi = np.ones(shape, dtype=np.bool_)
    roi[-BOTTOM_CROP_PX:] = False
    return roi


def _union(instances: Sequence[NormalizedInstance], shape: tuple[int, ...]) -> np.ndarray:
    union = np.zeros(shape, dtype=np.bool_)
    for instance in instances:
        union |= instance.mask
    return union


def _dice(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = int(np.count_nonzero(prediction & target))
    denominator = int(np.count_nonzero(prediction)) + int(np.count_nonzero(target))
    return 2.0 * intersection / denominator if denominator else 1.0


def _mape(observed: float | int | None, baseline: float | int | None, *, name: str) -> float:
    if observed is None or baseline is None or not math.isfinite(float(baseline)):
        raise ValueError(f"{name} baseline is unavailable")
    if float(baseline) == 0.0:
        raise ValueError(f"{name} baseline is zero; MAPE is undefined")
    return abs(float(observed) - float(baseline)) / abs(float(baseline))


def _summarize(
    result: PostprocessResult,
    *,
    filename: str,
    roi_mask: np.ndarray,
    run_suffix: str,
) -> dict[str, int | float]:
    measured = measure(
        run_id=f"calibration-{Path(filename).stem}-{run_suffix}",
        instances=result.instances,
        roi_mask=roi_mask,
        scale_nm_per_pixel=SCALE_NM_PER_PIXEL,
        config=MorphometryConfig(perimeter_neighborhood=PERIMETER_NEIGHBORHOOD),
    ).image_summary
    if (
        measured.mean_equivalent_diameter_nm is None
        or measured.number_density_um2 is None
        or measured.perimeter_density_um is None
    ):
        raise ValueError(f"physical morphometry is unavailable for {filename}")
    return {
        "particle_count": measured.particle_count,
        "mean_equivalent_diameter_nm": measured.mean_equivalent_diameter_nm,
        "perimeter_density_um": measured.perimeter_density_um,
        "number_density_um2": measured.number_density_um2,
        "coverage_ratio": measured.coverage_ratio,
        "excluded_border_count": result.excluded_border_count,
    }


def evaluate_min_areas(
    probabilities: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    *,
    candidates: Sequence[int] = CANDIDATE_MIN_AREAS,
) -> list[dict[str, object]]:
    if tuple(probabilities) != tuple(targets):
        raise ValueError("probability and target image order differs")
    if not candidates or any(value < 0 for value in candidates):
        raise ValueError("min_area candidates must be non-negative")

    baselines: dict[str, tuple[PostprocessResult, dict[str, int | float], np.ndarray]] = {}
    for filename in probabilities:
        probability = np.asarray(probabilities[filename], dtype=np.float32)
        target = np.asarray(targets[filename], dtype=np.bool_)
        if probability.shape != target.shape:
            raise ValueError(f"probability/GT shape mismatch for {filename}")
        if not np.isfinite(probability).all():
            raise ValueError(f"probability contains non-finite values for {filename}")
        roi_mask = _roi(probability.shape)
        baseline = normalize_semantic_mask_detailed(
            target,
            roi_mask=roi_mask,
            profile=_profile(0),
        )
        baselines[filename] = (
            baseline,
            _summarize(baseline, filename=filename, roi_mask=roi_mask, run_suffix="gt"),
            roi_mask,
        )

    results: list[dict[str, object]] = []
    for min_area_px in candidates:
        image_results: list[dict[str, object]] = []
        retained_gt = baseline_gt = 0
        for filename in probabilities:
            probability = np.asarray(probabilities[filename], dtype=np.float32)
            target = np.asarray(targets[filename], dtype=np.bool_)
            baseline_result, baseline_summary, roi_mask = baselines[filename]
            prediction = probability > THRESHOLD
            predicted_result = normalize_semantic_mask_detailed(
                prediction,
                roi_mask=roi_mask,
                profile=_profile(min_area_px),
                probability=probability,
            )
            predicted_summary = _summarize(
                predicted_result,
                filename=filename,
                roi_mask=roi_mask,
                run_suffix=f"pred-{min_area_px}",
            )
            retained_result = normalize_semantic_mask_detailed(
                target,
                roi_mask=roi_mask,
                profile=_profile(min_area_px),
            )
            retained_gt += len(retained_result.instances)
            baseline_gt += len(baseline_result.instances)
            mapes = {
                "count_mape": _mape(
                    predicted_summary["particle_count"],
                    baseline_summary["particle_count"],
                    name=f"{filename} particle count",
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
            image_results.append(
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
            metric: float(np.mean([float(item[metric]) for item in image_results]))
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
                "min_area_px": min_area_px,
                "physical_area_nm2": min_area_px * SCALE_NM_PER_PIXEL**2,
                "equivalent_diameter_nm": (
                    2.0 * math.sqrt(min_area_px / math.pi) * SCALE_NM_PER_PIXEL
                ),
                "images": image_results,
                "macro": macro,
                "gt_baseline_count": baseline_gt,
                "gt_retained_count": retained_gt,
                "gt_retention": retained_gt / baseline_gt if baseline_gt else 1.0,
            }
        )
    return results


def _macro_metric(result: Mapping[str, object], name: str) -> float:
    macro = result.get("macro")
    if not isinstance(macro, Mapping) or name not in macro:
        raise ValueError(f"min_area result is missing macro {name}")
    return float(macro[name])


def select_min_area(results: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if not results:
        raise ValueError("min_area results are empty")
    best_composite = min(_macro_metric(item, "composite_mape") for item in results)
    composite_winners = [
        item
        for item in results
        if _macro_metric(item, "composite_mape") == best_composite
    ]
    best_dice = max(_macro_metric(item, "dice") for item in composite_winners)
    dice_winners = [
        item for item in composite_winners if _macro_metric(item, "dice") == best_dice
    ]
    smallest_area = min(int(item["min_area_px"]) for item in dice_winners)
    winners = [item for item in dice_winners if int(item["min_area_px"]) == smallest_area]
    if len(winners) != 1:
        raise ValueError("the prespecified min_area selection rule did not resolve the tie")
    return winners[0]


def _write_csv(path: Path, results: Sequence[Mapping[str, object]]) -> None:
    fields = [
        "min_area_px",
        "scope",
        "filename",
        "particle_count",
        "mean_equivalent_diameter_nm",
        "perimeter_density_um",
        "number_density_um2",
        "coverage_ratio",
        "excluded_border_count",
        "count_mape",
        "mean_diameter_mape",
        "perimeter_density_mape",
        "composite_mape",
        "dice",
        "gt_retained_count",
        "gt_retention",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            image_results = result.get("images")
            macro = result.get("macro")
            if not isinstance(image_results, list) or not isinstance(macro, Mapping):
                raise ValueError("min_area result has an invalid output structure")
            for image_result in image_results:
                if not isinstance(image_result, Mapping):
                    raise ValueError("min_area image result is invalid")
                row = {key: image_result.get(key) for key in fields}
                row.update(min_area_px=result["min_area_px"], scope="image")
                writer.writerow(row)
            macro_row = {key: macro.get(key) for key in fields}
            macro_row.update(
                min_area_px=result["min_area_px"],
                scope="macro",
                gt_retained_count=result["gt_retained_count"],
                gt_retention=result["gt_retention"],
            )
            writer.writerow(
                macro_row
            )


def _write_selected_visualizations(
    *,
    paths: CalibrationPaths,
    probabilities: Mapping[str, np.ndarray],
    selected_min_area: int,
) -> dict[str, dict[str, str]]:
    artifacts: dict[str, dict[str, str]] = {}
    for filename in VALIDATION_FILENAMES:
        probability = np.asarray(probabilities[filename], dtype=np.float32)
        roi_mask = _roi(probability.shape)
        result = normalize_semantic_mask_detailed(
            probability > THRESHOLD,
            roi_mask=roi_mask,
            profile=_profile(selected_min_area),
            probability=probability,
        )
        union = _union(result.instances, probability.shape)
        output_dir = paths.output_root / "selected-visualizations" / Path(filename).stem
        overlay, labeled = write_review_visualizations(
            image_path=paths.image_dir / filename,
            binary_mask=union,
            instances=result.instances,
            output_dir=output_dir,
        )
        artifacts[filename] = {"overlay": str(overlay), "labeled_particles": str(labeled)}
    return artifacts


def run(paths: CalibrationPaths) -> dict[str, object]:
    output_root = _validate_output_root(paths.output_root)
    probabilities, targets = _load_arrays(paths)
    results = evaluate_min_areas(probabilities, targets)
    selected = select_min_area(results)
    output_root.mkdir(parents=True, exist_ok=False)
    paths = CalibrationPaths(
        paths.threshold_calibration_root,
        paths.image_dir,
        paths.mask_dir,
        output_root,
    )
    visualizations = _write_selected_visualizations(
        paths=paths,
        probabilities=probabilities,
        selected_min_area=int(selected["min_area_px"]),
    )
    csv_path = output_root / "min-area-calibration.csv"
    json_path = output_root / "min-area-calibration.json"
    _write_csv(csv_path, results)
    payload: dict[str, object] = {
        "model_id": MODEL_ID,
        "torchscript_sha256": TORCHSCRIPT_SHA256,
        "probability_source": "threshold-calibration-v1 cached .npy; no repeated inference",
        "validation_images": list(VALIDATION_FILENAMES),
        "validation_scope": "field-of-view validation; not sample-level independent",
        "known_domain_biases": [
            "BaCo-3 has known under-segmentation",
            "BaCu-1 has known false-positive segmentation",
        ],
        "fixed_parameters": {
            "threshold": THRESHOLD,
            "threshold_comparison": "gt",
            "bottom_crop_px": BOTTOM_CROP_PX,
            "scale_nm_per_pixel": SCALE_NM_PER_PIXEL,
            "seed": SEED,
            **POSTPROCESS_FIXED,
            "perimeter_neighborhood": PERIMETER_NEIGHBORHOOD,
        },
        "candidate_min_area_px": list(CANDIDATE_MIN_AREAS),
        "unit_conversion": {
            "area_nm2": "min_area_px * scale_nm_per_pixel^2",
            "equivalent_diameter_nm": (
                "2 * sqrt(min_area_px / pi) * scale_nm_per_pixel"
            ),
        },
        "gt_policy": (
            "GT metrics use min_area_px=0; candidate-filtered GT is diagnostic retention only"
        ),
        "selection_rule": [
            "minimize six-image macro composite MAPE",
            "composite MAPE equally weights count, mean equivalent diameter, and perimeter density",
            "then maximize macro Dice",
            "then choose the smaller min_area_px",
            "fail if the prespecified rules still leave a tie",
        ],
        "selected": selected,
        "candidate_results": results,
        "artifacts": {
            "json": str(json_path),
            "csv": str(csv_path),
            "selected_visualizations": visualizations,
        },
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-calibration-root", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
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

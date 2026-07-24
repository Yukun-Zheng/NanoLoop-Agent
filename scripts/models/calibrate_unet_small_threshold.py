"""Calibrate the Small U-Net threshold from the private calibration split only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from app.contracts.analysis_config import MorphometryConfig, PostprocessProfile
from scripts.models.evaluate_unet_large_independent_test import compute_scientific_metrics
from scripts.models.small_b_contracts import (
    ManifestSplit,
    SmallBSplitManifest,
    SplitManifestRecord,
    load_split_manifest,
)

MODEL_ID = "unet-small-balanced-v1"
BOTTOM_CROP_PX = 130
_CALIBRATION_METRICS = {
    "instance_precision",
    "instance_recall",
    "instance_f1",
    "count_absolute_error",
    "count_relative_error",
}
_DIRECTIONS = {"maximize", "minimize", "closest_to"}


class ProbabilityProvider(Protocol):
    """Return one full-image probability array for one calibration record."""

    def __call__(self, record: SplitManifestRecord) -> np.ndarray: ...


class TruthProvider(Protocol):
    """Return one full-image binary GT array for one calibration record."""

    def __call__(self, record: SplitManifestRecord) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class SelectionRule:
    metric: str
    direction: str
    target: float | None = None


@dataclass(frozen=True, slots=True)
class ThresholdCalibrationPlan:
    candidate_thresholds: tuple[float, ...]
    selection_metric: SelectionRule
    ordered_tie_break: tuple[SelectionRule, ...]
    bottom_crop_px: int = BOTTOM_CROP_PX


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _selection_rule(value: object, *, field: str) -> SelectionRule:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    metric = value.get("metric")
    direction = value.get("direction")
    if not isinstance(metric, str) or metric not in _CALIBRATION_METRICS | {"threshold"}:
        raise ValueError(f"{field}.metric is not supported")
    if not isinstance(direction, str) or direction not in _DIRECTIONS:
        raise ValueError(f"{field}.direction must be one of: {sorted(_DIRECTIONS)}")
    target_value = value.get("target")
    target = (
        _finite_number(target_value, field=f"{field}.target")
        if target_value is not None
        else None
    )
    if direction == "closest_to" and target is None:
        raise ValueError(f"{field}.target is required for closest_to")
    if direction != "closest_to" and target is not None:
        raise ValueError(f"{field}.target is only allowed for closest_to")
    if metric == "threshold" and direction not in {"minimize", "maximize", "closest_to"}:
        raise ValueError(f"{field} has an invalid threshold direction")
    if metric != "threshold" and direction == "closest_to":
        raise ValueError(f"{field}.closest_to is only supported for threshold")
    return SelectionRule(metric=metric, direction=direction, target=target)


def load_threshold_plan(path: Path) -> ThresholdCalibrationPlan:
    """Read and validate a frozen Small-B threshold calibration plan."""

    resolved = path.expanduser().resolve(strict=True)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid threshold calibration plan JSON: {resolved}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("threshold calibration plan must be a JSON object")

    raw_candidates = payload.get("candidate_thresholds")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("candidate_thresholds must be a non-empty array")
    candidates = tuple(
        _finite_number(value, field=f"candidate_thresholds[{index}]")
        for index, value in enumerate(raw_candidates)
    )
    if any(not 0.0 < value < 1.0 for value in candidates):
        raise ValueError("candidate_thresholds must all be in the open interval (0, 1)")
    if any(left >= right for left, right in pairwise(candidates)):
        raise ValueError("candidate_thresholds must be unique and strictly increasing")

    bottom_crop_px = payload.get("bottom_crop_px")
    if bottom_crop_px != BOTTOM_CROP_PX:
        raise ValueError(f"bottom_crop_px must be frozen at {BOTTOM_CROP_PX}")
    selection = _selection_rule(payload.get("selection_metric"), field="selection_metric")
    if selection.metric == "threshold":
        raise ValueError("selection_metric must be a canonical scientific metric")
    raw_tie_break = payload.get("ordered_tie_break")
    if not isinstance(raw_tie_break, list):
        raise ValueError("ordered_tie_break must be an array")
    tie_break = tuple(
        _selection_rule(value, field=f"ordered_tie_break[{index}]")
        for index, value in enumerate(raw_tie_break)
    )
    return ThresholdCalibrationPlan(
        candidate_thresholds=candidates,
        selection_metric=selection,
        ordered_tie_break=tie_break,
        bottom_crop_px=bottom_crop_px,
    )


def _metric_values(metrics: Mapping[str, Any]) -> dict[str, float]:
    matching = metrics["instance_matching"]
    count = metrics["count"]
    return {
        "instance_precision": float(matching["precision"]),
        "instance_recall": float(matching["recall"]),
        "instance_f1": float(matching["f1"]),
        "count_absolute_error": float(count["absolute_error"]),
        "count_relative_error": float(count["relative_error"]),
    }


def _postprocess_profile() -> PostprocessProfile:
    return PostprocessProfile(
        profile_id="unet_small_threshold_calibration_v1",
        min_area_px=0,
        fill_holes=True,
        watershed_enabled=False,
        exclude_border=True,
        connectivity=2,
        instance_iou_threshold=0.7,
    )


def _validate_arrays(
    probability: np.ndarray,
    truth: np.ndarray,
    *,
    sample_id: str,
    bottom_crop_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    probability = np.asarray(probability, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.bool_)
    if probability.shape != truth.shape:
        raise ValueError(f"probability/GT shape mismatch for {sample_id}")
    if probability.ndim != 2 or probability.shape[0] <= bottom_crop_px:
        raise ValueError(f"invalid full-image shape for {sample_id}: {probability.shape}")
    if not np.isfinite(probability).all():
        raise ValueError(f"probability contains non-finite values for {sample_id}")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError(f"probability is outside [0, 1] for {sample_id}")
    return probability[:-bottom_crop_px], truth[:-bottom_crop_px]


def _rule_value(result: Mapping[str, Any], rule: SelectionRule) -> float:
    if rule.metric == "threshold":
        value = float(result["threshold"])
    else:
        value = float(result["aggregate"][rule.metric])
    return abs(value - rule.target) if rule.direction == "closest_to" else value


def select_threshold(
    results: Sequence[Mapping[str, Any]],
    plan: ThresholdCalibrationPlan,
) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    """Apply the prespecified ordered selection rules and fail on an unresolved tie."""

    if not results:
        raise ValueError("threshold results are empty")
    winners = list(results)
    trace: list[dict[str, Any]] = []
    for rule in (plan.selection_metric, *plan.ordered_tie_break):
        values = [_rule_value(result, rule) for result in winners]
        best = max(values) if rule.direction == "maximize" else min(values)
        winners = [
            result for result in winners if math.isclose(_rule_value(result, rule), best)
        ]
        trace.append(
            {
                "metric": rule.metric,
                "direction": rule.direction,
                "target": rule.target,
                "best_value": best,
                "remaining_thresholds": [float(result["threshold"]) for result in winners],
            }
        )
        if len(winners) == 1:
            return winners[0], trace
    raise ValueError("the prespecified threshold selection rules did not resolve the tie")


def evaluate_threshold_calibration(
    manifest: SmallBSplitManifest,
    plan: ThresholdCalibrationPlan,
    *,
    probability_provider: ProbabilityProvider,
    truth_provider: TruthProvider,
    manifest_sha256: str,
    plan_sha256: str,
) -> dict[str, Any]:
    """Evaluate all candidates from calibration rows using one inference per image."""

    records = manifest.select(ManifestSplit.CALIBRATION)
    if not records:
        raise ValueError("calibration split contains no included records")

    cached: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for record in records:
        probability = probability_provider(record)
        truth = truth_provider(record)
        cached[record.sample_id] = _validate_arrays(
            probability,
            truth,
            sample_id=record.sample_id,
            bottom_crop_px=plan.bottom_crop_px,
        )

    profile = _postprocess_profile()
    morphometry = MorphometryConfig(perimeter_neighborhood=8)
    candidate_results: list[dict[str, Any]] = []
    for threshold in plan.candidate_thresholds:
        per_image: list[dict[str, Any]] = []
        for record in records:
            probability, truth = cached[record.sample_id]
            prediction = probability > threshold
            scientific = compute_scientific_metrics(
                prediction,
                truth,
                profile=profile,
                morphometry=morphometry,
                scale_nm_per_pixel=1.0,
                iou_threshold=profile.instance_iou_threshold,
                sample_id=record.sample_id,
            )
            per_image.append(
                {
                    "sample_id": record.sample_id,
                    "image_sha256": record.image_sha256,
                    "mask_sha256": record.mask_sha256,
                    **_metric_values(scientific),
                }
            )
        aggregate = {
            metric: float(statistics.fmean(item[metric] for item in per_image))
            for metric in sorted(_CALIBRATION_METRICS)
        }
        candidate_results.append(
            {
                "threshold": threshold,
                "per_image": per_image,
                "aggregate": aggregate,
            }
        )

    selected, trace = select_threshold(candidate_results, plan)
    return {
        "candidate_results": candidate_results,
        "selected_threshold": float(selected["threshold"]),
        "selection_status": "SELECTED",
        "selection_trace": trace,
        "calibration_sample_ids": [record.sample_id for record in records],
        "calibration_inputs": [
            {
                "sample_id": record.sample_id,
                "image_sha256": record.image_sha256,
                "mask_sha256": record.mask_sha256,
            }
            for record in records
        ],
        "split_manifest_sha256": manifest_sha256,
        "plan_sha256": plan_sha256,
        "independent_test_accessed": False,
        "comparison_rule": "probability > threshold",
        "prediction_min_area_px": 0,
        "ground_truth_min_area_px": 0,
        "bottom_crop_px": plan.bottom_crop_px,
    }


def not_evaluated_payload(
    manifest: SmallBSplitManifest,
    plan: ThresholdCalibrationPlan,
    *,
    manifest_sha256: str,
    plan_sha256: str,
) -> dict[str, Any]:
    records = manifest.select(ManifestSplit.CALIBRATION)
    return {
        "schema_version": "1",
        "model_id": MODEL_ID,
        "selection_status": "NOT_EVALUATED",
        "selected_threshold": None,
        "candidate_thresholds": list(plan.candidate_thresholds),
        "selection_metric": {
            "metric": plan.selection_metric.metric,
            "direction": plan.selection_metric.direction,
        },
        "ordered_tie_break": [
            {"metric": rule.metric, "direction": rule.direction, "target": rule.target}
            for rule in plan.ordered_tie_break
        ],
        "bottom_crop_px": plan.bottom_crop_px,
        "comparison_rule": "probability > threshold",
        "prediction_min_area_px": 0,
        "ground_truth_min_area_px": 0,
        "split_manifest_sha256": manifest_sha256,
        "plan_sha256": plan_sha256,
        "calibration_inputs": [
            {
                "sample_id": record.sample_id,
                "image_sha256": record.image_sha256,
                "mask_sha256": record.mask_sha256,
            }
            for record in records
        ],
        "independent_test_accessed": False,
        "candidate_results": [],
        "selection_trace": [],
    }


def write_calibration_outputs(output_root: Path, payload: Mapping[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=False)
    json_path = output_root / "threshold-calibration.json"
    csv_path = output_root / "threshold-calibration.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        columns = (
            "scope",
            "sample_id",
            "threshold",
            *sorted(_CALIBRATION_METRICS),
        )
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for candidate in payload.get("candidate_results", []):
            for item in candidate["per_image"]:
                writer.writerow(
                    {
                        "scope": "per_image",
                        "sample_id": item["sample_id"],
                        "threshold": candidate["threshold"],
                        **{metric: item[metric] for metric in _CALIBRATION_METRICS},
                    }
                )
            writer.writerow(
                {
                    "scope": "aggregate",
                    "sample_id": "",
                    "threshold": candidate["threshold"],
                    **candidate["aggregate"],
                }
            )
    complete_payload = {
        **payload,
        "artifacts": {
            "json": json_path.name,
            "csv": {
                "path": csv_path.name,
                "sha256": _sha256(csv_path),
            },
        },
    }
    json_path.write_text(
        json.dumps(complete_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--not-evaluated",
        action="store_true",
        help="Write the validated Phase-1 contract without accessing private data or a model.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    try:
        manifest = load_split_manifest(namespace.split_manifest)
        plan = load_threshold_plan(namespace.plan)
        if not namespace.not_evaluated:
            raise ValueError(
                "real calibration requires an approved private probability provider; "
                "use the Python evaluate_threshold_calibration interface"
            )
        payload = not_evaluated_payload(
            manifest,
            plan,
            manifest_sha256=_sha256(namespace.split_manifest.resolve(strict=True)),
            plan_sha256=_sha256(namespace.plan.resolve(strict=True)),
        )
        write_calibration_outputs(namespace.output_root, payload)
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

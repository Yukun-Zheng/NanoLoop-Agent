"""Calibrate Small U-Net min-area from frozen threshold evidence and cached probabilities."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from app.analysis.postprocessing import normalize_semantic_mask_detailed
from app.contracts.analysis_config import MorphometryConfig, PostprocessProfile
from scripts.models.calibrate_unet_small_threshold import SelectionRule
from scripts.models.evaluate_unet_large_independent_test import compute_scientific_metrics
from scripts.models.small_b_contracts import (
    ManifestSplit,
    SmallBSplitManifest,
    SplitManifestRecord,
)

BOTTOM_CROP_PX = 130
_SHA256_LENGTH = 64
_SELECTION_METRICS = {
    "instance_precision",
    "instance_recall",
    "instance_f1",
    "count_absolute_error",
    "count_relative_error",
    "gt_retention",
}
_DIRECTIONS = {"maximize", "minimize", "closest_to"}


class ProbabilityCacheProvider(Protocol):
    """Load one threshold-stage probability cache without running inference."""

    def __call__(self, record: SplitManifestRecord) -> np.ndarray: ...


class TruthProvider(Protocol):
    def __call__(self, record: SplitManifestRecord) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class MinAreaCalibrationPlan:
    candidate_min_area_px: tuple[int, ...]
    selection_metric: SelectionRule
    ordered_tie_break: tuple[SelectionRule, ...]
    minimum_gt_retention: float
    threshold_evidence_sha256: str


@dataclass(frozen=True, slots=True)
class ThresholdEvidence:
    payload: Mapping[str, Any]
    sha256: str
    selected_threshold: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be 64 lowercase hexadecimal characters")
    return value


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
    if not isinstance(metric, str) or metric not in _SELECTION_METRICS | {"min_area_px"}:
        raise ValueError(f"{field}.metric is not supported")
    if not isinstance(direction, str) or direction not in _DIRECTIONS:
        raise ValueError(f"{field}.direction must be one of: {sorted(_DIRECTIONS)}")
    target_value = value.get("target")
    target = (
        _finite_number(target_value, field=f"{field}.target")
        if target_value is not None
        else None
    )
    if direction == "closest_to" and (metric != "min_area_px" or target is None):
        raise ValueError(f"{field}.closest_to requires min_area_px and a target")
    if direction != "closest_to" and target is not None:
        raise ValueError(f"{field}.target is only allowed for closest_to")
    return SelectionRule(metric=metric, direction=direction, target=target)


def load_min_area_plan(path: Path) -> MinAreaCalibrationPlan:
    resolved = path.expanduser().resolve(strict=True)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid min-area calibration plan JSON: {resolved}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("min-area calibration plan must be a JSON object")

    raw_candidates = payload.get("candidate_min_area_px")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("candidate_min_area_px must be a non-empty array")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_candidates):
        raise ValueError("candidate_min_area_px must contain only integers")
    candidates = tuple(raw_candidates)
    if any(value < 0 for value in candidates):
        raise ValueError("candidate_min_area_px must be non-negative")
    if any(left >= right for left, right in pairwise(candidates)):
        raise ValueError("candidate_min_area_px must be unique and strictly increasing")

    selection = _selection_rule(payload.get("selection_metric"), field="selection_metric")
    if selection.metric == "min_area_px":
        raise ValueError("selection_metric must be a canonical scientific metric")
    raw_tie_break = payload.get("ordered_tie_break")
    if not isinstance(raw_tie_break, list):
        raise ValueError("ordered_tie_break must be an array")
    tie_break = tuple(
        _selection_rule(value, field=f"ordered_tie_break[{index}]")
        for index, value in enumerate(raw_tie_break)
    )
    retention = _finite_number(
        payload.get("minimum_gt_retention"),
        field="minimum_gt_retention",
    )
    if not 0.0 <= retention <= 1.0:
        raise ValueError("minimum_gt_retention must be in [0, 1]")
    evidence_sha = _validate_sha256(
        payload.get("threshold_evidence_sha256"),
        field="threshold_evidence_sha256",
    )
    return MinAreaCalibrationPlan(
        candidate_min_area_px=candidates,
        selection_metric=selection,
        ordered_tie_break=tie_break,
        minimum_gt_retention=retention,
        threshold_evidence_sha256=evidence_sha,
    )


def load_threshold_evidence(
    path: Path,
    *,
    expected_sha256: str,
) -> ThresholdEvidence:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"threshold calibration evidence is missing: {resolved}")
    expected = _validate_sha256(expected_sha256, field="expected threshold evidence SHA")
    observed = _sha256(resolved)
    if observed != expected:
        raise ValueError(
            "threshold calibration evidence SHA mismatch: "
            f"expected={expected}, observed={observed}"
        )
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid threshold calibration evidence JSON: {resolved}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("threshold calibration evidence must be a JSON object")
    if payload.get("selection_status") != "SELECTED":
        raise ValueError("threshold calibration evidence has no frozen selected threshold")
    selected = payload.get("selected_threshold")
    threshold = _finite_number(selected, field="selected_threshold")
    if not 0.0 < threshold < 1.0:
        raise ValueError("selected_threshold must be in the open interval (0, 1)")
    if payload.get("comparison_rule") != "probability > threshold":
        raise ValueError("threshold evidence comparison rule must be strict greater-than")
    if payload.get("prediction_min_area_px") != 0:
        raise ValueError("threshold evidence prediction_min_area_px must be zero")
    if payload.get("ground_truth_min_area_px") != 0:
        raise ValueError("threshold evidence ground_truth_min_area_px must be zero")
    if payload.get("bottom_crop_px") != BOTTOM_CROP_PX:
        raise ValueError(f"threshold evidence bottom_crop_px must be {BOTTOM_CROP_PX}")
    if payload.get("independent_test_accessed") is not False:
        raise ValueError("threshold evidence must prove independent_test_accessed=false")
    return ThresholdEvidence(payload=payload, sha256=observed, selected_threshold=threshold)


def _profile(min_area_px: int) -> PostprocessProfile:
    return PostprocessProfile(
        profile_id="unet_small_min_area_calibration_v1",
        min_area_px=min_area_px,
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
) -> tuple[np.ndarray, np.ndarray]:
    probability = np.asarray(probability, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.bool_)
    if probability.shape != truth.shape:
        raise ValueError(f"probability/GT shape mismatch for {sample_id}")
    if probability.ndim != 2 or probability.shape[0] <= BOTTOM_CROP_PX:
        raise ValueError(f"invalid full-image shape for {sample_id}: {probability.shape}")
    if not np.isfinite(probability).all():
        raise ValueError(f"probability contains non-finite values for {sample_id}")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError(f"probability is outside [0, 1] for {sample_id}")
    return probability[:-BOTTOM_CROP_PX], truth[:-BOTTOM_CROP_PX]


def _metric_values(metrics: Mapping[str, Any]) -> dict[str, float]:
    matching = metrics["instance_matching"]
    count = metrics["count"]
    return {
        "instance_precision": float(matching["precision"]),
        "instance_recall": float(matching["recall"]),
        "instance_f1": float(matching["f1"]),
        "count_absolute_error": float(count["absolute_error"]),
        "count_relative_error": float(count["relative_error"]),
        "prediction_count": float(matching["prediction_count"]),
        "ground_truth_count": float(matching["ground_truth_count"]),
    }


def _gt_retention(
    truth: np.ndarray,
    *,
    candidate_min_area_px: int,
) -> tuple[int, int, float]:
    roi = np.ones(truth.shape, dtype=np.bool_)
    baseline = normalize_semantic_mask_detailed(
        truth,
        roi_mask=roi,
        profile=_profile(0),
    )
    retained = normalize_semantic_mask_detailed(
        truth,
        roi_mask=roi,
        profile=_profile(candidate_min_area_px),
    )
    baseline_count = len(baseline.instances)
    retained_count = len(retained.instances)
    retention = retained_count / baseline_count if baseline_count else 1.0
    return baseline_count, retained_count, float(retention)


def _rule_value(result: Mapping[str, Any], rule: SelectionRule) -> float:
    if rule.metric == "min_area_px":
        value = float(result["min_area_px"])
    else:
        value = float(result["aggregate"][rule.metric])
    return abs(value - rule.target) if rule.direction == "closest_to" else value


def select_min_area(
    results: Sequence[Mapping[str, Any]],
    plan: MinAreaCalibrationPlan,
) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    eligible = [
        result
        for result in results
        if float(result["aggregate"]["gt_retention"]) >= plan.minimum_gt_retention
    ]
    trace: list[dict[str, Any]] = [
        {
            "metric": "gt_retention",
            "direction": "minimum",
            "minimum": plan.minimum_gt_retention,
            "remaining_min_area_px": [int(result["min_area_px"]) for result in eligible],
        }
    ]
    if not eligible:
        raise ValueError("no min-area candidate satisfies minimum GT retention")
    winners = eligible
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
                "remaining_min_area_px": [int(result["min_area_px"]) for result in winners],
            }
        )
        if len(winners) == 1:
            return winners[0], trace
    raise ValueError("the prespecified min-area selection rules did not resolve the tie")


def _validate_evidence_dataset(
    records: Sequence[SplitManifestRecord],
    evidence: ThresholdEvidence,
    *,
    manifest_sha256: str,
) -> None:
    if evidence.payload.get("split_manifest_sha256") != manifest_sha256:
        raise ValueError("threshold evidence split-manifest SHA differs from the current manifest")
    recorded_ids = evidence.payload.get("calibration_sample_ids")
    expected_ids = [record.sample_id for record in records]
    if recorded_ids != expected_ids:
        raise ValueError(
            "threshold evidence calibration sample IDs differ from the current manifest"
        )


def evaluate_min_area_calibration(
    manifest: SmallBSplitManifest,
    plan: MinAreaCalibrationPlan,
    threshold_evidence: ThresholdEvidence,
    *,
    probability_cache_provider: ProbabilityCacheProvider,
    truth_provider: TruthProvider,
    manifest_sha256: str,
    plan_sha256: str,
) -> dict[str, Any]:
    """Evaluate candidates using threshold-stage caches and canonical scientific metrics."""

    if threshold_evidence.sha256 != plan.threshold_evidence_sha256:
        raise ValueError("threshold evidence SHA differs from the frozen min-area plan")
    records = manifest.select(ManifestSplit.CALIBRATION)
    if not records:
        raise ValueError("calibration split contains no included records")
    _validate_evidence_dataset(
        records,
        threshold_evidence,
        manifest_sha256=manifest_sha256,
    )

    cached: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for record in records:
        cached[record.sample_id] = _validate_arrays(
            probability_cache_provider(record),
            truth_provider(record),
            sample_id=record.sample_id,
        )

    morphometry = MorphometryConfig(perimeter_neighborhood=8)
    candidate_results: list[dict[str, Any]] = []
    for candidate in plan.candidate_min_area_px:
        per_image: list[dict[str, Any]] = []
        total_baseline = total_retained = 0
        for record in records:
            probability, truth = cached[record.sample_id]
            prediction = probability > threshold_evidence.selected_threshold
            scientific = compute_scientific_metrics(
                prediction,
                truth,
                profile=_profile(candidate),
                morphometry=morphometry,
                scale_nm_per_pixel=1.0,
                iou_threshold=0.7,
                sample_id=record.sample_id,
            )
            baseline_count, retained_count, retention = _gt_retention(
                truth,
                candidate_min_area_px=candidate,
            )
            total_baseline += baseline_count
            total_retained += retained_count
            per_image.append(
                {
                    "sample_id": record.sample_id,
                    "image_sha256": record.image_sha256,
                    "mask_sha256": record.mask_sha256,
                    **_metric_values(scientific),
                    "gt_baseline_count": baseline_count,
                    "gt_retained_count": retained_count,
                    "gt_retention": retention,
                }
            )
        aggregate = {
            metric: float(statistics.fmean(item[metric] for item in per_image))
            for metric in sorted(_SELECTION_METRICS - {"gt_retention"})
        }
        aggregate["gt_retention"] = (
            float(total_retained / total_baseline) if total_baseline else 1.0
        )
        candidate_results.append(
            {
                "min_area_px": candidate,
                "per_image": per_image,
                "aggregate": aggregate,
                "gt_baseline_count": total_baseline,
                "gt_retained_count": total_retained,
            }
        )

    selected, trace = select_min_area(candidate_results, plan)
    return {
        "schema_version": "1",
        "selection_status": "SELECTED",
        "selected_min_area_px": int(selected["min_area_px"]),
        "selected_threshold": threshold_evidence.selected_threshold,
        "threshold_evidence_sha256": threshold_evidence.sha256,
        "split_manifest_sha256": manifest_sha256,
        "plan_sha256": plan_sha256,
        "minimum_gt_retention": plan.minimum_gt_retention,
        "candidate_results": candidate_results,
        "selection_trace": trace,
        "calibration_sample_ids": [record.sample_id for record in records],
        "probability_source": "threshold-stage cache; no model inference",
        "ground_truth_min_area_px": 0,
        "bottom_crop_px": BOTTOM_CROP_PX,
        "comparison_rule": "probability > threshold",
        "independent_test_accessed": False,
    }


def write_min_area_outputs(output_root: Path, payload: Mapping[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=False)
    json_path = output_root / "min-area-calibration.json"
    csv_path = output_root / "min-area-calibration.csv"
    metric_columns = tuple(sorted(_SELECTION_METRICS))
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        columns = (
            "scope",
            "sample_id",
            "min_area_px",
            *metric_columns,
            "prediction_count",
            "ground_truth_count",
            "gt_baseline_count",
            "gt_retained_count",
        )
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for candidate in payload["candidate_results"]:
            for item in candidate["per_image"]:
                writer.writerow(
                    {
                        "scope": "per_image",
                        "sample_id": item["sample_id"],
                        "min_area_px": candidate["min_area_px"],
                        **{field: item[field] for field in metric_columns},
                        "prediction_count": item["prediction_count"],
                        "ground_truth_count": item["ground_truth_count"],
                        "gt_baseline_count": item["gt_baseline_count"],
                        "gt_retained_count": item["gt_retained_count"],
                    }
                )
            writer.writerow(
                {
                    "scope": "aggregate",
                    "sample_id": "",
                    "min_area_px": candidate["min_area_px"],
                    **candidate["aggregate"],
                    "gt_baseline_count": candidate["gt_baseline_count"],
                    "gt_retained_count": candidate["gt_retained_count"],
                }
            )
    complete_payload = {
        **payload,
        "artifacts": {
            "json": json_path.name,
            "csv": {"path": csv_path.name, "sha256": _sha256(csv_path)},
        },
    }
    json_path.write_text(
        json.dumps(complete_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

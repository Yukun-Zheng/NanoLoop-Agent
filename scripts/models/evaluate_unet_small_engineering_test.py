"""Run the Small-B ENGINEERING_ONLY exploratory test.

This entry point is intentionally separate from the formal scientific
Independent Test contract.  It accepts only the frozen engineering presets and
can never emit a scientific PASS/FAIL verdict.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from app.contracts.analysis_config import MorphometryConfig, PostprocessProfile
from scripts.models.evaluate_unet_large_independent_test import (
    _scientific_metric_values,
    compute_scientific_metrics,
)
from scripts.models.small_b_contracts import (
    REQUIRED_COLUMNS,
    ManifestSplit,
    SmallBSplitManifest,
    SplitManifestRecord,
    _parse_record,
    _validate_calibration_test_disjoint,
    _validate_no_cross_split_source_or_sha,
    _validate_unique_sample_ids,
)
from scripts.models.small_b_tolerance_policy import sha256_file

MODEL_ID = "unet-small-balanced-v1"
ENGINEERING_THRESHOLD = 0.30
ENGINEERING_MIN_AREA_PX = 64
BOTTOM_CROP_PX = 130
INSTANCE_IOU_THRESHOLD = 0.7
EXPECTED_SOURCE_SAMPLE_ID = "SrNi"
EXPECTED_SOURCE_IMAGE_IDS = frozenset({"SrNi-1", "SrNi-2", "SrNi-3"})
EVALUATION_SCOPE = "ENGINEERING_ONLY"
SCIENTIFIC_VERDICT = "NOT_EVALUATED"
CALIBRATION_STATUS = "BLOCKED_NO_CALIBRATION_DATA"

_METRIC_FIELDS = (
    "instance_precision",
    "instance_recall",
    "instance_f1",
    "count_absolute_error",
    "count_relative_error",
    "mean_area_relative_error",
    "mean_equivalent_diameter_relative_error",
    "number_density_relative_error",
    "perimeter_density_relative_error",
)
_PER_IMAGE_COLUMNS = (
    "evaluation_id",
    "sample_id",
    "source_sample_id",
    "source_image_id",
    "field_of_view_id",
    "image_sha256",
    "mask_sha256",
    "prediction_count",
    "ground_truth_count",
    "matched_count",
    "false_positive_count",
    "false_negative_count",
    *_METRIC_FIELDS,
    "failure_case_count",
    "evaluation_status",
    "evaluation_scope",
    "scientific_verdict",
)
_FAILURE_COLUMNS = (
    "evaluation_id",
    "sample_id",
    "metric",
    "reason_code",
    "message",
)


class PredictionProvider(Protocol):
    """Return an existing canonical prediction mask; never run inference."""

    def __call__(self, record: SplitManifestRecord) -> np.ndarray: ...


class TruthProvider(Protocol):
    def __call__(self, record: SplitManifestRecord) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class SmallBEngineeringContract:
    manifest: SmallBSplitManifest
    manifest_sha256: str
    threshold_evidence: Mapping[str, Any]
    min_area_evidence: Mapping[str, Any]
    threshold_evidence_sha256: str
    min_area_evidence_sha256: str


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {resolved}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def load_engineering_manifest(path: Path) -> SmallBSplitManifest:
    """Load the shared manifest rules without requiring a Calibration split."""

    resolved = path.expanduser().resolve(strict=True)
    with resolved.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        for field in (*REQUIRED_COLUMNS, "annotator_review_status"):
            if field not in fieldnames:
                raise ValueError(f"line 1, field '{field}': required column is missing")
        rows = tuple((reader.line_num, row) for row in reader)

    records = tuple(
        _parse_record(row, line_number=line_number)
        for line_number, row in rows
    )
    _validate_unique_sample_ids(records)
    _validate_no_cross_split_source_or_sha(records)
    _validate_calibration_test_disjoint(records)
    manifest = SmallBSplitManifest(records)
    selected = manifest.select(ManifestSplit.INDEPENDENT_TEST)
    selected_ids = {record.source_image_id for record in selected}
    if selected_ids != EXPECTED_SOURCE_IMAGE_IDS or len(selected) != 3:
        raise ValueError(
            "included independent_test rows must be exactly SrNi-1, SrNi-2, and SrNi-3"
        )
    if {record.source_sample_id for record in selected} != {EXPECTED_SOURCE_SAMPLE_ID}:
        raise ValueError("independent_test rows must share source_sample_id SrNi")
    if len({record.field_of_view_id for record in selected}) != len(selected):
        raise ValueError("independent_test field_of_view_id values must be unique")

    approval_by_sample = {
        (row.get("sample_id") or "").strip(): (
            row.get("annotator_review_status") or ""
        ).strip()
        for _, row in rows
    }
    for record in selected:
        if approval_by_sample.get(record.sample_id) != "approved":
            raise ValueError(
                f"independent_test row {record.sample_id} requires "
                "annotator_review_status=approved"
            )
    return manifest


def _validate_common_evidence(
    payload: Mapping[str, Any],
    *,
    parameter: str,
    manifest_sha256: str,
) -> None:
    if payload.get("model_id") != MODEL_ID:
        raise ValueError(f"{parameter} evidence model_id must be {MODEL_ID}")
    if payload.get("evidence_type") != EVALUATION_SCOPE:
        raise ValueError(f"{parameter} evidence_type must be ENGINEERING_ONLY")
    if payload.get("parameter") != parameter:
        raise ValueError(f"{parameter} evidence parameter identity differs")
    if payload.get("selection_status") != "ENGINEERING_PRESET":
        raise ValueError(f"{parameter} selection_status must be ENGINEERING_PRESET")
    if payload.get("calibration_performed") is not False:
        raise ValueError(f"{parameter} calibration_performed must be false")
    if payload.get("calibration_status") != CALIBRATION_STATUS:
        raise ValueError(
            f"{parameter} calibration_status must be BLOCKED_NO_CALIBRATION_DATA"
        )
    if payload.get("scientific_verdict") != SCIENTIFIC_VERDICT:
        raise ValueError(f"{parameter} scientific_verdict must be NOT_EVALUATED")
    if payload.get("independent_test_may_adjust_parameter") is not False:
        raise ValueError(
            f"{parameter} independent_test_may_adjust_parameter must be false"
        )
    if payload.get("split_manifest_sha256") != manifest_sha256:
        raise ValueError(f"{parameter} evidence split-manifest SHA differs")


def validate_engineering_contract(contract: SmallBEngineeringContract) -> None:
    """Accept only the declared engineering presets and reject scientific claims."""

    threshold = contract.threshold_evidence
    min_area = contract.min_area_evidence
    _validate_common_evidence(
        threshold,
        parameter="threshold",
        manifest_sha256=contract.manifest_sha256,
    )
    _validate_common_evidence(
        min_area,
        parameter="min_area_px",
        manifest_sha256=contract.manifest_sha256,
    )
    threshold_value = threshold.get("engineering_value")
    if (
        isinstance(threshold_value, bool)
        or not isinstance(threshold_value, int | float)
        or not math.isfinite(float(threshold_value))
        or float(threshold_value) != ENGINEERING_THRESHOLD
    ):
        raise ValueError("threshold engineering_value must be exactly 0.30")
    min_area_value = min_area.get("engineering_value")
    if (
        isinstance(min_area_value, bool)
        or not isinstance(min_area_value, int)
        or min_area_value != ENGINEERING_MIN_AREA_PX
    ):
        raise ValueError("min_area_px engineering_value must be exactly 64")


def load_engineering_contract(
    *,
    split_manifest_path: Path,
    threshold_evidence_path: Path,
    min_area_evidence_path: Path,
) -> SmallBEngineeringContract:
    manifest_path = split_manifest_path.expanduser().resolve(strict=True)
    threshold_path = threshold_evidence_path.expanduser().resolve(strict=True)
    min_area_path = min_area_evidence_path.expanduser().resolve(strict=True)
    contract = SmallBEngineeringContract(
        manifest=load_engineering_manifest(manifest_path),
        manifest_sha256=sha256_file(manifest_path),
        threshold_evidence=_load_json(threshold_path, label="threshold evidence"),
        min_area_evidence=_load_json(min_area_path, label="min-area evidence"),
        threshold_evidence_sha256=sha256_file(threshold_path),
        min_area_evidence_sha256=sha256_file(min_area_path),
    )
    validate_engineering_contract(contract)
    return contract


def _validate_arrays(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    sample_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.bool_)
    truth = np.asarray(truth, dtype=np.bool_)
    if prediction.shape != truth.shape:
        raise ValueError(f"prediction/GT shape mismatch for {sample_id}")
    if prediction.ndim != 2 or prediction.shape[0] <= BOTTOM_CROP_PX:
        raise ValueError(f"invalid full-image shape for {sample_id}: {prediction.shape}")
    if np.any(prediction[-BOTTOM_CROP_PX:]):
        raise ValueError(f"prediction contains foreground in excluded bottom rows for {sample_id}")
    return prediction[:-BOTTOM_CROP_PX], truth[:-BOTTOM_CROP_PX]


def _profile() -> PostprocessProfile:
    return PostprocessProfile(
        profile_id="unet_small_engineering_test_v1",
        min_area_px=ENGINEERING_MIN_AREA_PX,
        fill_holes=True,
        watershed_enabled=False,
        exclude_border=True,
        connectivity=2,
        instance_iou_threshold=INSTANCE_IOU_THRESHOLD,
    )


def _failure_cases(
    *,
    evaluation_id: str,
    sample_id: str,
    values: Mapping[str, float | None],
) -> list[dict[str, Any]]:
    return [
        {
            "evaluation_id": evaluation_id,
            "sample_id": sample_id,
            "metric": metric,
            "reason_code": "METRIC_NOT_EVALUABLE",
            "message": "metric is not evaluable for this image; no tolerance verdict was applied",
        }
        for metric in _METRIC_FIELDS
        if values.get(metric) is None
    ]


def _summary(per_image: Sequence[Mapping[str, Any]], failure_count: int) -> dict[str, Any]:
    means: dict[str, float | None] = {}
    for metric in _METRIC_FIELDS:
        observed = [
            float(row[metric])
            for row in per_image
            if row.get(metric) is not None
        ]
        means[metric] = float(np.mean(observed)) if observed else None
    return {
        "total_images": len(per_image),
        "source_sample_count": 1 if per_image else 0,
        "independent_region_count": len(per_image),
        "failure_case_count": failure_count,
        "metric_means": means,
    }


def evaluate_engineering_test(
    contract: SmallBEngineeringContract,
    *,
    prediction_provider: PredictionProvider,
    truth_provider: TruthProvider,
    evaluation_id: str = "small-b-engineering-test-v1",
) -> dict[str, Any]:
    """Evaluate canonical masks without producing a scientific verdict."""

    validate_engineering_contract(contract)
    records = contract.manifest.select(ManifestSplit.INDEPENDENT_TEST)
    per_image: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    profile = _profile()
    morphometry = MorphometryConfig(perimeter_neighborhood=8)
    for record in records:
        prediction, truth = _validate_arrays(
            prediction_provider(record),
            truth_provider(record),
            sample_id=record.sample_id,
        )
        scientific = compute_scientific_metrics(
            prediction,
            truth,
            profile=profile,
            morphometry=morphometry,
            scale_nm_per_pixel=1.0,
            iou_threshold=INSTANCE_IOU_THRESHOLD,
            sample_id=record.sample_id,
        )
        values = _scientific_metric_values(scientific)
        failures = _failure_cases(
            evaluation_id=evaluation_id,
            sample_id=record.sample_id,
            values=values,
        )
        all_failures.extend(failures)
        matching = scientific["instance_matching"]
        per_image.append(
            {
                "evaluation_id": evaluation_id,
                "sample_id": record.sample_id,
                "source_sample_id": record.source_sample_id,
                "source_image_id": record.source_image_id,
                "field_of_view_id": record.field_of_view_id,
                "image_sha256": record.image_sha256,
                "mask_sha256": record.mask_sha256,
                "prediction_count": matching["prediction_count"],
                "ground_truth_count": matching["ground_truth_count"],
                "matched_count": matching["matched_count"],
                "false_positive_count": matching["false_positive_count"],
                "false_negative_count": matching["false_negative_count"],
                **values,
                "failure_case_count": len(failures),
                "evaluation_status": (
                    "COMPLETED_WITH_NOT_EVALUABLE_METRICS"
                    if failures
                    else "COMPLETED"
                ),
                "evaluation_scope": EVALUATION_SCOPE,
                "scientific_verdict": SCIENTIFIC_VERDICT,
            }
        )

    return {
        "schema_version": "1",
        "evaluation_id": evaluation_id,
        "evaluation_scope": EVALUATION_SCOPE,
        "evaluation_status": "COMPLETED",
        "scientific_verdict": SCIENTIFIC_VERDICT,
        "model_id": MODEL_ID,
        "engineering_parameters": {
            "threshold": ENGINEERING_THRESHOLD,
            "threshold_comparison": "gt",
            "min_area_px": ENGINEERING_MIN_AREA_PX,
            "bottom_crop_px": BOTTOM_CROP_PX,
            "selection_status": "ENGINEERING_PRESET",
            "calibration_performed": False,
        },
        "summary": _summary(per_image, len(all_failures)),
        "per_image": per_image,
        "failure_cases": all_failures,
        "report_statements": [
            "Calibration was not performed.",
            (
                "Threshold and min-area are engineering presets, "
                "not scientifically calibrated parameters."
            ),
            "The three images are independent regions from the same physical sample.",
            "Results represent same-sample, independent-region generalization only.",
            "Independent Test results must not be used to adjust parameters.",
        ],
        "independent_test_used_for_tuning": False,
        "parameters_changed_after_test": False,
        "parameter_writeback_performed": False,
        "policy_writeback_performed": False,
        "formal_scientific_verdict_path_called": False,
    }


def _write_csv(
    path: Path,
    columns: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_engineering_outputs(
    output_root: Path,
    contract: SmallBEngineeringContract,
    result: Mapping[str, Any],
) -> None:
    """Write engineering-only metrics; never write parameters or policy files."""

    if result.get("evaluation_scope") != EVALUATION_SCOPE:
        raise ValueError("evaluation_scope must be ENGINEERING_ONLY")
    if result.get("scientific_verdict") != SCIENTIFIC_VERDICT:
        raise ValueError("scientific_verdict must be NOT_EVALUATED")
    output_root.mkdir(parents=True, exist_ok=False)
    per_image_path = output_root / "engineering-test-per-image.csv"
    failures_path = output_root / "failure-cases.csv"
    metrics_path = output_root / "engineering-test-metrics.json"
    _write_csv(per_image_path, _PER_IMAGE_COLUMNS, result["per_image"])
    _write_csv(failures_path, _FAILURE_COLUMNS, result["failure_cases"])
    payload = {
        **result,
        "split_manifest_sha256": contract.manifest_sha256,
        "threshold_evidence_sha256": contract.threshold_evidence_sha256,
        "min_area_evidence_sha256": contract.min_area_evidence_sha256,
        "artifacts": {
            "per_image_csv": {
                "path": per_image_path.name,
                "sha256": sha256_file(per_image_path),
            },
            "failure_cases_csv": {
                "path": failures_path.name,
                "sha256": sha256_file(failures_path),
            },
        },
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--threshold-evidence", required=True, type=Path)
    parser.add_argument("--min-area-evidence", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--not-evaluated",
        action="store_true",
        help="Validate inputs and write a NOT_EVALUATED readiness report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    try:
        contract = load_engineering_contract(
            split_manifest_path=namespace.split_manifest,
            threshold_evidence_path=namespace.threshold_evidence,
            min_area_evidence_path=namespace.min_area_evidence,
        )
        if not namespace.not_evaluated:
            raise ValueError(
                "real evaluation requires approved canonical prediction and GT providers; "
                "use evaluate_engineering_test"
            )
        result = {
            "schema_version": "1",
            "evaluation_id": "small-b-engineering-test-v1",
            "evaluation_scope": EVALUATION_SCOPE,
            "evaluation_status": "NOT_RUN",
            "scientific_verdict": SCIENTIFIC_VERDICT,
            "engineering_parameters": {
                "threshold": ENGINEERING_THRESHOLD,
                "min_area_px": ENGINEERING_MIN_AREA_PX,
                "selection_status": "ENGINEERING_PRESET",
                "calibration_performed": False,
            },
            "summary": _summary([], 0),
            "per_image": [],
            "failure_cases": [],
            "report_statements": [
                "Calibration was not performed.",
                "Engineering evaluation was not run.",
            ],
            "independent_test_used_for_tuning": False,
            "parameters_changed_after_test": False,
            "parameter_writeback_performed": False,
            "policy_writeback_performed": False,
            "formal_scientific_verdict_path_called": False,
        }
        write_engineering_outputs(namespace.output_root, contract, result)
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

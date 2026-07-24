"""Evaluate frozen Small-B Analysis masks on the independent-test split."""

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
    ManifestSplit,
    SmallBSplitManifest,
    SplitManifestRecord,
    load_split_manifest,
)
from scripts.models.small_b_tolerance_policy import (
    BOTTOM_CROP_PX,
    FROZEN_PREDEFINED,
    SELECTED,
    SmallBTolerancePolicy,
    evaluate_per_image_tolerances,
    load_tolerance_policy,
    sha256_file,
)

PREDEFINED_THRESHOLD = 0.30
PREDEFINED_MIN_AREA_PX = 64

_PER_IMAGE_COLUMNS = (
    "evaluation_id",
    "sample_id",
    "image_sha256",
    "mask_sha256",
    "prediction_count",
    "ground_truth_count",
    "matched_count",
    "false_positive_count",
    "false_negative_count",
    "instance_precision",
    "instance_recall",
    "instance_f1",
    "count_absolute_error",
    "count_relative_error",
    "mean_area_relative_error",
    "mean_equivalent_diameter_relative_error",
    "number_density_relative_error",
    "perimeter_density_relative_error",
    "failure_count",
    "status",
)
_FAILURE_COLUMNS = (
    "evaluation_id",
    "sample_id",
    "metric",
    "operator",
    "observed",
    "tolerance",
    "reason_code",
)


class PredictionProvider(Protocol):
    """Load an existing formal Analysis prediction; never run model inference."""

    def __call__(self, record: SplitManifestRecord) -> np.ndarray: ...


class TruthProvider(Protocol):
    def __call__(self, record: SplitManifestRecord) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class SmallBEvaluationContract:
    manifest: SmallBSplitManifest
    manifest_sha256: str
    threshold_evidence: Mapping[str, Any]
    min_area_evidence: Mapping[str, Any]
    tolerance_policy: SmallBTolerancePolicy
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


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _evidence_selection_status(
    threshold_evidence: Mapping[str, Any],
    min_area_evidence: Mapping[str, Any],
) -> str:
    selection_status = threshold_evidence.get("selection_status")
    if min_area_evidence.get("selection_status") != selection_status:
        raise ValueError("threshold and min-area evidence selection_status differs")
    if selection_status not in {SELECTED, FROZEN_PREDEFINED}:
        raise ValueError(
            "evidence selection_status must be SELECTED or FROZEN_PREDEFINED"
        )
    return selection_status


def validate_evaluation_contract(
    contract: SmallBEvaluationContract,
    *,
    expected_policy_sha256: str,
) -> None:
    """Fail closed on evidence, policy, manifest, or frozen-parameter drift."""

    if contract.tolerance_policy.sha256 != expected_policy_sha256:
        raise ValueError("tolerance policy SHA differs from the expected frozen SHA")
    if contract.threshold_evidence_sha256 != (
        contract.tolerance_policy.threshold_evidence_sha256
    ):
        raise ValueError("threshold evidence SHA differs from the tolerance policy")
    if contract.min_area_evidence_sha256 != contract.tolerance_policy.min_area_evidence_sha256:
        raise ValueError("min-area evidence SHA differs from the tolerance policy")
    selection_status = _evidence_selection_status(
        contract.threshold_evidence,
        contract.min_area_evidence,
    )
    if selection_status == SELECTED:
        manifest_field = "split_manifest_sha256"
        threshold_field = "selected_threshold"
        min_area_threshold_field = "selected_threshold"
        min_area_field = "selected_min_area_px"
    elif selection_status == FROZEN_PREDEFINED:
        manifest_field = "manifest_sha256"
        threshold_field = "threshold"
        min_area_threshold_field = "threshold"
        min_area_field = "min_area_px"
        if contract.threshold_evidence.get("frozen_before_test") is not True:
            raise ValueError("threshold evidence frozen_before_test must be true")
        if contract.min_area_evidence.get("frozen_before_test") is not True:
            raise ValueError("min-area evidence frozen_before_test must be true")
        if contract.threshold_evidence.get("parameters_changed_after_test") is not False:
            raise ValueError("threshold parameters_changed_after_test must be false")
        if contract.min_area_evidence.get("parameters_changed_after_test") is not False:
            raise ValueError("min-area parameters_changed_after_test must be false")
    if contract.threshold_evidence.get(manifest_field) != contract.manifest_sha256:
        raise ValueError("threshold evidence manifest SHA differs from the current manifest")
    if contract.min_area_evidence.get(manifest_field) != contract.manifest_sha256:
        raise ValueError("min-area evidence manifest SHA differs from the current manifest")

    frozen = contract.tolerance_policy.frozen_scientific_parameters
    threshold = _finite_number(
        contract.threshold_evidence.get(threshold_field),
        field=f"threshold evidence {threshold_field}",
    )
    min_area_threshold = _finite_number(
        contract.min_area_evidence.get(min_area_threshold_field),
        field=f"min-area evidence {min_area_threshold_field}",
    )
    if not math.isclose(threshold, frozen.threshold) or not math.isclose(
        min_area_threshold, frozen.threshold
    ):
        raise ValueError("selected threshold differs from the frozen tolerance policy")
    selected_min_area = contract.min_area_evidence.get(min_area_field)
    if (
        isinstance(selected_min_area, bool)
        or not isinstance(selected_min_area, int)
        or selected_min_area != frozen.min_area_px
    ):
        raise ValueError("selected min-area differs from the frozen tolerance policy")
    if selection_status == FROZEN_PREDEFINED and (
        not math.isclose(threshold, PREDEFINED_THRESHOLD)
        or selected_min_area != PREDEFINED_MIN_AREA_PX
    ):
        raise ValueError(
            "FROZEN_PREDEFINED parameters must be threshold=0.30 and min_area_px=64"
        )
    if (
        contract.min_area_evidence.get("threshold_evidence_sha256")
        != contract.threshold_evidence_sha256
    ):
        raise ValueError("min-area evidence references a different threshold evidence")
    if not contract.tolerance_policy.approval.frozen_before_independent_test:
        raise ValueError("tolerance policy was not frozen before independent test")


def load_evaluation_contract(
    *,
    split_manifest_path: Path,
    threshold_evidence_path: Path,
    min_area_evidence_path: Path,
    tolerance_policy_path: Path,
    expected_policy_sha256: str,
) -> SmallBEvaluationContract:
    """Load and cross-check every frozen input before predictions or GT are accessed."""

    manifest_path = split_manifest_path.expanduser().resolve(strict=True)
    threshold_path = threshold_evidence_path.expanduser().resolve(strict=True)
    min_area_path = min_area_evidence_path.expanduser().resolve(strict=True)
    policy_path = tolerance_policy_path.expanduser().resolve(strict=True)
    threshold_evidence = _load_json(threshold_path, label="threshold evidence")
    min_area_evidence = _load_json(min_area_path, label="min-area evidence")
    selection_status = _evidence_selection_status(
        threshold_evidence,
        min_area_evidence,
    )
    manifest = load_split_manifest(
        manifest_path,
        require_calibration=selection_status == SELECTED,
    )
    policy = load_tolerance_policy(
        policy_path,
        threshold_evidence_path=threshold_path,
        min_area_evidence_path=min_area_path,
    )
    contract = SmallBEvaluationContract(
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
        threshold_evidence=threshold_evidence,
        min_area_evidence=min_area_evidence,
        tolerance_policy=policy,
        threshold_evidence_sha256=sha256_file(threshold_path),
        min_area_evidence_sha256=sha256_file(min_area_path),
    )
    validate_evaluation_contract(contract, expected_policy_sha256=expected_policy_sha256)
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


def _postprocess_profile(policy: SmallBTolerancePolicy) -> PostprocessProfile:
    frozen = policy.frozen_scientific_parameters
    return PostprocessProfile(
        profile_id="unet_small_independent_test_v1",
        min_area_px=frozen.min_area_px,
        fill_holes=True,
        watershed_enabled=False,
        exclude_border=True,
        connectivity=2,
        instance_iou_threshold=policy.instance_matching.mask_iou_threshold,
    )


def _per_image_row(
    *,
    evaluation_id: str,
    record: SplitManifestRecord,
    scientific: Mapping[str, Any],
    values: Mapping[str, float | None],
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    matching = scientific["instance_matching"]
    return {
        "evaluation_id": evaluation_id,
        "sample_id": record.sample_id,
        "image_sha256": record.image_sha256,
        "mask_sha256": record.mask_sha256,
        "prediction_count": matching["prediction_count"],
        "ground_truth_count": matching["ground_truth_count"],
        "matched_count": matching["matched_count"],
        "false_positive_count": matching["false_positive_count"],
        "false_negative_count": matching["false_negative_count"],
        **values,
        "failure_count": len(failures),
        "status": "PASS" if not failures else "FAIL",
    }


def evaluate_independent_test(
    contract: SmallBEvaluationContract,
    *,
    expected_policy_sha256: str,
    prediction_provider: PredictionProvider,
    truth_provider: TruthProvider,
    evaluation_id: str = "small-b-independent-test-v1",
) -> dict[str, Any]:
    """Evaluate only included independent-test rows using frozen canonical settings."""

    validate_evaluation_contract(contract, expected_policy_sha256=expected_policy_sha256)
    records = contract.manifest.select(ManifestSplit.INDEPENDENT_TEST)
    if not records:
        return {
            "evaluation_id": evaluation_id,
            "verdict": "NOT_EVALUATED",
            "selection_status": contract.threshold_evidence["selection_status"],
            "per_image": [],
            "failures": [],
            "reason_codes": ["INDEPENDENT_TEST_EMPTY"],
            "generalization_scope": {
                "physical_sample_count": 1,
                "independent_region_count": 0,
                "description": "same-sample, independent-region generalization",
                "cross_sample_generalization": False,
                "statement": "three independent regions from one physical sample",
            },
            "independent_test_used_for_tuning": False,
            "parameters_changed_after_test": False,
            "policy_changed_after_test": False,
        }

    profile = _postprocess_profile(contract.tolerance_policy)
    morphometry = MorphometryConfig(perimeter_neighborhood=8)
    per_image: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
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
            iou_threshold=contract.tolerance_policy.instance_matching.mask_iou_threshold,
            sample_id=record.sample_id,
        )
        values = _scientific_metric_values(scientific)
        failures = evaluate_per_image_tolerances(
            record.sample_id,
            values,
            contract.tolerance_policy,
        )
        normalized_failures = [
            {"evaluation_id": evaluation_id, **failure} for failure in failures
        ]
        all_failures.extend(normalized_failures)
        per_image.append(
            _per_image_row(
                evaluation_id=evaluation_id,
                record=record,
                scientific=scientific,
                values=values,
                failures=failures,
            )
        )

    verdict = "FAIL" if all_failures else "PASS"
    return {
        "evaluation_id": evaluation_id,
        "verdict": verdict,
        "selection_status": contract.threshold_evidence["selection_status"],
        "per_image": per_image,
        "failures": all_failures,
        "reason_codes": sorted({failure["reason_code"] for failure in all_failures}),
        "generalization_scope": {
            "physical_sample_count": 1,
            "independent_region_count": len(records),
            "description": "same-sample, independent-region generalization",
            "cross_sample_generalization": False,
            "statement": "three independent regions from one physical sample",
        },
        "independent_test_used_for_tuning": False,
        "parameters_changed_after_test": False,
        "policy_changed_after_test": False,
    }


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation_outputs(
    output_root: Path,
    contract: SmallBEvaluationContract,
    result: Mapping[str, Any],
) -> None:
    """Write the four evidence files with a non-cyclic, content-addressed reference chain."""

    output_root.mkdir(parents=True, exist_ok=False)
    per_image_path = output_root / "independent-test-per-image.csv"
    failure_path = output_root / "failure-cases.csv"
    metrics_path = output_root / "independent-test-metrics.json"
    verdict_path = output_root / "scientific-verdict.json"
    _write_csv(per_image_path, _PER_IMAGE_COLUMNS, result["per_image"])
    _write_csv(failure_path, _FAILURE_COLUMNS, result["failures"])

    policy = contract.tolerance_policy
    metrics_payload = {
        "schema_version": "1",
        "evaluation_id": result["evaluation_id"],
        "status": "COMPLETED" if result["verdict"] != "NOT_EVALUATED" else "NOT_EVALUATED",
        "verdict": result["verdict"],
        "selection_status": result["selection_status"],
        "model_id": policy.model_id,
        "split_manifest_sha256": contract.manifest_sha256,
        "threshold_evidence_sha256": contract.threshold_evidence_sha256,
        "min_area_evidence_sha256": contract.min_area_evidence_sha256,
        "tolerance_policy_sha256": policy.sha256,
        "frozen_scientific_parameters": {
            "threshold": policy.frozen_scientific_parameters.threshold,
            "threshold_comparison": policy.frozen_scientific_parameters.threshold_comparison,
            "min_area_px": policy.frozen_scientific_parameters.min_area_px,
            "bottom_crop_px": policy.frozen_scientific_parameters.bottom_crop_px,
            "mask_iou_threshold": policy.instance_matching.mask_iou_threshold,
        },
        "per_image": result["per_image"],
        "failure_count": len(result["failures"]),
        "reason_codes": result["reason_codes"],
        "generalization_scope": result["generalization_scope"],
        "independent_test_used_for_tuning": False,
        "parameters_changed_after_test": False,
        "policy_changed_after_test": False,
        "artifacts": {
            "per_image_csv": {
                "path": per_image_path.name,
                "sha256": sha256_file(per_image_path),
            },
            "failure_cases_csv": {
                "path": failure_path.name,
                "sha256": sha256_file(failure_path),
            },
            "scientific_verdict": {"path": verdict_path.name},
        },
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verdict_payload = {
        "schema_version": "1",
        "evaluation_id": result["evaluation_id"],
        "verdict": result["verdict"],
        "selection_status": result["selection_status"],
        "total_images": len(result["per_image"]),
        "passed_images": sum(item["status"] == "PASS" for item in result["per_image"]),
        "failed_images": sum(item["status"] == "FAIL" for item in result["per_image"]),
        "failure_count": len(result["failures"]),
        "reason_codes": result["reason_codes"],
        "generalization_scope": result["generalization_scope"],
        "independent_test_metrics": {
            "path": metrics_path.name,
            "sha256": sha256_file(metrics_path),
        },
        "per_image_csv": {
            "path": per_image_path.name,
            "sha256": sha256_file(per_image_path),
        },
        "failure_cases_csv": {
            "path": failure_path.name,
            "sha256": sha256_file(failure_path),
        },
        "split_manifest_sha256": contract.manifest_sha256,
        "threshold_evidence_sha256": contract.threshold_evidence_sha256,
        "min_area_evidence_sha256": contract.min_area_evidence_sha256,
        "tolerance_policy_sha256": policy.sha256,
        "independent_test_used_for_tuning": False,
        "parameters_changed_after_test": False,
        "policy_changed_after_test": False,
    }
    verdict_path.write_text(
        json.dumps(verdict_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--threshold-evidence", required=True, type=Path)
    parser.add_argument("--min-area-evidence", required=True, type=Path)
    parser.add_argument("--tolerance-policy", required=True, type=Path)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--not-evaluated",
        action="store_true",
        help="Validate frozen inputs and write NOT_EVALUATED outputs without reading test data.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    try:
        contract = load_evaluation_contract(
            split_manifest_path=namespace.split_manifest,
            threshold_evidence_path=namespace.threshold_evidence,
            min_area_evidence_path=namespace.min_area_evidence,
            tolerance_policy_path=namespace.tolerance_policy,
            expected_policy_sha256=namespace.expected_policy_sha256,
        )
        if not namespace.not_evaluated:
            raise ValueError(
                "real evaluation requires approved prediction and GT providers; "
                "use evaluate_independent_test"
            )
        result = {
            "evaluation_id": "small-b-independent-test-v1",
            "verdict": "NOT_EVALUATED",
            "selection_status": contract.threshold_evidence["selection_status"],
            "per_image": [],
            "failures": [],
            "reason_codes": ["INDEPENDENT_TEST_NOT_RUN"],
            "generalization_scope": {
                "physical_sample_count": 1,
                "independent_region_count": 3,
                "description": "same-sample, independent-region generalization",
                "cross_sample_generalization": False,
                "statement": "three independent regions from one physical sample",
            },
        }
        write_evaluation_outputs(namespace.output_root, contract, result)
    except Exception as error:
        print(
            json.dumps(
                {"status": "error", "error_type": type(error).__name__, "message": str(error)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

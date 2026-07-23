"""Generate and formally validate the frozen Small-B tolerance policy."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.models.evaluate_unet_small_independent_test import (
    PREDEFINED_MIN_AREA_PX,
    PREDEFINED_THRESHOLD,
)
from scripts.models.generate_unet_small_frozen_predefined_evidence import (
    COMPARISON_RULE,
    _json_bytes,
    _restore_output,
    _stage_bytes,
)
from scripts.models.small_b_tolerance_policy import (
    BOTTOM_CROP_PX,
    FROZEN_PREDEFINED,
    MODEL_ID,
    SELECTED,
    _approval,
    _instance_matching,
    _tolerances,
    load_tolerance_policy,
    sha256_file,
)
from scripts.models.small_b_tolerance_policy import (
    _frozen_parameters as validate_frozen_parameters,
)


def _load_json_file(path: Path, *, label: str) -> tuple[Path, Mapping[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must be a readable file: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid UTF-8 JSON: {resolved}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return resolved, payload


def _non_empty(value: str, *, field: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must be a non-empty string")
    return normalized


def _approved_at(value: str | None) -> str:
    resolved = value or datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        timestamp = datetime.fromisoformat(resolved.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("approved_at must be an ISO-8601 timestamp") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("approved_at must include a timezone")
    return resolved


def _frozen_parameters(
    threshold_evidence: Mapping[str, Any],
    min_area_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    status = threshold_evidence.get("selection_status")
    if min_area_evidence.get("selection_status") != status:
        raise ValueError("threshold and min-area evidence selection_status differs")
    if status == FROZEN_PREDEFINED:
        threshold = threshold_evidence.get("threshold")
        min_area_threshold = min_area_evidence.get("threshold")
        min_area_px = min_area_evidence.get("min_area_px")
    elif status == SELECTED:
        threshold = threshold_evidence.get("selected_threshold")
        min_area_threshold = min_area_evidence.get("selected_threshold")
        min_area_px = min_area_evidence.get("selected_min_area_px")
    else:
        raise ValueError(
            "evidence selection_status must be SELECTED or FROZEN_PREDEFINED"
        )
    if threshold != PREDEFINED_THRESHOLD or min_area_threshold != PREDEFINED_THRESHOLD:
        raise ValueError("evidence threshold must be exactly 0.30")
    if (
        isinstance(min_area_px, bool)
        or not isinstance(min_area_px, int)
        or min_area_px != PREDEFINED_MIN_AREA_PX
    ):
        raise ValueError("evidence min_area_px must be exactly 64")
    for label, evidence in (
        ("threshold", threshold_evidence),
        ("min-area", min_area_evidence),
    ):
        if evidence.get("bottom_crop_px") != BOTTOM_CROP_PX:
            raise ValueError(f"{label} evidence bottom_crop_px must be exactly 130")
        if evidence.get("comparison_rule") != COMPARISON_RULE:
            raise ValueError(
                f"{label} evidence comparison_rule must be '{COMPARISON_RULE}'"
            )
    return {
        "threshold": PREDEFINED_THRESHOLD,
        "threshold_comparison": "gt",
        "min_area_px": PREDEFINED_MIN_AREA_PX,
        "bottom_crop_px": BOTTOM_CROP_PX,
    }


def _publish_and_validate(
    *,
    staged: Path,
    output: Path,
    threshold_evidence: Path,
    min_area_evidence: Path,
) -> None:
    previous = output.read_bytes() if output.exists() else None
    try:
        os.replace(staged, output)
        load_tolerance_policy(
            output,
            threshold_evidence_path=threshold_evidence,
            min_area_evidence_path=min_area_evidence,
        )
    except Exception:
        _restore_output(output, previous)
        raise


def generate_tolerance_policy(
    *,
    threshold_evidence: Path,
    min_area_evidence: Path,
    output: Path,
    policy_id: str,
    policy_version: str,
    approved_by: str,
    rationale: str,
    mask_iou_threshold: float,
    minimum_instance_precision: float,
    minimum_instance_recall: float,
    minimum_instance_f1: float,
    maximum_count_absolute_error: int,
    maximum_count_relative_error: float,
    maximum_mean_area_relative_error: float,
    maximum_mean_equivalent_diameter_relative_error: float,
    maximum_number_density_relative_error: float,
    maximum_perimeter_density_relative_error: float,
    approved_at: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate one policy without reading Independent Test data or results."""

    threshold_path, threshold_payload = _load_json_file(
        threshold_evidence,
        label="threshold evidence",
    )
    min_area_path, min_area_payload = _load_json_file(
        min_area_evidence,
        label="min-area evidence",
    )
    resolved_output = output.expanduser().resolve()
    if resolved_output in {threshold_path, min_area_path}:
        raise ValueError("output must not overwrite an evidence file")
    if resolved_output.exists() and not overwrite:
        raise FileExistsError("output already exists; pass --overwrite explicitly")

    frozen_parameters = _frozen_parameters(threshold_payload, min_area_payload)
    threshold_sha256 = sha256_file(threshold_path)
    min_area_sha256 = sha256_file(min_area_path)
    payload = {
        "schema_version": "1",
        "policy_id": _non_empty(policy_id, field="policy_id"),
        "policy_version": _non_empty(policy_version, field="policy_version"),
        "model_id": MODEL_ID,
        "threshold_evidence_sha256": threshold_sha256,
        "min_area_evidence_sha256": min_area_sha256,
        "frozen_scientific_parameters": frozen_parameters,
        "instance_matching": {
            "metric": "mask_iou",
            "mask_iou_threshold": mask_iou_threshold,
        },
        "per_image_tolerances": {
            "minimum_instance_precision": minimum_instance_precision,
            "minimum_instance_recall": minimum_instance_recall,
            "minimum_instance_f1": minimum_instance_f1,
            "maximum_count_absolute_error": maximum_count_absolute_error,
            "maximum_count_relative_error": maximum_count_relative_error,
            "maximum_mean_area_relative_error": maximum_mean_area_relative_error,
            "maximum_mean_equivalent_diameter_relative_error": (
                maximum_mean_equivalent_diameter_relative_error
            ),
            "maximum_number_density_relative_error": (
                maximum_number_density_relative_error
            ),
            "maximum_perimeter_density_relative_error": (
                maximum_perimeter_density_relative_error
            ),
        },
        "not_evaluable_rule": "fail",
        "approval": {
            "frozen_before_independent_test": True,
            "approved_by": _non_empty(approved_by, field="approved_by"),
            "approved_at": _approved_at(approved_at),
            "rationale": _non_empty(rationale, field="rationale"),
        },
    }
    validate_frozen_parameters(payload)
    _instance_matching(payload)
    _tolerances(payload)
    _approval(payload)

    staged: Path | None = None
    try:
        staged = _stage_bytes(resolved_output, _json_bytes(payload))
        load_tolerance_policy(
            staged,
            threshold_evidence_path=threshold_path,
            min_area_evidence_path=min_area_path,
        )
        if sha256_file(threshold_path) != threshold_sha256:
            raise ValueError("threshold evidence changed while policy was generated")
        if sha256_file(min_area_path) != min_area_sha256:
            raise ValueError("min-area evidence changed while policy was generated")
        _publish_and_validate(
            staged=staged,
            output=resolved_output,
            threshold_evidence=threshold_path,
            min_area_evidence=min_area_path,
        )
        staged = None
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)

    validated = load_tolerance_policy(
        resolved_output,
        threshold_evidence_path=threshold_path,
        min_area_evidence_path=min_area_path,
    )
    return {
        "status": "GENERATED",
        "output": str(resolved_output),
        "policy_sha256": validated.sha256,
        "threshold_evidence_sha256": validated.threshold_evidence_sha256,
        "min_area_evidence_sha256": validated.min_area_evidence_sha256,
        "frozen_scientific_parameters": {
            "threshold": validated.frozen_scientific_parameters.threshold,
            "threshold_comparison": (
                validated.frozen_scientific_parameters.threshold_comparison
            ),
            "min_area_px": validated.frozen_scientific_parameters.min_area_px,
            "bottom_crop_px": validated.frozen_scientific_parameters.bottom_crop_px,
        },
        "not_evaluable_rule": validated.not_evaluable_rule,
        "formal_policy_validation": "PASSED",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-evidence", required=True, type=Path)
    parser.add_argument("--min-area-evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--policy-version", required=True)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--approved-at")
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--mask-iou-threshold", required=True, type=float)
    parser.add_argument("--minimum-instance-precision", required=True, type=float)
    parser.add_argument("--minimum-instance-recall", required=True, type=float)
    parser.add_argument("--minimum-instance-f1", required=True, type=float)
    parser.add_argument("--maximum-count-absolute-error", required=True, type=int)
    parser.add_argument("--maximum-count-relative-error", required=True, type=float)
    parser.add_argument("--maximum-mean-area-relative-error", required=True, type=float)
    parser.add_argument(
        "--maximum-mean-equivalent-diameter-relative-error",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--maximum-number-density-relative-error",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--maximum-perimeter-density-relative-error",
        required=True,
        type=float,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    try:
        result = generate_tolerance_policy(
            threshold_evidence=namespace.threshold_evidence,
            min_area_evidence=namespace.min_area_evidence,
            output=namespace.output,
            policy_id=namespace.policy_id,
            policy_version=namespace.policy_version,
            approved_by=namespace.approved_by,
            approved_at=namespace.approved_at,
            rationale=namespace.rationale,
            mask_iou_threshold=namespace.mask_iou_threshold,
            minimum_instance_precision=namespace.minimum_instance_precision,
            minimum_instance_recall=namespace.minimum_instance_recall,
            minimum_instance_f1=namespace.minimum_instance_f1,
            maximum_count_absolute_error=namespace.maximum_count_absolute_error,
            maximum_count_relative_error=namespace.maximum_count_relative_error,
            maximum_mean_area_relative_error=(
                namespace.maximum_mean_area_relative_error
            ),
            maximum_mean_equivalent_diameter_relative_error=(
                namespace.maximum_mean_equivalent_diameter_relative_error
            ),
            maximum_number_density_relative_error=(
                namespace.maximum_number_density_relative_error
            ),
            maximum_perimeter_density_relative_error=(
                namespace.maximum_perimeter_density_relative_error
            ),
            overwrite=namespace.overwrite,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

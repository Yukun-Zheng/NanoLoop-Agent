"""Generate validated Small-B FROZEN_PREDEFINED evidence files."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.models.evaluate_unet_small_independent_test import (
    PREDEFINED_MIN_AREA_PX,
    PREDEFINED_THRESHOLD,
    SmallBEvaluationContract,
    validate_evaluation_contract,
)
from scripts.models.small_b_contracts import SmallBSplitManifest
from scripts.models.small_b_tolerance_policy import (
    BOTTOM_CROP_PX,
    FROZEN_PREDEFINED,
    MODEL_ID,
    FrozenScientificParameters,
    InstanceMatchingPolicy,
    PolicyApproval,
    SmallBTolerancePolicy,
    _validate_evidence,
    sha256_file,
)

COMPARISON_RULE = "probability > threshold"
THRESHOLD_PARAMETER = "threshold"
MIN_AREA_PARAMETER = "min_area_px"


def _readable_file(path: Path, *, field: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{field} must be a readable file: {resolved}")
    try:
        with resolved.open("rb") as stream:
            stream.read(1)
    except OSError as error:
        raise ValueError(f"{field} must be a readable file: {resolved}") from error
    return resolved


def _non_empty(value: str, *, field: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must be a non-empty string")
    return normalized


def _frozen_at(value: str | None) -> str:
    resolved = value or datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        timestamp = datetime.fromisoformat(resolved.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("frozen_at must be an ISO-8601 timestamp") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("frozen_at must include a timezone")
    return resolved


def _validate_fixed_parameters(
    *,
    threshold: float,
    min_area_px: int,
    bottom_crop_px: int,
    comparison_rule: str,
) -> None:
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
        or not math.isfinite(float(threshold))
        or float(threshold) != PREDEFINED_THRESHOLD
    ):
        raise ValueError("threshold must be exactly 0.30")
    if (
        isinstance(min_area_px, bool)
        or not isinstance(min_area_px, int)
        or min_area_px != PREDEFINED_MIN_AREA_PX
    ):
        raise ValueError("min_area_px must be exactly 64")
    if bottom_crop_px != BOTTOM_CROP_PX:
        raise ValueError(f"bottom_crop_px must be exactly {BOTTOM_CROP_PX}")
    if comparison_rule != COMPARISON_RULE:
        raise ValueError(f"comparison_rule must be exactly '{COMPARISON_RULE}'")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    return f"{text}\n".encode()


def _stage_bytes(output: Path, content: bytes) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _load_staged_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _validation_policy(
    *,
    threshold_evidence_sha256: str,
    min_area_evidence_sha256: str,
) -> SmallBTolerancePolicy:
    return SmallBTolerancePolicy(
        schema_version="1",
        policy_id="small-b-frozen-predefined-evidence-validation",
        policy_version="1",
        model_id=MODEL_ID,
        threshold_evidence_sha256=threshold_evidence_sha256,
        min_area_evidence_sha256=min_area_evidence_sha256,
        frozen_scientific_parameters=FrozenScientificParameters(
            threshold=PREDEFINED_THRESHOLD,
            threshold_comparison="gt",
            min_area_px=PREDEFINED_MIN_AREA_PX,
            bottom_crop_px=BOTTOM_CROP_PX,
        ),
        instance_matching=InstanceMatchingPolicy(
            metric="mask_iou",
            mask_iou_threshold=0.7,
        ),
        per_image_tolerances={},
        not_evaluable_rule="fail",
        approval=PolicyApproval(
            frozen_before_independent_test=True,
            approved_by="generator-contract-validation",
            approved_at="2000-01-01T00:00:00+00:00",
            rationale="In-memory validation only; no tolerance policy is generated.",
        ),
        sha256="0" * 64,
    )


def _validate_generated_pair(
    *,
    threshold_path: Path,
    min_area_path: Path,
    manifest_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], str, str]:
    threshold_payload = _load_staged_json(
        threshold_path,
        label="threshold evidence",
    )
    min_area_payload = _load_staged_json(
        min_area_path,
        label="min-area evidence",
    )
    threshold_sha256 = sha256_file(threshold_path)
    min_area_sha256 = sha256_file(min_area_path)
    frozen = FrozenScientificParameters(
        threshold=PREDEFINED_THRESHOLD,
        threshold_comparison="gt",
        min_area_px=PREDEFINED_MIN_AREA_PX,
        bottom_crop_px=BOTTOM_CROP_PX,
    )
    _validate_evidence(
        threshold_payload=threshold_payload,
        min_area_payload=min_area_payload,
        threshold_sha256=threshold_sha256,
        min_area_sha256=min_area_sha256,
        policy_threshold_sha256=threshold_sha256,
        policy_min_area_sha256=min_area_sha256,
        frozen=frozen,
    )
    policy = _validation_policy(
        threshold_evidence_sha256=threshold_sha256,
        min_area_evidence_sha256=min_area_sha256,
    )
    contract = SmallBEvaluationContract(
        manifest=SmallBSplitManifest(records=()),
        manifest_sha256=manifest_sha256,
        threshold_evidence=threshold_payload,
        min_area_evidence=min_area_payload,
        tolerance_policy=policy,
        threshold_evidence_sha256=threshold_sha256,
        min_area_evidence_sha256=min_area_sha256,
    )
    validate_evaluation_contract(contract, expected_policy_sha256=policy.sha256)
    return (
        threshold_payload,
        min_area_payload,
        threshold_sha256,
        min_area_sha256,
    )


def _assert_inputs_unchanged(inputs: Mapping[str, Path], expected: Mapping[str, str]) -> None:
    for field, path in inputs.items():
        if sha256_file(path) != expected[field]:
            raise ValueError(f"{field} changed while evidence was being generated")


def _restore_output(path: Path, previous: bytes | None) -> None:
    if previous is None:
        path.unlink(missing_ok=True)
        return
    temporary = _stage_bytes(path, previous)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_pair(
    *,
    threshold_staged: Path,
    min_area_staged: Path,
    threshold_output: Path,
    min_area_output: Path,
) -> None:
    previous_threshold = (
        threshold_output.read_bytes() if threshold_output.exists() else None
    )
    previous_min_area = (
        min_area_output.read_bytes() if min_area_output.exists() else None
    )
    try:
        os.replace(threshold_staged, threshold_output)
        os.replace(min_area_staged, min_area_output)
    except Exception:
        _restore_output(threshold_output, previous_threshold)
        _restore_output(min_area_output, previous_min_area)
        raise


def generate_evidence(
    *,
    manifest: Path,
    model: Path,
    config: Path,
    weight: Path,
    threshold_output: Path,
    min_area_output: Path,
    threshold_parameter_source: str,
    min_area_parameter_source: str,
    threshold: float = PREDEFINED_THRESHOLD,
    min_area_px: int = PREDEFINED_MIN_AREA_PX,
    bottom_crop_px: int = BOTTOM_CROP_PX,
    comparison_rule: str = COMPARISON_RULE,
    frozen_at: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate and validate both evidence files as one controlled operation."""

    _validate_fixed_parameters(
        threshold=threshold,
        min_area_px=min_area_px,
        bottom_crop_px=bottom_crop_px,
        comparison_rule=comparison_rule,
    )
    resolved_frozen_at = _frozen_at(frozen_at)
    sources = {
        "manifest": _readable_file(manifest, field="manifest"),
        "model": _readable_file(model, field="model"),
        "config": _readable_file(config, field="config"),
        "weight": _readable_file(weight, field="weight"),
    }
    outputs = {
        "threshold_output": threshold_output.expanduser().resolve(),
        "min_area_output": min_area_output.expanduser().resolve(),
    }
    if outputs["threshold_output"] == outputs["min_area_output"]:
        raise ValueError("threshold_output and min_area_output must differ")
    source_paths = set(sources.values())
    for field, output in outputs.items():
        if output in source_paths:
            raise ValueError(f"{field} must not overwrite an input file")
        if output.exists() and not overwrite:
            raise FileExistsError(f"{field} already exists; pass --overwrite explicitly")

    input_sha256 = {
        field: sha256_file(path) for field, path in sources.items()
    }
    common = {
        "selection_status": FROZEN_PREDEFINED,
        "calibration_performed": False,
        "frozen_before_test": True,
        "parameters_changed_after_test": False,
        "independent_test_used_for_tuning": False,
        "frozen_at": resolved_frozen_at,
        "manifest_sha256": input_sha256["manifest"],
        "model_sha256": input_sha256["model"],
        "config_sha256": input_sha256["config"],
        "weight_sha256": input_sha256["weight"],
        "comparison_rule": comparison_rule,
        "bottom_crop_px": bottom_crop_px,
    }
    threshold_payload = {
        **common,
        "parameter_source": _non_empty(
            threshold_parameter_source,
            field="threshold_parameter_source",
        ),
        "threshold": float(threshold),
    }
    threshold_staged: Path | None = None
    min_area_staged: Path | None = None
    try:
        threshold_staged = _stage_bytes(
            outputs["threshold_output"],
            _json_bytes(threshold_payload),
        )
        threshold_evidence_sha256 = sha256_file(threshold_staged)
        min_area_payload = {
            **common,
            "parameter_source": _non_empty(
                min_area_parameter_source,
                field="min_area_parameter_source",
            ),
            "threshold": float(threshold),
            "min_area_px": min_area_px,
            "threshold_evidence_sha256": threshold_evidence_sha256,
        }
        min_area_staged = _stage_bytes(
            outputs["min_area_output"],
            _json_bytes(min_area_payload),
        )
        (
            _threshold_reloaded,
            _min_area_reloaded,
            validated_threshold_sha256,
            validated_min_area_sha256,
        ) = _validate_generated_pair(
            threshold_path=threshold_staged,
            min_area_path=min_area_staged,
            manifest_sha256=input_sha256["manifest"],
        )
        if validated_threshold_sha256 != threshold_evidence_sha256:
            raise ValueError("threshold evidence SHA changed after staged write")
        _assert_inputs_unchanged(sources, input_sha256)
        _publish_pair(
            threshold_staged=threshold_staged,
            min_area_staged=min_area_staged,
            threshold_output=outputs["threshold_output"],
            min_area_output=outputs["min_area_output"],
        )
        threshold_staged = None
        min_area_staged = None
    finally:
        if threshold_staged is not None:
            threshold_staged.unlink(missing_ok=True)
        if min_area_staged is not None:
            min_area_staged.unlink(missing_ok=True)

    return {
        "status": "GENERATED",
        "selection_status": FROZEN_PREDEFINED,
        "threshold_output": str(outputs["threshold_output"]),
        "min_area_output": str(outputs["min_area_output"]),
        "manifest_sha256": input_sha256["manifest"],
        "model_sha256": input_sha256["model"],
        "config_sha256": input_sha256["config"],
        "weight_sha256": input_sha256["weight"],
        "threshold_evidence_sha256": validated_threshold_sha256,
        "min_area_evidence_sha256": validated_min_area_sha256,
        "formal_contract_validation": "PASSED",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--weight", required=True, type=Path)
    parser.add_argument("--threshold-output", required=True, type=Path)
    parser.add_argument("--min-area-output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=PREDEFINED_THRESHOLD)
    parser.add_argument("--min-area-px", type=int, default=PREDEFINED_MIN_AREA_PX)
    parser.add_argument("--bottom-crop-px", type=int, default=BOTTOM_CROP_PX)
    parser.add_argument("--comparison-rule", default=COMPARISON_RULE)
    parser.add_argument("--threshold-parameter-source", required=True)
    parser.add_argument("--min-area-parameter-source", required=True)
    parser.add_argument("--frozen-at")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    try:
        result = generate_evidence(
            manifest=namespace.manifest,
            model=namespace.model,
            config=namespace.config,
            weight=namespace.weight,
            threshold_output=namespace.threshold_output,
            min_area_output=namespace.min_area_output,
            threshold_parameter_source=namespace.threshold_parameter_source,
            min_area_parameter_source=namespace.min_area_parameter_source,
            threshold=namespace.threshold,
            min_area_px=namespace.min_area_px,
            bottom_crop_px=namespace.bottom_crop_px,
            comparison_rule=namespace.comparison_rule,
            frozen_at=namespace.frozen_at,
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

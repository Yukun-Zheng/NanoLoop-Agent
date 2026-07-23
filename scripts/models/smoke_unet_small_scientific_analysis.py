"""Validate frozen Small-B scientific Analysis runs without reimplementing Analysis."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from scripts.models.evaluate_unet_small_independent_test import (
    SmallBEvaluationContract,
    load_evaluation_contract,
    validate_evaluation_contract,
)
from scripts.models.small_b_contracts import SplitManifestRecord
from scripts.models.small_b_tolerance_policy import sha256_file

__all__ = [
    "execute_existing_small_analysis",
    "run_scientific_smoke",
    "validate_analysis_result",
]

MODEL_ID = "unet-small-balanced-v1"
BOTTOM_CROP_PX = 130
_REQUIRED_ARTIFACTS = ("mask", "instances", "run_configuration")
_IDENTITY_FIELDS = (
    "weight_sha256",
    "config_sha256",
    "model_card_sha256",
    "adapter_sha256",
)


@dataclass(frozen=True, slots=True)
class FrozenSmokeParameters:
    threshold: float
    min_area_px: int
    mask_iou_threshold: float
    bottom_crop_px: int = BOTTOM_CROP_PX
    threshold_comparison: str = "gt"
    fill_holes: bool = True
    watershed_enabled: bool = False
    exclude_border: bool = True
    connectivity: int = 2
    perimeter_neighborhood: int = 8


class AnalysisExecutor(Protocol):
    """Adapter around the existing Small Analysis execution service."""

    def __call__(
        self,
        record: SplitManifestRecord,
        parameters: FrozenSmokeParameters,
    ) -> Mapping[str, Any]: ...


def execute_existing_small_analysis(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    """Lazily call the existing Small Analysis executor on its supported runtime."""

    from scripts.models.smoke_unet_small_analysis import execute_analysis

    return execute_analysis(*args, **kwargs)


def frozen_parameters(contract: SmallBEvaluationContract) -> FrozenSmokeParameters:
    policy = contract.tolerance_policy
    frozen = policy.frozen_scientific_parameters
    return FrozenSmokeParameters(
        threshold=frozen.threshold,
        min_area_px=frozen.min_area_px,
        mask_iou_threshold=policy.instance_matching.mask_iou_threshold,
        bottom_crop_px=frozen.bottom_crop_px,
        threshold_comparison=frozen.threshold_comparison,
    )


def select_manifest_records(
    contract: SmallBEvaluationContract,
    sample_ids: Sequence[str],
) -> tuple[SplitManifestRecord, ...]:
    """Resolve explicitly requested included records without enumerating private data."""

    requested = tuple(sample_ids)
    if len(requested) != len(set(requested)):
        raise ValueError("sample IDs must be unique")
    included = {
        record.sample_id: record
        for record in contract.manifest.records
        if record.included and record.split.value != "excluded"
    }
    missing = [sample_id for sample_id in requested if sample_id not in included]
    if missing:
        raise ValueError(f"sample IDs are absent, excluded, or not included: {missing}")
    return tuple(included[sample_id] for sample_id in requested)


def _json_object(path: Path, *, artifact: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {artifact} artifact: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{artifact} artifact must be a JSON object")
    return payload


def _artifact_paths(result: Mapping[str, Any]) -> dict[str, Path]:
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Analysis result is missing artifacts")
    paths: dict[str, Path] = {}
    for name in _REQUIRED_ARTIFACTS:
        value = artifacts.get(name)
        if not isinstance(value, str | Path):
            raise ValueError(f"Analysis result is missing canonical artifact: {name}")
        path = Path(value).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"canonical artifact is not a file: {name}")
        paths[name] = path
    return paths


def _validate_identity(
    record: SplitManifestRecord,
    result: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> dict[str, str]:
    model = result.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Analysis result is missing model identity")
    if model.get("model_id") != MODEL_ID or run_config.get("model_id") != MODEL_ID:
        raise ValueError("Analysis result used a different model_id")
    if run_config.get("image_sha256") != record.image_sha256:
        raise ValueError("run configuration image SHA differs from the manifest")

    identity: dict[str, str] = {}
    for field in _IDENTITY_FIELDS:
        observed = model.get(field)
        stored = run_config.get(field)
        if (
            not isinstance(observed, str)
            or len(observed) != 64
            or observed != stored
        ):
            raise ValueError(f"model/run configuration {field} identity mismatch")
        identity[field] = observed
    adapter_path = model.get("adapter_path")
    if (
        not isinstance(adapter_path, str)
        or not adapter_path
        or adapter_path != run_config.get("adapter_path")
    ):
        raise ValueError("model/run configuration adapter_path identity mismatch")
    identity["adapter_path"] = adapter_path
    return identity


def _validate_frozen_configuration(
    result: Mapping[str, Any],
    run_config: Mapping[str, Any],
    parameters: FrozenSmokeParameters,
) -> None:
    inference = run_config.get("inference")
    postprocess = run_config.get("resolved_postprocess")
    roi = run_config.get("analysis_roi")
    if not isinstance(inference, Mapping):
        raise ValueError("run configuration is missing inference")
    if not isinstance(postprocess, Mapping):
        raise ValueError("run configuration is missing resolved_postprocess")
    if not isinstance(roi, Mapping):
        raise ValueError("run configuration is missing analysis_roi")
    if not math.isclose(float(inference.get("threshold", -1.0)), parameters.threshold):
        raise ValueError("run threshold differs from frozen threshold evidence")
    if inference.get("min_area_px") != parameters.min_area_px:
        raise ValueError("run min_area_px differs from frozen min-area evidence")
    expected_postprocess = {
        "min_area_px": parameters.min_area_px,
        "fill_holes": parameters.fill_holes,
        "watershed_enabled": parameters.watershed_enabled,
        "exclude_border": parameters.exclude_border,
        "connectivity": parameters.connectivity,
    }
    for field, expected in expected_postprocess.items():
        if postprocess.get(field) != expected:
            raise ValueError(f"resolved postprocess {field} differs from frozen settings")

    invalid_rects = roi.get("invalid_rects")
    if not isinstance(invalid_rects, list) or not any(
        isinstance(rect, Mapping)
        and rect.get("y2", 0) - rect.get("y1", 0) == parameters.bottom_crop_px
        and rect.get("reason") == "instrument_bar"
        for rect in invalid_rects
    ):
        raise ValueError("run configuration did not freeze the bottom 130 px exclusion")

    frozen_inference = result.get("frozen_inference")
    resolved_postprocess = result.get("resolved_postprocess")
    if not isinstance(frozen_inference, Mapping) or not isinstance(
        resolved_postprocess, Mapping
    ):
        raise ValueError("Analysis result omitted frozen configuration summaries")
    if frozen_inference.get("threshold") != parameters.threshold:
        raise ValueError("Analysis result threshold summary differs from frozen evidence")
    if frozen_inference.get("min_area_px") != parameters.min_area_px:
        raise ValueError("Analysis result min-area summary differs from frozen evidence")
    for field, expected in expected_postprocess.items():
        if resolved_postprocess.get(field) != expected:
            raise ValueError(f"Analysis result postprocess {field} differs from run config")


def _validate_canonical_artifacts(paths: Mapping[str, Path]) -> tuple[int, int]:
    try:
        with Image.open(paths["mask"]) as image:
            image.load()
            width, height = image.size
    except OSError as error:
        raise ValueError("canonical mask is unreadable") from error
    if width <= 0 or height <= BOTTOM_CROP_PX:
        raise ValueError("canonical mask dimensions are invalid")

    instances = _json_object(paths["instances"], artifact="instances")
    records = instances.get("instances")
    if (
        instances.get("coordinate_space") != "original_px"
        or instances.get("width") != width
        or instances.get("height") != height
        or not isinstance(records, list)
        or instances.get("instance_count") != len(records)
    ):
        raise ValueError("canonical instances artifact is inconsistent with the mask")
    return width, height


def validate_analysis_result(
    record: SplitManifestRecord,
    result: Mapping[str, Any],
    parameters: FrozenSmokeParameters,
) -> dict[str, Any]:
    """Validate canonical artifacts, frozen run settings, and content identities."""

    if result.get("final_status") != "completed":
        raise ValueError("Analysis run did not complete")
    paths = _artifact_paths(result)
    run_config = _json_object(paths["run_configuration"], artifact="run configuration")
    if run_config.get("contract_schema_version") != 3:
        raise ValueError("run configuration is not schema-v3")
    identity = _validate_identity(record, result, run_config)
    _validate_frozen_configuration(result, run_config, parameters)
    width, height = _validate_canonical_artifacts(paths)
    return {
        "sample_id": record.sample_id,
        "run_id": result.get("run_id"),
        "status": "PASS",
        "image_sha256": record.image_sha256,
        "model_identity": identity,
        "image_size": {"width": width, "height": height},
        "artifact_sha256": {
            name: sha256_file(path) for name, path in sorted(paths.items())
        },
    }


def run_scientific_smoke(
    contract: SmallBEvaluationContract,
    *,
    expected_policy_sha256: str,
    sample_ids: Sequence[str],
    analysis_executor: AnalysisExecutor,
) -> dict[str, Any]:
    """Execute selected records through existing Analysis and return a compact smoke report."""

    try:
        validate_evaluation_contract(
            contract,
            expected_policy_sha256=expected_policy_sha256,
        )
        records = select_manifest_records(contract, sample_ids)
    except Exception as error:
        return {
            "status": "FAIL",
            "samples": [],
            "failures": [{"reason_code": "FROZEN_CONTRACT_INVALID", "message": str(error)}],
        }
    if not records:
        return {
            "status": "NOT_EVALUATED",
            "samples": [],
            "failures": [],
            "reason_codes": ["NO_MANIFEST_RECORD_SELECTED"],
        }

    parameters = frozen_parameters(contract)
    samples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in records:
        try:
            result = analysis_executor(record, parameters)
            samples.append(validate_analysis_result(record, result, parameters))
        except Exception as error:
            failures.append(
                {
                    "sample_id": record.sample_id,
                    "reason_code": "SCIENTIFIC_ANALYSIS_INVALID",
                    "message": str(error),
                }
            )
    return {
        "status": "FAIL" if failures else "PASS",
        "model_id": MODEL_ID,
        "split_manifest_sha256": contract.manifest_sha256,
        "threshold_evidence_sha256": contract.threshold_evidence_sha256,
        "min_area_evidence_sha256": contract.min_area_evidence_sha256,
        "tolerance_policy_sha256": contract.tolerance_policy.sha256,
        "frozen_parameters": {
            "threshold": parameters.threshold,
            "threshold_comparison": parameters.threshold_comparison,
            "min_area_px": parameters.min_area_px,
            "mask_iou_threshold": parameters.mask_iou_threshold,
            "bottom_crop_px": parameters.bottom_crop_px,
        },
        "samples": samples,
        "failures": failures,
    }


def write_smoke_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--threshold-evidence", required=True, type=Path)
    parser.add_argument("--min-area-evidence", required=True, type=Path)
    parser.add_argument("--tolerance-policy", required=True, type=Path)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--not-evaluated",
        action="store_true",
        help="Validate frozen inputs and write NOT_EVALUATED without running Analysis.",
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
                "real smoke requires an approved existing-Small-Analysis executor; "
                "use run_scientific_smoke"
            )
        select_manifest_records(contract, namespace.sample_id)
        report = {
            "status": "NOT_EVALUATED",
            "model_id": MODEL_ID,
            "selected_sample_ids": namespace.sample_id,
            "split_manifest_sha256": contract.manifest_sha256,
            "threshold_evidence_sha256": contract.threshold_evidence_sha256,
            "min_area_evidence_sha256": contract.min_area_evidence_sha256,
            "tolerance_policy_sha256": contract.tolerance_policy.sha256,
        }
        write_smoke_report(namespace.output, report)
    except Exception as error:
        print(
            json.dumps(
                {"status": "error", "error_type": type(error).__name__, "message": str(error)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

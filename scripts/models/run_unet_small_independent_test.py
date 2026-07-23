"""Run Small-B Independent Test from approved, pre-existing Analysis artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.analysis.instance_artifacts import decode_binary_mask
from scripts.models.evaluate_unet_small_independent_test import (
    SmallBEvaluationContract,
    evaluate_independent_test,
    load_evaluation_contract,
    write_evaluation_outputs,
)
from scripts.models.small_b_contracts import ManifestSplit, SplitManifestRecord
from scripts.models.small_b_tolerance_policy import BOTTOM_CROP_PX, MODEL_ID, sha256_file

_SHA256_LENGTH = 64
_TERMINAL_ANALYSIS_STATUSES = {"completed", "completed_with_warnings"}
_CANONICAL_ARTIFACTS = ("mask", "instances", "run_configuration")
_IDENTITY_FIELDS = (
    "weight_sha256",
    "config_sha256",
    "model_card_sha256",
    "adapter_sha256",
)


@dataclass(frozen=True, slots=True)
class AnalysisResultBinding:
    sample_id: str
    result_path: Path


def _json_object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _inside(root: Path, path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(root):
        raise ValueError(f"{label} must be a file beneath analysis-root: {resolved}")
    return resolved


def _artifact_path(
    analysis_root: Path,
    value: object,
    *,
    artifact: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Analysis result is missing canonical artifact: {artifact}")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = analysis_root / candidate
    return _inside(analysis_root, candidate, label=f"canonical {artifact}")


def _sha256_string(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _load_binary_image(path: Path, *, label: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            image.load()
            array = np.asarray(image)
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"{label} is unreadable: {path}") from error
    if array.ndim != 2:
        raise ValueError(f"{label} must be a two-dimensional single-channel mask")
    values = np.unique(array)
    if not set(values.tolist()).issubset({0, 1, 255}):
        raise ValueError(f"{label} must be binary")
    return np.asarray(array != 0, dtype=np.bool_)


def _expected_asset_sha(
    contract: SmallBEvaluationContract,
    field: str,
) -> str:
    threshold_value = contract.threshold_evidence.get(field)
    min_area_value = contract.min_area_evidence.get(field)
    threshold_sha = _sha256_string(threshold_value, field=f"threshold evidence {field}")
    min_area_sha = _sha256_string(min_area_value, field=f"min-area evidence {field}")
    if threshold_sha != min_area_sha:
        raise ValueError(f"threshold/min-area evidence {field} differs")
    return threshold_sha


def _validate_instance_union(
    instances_path: Path,
    prediction: np.ndarray,
) -> None:
    payload = _json_object(instances_path, label="canonical instances")
    height, width = prediction.shape
    records = payload.get("instances")
    if (
        payload.get("coordinate_space") != "original_px"
        or payload.get("width") != width
        or payload.get("height") != height
        or not isinstance(records, list)
        or payload.get("instance_count") != len(records)
    ):
        raise ValueError("canonical instances metadata differs from prediction mask")

    union = np.zeros(prediction.shape, dtype=np.bool_)
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"canonical instance {index} must be an object")
        encoded = record.get("mask")
        if (
            not isinstance(encoded, Mapping)
            or encoded.get("encoding") != "flat_rle_v1"
            or encoded.get("order") != "row_major"
            or not isinstance(encoded.get("starts"), list)
            or not isinstance(encoded.get("lengths"), list)
        ):
            raise ValueError(f"canonical instance {index} has an invalid mask encoding")
        instance = decode_binary_mask(
            starts=encoded["starts"],
            lengths=encoded["lengths"],
            width=width,
            height=height,
        )
        expected_sha = _sha256_string(
            encoded.get("sha256"),
            field=f"canonical instance {index} mask sha256",
        )
        observed_sha = hashlib.sha256(
            np.packbits(instance, bitorder="little").tobytes()
        ).hexdigest()
        if observed_sha != expected_sha:
            raise ValueError(f"canonical instance {index} mask SHA mismatch")
        union |= instance
    if not np.array_equal(union, prediction):
        raise ValueError("canonical prediction mask differs from canonical instances union")


def _validate_bottom_exclusion(
    run_config: Mapping[str, Any],
    *,
    width: int,
    height: int,
) -> None:
    if run_config.get("roi_mode") != "full_image":
        raise ValueError("approved prediction must use the full_image ROI contract")
    roi = run_config.get("analysis_roi")
    if not isinstance(roi, Mapping):
        raise ValueError("run configuration is missing analysis_roi")
    expected = {
        "x1": 0,
        "y1": height - BOTTOM_CROP_PX,
        "x2": width,
        "y2": height,
        "reason": "model_bottom_information_bar",
    }
    invalid = roi.get("invalid_rects")
    if not isinstance(invalid, list) or not any(
        isinstance(region, Mapping)
        and all(region.get(field) == value for field, value in expected.items())
        for region in invalid
    ):
        raise ValueError("run configuration lacks the exact bottom 130 px exclusion")


def _validate_frozen_parameters(
    contract: SmallBEvaluationContract,
    result: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> None:
    frozen = contract.tolerance_policy.frozen_scientific_parameters
    inference = run_config.get("inference")
    postprocess = run_config.get("resolved_postprocess")
    if not isinstance(inference, Mapping) or not isinstance(postprocess, Mapping):
        raise ValueError("run configuration is missing frozen inference/postprocess settings")
    threshold = inference.get("threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
        or not math.isclose(float(threshold), frozen.threshold)
    ):
        raise ValueError("prediction threshold differs from frozen evidence")
    if inference.get("min_area_px") != frozen.min_area_px:
        raise ValueError("prediction min_area_px differs from frozen evidence")
    expected_postprocess = {
        "min_area_px": frozen.min_area_px,
        "fill_holes": True,
        "watershed_enabled": False,
        "exclude_border": True,
        "connectivity": 2,
    }
    for field, expected in expected_postprocess.items():
        if postprocess.get(field) != expected:
            raise ValueError(f"prediction postprocess {field} differs from frozen contract")
    if frozen.threshold_comparison != "gt":
        raise ValueError("prediction contract requires strict probability > threshold")
    for evidence in (contract.threshold_evidence, contract.min_area_evidence):
        if evidence.get("comparison_rule") != "probability > threshold":
            raise ValueError("evidence comparison_rule is not strict probability > threshold")
        if evidence.get("bottom_crop_px") != frozen.bottom_crop_px:
            raise ValueError("evidence bottom_crop_px differs from frozen policy")

    frozen_summary = result.get("frozen_inference")
    resolved_summary = result.get("resolved_postprocess")
    if not isinstance(frozen_summary, Mapping) or not isinstance(
        resolved_summary, Mapping
    ):
        raise ValueError("Analysis result omitted frozen configuration summaries")
    if frozen_summary.get("threshold") != frozen.threshold:
        raise ValueError("Analysis result threshold summary differs from frozen evidence")
    if frozen_summary.get("min_area_px") != frozen.min_area_px:
        raise ValueError("Analysis result min-area summary differs from frozen evidence")
    for field, expected in expected_postprocess.items():
        if resolved_summary.get(field) != expected:
            raise ValueError(f"Analysis result postprocess {field} differs from run config")


class ApprovedAnalysisPredictionProvider:
    """Read approved canonical Analysis masks without importing or running inference."""

    def __init__(
        self,
        contract: SmallBEvaluationContract,
        *,
        analysis_root: Path,
        bindings: Sequence[AnalysisResultBinding],
    ) -> None:
        self._contract = contract
        self._analysis_root = analysis_root.expanduser().resolve(strict=True)
        if not self._analysis_root.is_dir():
            raise ValueError("analysis-root must be a directory")
        records = {
            record.sample_id: record
            for record in contract.manifest.select(ManifestSplit.INDEPENDENT_TEST)
        }
        supplied: dict[str, Path] = {}
        for binding in bindings:
            if binding.sample_id in supplied:
                raise ValueError(f"duplicate prediction for sample_id: {binding.sample_id}")
            if binding.sample_id not in records:
                raise ValueError(
                    f"prediction sample_id is not an included independent_test row: "
                    f"{binding.sample_id}"
                )
            result_path = binding.result_path
            if not result_path.is_absolute():
                result_path = self._analysis_root / result_path
            supplied[binding.sample_id] = _inside(
                self._analysis_root,
                result_path,
                label="Analysis result",
            )
        missing = sorted(set(records) - set(supplied))
        if missing:
            raise ValueError(f"missing approved predictions for sample_id: {missing}")
        self._result_paths = supplied
        self._used_masks: set[Path] = set()
        self._provenance: dict[str, dict[str, str]] = {}

    def __call__(self, record: SplitManifestRecord) -> np.ndarray:
        result_path = self._result_paths.get(record.sample_id)
        if result_path is None:
            raise ValueError(f"no approved prediction binding for {record.sample_id}")
        result = _json_object(result_path, label="Analysis result")
        declared_sample_id = result.get("sample_id")
        if declared_sample_id is not None and declared_sample_id != record.sample_id:
            raise ValueError(f"prediction sample_id mismatch for {record.sample_id}")
        declared_image_sha = result.get("source_image_sha256")
        if declared_image_sha is not None and declared_image_sha != record.image_sha256:
            raise ValueError(f"prediction source image SHA mismatch for {record.sample_id}")
        if result.get("final_status") not in _TERMINAL_ANALYSIS_STATUSES:
            raise ValueError(f"Analysis run is not complete for {record.sample_id}")

        artifacts = result.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError("Analysis result is missing artifacts")
        paths = {
            name: _artifact_path(
                self._analysis_root,
                artifacts.get(name),
                artifact=name,
            )
            for name in _CANONICAL_ARTIFACTS
        }
        if paths["mask"] in self._used_masks:
            raise ValueError(f"duplicate canonical prediction artifact: {paths['mask']}")
        self._used_masks.add(paths["mask"])

        run_config = _json_object(paths["run_configuration"], label="run configuration")
        if (
            run_config.get("contract_schema_version") != 3
            or run_config.get("provenance_status") != "complete"
        ):
            raise ValueError("run configuration must have complete schema-v3 provenance")
        if run_config.get("model_id") != MODEL_ID:
            raise ValueError("prediction model_id mismatch")
        if run_config.get("image_sha256") != record.image_sha256:
            raise ValueError(f"run configuration image SHA mismatch for {record.sample_id}")

        model = result.get("model")
        if not isinstance(model, Mapping) or model.get("model_id") != MODEL_ID:
            raise ValueError("Analysis result model_id mismatch")
        expected_identity = {
            "weight_sha256": _expected_asset_sha(self._contract, "weight_sha256"),
            "config_sha256": _expected_asset_sha(self._contract, "config_sha256"),
        }
        for field in _IDENTITY_FIELDS:
            result_sha = _sha256_string(model.get(field), field=f"Analysis result {field}")
            run_sha = _sha256_string(run_config.get(field), field=f"run config {field}")
            if result_sha != run_sha:
                raise ValueError(f"prediction model/run {field} mismatch")
            if field in expected_identity and result_sha != expected_identity[field]:
                raise ValueError(f"prediction {field} differs from frozen evidence")
        if (
            not isinstance(model.get("adapter_path"), str)
            or model.get("adapter_path") != run_config.get("adapter_path")
        ):
            raise ValueError("prediction adapter identity mismatch")

        _validate_frozen_parameters(self._contract, result, run_config)
        prediction = _load_binary_image(paths["mask"], label="canonical prediction mask")
        height, width = prediction.shape
        if height <= BOTTOM_CROP_PX or np.any(prediction[-BOTTOM_CROP_PX:]):
            raise ValueError("canonical prediction violates the bottom 130 px exclusion")
        _validate_bottom_exclusion(run_config, width=width, height=height)
        _validate_instance_union(paths["instances"], prediction)
        self._provenance[record.sample_id] = {
            "analysis_result_path": str(result_path),
            "analysis_result_sha256": sha256_file(result_path),
            "prediction_mask_path": str(paths["mask"]),
            "prediction_mask_sha256": sha256_file(paths["mask"]),
            "instances_path": str(paths["instances"]),
            "instances_sha256": sha256_file(paths["instances"]),
            "run_configuration_path": str(paths["run_configuration"]),
            "run_configuration_sha256": sha256_file(paths["run_configuration"]),
        }
        return prediction

    def provenance(self) -> Mapping[str, Mapping[str, str]]:
        return dict(self._provenance)


class ManifestTruthProvider:
    """Read only the manifest-bound formal GT masks and verify their SHA-256."""

    def __init__(self, *, private_data_root: Path) -> None:
        self._root = private_data_root.expanduser().resolve(strict=True)
        if not self._root.is_dir():
            raise ValueError("private-data-root must be a directory")

    def __call__(self, record: SplitManifestRecord) -> np.ndarray:
        if record.mask_path is None or record.mask_sha256 is None:
            raise ValueError(f"manifest GT is missing for {record.sample_id}")
        posix = PurePosixPath(record.mask_path)
        windows = PureWindowsPath(record.mask_path)
        if posix.is_absolute() or windows.is_absolute() or windows.drive or ".." in posix.parts:
            raise ValueError(f"manifest GT path is unsafe for {record.sample_id}")
        path = _inside(
            self._root,
            self._root / Path(*posix.parts),
            label=f"GT for {record.sample_id}",
        )
        if sha256_file(path) != record.mask_sha256:
            raise ValueError(f"GT SHA mismatch for {record.sample_id}")
        return _load_binary_image(path, label=f"GT for {record.sample_id}")


def _parse_binding(value: str) -> AnalysisResultBinding:
    sample_id, separator, raw_path = value.partition("=")
    if not separator or not sample_id.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(
            "analysis-result must use SAMPLE_ID=/path/to/analysis-result.json"
        )
    return AnalysisResultBinding(sample_id=sample_id.strip(), result_path=Path(raw_path))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode()
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_formal_evaluation(
    *,
    split_manifest: Path,
    threshold_evidence: Path,
    min_area_evidence: Path,
    tolerance_policy: Path,
    expected_policy_sha256: str,
    analysis_root: Path,
    analysis_results: Sequence[AnalysisResultBinding],
    private_data_root: Path,
    output_root: Path,
    evaluation_id: str = "small-b-independent-test-v1",
) -> dict[str, Any]:
    """Wire approved artifacts into the unchanged formal evaluator."""

    resolved_output = output_root.expanduser().resolve(strict=False)
    if resolved_output.exists():
        raise FileExistsError(f"output-root already exists: {resolved_output}")
    contract = load_evaluation_contract(
        split_manifest_path=split_manifest,
        threshold_evidence_path=threshold_evidence,
        min_area_evidence_path=min_area_evidence,
        tolerance_policy_path=tolerance_policy,
        expected_policy_sha256=expected_policy_sha256,
    )
    prediction_provider = ApprovedAnalysisPredictionProvider(
        contract,
        analysis_root=analysis_root,
        bindings=analysis_results,
    )
    result = evaluate_independent_test(
        contract,
        expected_policy_sha256=expected_policy_sha256,
        prediction_provider=prediction_provider,
        truth_provider=ManifestTruthProvider(private_data_root=private_data_root),
        evaluation_id=evaluation_id,
    )
    write_evaluation_outputs(resolved_output, contract, result)
    _atomic_write_json(
        resolved_output / "prediction-artifact-provenance.json",
        {
            "schema_version": "1",
            "evaluation_id": evaluation_id,
            "split_manifest_sha256": contract.manifest_sha256,
            "tolerance_policy_sha256": contract.tolerance_policy.sha256,
            "independent_test_used_for_tuning": False,
            "artifacts": prediction_provider.provenance(),
        },
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--threshold-evidence", required=True, type=Path)
    parser.add_argument("--min-area-evidence", required=True, type=Path)
    parser.add_argument("--tolerance-policy", required=True, type=Path)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--analysis-root", required=True, type=Path)
    parser.add_argument(
        "--analysis-result",
        required=True,
        action="append",
        type=_parse_binding,
        help="Repeat SAMPLE_ID=/path/to/formal-analysis-result.json for every test row.",
    )
    parser.add_argument("--private-data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--evaluation-id", default="small-b-independent-test-v1")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    try:
        result = run_formal_evaluation(
            split_manifest=namespace.split_manifest,
            threshold_evidence=namespace.threshold_evidence,
            min_area_evidence=namespace.min_area_evidence,
            tolerance_policy=namespace.tolerance_policy,
            expected_policy_sha256=namespace.expected_policy_sha256,
            analysis_root=namespace.analysis_root,
            analysis_results=namespace.analysis_result,
            private_data_root=namespace.private_data_root,
            output_root=namespace.output_root,
            evaluation_id=namespace.evaluation_id,
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
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from app.analysis.config import MorphometryConfig, PostprocessProfile
from app.analysis.morphometry import measure
from app.analysis.postprocessing import (
    NormalizedInstance,
    normalize_semantic_mask_detailed,
)
from scripts.models.evaluate_unet_large_independent_test import (
    _TOLERANCE_FIELDS,
    _require_finite_number,
    _scientific_metric_values,
)
from scripts.models.evaluate_unet_large_independent_test import (
    _load_tolerance_policy as _load_large_tolerance_policy,
)
from scripts.models.evaluate_unet_large_independent_test import (
    _match_instances as _large_match_instances,
)
from scripts.models.evaluate_unet_large_independent_test import (
    compute_scientific_metrics as _large_compute_scientific_metrics,
)
from scripts.models.small_b_contracts import (
    ManifestSplit,
    SplitManifestRecord,
    load_split_manifest,
)

_match_instances = _large_match_instances
compute_scientific_metrics = _large_compute_scientific_metrics

MODEL_ID = "unet-agglomerated-specialized-v1"
MODEL_VERSION = "1"
ADAPTER_PATH = "app.inference.adapters.unet:UNetAdapter"
TORCHSCRIPT_SHA256 = "d36cf627dc6d1eea83b769743b657e6bb3d9a1af39ddc6a00ae95acc3b20ffc9"
CHECKPOINT_SHA256 = "e2be19c6fe1e843856fb339d13de8baed8d748f88558ba7bd3eaaa20b90ede21"
CONFIG_SHA256 = "54f5113d0de4b5d2a26e48e8c231b5223c43b373ba8a496934f2b4bbd1bfc524"
MODEL_CARD_SHA256 = "6a56b5b29cb2f8c1b8d1893ca072a2dab0ad309708d9f682f9374f6ea279fef1"
ADAPTER_SHA256 = "6055db452f0a78a0352732d66ea3436f16a558cf19d1a6f022a78627136dfab6"
IMAGE_WIDTH = 2048
IMAGE_HEIGHT = 1536
BOTTOM_CROP_PX = 130
VALID_HEIGHT = IMAGE_HEIGHT - BOTTOM_CROP_PX
THRESHOLD = 0.25
MIN_AREA_PX = 1024
SCALE_NM_PER_PIXEL = 100 / 184
PERIMETER_NEIGHBORHOOD = 8
SEED = 2026
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORMAL_PREDICTION_GLOB = "artifacts/job_*/images/img_*/runs/run_*/pred_mask.png"
STATISTIC_FIELDS = (
    "agglomerate_count",
    "mean_equivalent_diameter_nm",
    "number_density_um2",
    "perimeter_density_um",
    "coverage_ratio",
)
_AGGLOMERATED_TOLERANCE_FIELDS = (
    *_TOLERANCE_FIELDS,
    ("coverage_relative_error", "maximum_coverage_relative_error", "lte"),
)
_ZERO_DENOMINATOR_RULES = {
    "instance_precision": "value_is_one_when_no_prediction_instances",
    "instance_recall": "value_is_one_when_no_ground_truth_instances",
    "instance_f1": "value_is_one_when_prediction_and_ground_truth_are_both_empty",
    "count_absolute_error": "not_applicable_metric_has_no_denominator",
    "count_relative_error": "denominator_is_maximum_of_ground_truth_count_and_one",
    "mean_area_relative_error": (
        "not_evaluable_when_ground_truth_value_is_zero_or_either_instance_set_is_empty"
    ),
    "mean_equivalent_diameter_relative_error": (
        "not_evaluable_when_ground_truth_value_is_zero_or_either_instance_set_is_empty"
    ),
    "number_density_relative_error": (
        "not_evaluable_when_ground_truth_value_is_zero_or_either_instance_set_is_empty"
    ),
    "perimeter_density_relative_error": (
        "not_evaluable_when_ground_truth_value_is_zero_or_either_instance_set_is_empty"
    ),
    "coverage_relative_error": (
        "not_evaluable_when_ground_truth_value_is_zero_or_either_instance_set_is_empty"
    ),
}


@dataclass(frozen=True)
class EvaluationParameters:
    analysis_output_root: Path
    mask_dir: Path
    output_root: Path
    independent_test_manifest_path: Path
    checkpoint_sha256: str
    tolerance_policy_path: Path | None = None
    instance_iou_threshold: float | None = None


@dataclass(frozen=True)
class ConfusionMetrics:
    tp: int
    fp: int
    fn: int
    tn: int
    dice: float
    iou: float
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class AgglomeratedTolerancePolicy:
    content: Mapping[str, Any]
    source_path: Path
    filename: str
    sha256: str
    instance_iou_threshold: float
    approved: bool
    metric_contracts: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class SampleInputs:
    sample_id: str
    filename: str
    prediction_path: Path
    truth_path: Path
    run_config_path: Path
    execution_provenance_path: Path
    metadata_path: Path
    image_path: Path
    manifest_record: SplitManifestRecord


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen Agglomerated U-Net Analysis masks selected by an approved manifest."
        )
    )
    parser.add_argument("--analysis-output-root", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--independent-test-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--tolerance-policy", type=Path)
    parser.add_argument("--instance-iou-threshold", type=float)
    return parser


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_output_root(output_root: Path, *, repository: Path | None = None) -> Path:
    resolved = output_root.expanduser().resolve(strict=False)
    repository = (repository or _repository_root()).resolve()
    if _is_relative_to(resolved, repository):
        raise ValueError("output-root must be outside the Git repository")
    if resolved.exists():
        raise ValueError("output-root must not already exist")
    return resolved


def _validated_parameters(namespace: argparse.Namespace) -> EvaluationParameters:
    analysis_output_root = namespace.analysis_output_root.expanduser().resolve()
    mask_dir = namespace.mask_dir.expanduser().resolve()
    if not analysis_output_root.is_dir():
        raise ValueError("analysis-output-root must be an existing directory")
    if not mask_dir.is_dir():
        raise ValueError("mask-dir must be an existing directory")
    instance_iou_threshold = namespace.instance_iou_threshold
    if instance_iou_threshold is not None and not 0.0 < instance_iou_threshold <= 1.0:
        raise ValueError("instance-iou-threshold must be in (0, 1]")
    return EvaluationParameters(
        analysis_output_root=analysis_output_root,
        mask_dir=mask_dir,
        output_root=_validate_output_root(namespace.output_root),
        independent_test_manifest_path=namespace.independent_test_manifest,
        checkpoint_sha256=namespace.checkpoint_sha256,
        tolerance_policy_path=namespace.tolerance_policy,
        instance_iou_threshold=instance_iou_threshold,
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return payload


def _file_identity(path: Path, *, relative_path: str) -> dict[str, Any]:
    return {
        "path": relative_path,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _load_independent_test_manifest(
    path: Path,
) -> tuple[Path, dict[str, SplitManifestRecord]]:
    resolved = path.expanduser().resolve(strict=True)
    manifest = load_split_manifest(resolved, require_calibration=False)
    selected = tuple(
        record
        for record in manifest.records
        if record.included and record.split is ManifestSplit.INDEPENDENT_TEST
    )
    if not selected:
        raise ValueError("independent-test manifest contains no included samples")
    return resolved, {record.sample_id: record for record in selected}


def _validate_manifest_files(
    manifest_path: Path,
    records: Mapping[str, SplitManifestRecord],
) -> None:
    root = manifest_path.parent
    for record in records.values():
        if record.mask_path is None or record.mask_sha256 is None:
            raise ValueError(f"manifest GT identity is missing for sample {record.sample_id}")
        image_path = (root / record.image_path).resolve(strict=False)
        truth_path = (root / record.mask_path).resolve(strict=False)
        if not image_path.is_file():
            raise ValueError(f"manifest image is missing for sample {record.sample_id}")
        if not truth_path.is_file():
            raise ValueError(f"manifest GT is missing for sample {record.sample_id}")
        if _sha256(image_path) != record.image_sha256:
            raise ValueError(f"manifest image_sha256 mismatch for sample {record.sample_id}")
        if _sha256(truth_path) != record.mask_sha256:
            raise ValueError(f"manifest gt_sha256 mismatch for sample {record.sample_id}")


def _mapping(value: Any, *, field: str, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object in {path}")
    return value


def _metadata_for_prediction(prediction_path: Path) -> Path:
    # Formal Analysis layout is artifacts/job_*/images/img_*/runs/run_*/pred_mask.png.
    metadata_path = prediction_path.parents[2] / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"pred_mask.png has no sibling image metadata.json: {prediction_path}")
    return metadata_path


def _match_sample(
    metadata: Mapping[str, Any],
    metadata_path: Path,
    manifest_records: Mapping[str, SplitManifestRecord],
) -> str:
    filename = metadata.get("filename")
    sample_id = metadata.get("sample_id")
    if not isinstance(filename, str) or not isinstance(sample_id, str):
        raise ValueError(f"image metadata is missing filename/sample_id: {metadata_path}")
    record = manifest_records.get(sample_id)
    if record is None:
        raise ValueError(f"image metadata sample_id is absent from the manifest: {metadata_path}")
    if filename != Path(record.image_path).name:
        raise ValueError(f"image metadata filename differs from the manifest: {sample_id}")
    return sample_id


def locate_sample_inputs(
    parameters: EvaluationParameters,
    manifest_path: Path,
    manifest_records: Mapping[str, SplitManifestRecord],
) -> dict[str, SampleInputs]:
    located: dict[str, SampleInputs] = {}
    all_predictions = {
        path.resolve() for path in parameters.analysis_output_root.rglob("pred_mask.png")
    }
    predictions = sorted(parameters.analysis_output_root.glob(_FORMAL_PREDICTION_GLOB))
    formal_predictions = {path.resolve() for path in predictions}
    if all_predictions != formal_predictions:
        unexpected = sorted(str(path) for path in all_predictions - formal_predictions)
        raise ValueError(
            "analysis-output-root contains pred_mask.png outside the formal Analysis layout: "
            f"{unexpected}"
        )
    if not predictions:
        raise ValueError("analysis-output-root contains no formal Analysis pred_mask.png")
    for prediction_path in predictions:
        metadata_path = _metadata_for_prediction(prediction_path)
        metadata = _load_json(metadata_path)
        sample_id = _match_sample(metadata, metadata_path, manifest_records)
        expected_filename = Path(manifest_records[sample_id].image_path).name
        if metadata.get("filename") != expected_filename or metadata.get("sample_id") != sample_id:
            raise ValueError(f"metadata does not exactly identify {expected_filename}")
        if metadata.get("width") != IMAGE_WIDTH or metadata.get("height") != IMAGE_HEIGHT:
            raise ValueError(f"original image dimensions are not 2048x1536: {sample_id}")
        image_sha256 = metadata.get("sha256")
        if not isinstance(image_sha256, str) or _SHA256_RE.fullmatch(image_sha256) is None:
            raise ValueError(f"image metadata has no valid sha256: {sample_id}")
        expected_image_id = prediction_path.parents[2].name
        expected_job_id = prediction_path.parents[4].name
        if (
            metadata.get("image_id") != expected_image_id
            or metadata.get("job_id") != expected_job_id
        ):
            raise ValueError(f"metadata identity does not match formal Analysis path: {sample_id}")
        if sample_id in located:
            raise ValueError(f"multiple pred_mask.png files found for sample {sample_id}")
        run_config_path = prediction_path.parent / "run_config.json"
        if not run_config_path.is_file():
            raise ValueError(f"run_config.json is missing for sample {sample_id}")
        execution_provenance_path = prediction_path.parent / "execution_provenance.json"
        if not execution_provenance_path.is_file():
            raise ValueError(f"execution_provenance.json is missing for sample {sample_id}")
        truth_filename = f"{sample_id}_mask.tif"
        truth_path = parameters.mask_dir / truth_filename
        if not truth_path.is_file():
            raise ValueError(f"human mask is missing for sample {sample_id}: {truth_filename}")
        manifest_record = manifest_records[sample_id]
        manifest_image_path = (manifest_path.parent / manifest_record.image_path).resolve()
        assert manifest_record.mask_path is not None
        manifest_truth_path = (manifest_path.parent / manifest_record.mask_path).resolve()
        if manifest_truth_path != truth_path.resolve():
            raise ValueError(f"manifest GT path differs from mask-dir for sample {sample_id}")
        if image_sha256 != manifest_record.image_sha256:
            raise ValueError(f"image metadata SHA differs from manifest for sample {sample_id}")
        located[sample_id] = SampleInputs(
            sample_id=sample_id,
            filename=expected_filename,
            prediction_path=prediction_path.resolve(),
            truth_path=truth_path.resolve(),
            run_config_path=run_config_path.resolve(),
            execution_provenance_path=execution_provenance_path.resolve(),
            metadata_path=metadata_path.resolve(),
            image_path=manifest_image_path,
            manifest_record=manifest_record,
        )
    expected = set(manifest_records)
    missing = sorted(expected - set(located))
    if missing:
        raise ValueError(f"pred_mask.png is missing for manifest samples: {missing}")
    return located


def _require_number(
    value: Any,
    expected: float,
    *,
    field: str,
    path: Path,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} is not numeric in {path}")
    if not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{field} is not frozen at {expected} in {path}")


def _require_sha256(value: Any, *, field: str, path: Path, expected: str | None = None) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} is not a lowercase SHA-256 in {path}")
    if expected is not None and value != expected:
        raise ValueError(f"{field} does not match the frozen Agglomerated asset in {path}")
    return value


def _validate_execution_build(value: Any, *, field: str, path: Path) -> Mapping[str, Any]:
    build = _mapping(value, field=field, path=path)
    for name in ("application_version", "python_version"):
        observed = build.get(name)
        if not isinstance(observed, str) or not observed.strip():
            raise ValueError(f"{field}.{name} is missing in {path}")
    for name in (
        "dependency_contract_sha256",
        "installed_dependencies_sha256",
        "application_source_sha256",
    ):
        _require_sha256(build.get(name), field=f"{field}.{name}", path=path)
    return build


def _validate_artifact_reference(
    value: Any,
    *,
    field: str,
    path: Path,
    expected: str,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"model_bundle.{field} is missing in {path}")
    reference = PurePosixPath(value)
    if reference.is_absolute() or ".." in reference.parts or str(reference) != value:
        raise ValueError(f"model_bundle.{field} is not a normalized relative path in {path}")
    if value != expected:
        raise ValueError(f"model_bundle.{field} does not identify the frozen asset in {path}")
    return value


def _validate_model_bundle(value: Any, *, path: Path) -> Mapping[str, Any]:
    bundle = _mapping(value, field="model_bundle", path=path)
    if bundle.get("schema_version") != 1:
        raise ValueError(f"model_bundle schema_version must be 1 in {path}")
    bundle_id = _require_sha256(bundle.get("bundle_id"), field="model_bundle.bundle_id", path=path)
    _require_sha256(
        bundle.get("adapter_sha256"),
        field="model_bundle.adapter_sha256",
        path=path,
        expected=ADAPTER_SHA256,
    )
    _validate_artifact_reference(
        bundle.get("manifest_ref"),
        field="manifest_ref",
        path=path,
        expected=f"bundles/{bundle_id}/manifest.json",
    )
    for field, digest, filename in (
        ("weight_ref", TORCHSCRIPT_SHA256, "weights.pt"),
        ("config_ref", CONFIG_SHA256, "config.yaml"),
        ("model_card_ref", MODEL_CARD_SHA256, "model-card.md"),
        ("adapter_ref", ADAPTER_SHA256, "adapter.py"),
    ):
        _validate_artifact_reference(
            bundle.get(field),
            field=field,
            path=path,
            expected=f"{digest}/{filename}",
        )
    return bundle


def validate_run_config(path: Path, *, image_sha256: str) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("contract_schema_version") != 3:
        raise ValueError(f"run configuration is not schema-v3 evidence in {path}")
    if payload.get("provenance_status") != "complete":
        raise ValueError(f"run configuration provenance is not complete in {path}")
    if payload.get("provenance_warnings") != []:
        raise ValueError(f"run configuration has provenance warnings in {path}")
    if payload.get("model_id") != MODEL_ID:
        raise ValueError(f"unexpected model_id in {path}")
    if payload.get("model_version") != MODEL_VERSION:
        raise ValueError(f"unexpected model_version in {path}")
    if payload.get("adapter_path") != ADAPTER_PATH:
        raise ValueError(f"unexpected adapter_path in {path}")
    for field, expected in (
        ("weight_sha256", TORCHSCRIPT_SHA256),
        ("config_sha256", CONFIG_SHA256),
        ("model_card_sha256", MODEL_CARD_SHA256),
        ("adapter_sha256", ADAPTER_SHA256),
    ):
        _require_sha256(payload.get(field), field=field, path=path, expected=expected)
    _require_sha256(payload.get("image_sha256"), field="image_sha256", path=path)
    if payload.get("image_sha256") != image_sha256:
        raise ValueError(f"run configuration image_sha256 differs from image metadata in {path}")
    if payload.get("roi_mode") != "full_image":
        raise ValueError(f"roi_mode must be full_image in {path}")
    if payload.get("review_source") != "model_inference":
        raise ValueError(f"review_source must be model_inference in {path}")
    _validate_model_bundle(payload.get("model_bundle"), path=path)
    _validate_execution_build(payload.get("execution_build"), field="execution_build", path=path)
    inference = _mapping(payload.get("inference"), field="inference", path=path)
    _require_number(inference.get("threshold"), THRESHOLD, field="threshold", path=path)
    if inference.get("threshold_comparison") != "gte":
        raise ValueError(f"threshold_comparison is not frozen at gte in {path}")
    if inference.get("min_area_px") != MIN_AREA_PX:
        raise ValueError(f"min_area_px is not frozen at {MIN_AREA_PX} in {path}")
    if inference.get("watershed_enabled") is not False:
        raise ValueError(f"watershed_enabled must be false in {path}")
    if inference.get("exclude_border") is not True:
        raise ValueError(f"exclude_border must be true in {path}")
    if inference.get("device") != "cpu" or inference.get("seed") != SEED:
        raise ValueError(f"inference device/seed must be cpu/{SEED} in {path}")
    _require_number(
        payload.get("scale_nm_per_pixel"),
        SCALE_NM_PER_PIXEL,
        field="scale_nm_per_pixel",
        path=path,
    )
    postprocess = _mapping(
        payload.get("resolved_postprocess"), field="resolved_postprocess", path=path
    )
    if postprocess.get("min_area_px") != MIN_AREA_PX:
        raise ValueError(f"resolved min_area_px differs in {path}")
    if postprocess.get("fill_holes") is not True:
        raise ValueError(f"fill_holes must be true in {path}")
    if postprocess.get("watershed_enabled") is not False:
        raise ValueError(f"resolved watershed_enabled must be false in {path}")
    if postprocess.get("exclude_border") is not True:
        raise ValueError(f"resolved exclude_border must be true in {path}")
    if postprocess.get("connectivity") != 2:
        raise ValueError(f"resolved connectivity must be 2 in {path}")
    morphometry = _mapping(
        payload.get("resolved_morphometry"), field="resolved_morphometry", path=path
    )
    if morphometry.get("perimeter_neighborhood") != PERIMETER_NEIGHBORHOOD:
        raise ValueError(f"perimeter_neighborhood must be 8 in {path}")
    analysis_roi = _mapping(payload.get("analysis_roi"), field="analysis_roi", path=path)
    invalid_rects = analysis_roi.get("invalid_rects")
    if not isinstance(invalid_rects, list):
        raise ValueError(f"analysis_roi.invalid_rects must be a list in {path}")
    expected_rect = {"x1": 0, "y1": VALID_HEIGHT, "x2": IMAGE_WIDTH, "y2": IMAGE_HEIGHT}
    matching_rects = [
        rect
        for rect in invalid_rects
        if isinstance(rect, Mapping)
        and all(rect.get(key) == value for key, value in expected_rect.items())
    ]
    if len(matching_rects) != 1:
        raise ValueError(f"exact bottom invalid rectangle y=1406..1536 is missing in {path}")
    return payload


def validate_execution_provenance(
    path: Path,
    *,
    run_config: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("contract_schema_version") != 1:
        raise ValueError(f"execution provenance is not schema-v1 evidence in {path}")
    if payload.get("build_identity_matches_contract") is not True:
        raise ValueError(f"execution build does not match the frozen run contract in {path}")
    executor_build = _validate_execution_build(
        payload.get("executor_build"), field="executor_build", path=path
    )
    if executor_build != run_config.get("execution_build"):
        raise ValueError(f"executor_build differs from the frozen run contract in {path}")
    if payload.get("requested_device") != "cpu" or payload.get("actual_device") != "cpu":
        raise ValueError(f"execution device must be cpu in {path}")
    if payload.get("seed") != SEED:
        raise ValueError(f"execution seed must be {SEED} in {path}")
    for field in (
        "python_random_seeded",
        "numpy_random_seeded",
        "torch_deterministic_algorithms",
        "global_inference_serialized",
    ):
        if payload.get(field) is not True:
            raise ValueError(f"{field} must be true in {path}")
    backend = payload.get("backend")
    if not isinstance(backend, str) or not backend.endswith(".UNetAdapter"):
        raise ValueError(f"execution backend is not UNetAdapter in {path}")
    bundle = _mapping(run_config.get("model_bundle"), field="model_bundle", path=path)
    if payload.get("model_bundle_id") != bundle.get("bundle_id"):
        raise ValueError(f"execution model_bundle_id differs from the run contract in {path}")
    _require_sha256(
        payload.get("adapter_sha256"),
        field="adapter_sha256",
        path=path,
        expected=ADAPTER_SHA256,
    )
    if payload.get("warnings") != []:
        raise ValueError(f"execution provenance has warnings in {path}")
    executed_at = payload.get("executed_at")
    if not isinstance(executed_at, str):
        raise ValueError(f"executed_at is missing in {path}")
    try:
        parsed = datetime.fromisoformat(executed_at)
    except ValueError as error:
        raise ValueError(f"executed_at is not an ISO-8601 datetime in {path}") from error
    if parsed.tzinfo is None:
        raise ValueError(f"executed_at must include a timezone in {path}")
    return payload


def _load_foreground(
    path: Path,
    *,
    sample_id: str,
    kind: str,
    require_zero_bottom: bool = False,
) -> np.ndarray:
    try:
        with Image.open(path) as image:
            if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                raise ValueError(f"{kind} dimensions are not 2048x1536: {sample_id}")
            pixels = np.asarray(image)
    except OSError as error:
        raise ValueError(f"cannot read {kind} image for {sample_id}: {path}") from error
    if pixels.ndim == 2:
        foreground = pixels != 0
    elif pixels.ndim == 3:
        foreground = np.any(pixels != 0, axis=2)
    else:
        raise ValueError(f"unsupported {kind} dimensions for {sample_id}: {pixels.shape}")
    if require_zero_bottom and np.any(foreground[VALID_HEIGHT:, :]):
        raise ValueError(f"prediction bottom {BOTTOM_CROP_PX} px is not zero: {sample_id}")
    return np.asarray(foreground[:VALID_HEIGHT, :], dtype=bool)


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 1.0


def compute_metrics(prediction: np.ndarray, truth: np.ndarray) -> ConfusionMetrics:
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth shapes differ")
    prediction = np.asarray(prediction, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    tn = int(np.count_nonzero(~prediction & ~truth))
    return ConfusionMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        dice=_ratio(2 * tp, 2 * tp + fp + fn),
        iou=_ratio(tp, tp + fp + fn),
        precision=_ratio(tp, tp + fp),
        recall=_ratio(tp, tp + fn),
        f1=_ratio(2 * tp, 2 * tp + fp + fn),
    )


def _macro(metrics: Sequence[ConfusionMetrics]) -> dict[str, float]:
    return {
        name: float(sum(getattr(item, name) for item in metrics) / len(metrics))
        for name in ("dice", "iou", "precision", "recall", "f1")
    }


def _micro(metrics: Sequence[ConfusionMetrics]) -> ConfusionMetrics:
    tp = sum(item.tp for item in metrics)
    fp = sum(item.fp for item in metrics)
    fn = sum(item.fn for item in metrics)
    tn = sum(item.tn for item in metrics)
    return ConfusionMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        dice=_ratio(2 * tp, 2 * tp + fp + fn),
        iou=_ratio(tp, tp + fp + fn),
        precision=_ratio(tp, tp + fp),
        recall=_ratio(tp, tp + fn),
        f1=_ratio(2 * tp, 2 * tp + fp + fn),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_agglomerated_tolerance_policy(
    path: Path | None,
    *,
    manifest_sha256: str,
    checkpoint_sha256: str,
) -> AgglomeratedTolerancePolicy | None:
    if path is None:
        return None
    payload, resolved, digest = _load_large_tolerance_policy(path)
    if payload.get("model_id") != MODEL_ID:
        raise ValueError(f"tolerance-policy model_id must be {MODEL_ID}")
    for field, expected in (
        ("checkpoint_sha256", checkpoint_sha256),
        ("torchscript_sha256", TORCHSCRIPT_SHA256),
        ("config_sha256", CONFIG_SHA256),
        ("model_card_sha256", MODEL_CARD_SHA256),
        ("adapter_sha256", ADAPTER_SHA256),
        ("independent_test_manifest_sha256", manifest_sha256),
    ):
        _require_sha256(
            payload.get(field),
            field=f"tolerance-policy {field}",
            path=resolved,
            expected=expected,
        )
    frozen = _mapping(
        payload.get("frozen_scientific_parameters"),
        field="frozen_scientific_parameters",
        path=resolved,
    )
    expected_frozen = {
        "threshold": THRESHOLD,
        "threshold_comparison": "gte",
        "min_area_px": MIN_AREA_PX,
        "bottom_crop_px": BOTTOM_CROP_PX,
    }
    for field, frozen_expected in expected_frozen.items():
        observed = frozen.get(field)
        if isinstance(frozen_expected, float):
            _require_number(
                observed,
                frozen_expected,
                field=f"tolerance-policy frozen_scientific_parameters.{field}",
                path=resolved,
            )
        elif observed != frozen_expected:
            raise ValueError(
                f"tolerance-policy frozen_scientific_parameters.{field} must be {frozen_expected}"
            )
    approval = _mapping(payload.get("approval"), field="approval", path=resolved)
    approval_status = approval.get("status")
    if approval_status not in {"APPROVED", "DRAFT", "REJECTED"}:
        raise ValueError("tolerance-policy approval.status must be APPROVED, DRAFT, or REJECTED")
    frozen_before_test = approval.get("frozen_before_independent_test")
    if not isinstance(frozen_before_test, bool):
        raise ValueError("tolerance-policy approval.frozen_before_independent_test must be boolean")
    if approval_status == "APPROVED" and frozen_before_test is not True:
        raise ValueError("approved tolerance-policy must be frozen before the independent test")
    for field in ("approved_by", "approved_at", "rationale"):
        value = approval.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"tolerance-policy approval.{field} must be non-empty")
    approved_at = str(approval["approved_at"])
    try:
        approved_timestamp = datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("tolerance-policy approval.approved_at must be ISO-8601") from error
    if approved_timestamp.tzinfo is None or approved_timestamp.utcoffset() is None:
        raise ValueError("tolerance-policy approval.approved_at must include a timezone")
    matching = _mapping(
        payload.get("instance_matching"),
        field="instance_matching",
        path=resolved,
    )
    tolerances = _mapping(
        payload.get("per_image_tolerances"),
        field="per_image_tolerances",
        path=resolved,
    )
    raw_contracts = payload.get("metric_contracts")
    if not isinstance(raw_contracts, list):
        raise ValueError("tolerance-policy metric_contracts must be a list")
    expected_rules = {
        metric: (tolerance_field, comparison)
        for metric, tolerance_field, comparison in _AGGLOMERATED_TOLERANCE_FIELDS
    }
    metric_contracts: list[Mapping[str, Any]] = []
    seen_metrics: set[str] = set()
    for index, raw_contract in enumerate(raw_contracts):
        contract = _mapping(
            raw_contract,
            field=f"metric_contracts[{index}]",
            path=resolved,
        )
        metric = contract.get("metric_name")
        if not isinstance(metric, str) or metric not in expected_rules:
            raise ValueError(f"metric_contracts[{index}].metric_name is unsupported")
        if metric in seen_metrics:
            raise ValueError(f"metric_contracts contains duplicate metric: {metric}")
        seen_metrics.add(metric)
        tolerance_field, comparison = expected_rules[metric]
        if contract.get("aggregation_scope") != "per-sample":
            raise ValueError(f"metric_contracts[{index}].aggregation_scope must be per-sample")
        if contract.get("operator") != comparison:
            raise ValueError(f"metric_contracts[{index}].operator must be {comparison}")
        if contract.get("tolerance_field") != tolerance_field:
            raise ValueError(f"metric_contracts[{index}].tolerance_field must be {tolerance_field}")
        tolerance = _require_finite_number(
            contract.get("tolerance"),
            field=f"metric_contracts[{index}].tolerance",
            minimum=0.0,
        )
        if not math.isclose(
            tolerance,
            float(tolerances[tolerance_field]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"metric_contracts[{index}].tolerance differs from per_image_tolerances"
            )
        if contract.get("requirement") not in {"required", "informational"}:
            raise ValueError(
                f"metric_contracts[{index}].requirement must be required or informational"
            )
        expected_applicability = (
            "always"
            if metric.startswith("instance_") or metric.startswith("count_")
            else "ground_truth_and_prediction_instances_present"
        )
        if contract.get("applicability") != expected_applicability:
            raise ValueError(
                f"metric_contracts[{index}].applicability must be {expected_applicability}"
            )
        if contract.get("undefined_rule") != "fail_if_required_otherwise_not_evaluated":
            raise ValueError(
                "metric contract undefined_rule must be fail_if_required_otherwise_not_evaluated"
            )
        if contract.get("zero_denominator_rule") != _ZERO_DENOMINATOR_RULES[metric]:
            raise ValueError(
                f"metric_contracts[{index}].zero_denominator_rule differs from metric semantics"
            )
        reason_codes = _mapping(
            contract.get("reason_codes"),
            field=f"metric_contracts[{index}].reason_codes",
            path=resolved,
        )
        expected_reason_codes = {
            "tolerance_met": "TOLERANCE_MET",
            "tolerance_not_met": "TOLERANCE_NOT_MET",
            "not_evaluable": "METRIC_NOT_EVALUABLE",
            "not_applicable": "METRIC_NOT_APPLICABLE",
        }
        if dict(reason_codes) != expected_reason_codes:
            raise ValueError(
                f"metric_contracts[{index}].reason_codes differs from the public contract"
            )
        metric_contracts.append(dict(contract))
    if seen_metrics != set(expected_rules):
        raise ValueError("metric_contracts must define every public tolerance metric exactly once")
    return AgglomeratedTolerancePolicy(
        content=payload,
        source_path=resolved,
        filename=resolved.name,
        sha256=digest,
        instance_iou_threshold=float(matching["mask_iou_threshold"]),
        approved=approval_status == "APPROVED" and frozen_before_test,
        metric_contracts=tuple(metric_contracts),
    )


def _resolved_instance_iou_threshold(
    *,
    policy: AgglomeratedTolerancePolicy | None,
    explicit_threshold: float | None,
) -> float | None:
    if explicit_threshold is not None and not 0.0 < explicit_threshold <= 1.0:
        raise ValueError("instance_iou_threshold must be in (0, 1]")
    if policy is None:
        return explicit_threshold
    if explicit_threshold is not None and not math.isclose(
        explicit_threshold,
        policy.instance_iou_threshold,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("explicit instance IoU threshold differs from tolerance policy")
    return policy.instance_iou_threshold


def _metric_assessments(
    scientific_metrics: Mapping[str, Any],
    statistical_errors: Mapping[str, Mapping[str, Any]],
    policy: AgglomeratedTolerancePolicy | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    values = _scientific_metric_values(scientific_metrics)
    coverage_relative_error = statistical_errors["coverage_ratio"]["relative_error"]
    values["coverage_relative_error"] = (
        abs(float(coverage_relative_error)) if coverage_relative_error is not None else None
    )
    assessments: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    matching = _mapping(
        scientific_metrics.get("instance_matching"),
        field="instance_matching",
        path=Path("scientific_metrics"),
    )
    contracts = (
        policy.metric_contracts
        if policy is not None
        else tuple(
            {
                "metric_name": metric,
                "aggregation_scope": "per-sample",
                "operator": comparison,
                "tolerance_field": tolerance_field,
                "tolerance": None,
                "requirement": "required",
                "applicability": (
                    "always"
                    if metric.startswith("instance_") or metric.startswith("count_")
                    else "ground_truth_and_prediction_instances_present"
                ),
                "undefined_rule": "fail_if_required_otherwise_not_evaluated",
                "zero_denominator_rule": "not_bound_without_policy",
            }
            for metric, tolerance_field, comparison in _AGGLOMERATED_TOLERANCE_FIELDS
        )
    )
    for contract in contracts:
        metric = str(contract["metric_name"])
        comparison = str(contract["operator"])
        observed = values[metric]
        operator = ">=" if comparison == "gte" else "<="
        requirement = str(contract["requirement"])
        assessment_base = {
            "metric": metric,
            "aggregation_scope": contract["aggregation_scope"],
            "operator": operator,
            "observed": observed,
            "tolerance": contract["tolerance"],
            "requirement": requirement,
            "applicability": contract["applicability"],
            "undefined_rule": contract["undefined_rule"],
            "zero_denominator_rule": contract["zero_denominator_rule"],
        }
        if policy is None or not policy.approved:
            assessments.append(
                {
                    **assessment_base,
                    "status": "NOT_EVALUATED",
                    "reason_code": (
                        "TOLERANCE_POLICY_NOT_PROVIDED"
                        if policy is None
                        else "TOLERANCE_POLICY_NOT_APPROVED"
                    ),
                }
            )
            continue
        tolerance = _require_finite_number(
            contract["tolerance"],
            field=f"metric_contracts.{metric}.tolerance",
            minimum=0.0,
        )
        applicable = contract["applicability"] == "always" or (
            int(matching["ground_truth_count"]) > 0 and int(matching["prediction_count"]) > 0
        )
        if not applicable or observed is None:
            assessment = {
                **assessment_base,
                "status": "FAIL" if requirement == "required" else "NOT_EVALUATED",
                "reason_code": (
                    "METRIC_NOT_EVALUABLE"
                    if requirement == "required" or applicable
                    else "METRIC_NOT_APPLICABLE"
                ),
            }
        else:
            passed = observed >= tolerance if comparison == "gte" else observed <= tolerance
            assessment = {
                **assessment_base,
                "status": (
                    "PASS" if passed else ("FAIL" if requirement == "required" else "INFORMATIONAL")
                ),
                "reason_code": "TOLERANCE_MET" if passed else "TOLERANCE_NOT_MET",
            }
        assessments.append(assessment)
        if assessment["status"] == "FAIL":
            failures.append(assessment)
    return assessments, failures


def _union(instances: Sequence[NormalizedInstance], shape: tuple[int, ...]) -> np.ndarray:
    result = np.zeros(shape, dtype=np.bool_)
    for instance in instances:
        result |= instance.mask
    return result


def _scientific_statistics(mask: np.ndarray, *, sample_id: str, kind: str) -> dict[str, Any]:
    roi = np.ones(mask.shape, dtype=np.bool_)
    normalized = normalize_semantic_mask_detailed(
        np.asarray(mask, dtype=np.bool_),
        roi_mask=roi,
        profile=PostprocessProfile(
            profile_id="agglomerated-independent-evaluation-v1",
            min_area_px=0,
            fill_holes=True,
            watershed_enabled=False,
            exclude_border=True,
            connectivity=2,
        ),
    )
    summary = measure(
        run_id=f"independent-{sample_id}-{kind}",
        instances=normalized.instances,
        roi_mask=roi,
        scale_nm_per_pixel=SCALE_NM_PER_PIXEL,
        config=MorphometryConfig(perimeter_neighborhood=PERIMETER_NEIGHBORHOOD),
    ).image_summary
    return {
        "agglomerate_count": summary.particle_count,
        "mean_equivalent_diameter_nm": summary.mean_equivalent_diameter_nm,
        "number_density_um2": summary.number_density_um2,
        "perimeter_density_um": summary.perimeter_density_um,
        "coverage_ratio": summary.coverage_ratio,
        "excluded_border_count": normalized.excluded_border_count,
        "normalized_foreground_px": int(np.count_nonzero(_union(normalized.instances, mask.shape))),
    }


def _statistical_errors(
    prediction: Mapping[str, Any], truth: Mapping[str, Any]
) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for field in STATISTIC_FIELDS:
        predicted = prediction.get(field)
        expected = truth.get(field)
        if not isinstance(predicted, int | float) or not isinstance(expected, int | float):
            result[field] = {
                "prediction": predicted,
                "ground_truth": expected,
                "signed_error": None,
                "absolute_error": None,
                "relative_error": None,
                "absolute_percentage_error": None,
            }
            continue
        signed = float(predicted) - float(expected)
        relative = signed / float(expected) if float(expected) != 0 else None
        result[field] = {
            "prediction": predicted,
            "ground_truth": expected,
            "signed_error": signed,
            "absolute_error": abs(signed),
            "relative_error": relative,
            "absolute_percentage_error": abs(relative) * 100 if relative is not None else None,
        }
    return result


def _macro_statistical_errors(per_image: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    macro: dict[str, Any] = {}
    for field in STATISTIC_FIELDS:
        field_errors = [item["statistical_errors"][field] for item in per_image]
        absolute_percentages = [
            float(item["absolute_percentage_error"])
            for item in field_errors
            if item["absolute_percentage_error"] is not None
        ]
        signed_percentages = [
            float(item["relative_error"]) * 100
            for item in field_errors
            if item["relative_error"] is not None
        ]
        macro[field] = {
            "mean_absolute_percentage_error": (
                float(np.mean(absolute_percentages)) if absolute_percentages else None
            ),
            "mean_signed_percentage_error": (
                float(np.mean(signed_percentages)) if signed_percentages else None
            ),
            "available_image_count": len(absolute_percentages),
        }
    return macro


def _write_review(
    path: Path,
    *,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> None:
    truth_panel = np.repeat((truth.astype(np.uint8) * 255)[..., None], 3, axis=2)
    prediction_panel = np.repeat((prediction.astype(np.uint8) * 255)[..., None], 3, axis=2)
    error_panel = np.zeros((*truth.shape, 3), dtype=np.uint8)
    error_panel[prediction & truth] = (0, 180, 0)
    error_panel[prediction & ~truth] = (255, 0, 0)
    error_panel[~prediction & truth] = (0, 80, 255)
    review = np.concatenate((truth_panel, prediction_panel, error_panel), axis=1)
    Image.fromarray(review, mode="RGB").save(path)


def _write_csv(
    path: Path,
    per_image: Sequence[dict[str, Any]],
    macro: Mapping[str, float],
    micro: ConfusionMetrics,
) -> None:
    columns = (
        "scope",
        "tp",
        "fp",
        "fn",
        "tn",
        "dice",
        "iou",
        "precision",
        "recall",
        "f1",
        "instance_precision",
        "instance_recall",
        "instance_f1",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for item in per_image:
            instance_matching = item.get("scientific_metrics", {}).get("instance_matching", {})
            writer.writerow(
                {
                    "scope": item["sample_id"],
                    **item["metrics"],
                    "instance_precision": instance_matching.get("precision"),
                    "instance_recall": instance_matching.get("recall"),
                    "instance_f1": instance_matching.get("f1"),
                }
            )
        writer.writerow({"scope": "macro_average", **macro})
        writer.writerow({"scope": "micro_average", **asdict(micro)})


def _write_statistics_csv(path: Path, per_image: Sequence[Mapping[str, Any]]) -> None:
    columns = (
        "sample_id",
        "statistic",
        "prediction",
        "ground_truth",
        "signed_error",
        "absolute_error",
        "relative_error",
        "absolute_percentage_error",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for item in per_image:
            errors = item["statistical_errors"]
            for field in STATISTIC_FIELDS:
                writer.writerow(
                    {"sample_id": item["sample_id"], "statistic": field, **errors[field]}
                )


def _write_failure_cases_csv(path: Path, failures: Sequence[Mapping[str, Any]]) -> None:
    columns = (
        "sample_id",
        "metric",
        "operator",
        "observed",
        "tolerance",
        "status",
        "reason_code",
        "aggregation_scope",
        "requirement",
        "applicability",
        "undefined_rule",
        "zero_denominator_rule",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(failures)


def _analysis_reference(path: Path, analysis_root: Path) -> str:
    return path.relative_to(analysis_root).as_posix()


def _evaluator_identity(run_configs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    git_commits = {
        str(
            _mapping(item["execution_build"], field="execution_build", path=Path("run_config"))[
                "git_commit"
            ]
        )
        for item in run_configs
    }
    if len(git_commits) != 1 or not next(iter(git_commits)).strip():
        raise ValueError("run configurations do not bind one evaluator git commit")
    return {
        "source_path": "scripts/models/evaluate_unet_agglomerated_independent_test.py",
        "source_sha256": _sha256(Path(__file__).resolve()),
        "git_commit": next(iter(git_commits)),
    }


def _write_evidence_manifest(
    path: Path,
    *,
    result: Mapping[str, Any],
    manifest_path: Path,
    manifest_records: Mapping[str, SplitManifestRecord],
    located: Mapping[str, SampleInputs],
    analysis_root: Path,
    policy: AgglomeratedTolerancePolicy | None,
    evaluator_identity: Mapping[str, Any],
) -> dict[str, Any]:
    inputs: list[dict[str, Any]] = [
        {
            "role": "independent_test_manifest",
            **_file_identity(manifest_path, relative_path=manifest_path.name),
        }
    ]
    for sample_id in sorted(located):
        sample = located[sample_id]
        record = manifest_records[sample_id]
        assert record.mask_path is not None
        entries = (
            ("image", sample.image_path, record.image_path),
            ("ground_truth", sample.truth_path, record.mask_path),
            (
                "prediction",
                sample.prediction_path,
                _analysis_reference(sample.prediction_path, analysis_root),
            ),
            (
                "run_config",
                sample.run_config_path,
                _analysis_reference(sample.run_config_path, analysis_root),
            ),
            (
                "execution_provenance",
                sample.execution_provenance_path,
                _analysis_reference(
                    sample.execution_provenance_path,
                    analysis_root,
                ),
            ),
            (
                "image_metadata",
                sample.metadata_path,
                _analysis_reference(sample.metadata_path, analysis_root),
            ),
        )
        for role, input_path, relative_path in entries:
            inputs.append(
                {
                    "role": role,
                    "sample_id": sample_id,
                    **_file_identity(input_path, relative_path=relative_path),
                }
            )
    if policy is not None:
        inputs.append(
            {
                "role": "tolerance_policy",
                **_file_identity(policy.source_path, relative_path=policy.filename),
            }
        )
    output_root = path.parent
    output_paths = [
        output_root / "metrics.json",
        output_root / "metrics.csv",
        output_root / "statistics.csv",
        output_root / "failure-cases.csv",
        *sorted(output_root.glob("*_gt_pred_error.png"), key=lambda item: item.name),
    ]
    outputs = [
        {
            "role": "review_image" if item.suffix == ".png" else item.name,
            **_file_identity(item, relative_path=item.name),
        }
        for item in output_paths
    ]
    payload = {
        "schema_version": "1",
        "evaluation": "agglomerated-independent-test",
        "input_manifest": {
            "filename": manifest_path.name,
            "sha256": _sha256(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
            "split": "independent-test",
            "sample_count": len(manifest_records),
            "sample_ids": sorted(manifest_records),
        },
        "evaluation_identity": {
            "model_id": MODEL_ID,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_evidence_semantics": (
                "declared SHA validated against the frozen identity and policy when provided; "
                "the checkpoint file is not accessible to or re-hashed by this evaluator"
            ),
            "torchscript_sha256": TORCHSCRIPT_SHA256,
            "config_sha256": CONFIG_SHA256,
            "model_card_sha256": MODEL_CARD_SHA256,
            "adapter_sha256": ADAPTER_SHA256,
            **evaluator_identity,
            "threshold": THRESHOLD,
            "threshold_comparison": "gte",
            "min_area_px": MIN_AREA_PX,
            "bottom_crop_px": BOTTOM_CROP_PX,
            "instance_iou_threshold": result["tolerance_policy"]["instance_iou_threshold"],
            "instance_iou_threshold_source": (
                "tolerance_policy"
                if policy is not None
                else (
                    "explicit_cli"
                    if result["tolerance_policy"]["instance_iou_threshold"] is not None
                    else None
                )
            ),
            "tolerance_policy_sha256": policy.sha256 if policy is not None else None,
        },
        "inputs": sorted(
            inputs,
            key=lambda item: (
                str(item["role"]),
                str(item.get("sample_id", "")),
                str(item["path"]),
            ),
        ),
        "outputs": sorted(outputs, key=lambda item: str(item["path"])),
        "overall_status": result["overall_status"],
    }
    path.write_bytes(
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return payload


def run_evaluation(parameters: EvaluationParameters) -> dict[str, Any]:
    parameters = EvaluationParameters(
        parameters.analysis_output_root.expanduser().resolve(),
        parameters.mask_dir.expanduser().resolve(),
        _validate_output_root(parameters.output_root),
        parameters.independent_test_manifest_path,
        parameters.checkpoint_sha256,
        parameters.tolerance_policy_path,
        parameters.instance_iou_threshold,
    )
    _require_sha256(
        parameters.checkpoint_sha256,
        field="checkpoint_sha256",
        path=parameters.independent_test_manifest_path,
        expected=CHECKPOINT_SHA256,
    )
    if not parameters.analysis_output_root.is_dir() or not parameters.mask_dir.is_dir():
        raise ValueError("analysis-output-root and mask-dir must be existing directories")
    manifest_path, manifest_records = _load_independent_test_manifest(
        parameters.independent_test_manifest_path
    )
    manifest_sha256 = _sha256(manifest_path)
    policy = _load_agglomerated_tolerance_policy(
        parameters.tolerance_policy_path,
        manifest_sha256=manifest_sha256,
        checkpoint_sha256=parameters.checkpoint_sha256,
    )
    _validate_manifest_files(manifest_path, manifest_records)
    instance_iou_threshold = _resolved_instance_iou_threshold(
        policy=policy,
        explicit_threshold=parameters.instance_iou_threshold,
    )
    located = locate_sample_inputs(parameters, manifest_path, manifest_records)
    prepared: list[dict[str, Any]] = []
    validated: list[dict[str, Any]] = []
    for sample_id in sorted(manifest_records):
        inputs = located[sample_id]
        metadata = _load_json(inputs.metadata_path)
        image_sha256 = _require_sha256(
            metadata.get("sha256"), field="image metadata sha256", path=inputs.metadata_path
        )
        run_config = validate_run_config(
            inputs.run_config_path,
            image_sha256=image_sha256,
        )
        execution = validate_execution_provenance(
            inputs.execution_provenance_path,
            run_config=run_config,
        )
        validated.append({"inputs": inputs, "run_config": run_config, "execution": execution})
    evaluator_identity = _evaluator_identity([item["run_config"] for item in validated])
    for validated_sample in validated:
        inputs = validated_sample["inputs"]
        run_config = validated_sample["run_config"]
        execution = validated_sample["execution"]
        prediction = _load_foreground(
            inputs.prediction_path,
            sample_id=inputs.sample_id,
            kind="prediction",
            require_zero_bottom=True,
        )
        truth = _load_foreground(inputs.truth_path, sample_id=inputs.sample_id, kind="truth")
        pixel_metrics = compute_metrics(prediction, truth)
        prediction_statistics = _scientific_statistics(
            prediction, sample_id=inputs.sample_id, kind="prediction"
        )
        truth_statistics = _scientific_statistics(
            truth, sample_id=inputs.sample_id, kind="ground-truth"
        )
        postprocess = PostprocessProfile.model_validate(run_config["resolved_postprocess"])
        morphometry = MorphometryConfig.model_validate(run_config["resolved_morphometry"])
        prepared.append(
            {
                "inputs": inputs,
                "prediction": prediction,
                "truth": truth,
                "metrics": pixel_metrics,
                "run_config": run_config,
                "execution": execution,
                "prediction_statistics": prediction_statistics,
                "ground_truth_statistics": truth_statistics,
                "statistical_errors": _statistical_errors(prediction_statistics, truth_statistics),
                "postprocess": postprocess,
                "morphometry": morphometry,
            }
        )

    parameters.output_root.mkdir(parents=True, exist_ok=False)
    per_image: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    for prepared_sample in prepared:
        inputs = prepared_sample["inputs"]
        prediction = prepared_sample["prediction"]
        truth = prepared_sample["truth"]
        pixel_metrics = prepared_sample["metrics"]
        run_config = prepared_sample["run_config"]
        execution = prepared_sample["execution"]
        review_path = parameters.output_root / f"{inputs.sample_id}_gt_pred_error.png"
        _write_review(review_path, truth=truth, prediction=prediction)
        if instance_iou_threshold is None:
            scientific_metrics: dict[str, Any] = {
                "status": "NOT_EVALUATED",
                "reason_code": "INSTANCE_IOU_THRESHOLD_NOT_PROVIDED",
                "instance_matching": {
                    "rule": "one_to_one_maximum_cardinality_mask_iou",
                    "iou_threshold": None,
                    "prediction_min_area_px": MIN_AREA_PX,
                    "ground_truth_min_area_px": 0,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "matches": [],
                },
            }
            assessments = [
                {
                    "metric": "instance_metrics",
                    "operator": None,
                    "observed": None,
                    "tolerance": None,
                    "status": "NOT_EVALUATED",
                    "reason_code": "INSTANCE_IOU_THRESHOLD_NOT_PROVIDED",
                }
            ]
            failures: list[dict[str, Any]] = []
        else:
            scientific_metrics = compute_scientific_metrics(
                prediction,
                truth,
                profile=prepared_sample["postprocess"],
                morphometry=prepared_sample["morphometry"],
                scale_nm_per_pixel=SCALE_NM_PER_PIXEL,
                iou_threshold=instance_iou_threshold,
                sample_id=inputs.sample_id,
            )
            assessments, failures = _metric_assessments(
                scientific_metrics,
                prepared_sample["statistical_errors"],
                policy,
            )
        annotated_failures = [{"sample_id": inputs.sample_id, **failure} for failure in failures]
        all_failures.extend(annotated_failures)
        assessment_status = (
            "NOT_EVALUATED"
            if policy is None or not policy.approved
            else ("FAIL" if failures else "PASS")
        )
        per_image.append(
            {
                "sample_id": inputs.sample_id,
                "filename": inputs.filename,
                "metrics": asdict(pixel_metrics),
                "prediction_statistics": prepared_sample["prediction_statistics"],
                "ground_truth_statistics": prepared_sample["ground_truth_statistics"],
                "statistical_errors": prepared_sample["statistical_errors"],
                "scientific_metrics": scientific_metrics,
                "tolerance_assessment": {
                    "status": assessment_status,
                    "assessments": assessments,
                    "failures": failures,
                },
                "input_provenance": {
                    "prediction_path": _analysis_reference(
                        inputs.prediction_path, parameters.analysis_output_root
                    ),
                    "prediction_sha256": _sha256(inputs.prediction_path),
                    "truth_path": inputs.truth_path.name,
                    "truth_sha256": _sha256(inputs.truth_path),
                    "run_config_path": _analysis_reference(
                        inputs.run_config_path, parameters.analysis_output_root
                    ),
                    "run_config_sha256": _sha256(inputs.run_config_path),
                    "execution_provenance_path": _analysis_reference(
                        inputs.execution_provenance_path,
                        parameters.analysis_output_root,
                    ),
                    "execution_provenance_sha256": _sha256(inputs.execution_provenance_path),
                    "image_metadata_path": _analysis_reference(
                        inputs.metadata_path, parameters.analysis_output_root
                    ),
                    "image_metadata_sha256": _sha256(inputs.metadata_path),
                },
                "verified_frozen_configuration": {
                    "model_id": run_config["model_id"],
                    "model_version": run_config["model_version"],
                    "checkpoint_sha256": parameters.checkpoint_sha256,
                    "weight_sha256": run_config["weight_sha256"],
                    "config_sha256": run_config["config_sha256"],
                    "model_card_sha256": run_config["model_card_sha256"],
                    "adapter_sha256": run_config["adapter_sha256"],
                    "model_bundle_id": execution["model_bundle_id"],
                    "threshold": run_config["inference"]["threshold"],
                    "threshold_comparison": "gte",
                    "min_area_px": run_config["inference"]["min_area_px"],
                    "bottom_invalid_rect": {
                        "x1": 0,
                        "y1": VALID_HEIGHT,
                        "x2": IMAGE_WIDTH,
                        "y2": IMAGE_HEIGHT,
                    },
                },
                "review_image": review_path.name,
            }
        )
    all_pixel_metrics = [item["metrics"] for item in prepared]
    macro = _macro(all_pixel_metrics)
    micro = _micro(all_pixel_metrics)
    result = {
        "schema_version": "1",
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation": "Agglomerated U-Net frozen independent test-set ground-truth evaluation",
        "input_manifest": {
            "filename": manifest_path.name,
            "sha256": _sha256(manifest_path),
            "split": "independent-test",
            "sample_count": len(manifest_records),
        },
        "model": {
            "model_id": MODEL_ID,
            "model_version": MODEL_VERSION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_evidence_semantics": (
                "declared SHA validated against the frozen identity and policy when provided; "
                "checkpoint bytes were not accessed or re-hashed"
            ),
            "weight_sha256": TORCHSCRIPT_SHA256,
            "config_sha256": CONFIG_SHA256,
            "model_card_sha256": MODEL_CARD_SHA256,
            "adapter_sha256": ADAPTER_SHA256,
            "evaluator": evaluator_identity,
            "threshold_rule": "probability >= 0.25 (verified from prior run config)",
            "threshold": THRESHOLD,
            "threshold_comparison": "gte",
            "min_area_px": MIN_AREA_PX,
            "scale_nm_per_pixel": SCALE_NM_PER_PIXEL,
            "watershed_enabled": False,
            "fill_holes": True,
            "exclude_border": True,
            "connectivity": 2,
            "perimeter_neighborhood": PERIMETER_NEIGHBORHOOD,
            "bottom_crop_px": BOTTOM_CROP_PX,
        },
        "evaluation_region": {
            "image_size_px": [IMAGE_WIDTH, IMAGE_HEIGHT],
            "evaluated_rect_half_open": [0, 0, IMAGE_WIDTH, VALID_HEIGHT],
            "excluded_bottom_rect_half_open": [
                0,
                VALID_HEIGHT,
                IMAGE_WIDTH,
                IMAGE_HEIGHT,
            ],
            "evaluated_pixels_per_image": IMAGE_WIDTH * VALID_HEIGHT,
            "foreground_rules": {
                "prediction": "nonzero pixel",
                "ground_truth": "nonzero pixel",
            },
            "zero_denominator_convention": "metric equals 1.0 when its denominator is zero",
        },
        "metric_definitions": {
            "pixel_f1": "2 * TP / (2 * TP + FP + FN)",
            "pixel_dice": "2 * TP / (2 * TP + FP + FN)",
            "pixel_f1_equals_dice": True,
            "instance_matching": "one-to-one maximum-cardinality mask-IoU matching",
            "ground_truth_instance_min_area_px": 0,
        },
        "tolerance_policy": (
            {
                "status": "APPROVED" if policy.approved else "NOT_APPROVED",
                "filename": policy.filename,
                "sha256": policy.sha256,
                "policy_id": policy.content["policy_id"],
                "policy_version": policy.content["policy_version"],
                "instance_iou_threshold": policy.instance_iou_threshold,
                "approval_status": policy.content["approval"]["status"],
                "approved_by": policy.content["approval"]["approved_by"],
                "approved_at": policy.content["approval"]["approved_at"],
                "metric_contracts": list(policy.metric_contracts),
            }
            if policy is not None
            else {
                "status": "NOT_PROVIDED",
                "filename": None,
                "sha256": None,
                "policy_id": None,
                "policy_version": None,
                "instance_iou_threshold": instance_iou_threshold,
            }
        ),
        "per_image": per_image,
        "macro_average": macro,
        "micro_average": asdict(micro),
        "overall_status": (
            "NOT_EVALUATED"
            if policy is None or not policy.approved
            else ("FAIL" if all_failures else "PASS")
        ),
        "failures": all_failures,
        "macro_statistical_errors": _macro_statistical_errors(per_image),
        "statistical_evaluation": {
            "object_definition": "whole agglomerate",
            "prediction_policy": (
                "measure the existing frozen Analysis pred_mask; no additional min-area filtering"
            ),
            "ground_truth_policy": (
                "nonzero GT, min_area_px=0, fill holes, exclude ROI-border components"
            ),
            "relative_error_definition": "(prediction - ground_truth) / ground_truth",
            "zero_ground_truth_policy": "relative and percentage errors are null",
        },
        "review_image_legend": {
            "layout": "ground truth | prediction | error",
            "error_colors": {
                "true_positive": "green",
                "false_positive": "red",
                "false_negative": "blue",
                "true_negative": "black",
            },
        },
        "limitations": [
            (
                "These results evaluate the already frozen model and are not used "
                "to tune threshold, min_area_px, or any other parameter."
            ),
            "No training or validation images or masks were read, and no inference was performed.",
            (
                "The evaluated sample set is selected exclusively by the SHA-bound "
                "independent-test manifest."
            ),
            "The scientific object is each whole agglomerate, not internal primary particles.",
        ],
    }
    metrics_json = parameters.output_root / "metrics.json"
    metrics_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(parameters.output_root / "metrics.csv", per_image, macro, micro)
    _write_statistics_csv(parameters.output_root / "statistics.csv", per_image)
    _write_failure_cases_csv(
        parameters.output_root / "failure-cases.csv",
        all_failures,
    )
    _write_evidence_manifest(
        parameters.output_root / "evidence-manifest.json",
        result=result,
        manifest_path=manifest_path,
        manifest_records=manifest_records,
        located=located,
        analysis_root=parameters.analysis_output_root,
        policy=policy,
        evaluator_identity=evaluator_identity,
    )
    return result


def _redact_cli_paths(message: str, namespace: argparse.Namespace | None) -> str:
    if namespace is None:
        return message
    redacted = message
    for field, placeholder in (
        ("analysis_output_root", "<analysis-output-root>"),
        ("mask_dir", "<mask-dir>"),
        ("independent_test_manifest", "<independent-test-manifest>"),
        ("output_root", "<output-root>"),
        ("tolerance_policy", "<tolerance-policy>"),
    ):
        value = getattr(namespace, field, None)
        if not isinstance(value, Path):
            continue
        for representation in {
            str(value),
            str(value.expanduser().resolve(strict=False)),
        }:
            redacted = redacted.replace(representation, placeholder)
    return redacted


def main(argv: Sequence[str] | None = None) -> int:
    namespace: argparse.Namespace | None = None
    try:
        namespace = build_parser().parse_args(argv)
        parameters = _validated_parameters(namespace)
        result = run_evaluation(parameters)
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "message": _redact_cli_paths(str(error), namespace),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

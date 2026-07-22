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

MODEL_ID = "unet-agglomerated-specialized-v1"
MODEL_VERSION = "1"
ADAPTER_PATH = "app.inference.adapters.unet:UNetAdapter"
TORCHSCRIPT_SHA256 = "d36cf627dc6d1eea83b769743b657e6bb3d9a1af39ddc6a00ae95acc3b20ffc9"
CONFIG_SHA256 = "54f5113d0de4b5d2a26e48e8c231b5223c43b373ba8a496934f2b4bbd1bfc524"
MODEL_CARD_SHA256 = "6a56b5b29cb2f8c1b8d1893ca072a2dab0ad309708d9f682f9374f6ea279fef1"
ADAPTER_SHA256 = "6055db452f0a78a0352732d66ea3436f16a558cf19d1a6f022a78627136dfab6"
TEST_FILENAMES = ("YCu-1.tif", "YCu-2.tif", "YCu-3.tif")
GROUND_TRUTH_FILENAMES = tuple(f"{Path(filename).stem}_mask.tif" for filename in TEST_FILENAMES)
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


@dataclass(frozen=True)
class EvaluationParameters:
    analysis_output_root: Path
    mask_dir: Path
    output_root: Path


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


@dataclass(frozen=True)
class SampleInputs:
    sample_id: str
    filename: str
    prediction_path: Path
    truth_path: Path
    run_config_path: Path
    execution_provenance_path: Path
    metadata_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen Agglomerated U-Net Analysis masks on three held-out fields of view."
        )
    )
    parser.add_argument("--analysis-output-root", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
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
    return EvaluationParameters(
        analysis_output_root=analysis_output_root,
        mask_dir=mask_dir,
        output_root=_validate_output_root(namespace.output_root),
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return payload


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


def _match_sample(metadata: Mapping[str, Any], metadata_path: Path) -> str | None:
    filename = metadata.get("filename")
    sample_id = metadata.get("sample_id")
    if not isinstance(filename, str) or not isinstance(sample_id, str):
        raise ValueError(f"image metadata is missing filename/sample_id: {metadata_path}")
    filename_stem = Path(filename).stem
    matches = {
        Path(expected).stem
        for expected in TEST_FILENAMES
        if Path(expected).stem in {filename_stem, sample_id}
    }
    if len(matches) > 1:
        raise ValueError(f"image metadata maps to multiple fixed samples: {metadata_path}")
    return next(iter(matches), None)


def locate_sample_inputs(parameters: EvaluationParameters) -> dict[str, SampleInputs]:
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
        sample_id = _match_sample(metadata, metadata_path)
        if sample_id is None:
            raise ValueError(f"analysis-output-root contains a non-YCu prediction: {metadata_path}")
        expected_filename = f"{sample_id}.tif"
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
        located[sample_id] = SampleInputs(
            sample_id=sample_id,
            filename=expected_filename,
            prediction_path=prediction_path.resolve(),
            truth_path=truth_path.resolve(),
            run_config_path=run_config_path.resolve(),
            execution_provenance_path=execution_provenance_path.resolve(),
            metadata_path=metadata_path.resolve(),
        )
    expected = {Path(filename).stem for filename in TEST_FILENAMES}
    missing = sorted(expected - set(located))
    if missing:
        raise ValueError(f"pred_mask.png is missing for fixed samples: {missing}")
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
    )


def _macro(metrics: Sequence[ConfusionMetrics]) -> dict[str, float]:
    return {
        name: float(sum(getattr(item, name) for item in metrics) / len(metrics))
        for name in ("dice", "iou", "precision", "recall")
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
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    columns = ("scope", "tp", "fp", "fn", "tn", "dice", "iou", "precision", "recall")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for item in per_image:
            writer.writerow({"scope": item["sample_id"], **item["metrics"]})
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


def run_evaluation(parameters: EvaluationParameters) -> dict[str, Any]:
    parameters = EvaluationParameters(
        parameters.analysis_output_root.expanduser().resolve(),
        parameters.mask_dir.expanduser().resolve(),
        _validate_output_root(parameters.output_root),
    )
    if not parameters.analysis_output_root.is_dir() or not parameters.mask_dir.is_dir():
        raise ValueError("analysis-output-root and mask-dir must be existing directories")
    located = locate_sample_inputs(parameters)
    prepared: list[dict[str, Any]] = []
    for filename in TEST_FILENAMES:
        sample_id = Path(filename).stem
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
        prediction = _load_foreground(
            inputs.prediction_path,
            sample_id=sample_id,
            kind="prediction",
            require_zero_bottom=True,
        )
        truth = _load_foreground(inputs.truth_path, sample_id=sample_id, kind="truth")
        metrics = compute_metrics(prediction, truth)
        prediction_statistics = _scientific_statistics(
            prediction, sample_id=sample_id, kind="prediction"
        )
        truth_statistics = _scientific_statistics(truth, sample_id=sample_id, kind="ground-truth")
        prepared.append(
            {
                "inputs": inputs,
                "prediction": prediction,
                "truth": truth,
                "metrics": metrics,
                "run_config": run_config,
                "execution": execution,
                "prediction_statistics": prediction_statistics,
                "ground_truth_statistics": truth_statistics,
                "statistical_errors": _statistical_errors(prediction_statistics, truth_statistics),
            }
        )

    parameters.output_root.mkdir(parents=True, exist_ok=False)
    per_image: list[dict[str, Any]] = []
    for prepared_sample in prepared:
        inputs = prepared_sample["inputs"]
        prediction = prepared_sample["prediction"]
        truth = prepared_sample["truth"]
        metrics = prepared_sample["metrics"]
        run_config = prepared_sample["run_config"]
        execution = prepared_sample["execution"]
        review_path = parameters.output_root / f"{inputs.sample_id}_gt_pred_error.png"
        _write_review(review_path, truth=truth, prediction=prediction)
        per_image.append(
            {
                "sample_id": inputs.sample_id,
                "filename": inputs.filename,
                "metrics": asdict(metrics),
                "prediction_statistics": prepared_sample["prediction_statistics"],
                "ground_truth_statistics": prepared_sample["ground_truth_statistics"],
                "statistical_errors": prepared_sample["statistical_errors"],
                "input_provenance": {
                    "prediction_path": str(inputs.prediction_path),
                    "prediction_sha256": _sha256(inputs.prediction_path),
                    "truth_path": str(inputs.truth_path),
                    "truth_sha256": _sha256(inputs.truth_path),
                    "run_config_path": str(inputs.run_config_path),
                    "run_config_sha256": _sha256(inputs.run_config_path),
                    "execution_provenance_path": str(inputs.execution_provenance_path),
                    "execution_provenance_sha256": _sha256(inputs.execution_provenance_path),
                    "image_metadata_path": str(inputs.metadata_path),
                    "image_metadata_sha256": _sha256(inputs.metadata_path),
                },
                "verified_frozen_configuration": {
                    "model_id": run_config["model_id"],
                    "model_version": run_config["model_version"],
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
                "review_image": str(review_path),
            }
        )
    metrics = [item["metrics"] for item in prepared]
    macro = _macro(metrics)
    micro = _micro(metrics)
    result = {
        "schema_version": "1",
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation": "Agglomerated U-Net frozen independent test-set ground-truth evaluation",
        "model": {
            "model_id": MODEL_ID,
            "model_version": MODEL_VERSION,
            "weight_sha256": TORCHSCRIPT_SHA256,
            "config_sha256": CONFIG_SHA256,
            "model_card_sha256": MODEL_CARD_SHA256,
            "adapter_sha256": ADAPTER_SHA256,
            "threshold_rule": "probability >= 0.25 (verified from prior run config)",
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
        "per_image": per_image,
        "macro_average": macro,
        "micro_average": asdict(micro),
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
            ("YCu-1/2/3 are independent fields of view, not sample-level independent materials."),
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
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        parameters = _validated_parameters(build_parser().parse_args(argv))
        result = run_evaluation(parameters)
    except Exception as error:
        print(
            json.dumps(
                {"status": "error", "error_type": type(error).__name__, "message": str(error)},
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

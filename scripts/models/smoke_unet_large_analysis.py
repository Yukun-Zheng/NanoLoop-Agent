"""Run the three fixed large U-Net test views through the full Analysis chain."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.analysis.application import (
    AnalysisApplicationService,
    AnalysisCreationService,
    AnalysisUpload,
    InferenceGatewayProtocol,
)
from app.analysis.config import MorphometryConfig, PostprocessProfile
from app.analysis.instance_artifacts import decode_binary_mask
from app.analysis.morphometry import measure
from app.analysis.postprocessing import NormalizedInstance
from app.analysis.preprocessing import build_analysis_roi
from app.analysis.quality import QualityInputs, evaluate
from app.analysis.reporting import ReportWriter
from app.contracts.analyses import (
    CreateAnalysisMetadata,
    CreateRunsRequest,
    ImageMetadataInput,
    ImageSummaryDTO,
    InferenceOptions,
    QualityReportDTO,
    RunConfiguration,
    ScaleInput,
)
from app.contracts.enums import DevicePreference, ModelStatus, RoiMode, ScaleMode
from app.contracts.execution import ExecutionRuntimeProvenance
from app.contracts.identity import AuthMode, PrincipalContext
from app.contracts.models import ModelBundleReference, ModelMetadata
from app.contracts.repositories import UnitOfWork
from app.core.config import Settings
from app.core.identity import legacy_principal_context
from app.db.base import Base
from app.db.repositories import SqlAlchemyUnitOfWork
from app.db.session import Database
from app.inference.gateway import InferenceGateway
from app.inference.model_sync import sync_model_registry
from app.inference.registry import ModelRegistryService
from app.storage import LocalFileStore, StoragePaths
from scripts.smoke_test import validate_export_zip

MODEL_ID = "unet-large-optimized-v1"
MODEL_VERSION = "1"
ADAPTER_PATH = "app.inference.adapters.unet:UNetAdapter"
TORCHSCRIPT_SHA256 = "007d9a16bf31e5f960160c52eefa938b83feeac2e6c0d7dec9c8670a38626e05"
# These three source-asset digests intentionally live together.  Refresh them only when
# the corresponding version-1 config/card/adapter is deliberately changed.
CONFIG_SHA256 = "4e48c75d960faaa17868f0318da5526a6ba72211396ec106c2e57ce7eecc8856"
MODEL_CARD_SHA256 = "bac2eacbc3569cc24e76aa90edf4a00ed1cfd7149180bf07113a4e905e2d7bfc"
ADAPTER_SHA256 = "a40d7df346675946a280df9a1f0eeecc42bc8d6261f10e375f6f2ab6631f184a"
TEST_FILENAMES = ("SrZr-3.tif", "BaCu-2.tif", "PrCu-3.tif")
IMAGE_WIDTH = 2048
IMAGE_HEIGHT = 1536
THRESHOLD = 0.50
MIN_AREA_PX = 512
BOTTOM_CROP_PX = 180
SCALE_NM_PER_PIXEL = 100.0 / 184.0
DEVICE = DevicePreference.CPU
SEED = 2026
REQUIRED_ARTIFACT_KEYS = frozenset(
    {
        "run_config_path",
        "execution_provenance_path",
        "transform_path",
        "particles_csv_path",
        "image_summary_path",
        "quality_report_path",
        "pred_mask_path",
        "overlay_path",
        "labeled_particles_path",
        "probability_path",
        "instances_path",
    }
)


@dataclass(frozen=True, slots=True)
class SmokeParameters:
    image_dir: Path
    registry: Path
    output_root: Path


@dataclass(frozen=True, slots=True)
class CalibratedAnalysis:
    postprocess_profile_id: str
    threshold: float
    threshold_comparison: str
    min_area_px: int
    min_area_nm2: float
    min_area_equivalent_diameter_nm: float
    watershed_enabled: bool
    fill_holes: bool
    exclude_border: bool
    connectivity: int
    perimeter_neighborhood: int
    bottom_crop_px: int
    scale_nm_per_pixel: float

    def postprocess_profile(self) -> PostprocessProfile:
        return PostprocessProfile(
            profile_id=self.postprocess_profile_id,
            min_area_px=self.min_area_px,
            fill_holes=self.fill_holes,
            watershed_enabled=self.watershed_enabled,
            exclude_border=self.exclude_border,
            connectivity=self.connectivity,
        )

    def morphometry_config(self) -> MorphometryConfig:
        return MorphometryConfig(perimeter_neighborhood=self.perimeter_neighborhood)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _validate_output_root(output_root: Path, *, repository: Path | None = None) -> Path:
    resolved = output_root.expanduser().resolve(strict=False)
    repository = (repository or _repository_root()).resolve(strict=True)
    if resolved.exists():
        raise ValueError(f"output-root already exists: {resolved}")
    if resolved == repository or resolved.is_relative_to(repository):
        raise ValueError("output-root must be outside the repository")
    return resolved


def _validated_parameters(namespace: argparse.Namespace) -> SmokeParameters:
    image_dir = namespace.image_dir.expanduser().resolve(strict=True)
    registry = namespace.registry.expanduser().resolve(strict=True)
    output_root = _validate_output_root(namespace.output_root)
    if not image_dir.is_dir():
        raise ValueError(f"image-dir is not a directory: {image_dir}")
    if not registry.is_file():
        raise ValueError(f"registry is not a file: {registry}")
    for filename in TEST_FILENAMES:
        if not (image_dir / filename).is_file():
            raise ValueError(f"large test image is missing: {filename}")
    return SmokeParameters(
        image_dir=image_dir,
        registry=registry,
        output_root=output_root,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def _require_bool(source: Mapping[str, Any], key: str) -> bool:
    value: object = source.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"calibrated_analysis.{key} must be boolean")
    return value


def _require_int(source: Mapping[str, Any], key: str) -> int:
    value: object = source.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"calibrated_analysis.{key} must be an integer")
    return value


def _require_float(source: Mapping[str, Any], key: str) -> float:
    value: object = source.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"calibrated_analysis.{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"calibrated_analysis.{key} must be finite")
    return result


def load_calibrated_analysis(
    config: Mapping[str, Any],
    metadata: ModelMetadata,
) -> CalibratedAnalysis:
    raw = config.get("calibrated_analysis")
    if not isinstance(raw, Mapping):
        raise ValueError("large model config is missing calibrated_analysis")
    comparison = raw.get("threshold_comparison")
    if comparison != "gt":
        raise ValueError("calibrated_analysis requires strict threshold_comparison=gt")
    calibrated = CalibratedAnalysis(
        postprocess_profile_id=metadata.postprocess_profile,
        threshold=_require_float(raw, "threshold"),
        threshold_comparison=comparison,
        min_area_px=_require_int(raw, "min_area_px"),
        min_area_nm2=_require_float(raw, "min_area_nm2"),
        min_area_equivalent_diameter_nm=_require_float(raw, "min_area_equivalent_diameter_nm"),
        watershed_enabled=_require_bool(raw, "watershed_enabled"),
        fill_holes=_require_bool(raw, "fill_holes"),
        exclude_border=_require_bool(raw, "exclude_border"),
        connectivity=_require_int(raw, "connectivity"),
        perimeter_neighborhood=_require_int(raw, "perimeter_neighborhood"),
        bottom_crop_px=_require_int(raw, "bottom_crop_px"),
        scale_nm_per_pixel=_require_float(raw, "scale_nm_per_pixel"),
    )
    if calibrated.threshold != metadata.default_threshold:
        raise ValueError("calibrated threshold differs from registry default_threshold")
    if calibrated.bottom_crop_px != metadata.inference_invalid_bottom_px:
        raise ValueError("calibrated bottom crop differs from registry invalid-bottom metadata")
    if config.get("bottom_crop_px") != calibrated.bottom_crop_px:
        raise ValueError("calibrated bottom crop differs from inference config")
    if config.get("threshold_comparison") != calibrated.threshold_comparison:
        raise ValueError("calibrated threshold comparison differs from inference config")
    if config.get("expected_image_size") != [IMAGE_HEIGHT, IMAGE_WIDTH]:
        raise ValueError("large model config is not frozen at the 2048x1536 image contract")
    exact_values = {
        "threshold": (calibrated.threshold, THRESHOLD),
        "min_area_px": (calibrated.min_area_px, MIN_AREA_PX),
        "bottom_crop_px": (calibrated.bottom_crop_px, BOTTOM_CROP_PX),
    }
    mismatches = {
        name: {"expected": expected, "observed": observed}
        for name, (observed, expected) in exact_values.items()
        if observed != expected
    }
    if not math.isclose(
        calibrated.scale_nm_per_pixel,
        SCALE_NM_PER_PIXEL,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        mismatches["scale_nm_per_pixel"] = {
            "expected": SCALE_NM_PER_PIXEL,
            "observed": calibrated.scale_nm_per_pixel,
        }
    if mismatches:
        raise ValueError(
            "calibrated_analysis differs from the frozen Large acceptance contract: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    expected_area = calibrated.min_area_px * calibrated.scale_nm_per_pixel**2
    expected_diameter = (
        2.0 * math.sqrt(calibrated.min_area_px / math.pi) * calibrated.scale_nm_per_pixel
    )
    if not math.isclose(calibrated.min_area_nm2, expected_area, rel_tol=1e-12):
        raise ValueError("calibrated min-area physical conversion is inconsistent")
    if not math.isclose(
        calibrated.min_area_equivalent_diameter_nm,
        expected_diameter,
        rel_tol=1e-12,
    ):
        raise ValueError("calibrated equivalent-diameter conversion is inconsistent")
    calibrated.postprocess_profile()
    calibrated.morphometry_config()
    return calibrated


def _validate_model_metadata(metadata: ModelMetadata) -> None:
    expected: dict[str, object] = {
        "model_id": MODEL_ID,
        "version": MODEL_VERSION,
        "status": ModelStatus.READY,
        "adapter_path": ADAPTER_PATH,
        "weight_sha256": TORCHSCRIPT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "model_card_sha256": MODEL_CARD_SHA256,
        "adapter_sha256": ADAPTER_SHA256,
        "default_threshold": THRESHOLD,
        "default_min_area_px": MIN_AREA_PX,
        "inference_invalid_bottom_px": BOTTOM_CROP_PX,
        "expected_input_width": IMAGE_WIDTH,
        "expected_input_height": IMAGE_HEIGHT,
    }
    mismatches = {
        name: {"expected": expected_value, "observed": getattr(metadata, name)}
        for name, expected_value in expected.items()
        if getattr(metadata, name) != expected_value
    }
    if mismatches:
        raise RuntimeError(
            "private registry does not match the frozen Large acceptance model: "
            + json.dumps(mismatches, default=str, ensure_ascii=False, sort_keys=True)
        )


def _validate_bundle(bundle: ModelBundleReference) -> None:
    expected_references = {
        "weight_ref": f"{TORCHSCRIPT_SHA256}/weights.pt",
        "config_ref": f"{CONFIG_SHA256}/config.yaml",
        "model_card_ref": f"{MODEL_CARD_SHA256}/model-card.md",
        "adapter_ref": f"{ADAPTER_SHA256}/adapter.py",
    }
    mismatches = {
        name: {"expected": expected, "observed": getattr(bundle, name)}
        for name, expected in expected_references.items()
        if getattr(bundle, name) != expected
    }
    if bundle.adapter_sha256 != ADAPTER_SHA256:
        mismatches["adapter_sha256"] = {
            "expected": ADAPTER_SHA256,
            "observed": bundle.adapter_sha256,
        }
    if bundle.manifest_ref != f"bundles/{bundle.bundle_id}/manifest.json":
        mismatches["manifest_ref"] = {
            "expected": f"bundles/{bundle.bundle_id}/manifest.json",
            "observed": bundle.manifest_ref,
        }
    if mismatches:
        raise RuntimeError(
            "frozen Large model bundle does not bind the expected assets: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def _load_contract_artifact(path: Path, model_type: type[Any]) -> Any:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid contract artifact: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"contract artifact is not an object: {path}")
    contract_schema_version = payload.pop("contract_schema_version", None)
    if contract_schema_version is None:
        raise RuntimeError(f"contract artifact has no contract_schema_version: {path}")
    payload["schema_version"] = contract_schema_version
    return model_type.model_validate(payload)


def _validate_frozen_run(
    configuration: RunConfiguration,
    execution: ExecutionRuntimeProvenance,
    *,
    run_config_path: Path,
    execution_path: Path,
) -> None:
    if configuration.schema_version != 3:
        raise RuntimeError("Large acceptance run is not schema-v3")
    if configuration.provenance_status != "complete" or configuration.provenance_warnings:
        raise RuntimeError("Large acceptance run has incomplete configuration provenance")
    if configuration.model_id != MODEL_ID or configuration.model_version != MODEL_VERSION:
        raise RuntimeError("Large acceptance run used a different model id/version")
    if configuration.adapter_path != ADAPTER_PATH:
        raise RuntimeError("Large acceptance run used a different adapter")
    for observed, expected, label in (
        (configuration.weight_sha256, TORCHSCRIPT_SHA256, "weight"),
        (configuration.config_sha256, CONFIG_SHA256, "config"),
        (configuration.model_card_sha256, MODEL_CARD_SHA256, "model card"),
        (configuration.adapter_sha256, ADAPTER_SHA256, "adapter"),
    ):
        if observed != expected:
            raise RuntimeError(f"Large acceptance run used a different {label} SHA-256")
    bundle = configuration.model_bundle
    if bundle is None:
        raise RuntimeError("Large acceptance run has no frozen model bundle")
    _validate_bundle(bundle)
    if configuration.inference.threshold != THRESHOLD:
        raise RuntimeError("Large acceptance threshold is not frozen at 0.50")
    if configuration.inference.min_area_px != MIN_AREA_PX:
        raise RuntimeError("Large acceptance min_area_px is not frozen at 512")
    if not math.isclose(
        configuration.scale_nm_per_pixel or 0.0,
        SCALE_NM_PER_PIXEL,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Large acceptance scale is not frozen at 100/184 nm/px")
    if not execution.build_identity_matches_contract or execution.warnings:
        raise RuntimeError("Large acceptance execution provenance is incomplete")
    if execution.model_bundle_id != bundle.bundle_id:
        raise RuntimeError("execution model bundle differs from the queued run contract")
    if execution.adapter_sha256 != ADAPTER_SHA256:
        raise RuntimeError("execution adapter digest differs from the queued run contract")
    if execution.requested_device != DEVICE or execution.actual_device != DEVICE.value:
        raise RuntimeError("Large acceptance execution did not remain on CPU")
    if execution.seed != SEED:
        raise RuntimeError("Large acceptance execution used a different seed")
    deterministic = (
        execution.python_random_seeded,
        execution.numpy_random_seeded,
        execution.torch_deterministic_algorithms,
        execution.global_inference_serialized,
    )
    if not all(deterministic):
        raise RuntimeError("Large acceptance execution controls are incomplete")
    if not execution.backend.endswith(".UNetAdapter"):
        raise RuntimeError("Large acceptance execution backend is not UNetAdapter")

    stored_configuration = _load_contract_artifact(run_config_path, RunConfiguration)
    stored_execution = _load_contract_artifact(
        execution_path,
        ExecutionRuntimeProvenance,
    )
    if stored_configuration != configuration or stored_execution != execution:
        raise RuntimeError("stored run/execution evidence differs from the completed run")


def _validate_mask_bottom(path: Path) -> None:
    try:
        with Image.open(path) as image:
            if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                raise RuntimeError("canonical prediction mask dimensions are not 2048x1536")
            pixels = np.asarray(image)
    except OSError as error:
        raise RuntimeError(f"cannot read canonical prediction mask: {path}") from error
    foreground = pixels != 0 if pixels.ndim == 2 else np.any(pixels != 0, axis=-1)
    if np.any(foreground[IMAGE_HEIGHT - BOTTOM_CROP_PX :]):
        raise RuntimeError("canonical prediction mask bottom 180 px is not zero")


def _run_context(*, sample_id: str, run_id: str) -> str:
    return f"sample={sample_id}, run={run_id}"


def _load_json_object(path: Path, *, artifact: str, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{context}: invalid {artifact}: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context}: {artifact} must be a JSON object")
    return payload


def _load_report_artifact(
    path: Path,
    model_type: type[Any],
    *,
    artifact: str,
    context: str,
) -> Any:
    payload = _load_json_object(path, artifact=artifact, context=context)
    file_schema_version = payload.pop("schema_version", None)
    if file_schema_version != LocalFileStore.schema_version:
        raise RuntimeError(
            f"{context}: {artifact}.schema_version must be {LocalFileStore.schema_version}"
        )
    try:
        return model_type.model_validate(payload)
    except ValueError as error:
        raise RuntimeError(f"{context}: {artifact} violates its DTO contract") from error


def _read_foreground_mask(path: Path, *, artifact: str, context: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            pixels = np.asarray(image)
    except OSError as error:
        raise RuntimeError(f"{context}: cannot read {artifact}: {path}") from error
    if pixels.ndim == 2:
        return np.asarray(pixels != 0, dtype=bool)
    if pixels.ndim == 3:
        return np.asarray(np.any(pixels != 0, axis=-1), dtype=bool)
    raise RuntimeError(f"{context}: {artifact} has unsupported shape {pixels.shape}")


def _required_int(value: object, *, field: str, artifact: str, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{context}: {artifact}.{field} must be an integer")
    return value


def _required_float(value: object, *, field: str, artifact: str, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{context}: {artifact}.{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{context}: {artifact}.{field} must be finite")
    return result


def _assert_close(
    observed: float | None,
    expected: float | None,
    *,
    field: str,
    artifact: str,
    context: str,
) -> None:
    if observed is None or expected is None:
        if observed != expected:
            raise RuntimeError(
                f"{context}: {artifact}.{field} differs (expected={expected}, observed={observed})"
            )
        return
    if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-15):
        raise RuntimeError(
            f"{context}: {artifact}.{field} differs (expected={expected}, observed={observed})"
        )


def _csv_optional_float(
    row: Mapping[str, str],
    *,
    field: str,
    context: str,
) -> float | None:
    value = row.get(field)
    if value is None:
        raise RuntimeError(f"{context}: particles.csv is missing column {field}")
    if value == "":
        return None
    try:
        result = float(value)
    except ValueError as error:
        raise RuntimeError(f"{context}: particles.csv.{field} is not numeric") from error
    if not math.isfinite(result):
        raise RuntimeError(f"{context}: particles.csv.{field} must be finite")
    return result


def _csv_required_int(row: Mapping[str, str], *, field: str, context: str) -> int:
    value = row.get(field)
    if value is None:
        raise RuntimeError(f"{context}: particles.csv is missing column {field}")
    try:
        return int(value)
    except ValueError as error:
        raise RuntimeError(f"{context}: particles.csv.{field} is not an integer") from error


def _load_canonical_instances(
    path: Path,
    *,
    width: int,
    height: int,
    roi_mask: np.ndarray,
    context: str,
) -> tuple[list[NormalizedInstance], np.ndarray]:
    payload = _load_json_object(path, artifact="instances.json", context=context)
    if payload.get("coordinate_space") != "original_px":
        raise RuntimeError(f"{context}: instances.json.coordinate_space is not original_px")
    if payload.get("width") != width or payload.get("height") != height:
        raise RuntimeError(f"{context}: instances.json dimensions differ from the source image")
    records = payload.get("instances")
    if not isinstance(records, list):
        raise RuntimeError(f"{context}: instances.json.instances must be a list")
    if payload.get("instance_count") != len(records):
        raise RuntimeError(f"{context}: instances.json.instance_count differs from instances")

    union = np.zeros((height, width), dtype=bool)
    instances: list[NormalizedInstance] = []
    seen_indices: set[int] = set()
    for position, value in enumerate(records):
        record_context = f"{context}, instances.json.instances[{position}]"
        if not isinstance(value, Mapping):
            raise RuntimeError(f"{record_context} must be an object")
        instance_index = _required_int(
            value.get("instance_index"),
            field="instance_index",
            artifact="instances.json",
            context=record_context,
        )
        if instance_index < 1 or instance_index in seen_indices:
            raise RuntimeError(f"{record_context} has a duplicate or invalid instance_index")
        seen_indices.add(instance_index)
        raw_bbox = value.get("bbox_xyxy")
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            raise RuntimeError(f"{record_context}.bbox_xyxy must contain four integers")
        bbox = tuple(
            _required_int(
                coordinate,
                field=f"bbox_xyxy[{coordinate_index}]",
                artifact="instances.json",
                context=record_context,
            )
            for coordinate_index, coordinate in enumerate(raw_bbox)
        )
        x1, y1, x2, y2 = bbox
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x1 >= x2 or y1 >= y2:
            raise RuntimeError(f"{record_context}.bbox_xyxy is outside the source image")
        raw_mask = value.get("mask")
        if not isinstance(raw_mask, Mapping):
            raise RuntimeError(f"{record_context}.mask must be an object")
        if raw_mask.get("encoding") != "flat_rle_v1" or raw_mask.get("order") != "row_major":
            raise RuntimeError(f"{record_context}.mask is not canonical flat_rle_v1 row-major data")
        starts = raw_mask.get("starts")
        lengths = raw_mask.get("lengths")
        if (
            not isinstance(starts, list)
            or not isinstance(lengths, list)
            or any(isinstance(item, bool) or not isinstance(item, int) for item in starts + lengths)
        ):
            raise RuntimeError(f"{record_context}.mask RLE contains non-integer values")
        try:
            mask = decode_binary_mask(starts=starts, lengths=lengths, width=width, height=height)
        except ValueError as error:
            raise RuntimeError(f"{record_context}.mask RLE is invalid") from error
        if not np.any(mask):
            raise RuntimeError(f"{record_context}.mask is empty")
        if np.any(mask & ~roi_mask):
            raise RuntimeError(f"{record_context}.mask escapes the effective ROI")
        if np.any(union & mask):
            raise RuntimeError(f"{record_context}.mask overlaps a prior canonical instance")
        area_px = _required_int(
            value.get("area_px"),
            field="area_px",
            artifact="instances.json",
            context=record_context,
        )
        measured_area = int(np.count_nonzero(mask))
        if area_px != measured_area:
            raise RuntimeError(f"{record_context}.area_px differs from its decoded RLE mask")
        ys, xs = np.nonzero(mask)
        measured_bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        if bbox != measured_bbox:
            raise RuntimeError(f"{record_context}.bbox_xyxy differs from its decoded RLE mask")
        confidence_raw = value.get("confidence")
        confidence = (
            None
            if confidence_raw is None
            else _required_float(
                confidence_raw,
                field="confidence",
                artifact="instances.json",
                context=record_context,
            )
        )
        touches = value.get("touches_roi_boundary")
        if not isinstance(touches, bool):
            raise RuntimeError(f"{record_context}.touches_roi_boundary must be boolean")
        union |= mask
        instances.append(
            NormalizedInstance(
                instance_index=instance_index,
                mask=mask,
                bbox=bbox,
                area_px=area_px,
                confidence=confidence,
                touches_roi_boundary=touches,
            )
        )
    return instances, union


def _validate_particles_csv(
    path: Path,
    *,
    instances: list[NormalizedInstance],
    configuration: RunConfiguration,
    roi_mask: np.ndarray,
    run_id: str,
    context: str,
) -> Any:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as error:
        raise RuntimeError(f"{context}: cannot read particles.csv: {path}") from error
    if len(rows) != len(instances):
        raise RuntimeError(f"{context}: particles.csv row count differs from instances.json")
    postprocess = configuration.resolved_postprocess
    morphometry = configuration.resolved_morphometry
    if postprocess is None or morphometry is None:
        raise RuntimeError(f"{context}: run configuration has no resolved scientific settings")
    measured = measure(
        run_id=run_id,
        instances=instances,
        roi_mask=roi_mask,
        scale_nm_per_pixel=configuration.scale_nm_per_pixel,
        config=morphometry,
    )
    # Particle IDs are intentionally random; compare the stable, canonical columns only.
    by_index = {instance.instance_index: instance for instance in instances}
    measured_by_index = {particle.instance_index: particle for particle in measured.particles}
    seen_indices: set[int] = set()
    for row_number, row in enumerate(rows, start=2):
        row_context = f"{context}, particles.csv row {row_number}"
        instance_index = _csv_required_int(row, field="instance_index", context=row_context)
        if instance_index in seen_indices or instance_index not in by_index:
            raise RuntimeError(f"{row_context}: instance_index does not match instances.json")
        seen_indices.add(instance_index)
        instance = by_index[instance_index]
        csv_area = _csv_optional_float(row, field="area_px", context=row_context)
        if csv_area is None or not math.isclose(
            csv_area,
            float(instance.area_px),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise RuntimeError(f"{row_context}: area_px differs from instances.json")
        bbox = tuple(
            _csv_required_int(row, field=field, context=row_context)
            for field in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
        )
        if bbox != instance.bbox:
            raise RuntimeError(f"{row_context}: bbox differs from instances.json")
        particle = measured_by_index[instance_index]
        if row.get("run_id") != particle.run_id:
            raise RuntimeError(f"{row_context}: run_id differs from recomputed morphometry")
        for field, expected in (
            ("perimeter_px", particle.perimeter_px),
            ("equivalent_diameter_px", particle.equivalent_diameter_px),
            ("equivalent_diameter_nm", particle.equivalent_diameter_nm),
            ("circularity", particle.circularity),
            ("confidence", particle.confidence),
        ):
            _assert_close(
                _csv_optional_float(row, field=field, context=row_context),
                expected,
                field=field,
                artifact="particles.csv",
                context=row_context,
            )
    if seen_indices != set(by_index):
        raise RuntimeError(f"{context}: particles.csv does not cover every canonical instance")
    return measured


def _validate_summary_and_quality(
    *,
    summary_path: Path,
    quality_path: Path,
    measured: Any,
    configuration: RunConfiguration,
    instances: list[NormalizedInstance],
    union: np.ndarray,
    roi_mask: np.ndarray,
    context: str,
) -> None:
    summary = _load_report_artifact(
        summary_path,
        ImageSummaryDTO,
        artifact="image_summary.json",
        context=context,
    )
    quality = _load_report_artifact(
        quality_path,
        QualityReportDTO,
        artifact="quality_report.json",
        context=context,
    )
    expected_summary = measured.image_summary
    if summary.run_id != expected_summary.run_id:
        raise RuntimeError(f"{context}: image_summary.json.run_id differs from particles.csv")
    if summary.particle_count != expected_summary.particle_count:
        raise RuntimeError(
            f"{context}: image_summary.json.particle_count differs from particles.csv"
        )
    if summary.roi_area_px != expected_summary.roi_area_px:
        raise RuntimeError(
            f"{context}: image_summary.json.roi_area_px differs from the effective ROI"
        )
    for field in (
        "number_density_px2",
        "number_density_um2",
        "mean_equivalent_diameter_px",
        "mean_equivalent_diameter_nm",
        "coverage_ratio",
        "perimeter_density_px",
        "perimeter_density_um",
    ):
        _assert_close(
            getattr(summary, field),
            getattr(expected_summary, field),
            field=field,
            artifact="image_summary.json",
            context=context,
        )
    if summary.quality_status != quality.status:
        raise RuntimeError(
            f"{context}: image_summary.json.quality_status differs from quality_report.json.status"
        )
    quality_gate = configuration.resolved_quality_gate
    postprocess = configuration.resolved_postprocess
    if quality_gate is None or postprocess is None:
        raise RuntimeError(f"{context}: run configuration has no resolved quality settings")
    candidate_count = quality.metrics.get("candidate_instance_count")
    boundary_count = quality.metrics.get("boundary_instance_count")
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or isinstance(boundary_count, bool)
        or not isinstance(boundary_count, int)
    ):
        raise RuntimeError(f"{context}: quality_report.json has invalid candidate diagnostics")
    recomputed_quality = evaluate(
        QualityInputs(
            roi_area_px=int(np.count_nonzero(roi_mask)),
            foreground_area_px=int(np.count_nonzero(union)),
            instances=instances,
            minimum_area_px=postprocess.min_area_px,
            validation_warnings=[],
            candidate_instance_count=candidate_count,
            boundary_instance_count=boundary_count,
        ),
        quality_gate,
    )
    if set(quality.metrics) != set(recomputed_quality.metrics):
        raise RuntimeError(f"{context}: quality_report.json metrics differ from canonical analysis")
    for field, expected in recomputed_quality.metrics.items():
        observed = quality.metrics[field]
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            if isinstance(observed, bool) or not isinstance(observed, (int, float)):
                raise RuntimeError(f"{context}: quality_report.json.{field} has an invalid type")
            _assert_close(
                float(observed),
                float(expected),
                field=field,
                artifact="quality_report.json",
                context=context,
            )
        elif observed != expected:
            raise RuntimeError(
                f"{context}: quality_report.json.{field} differs from canonical analysis"
            )


def _canonical_artifact_identities(
    file_store: LocalFileStore,
    artifacts: Mapping[str, str | None],
    *,
    output_root: Path,
    context: str,
) -> dict[str, dict[str, str]]:
    resolved_root = output_root.resolve(strict=True)
    identities: dict[str, dict[str, str]] = {}
    for key in sorted(REQUIRED_ARTIFACT_KEYS):
        raw_path = artifacts.get(key)
        if raw_path is None:
            raise RuntimeError(f"{context}: missing canonical artifact {key}")
        path = Path(raw_path).resolve(strict=True)
        try:
            relative = path.relative_to(resolved_root)
        except ValueError as error:
            raise RuntimeError(
                f"{context}: canonical artifact escaped smoke output: {key}"
            ) from error
        identities[key] = {
            "path": relative.as_posix(),
            "sha256": file_store.calculate_sha256(path),
        }
    return identities


def _validate_canonical_chain(
    *,
    artifacts: Mapping[str, str | None],
    configuration: RunConfiguration,
    width: int,
    height: int,
    sample_id: str,
    run_id: str,
    output_root: Path,
    file_store: LocalFileStore,
) -> dict[str, dict[str, str]]:
    context = _run_context(sample_id=sample_id, run_id=run_id)
    pred_mask_path = artifacts.get("pred_mask_path")
    instances_path = artifacts.get("instances_path")
    particles_csv_path = artifacts.get("particles_csv_path")
    summary_path = artifacts.get("image_summary_path")
    quality_path = artifacts.get("quality_report_path")
    if None in (pred_mask_path, instances_path, particles_csv_path, summary_path, quality_path):
        raise RuntimeError(f"{context}: canonical artifact chain is incomplete")
    roi_mask = build_analysis_roi(
        width=width,
        height=height,
        analysis_roi=configuration.analysis_roi,
        roi_mode=RoiMode.FULL_IMAGE,
        boxes=[],
    )
    pred_mask = _read_foreground_mask(
        Path(str(pred_mask_path)), artifact="pred_mask.png", context=context
    )
    if pred_mask.shape != roi_mask.shape:
        raise RuntimeError(f"{context}: pred_mask.png dimensions differ from the effective ROI")
    instances, union = _load_canonical_instances(
        Path(str(instances_path)),
        width=width,
        height=height,
        roi_mask=roi_mask,
        context=context,
    )
    if not np.array_equal(union, pred_mask):
        raise RuntimeError(f"{context}: instances.json union differs from pred_mask.png")
    measured = _validate_particles_csv(
        Path(str(particles_csv_path)),
        instances=instances,
        configuration=configuration,
        roi_mask=roi_mask,
        run_id=run_id,
        context=context,
    )
    _validate_summary_and_quality(
        summary_path=Path(str(summary_path)),
        quality_path=Path(str(quality_path)),
        measured=measured,
        configuration=configuration,
        instances=instances,
        union=union,
        roi_mask=roi_mask,
        context=context,
    )
    return _canonical_artifact_identities(
        file_store,
        artifacts,
        output_root=output_root,
        context=context,
    )


def _validate_export_canonical_artifacts(
    manifest: Mapping[str, Any],
    *,
    runs: Sequence[Mapping[str, Any]],
    file_store: LocalFileStore,
    job_id: str,
) -> None:
    records = manifest.get("files")
    if not isinstance(records, list):
        raise RuntimeError("export.zip: export manifest files must be a list")
    manifest_hashes: dict[str, str] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise RuntimeError("export.zip: export manifest contains a non-object file record")
        path = record.get("path")
        sha256 = record.get("sha256")
        if not isinstance(path, str) or not isinstance(sha256, str) or path in manifest_hashes:
            raise RuntimeError(
                "export.zip: export manifest has an invalid or duplicate file record"
            )
        manifest_hashes[path] = sha256
    job_dir = file_store.paths.job_dir(job_id).resolve(strict=True)
    for run in runs:
        context = _run_context(sample_id=str(run["sample_id"]), run_id=str(run["run_id"]))
        identities = run.get("canonical_artifact_identities")
        artifacts = run.get("artifacts")
        if not isinstance(identities, Mapping) or not isinstance(artifacts, Mapping):
            raise RuntimeError(f"{context}: canonical export evidence is incomplete")
        if set(identities) != REQUIRED_ARTIFACT_KEYS:
            raise RuntimeError(
                f"{context}: canonical export evidence has an unexpected artifact mapping"
            )
        expected_paths: set[str] = set()
        for key in REQUIRED_ARTIFACT_KEYS:
            identity = identities.get(key)
            raw_path = artifacts.get(key)
            if not isinstance(identity, Mapping) or not isinstance(raw_path, str):
                raise RuntimeError(f"{context}: canonical export evidence is invalid for {key}")
            source = Path(raw_path).resolve(strict=True)
            try:
                member_path = source.relative_to(job_dir).as_posix()
            except ValueError as error:
                raise RuntimeError(
                    f"{context}: canonical artifact is outside this export: {key}"
                ) from error
            if member_path in expected_paths:
                raise RuntimeError(f"{context}: canonical artifacts map to the same export member")
            expected_paths.add(member_path)
            source_sha256 = file_store.calculate_sha256(source)
            if identity.get("sha256") != source_sha256:
                raise RuntimeError(
                    f"{context}: canonical artifact SHA changed before export: {key}"
                )
            if manifest_hashes.get(member_path) != source_sha256:
                raise RuntimeError(
                    f"{context}: export.zip manifest SHA differs from canonical artifact {key}"
                )


def _sqlite_url(path: Path) -> str:
    return f"sqlite:///{path.resolve(strict=False).as_posix()}"


def _absolute_artifact_paths(
    file_store: LocalFileStore,
    paths: dict[str, str | None],
) -> dict[str, str | None]:
    return {
        key: (
            str(file_store.paths.require_managed(value, must_exist=True))
            if value is not None
            else None
        )
        for key, value in paths.items()
    }


def _bottom_exclusion_evidence(
    configuration: Any,
    *,
    width: int,
    height: int,
    bottom_crop_px: int,
) -> dict[str, Any]:
    expected_rect = (0, height - bottom_crop_px, width, height)
    regions = [
        region
        for region in configuration.analysis_roi.invalid_rects
        if region.reason == "model_bottom_information_bar"
    ]
    exact = [
        region
        for region in regions
        if (region.x1, region.y1, region.x2, region.y2) == expected_rect
    ]
    return {
        "expected_invalid_bottom_px": bottom_crop_px,
        "expected_bottom_area_px": width * bottom_crop_px,
        "expected_effective_roi_area_px": width * (height - bottom_crop_px),
        "expected_rect": {
            "x1": expected_rect[0],
            "y1": expected_rect[1],
            "x2": expected_rect[2],
            "y2": expected_rect[3],
        },
        "matching_model_bottom_region_present": bool(exact),
        "model_bottom_regions": [region.model_dump(mode="json") for region in regions],
    }


def _density_evidence(
    *,
    particles_csv: Path,
    particle_count: int,
    roi_area_px: int,
    scale_nm_per_pixel: float,
    coverage_ratio: float,
    number_density_um2: float | None,
    perimeter_density_um: float | None,
    valid_height: int,
) -> dict[str, Any]:
    with particles_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != particle_count:
        raise RuntimeError("particle CSV count differs from Analysis summary")
    area_total_px = sum(float(row["area_px"]) for row in rows)
    perimeter_total_px = sum(float(row["perimeter_px"]) for row in rows)
    all_particles_above_bottom = all(int(row["bbox_y2"]) <= valid_height for row in rows)
    roi_area_um2 = roi_area_px * (scale_nm_per_pixel / 1000.0) ** 2
    expected_coverage = area_total_px / roi_area_px
    expected_number_density = particle_count / roi_area_um2
    expected_perimeter_density = perimeter_total_px * scale_nm_per_pixel / 1000.0 / roi_area_um2
    checks = {
        "all_particle_bboxes_within_effective_roi": all_particles_above_bottom,
        "coverage_uses_effective_roi": math.isclose(
            coverage_ratio, expected_coverage, rel_tol=1e-12, abs_tol=1e-15
        ),
        "number_density_uses_effective_roi": (
            number_density_um2 is not None
            and math.isclose(
                number_density_um2,
                expected_number_density,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ),
        "perimeter_density_uses_effective_roi": (
            perimeter_density_um is not None
            and math.isclose(
                perimeter_density_um,
                expected_perimeter_density,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"scientific ROI consistency check failed: {checks}")
    return {
        "roi_area_um2": roi_area_um2,
        "particle_area_total_px": area_total_px,
        "particle_perimeter_total_px": perimeter_total_px,
        "expected_coverage_ratio": expected_coverage,
        "expected_number_density_um2": expected_number_density,
        "expected_perimeter_density_um": expected_perimeter_density,
        **checks,
    }


def execute_analyses(
    parameters: SmokeParameters,
    calibrated: CalibratedAnalysis,
    *,
    database: Database,
    file_store: LocalFileStore,
    gateway: InferenceGatewayProtocol,
    principal: PrincipalContext | None = None,
) -> dict[str, Any]:
    """Execute creation -> three FULL_IMAGE runs -> execution through public services."""

    principal = principal or legacy_principal_context(AuthMode.DISABLED)

    def uow_factory() -> UnitOfWork:
        return SqlAlchemyUnitOfWork(database.session_factory)

    creation_service = AnalysisCreationService(uow_factory=uow_factory, file_store=file_store)
    with ExitStack() as stack:
        uploads = [
            AnalysisUpload(
                filename=filename,
                stream=stack.enter_context((parameters.image_dir / filename).open("rb")),
            )
            for filename in TEST_FILENAMES
        ]
        job = creation_service.create_analysis(
            CreateAnalysisMetadata(
                job_name="Large U-Net independent full-analysis smoke",
                images=[
                    ImageMetadataInput(
                        filename=filename,
                        sample_id=Path(filename).stem,
                        scale=ScaleInput(
                            mode=ScaleMode.NM_PER_PIXEL,
                            value=calibrated.scale_nm_per_pixel,
                        ),
                    )
                    for filename in TEST_FILENAMES
                ],
            ),
            uploads,
            principal=principal,
        )
    images_by_name = {image.filename: image for image in job.images}
    if tuple(images_by_name) != TEST_FILENAMES:
        raise RuntimeError("created Analysis image order differs from fixed test order")
    dimension_mismatches = {
        filename: [image.width, image.height]
        for filename, image in images_by_name.items()
        if (image.width, image.height) != (IMAGE_WIDTH, IMAGE_HEIGHT)
    }
    if dimension_mismatches:
        raise RuntimeError(
            "Large acceptance images are not exactly 2048x1536: "
            + json.dumps(dimension_mismatches, sort_keys=True)
        )
    service = AnalysisApplicationService(
        uow_factory=uow_factory,
        file_store=file_store,
        inference_gateway=gateway,
        postprocess_config=calibrated.postprocess_profile(),
        morphometry_config=calibrated.morphometry_config(),
    )
    run_ids = service.create_runs(
        job.job.job_id,
        CreateRunsRequest(
            image_ids=[images_by_name[name].image_id for name in TEST_FILENAMES],
            model_ids=[MODEL_ID],
            roi_mode=RoiMode.FULL_IMAGE,
            inference=InferenceOptions(
                threshold=calibrated.threshold,
                min_area_px=calibrated.min_area_px,
                watershed_enabled=calibrated.watershed_enabled,
                exclude_border=calibrated.exclude_border,
                device=DEVICE,
                seed=SEED,
            ),
        ),
        principal=principal,
    )
    if len(run_ids) != len(TEST_FILENAMES):
        raise RuntimeError(f"expected three runs, got {len(run_ids)}")

    run_results: list[dict[str, Any]] = []
    model_identity: dict[str, Any] | None = None
    for filename, run_id in zip(TEST_FILENAMES, run_ids, strict=True):
        image = images_by_name[filename]
        completed = service.execute_run(run_id)
        if completed.summary is None or completed.quality is None or completed.execution is None:
            raise RuntimeError(f"completed run is missing acceptance results: {filename}")
        with uow_factory() as uow:
            relative_paths = uow.repositories.runs.get_artifact_paths(completed.run_id)
        if set(relative_paths) != REQUIRED_ARTIFACT_KEYS:
            raise RuntimeError(
                "Large acceptance run does not expose the exact canonical artifact set: "
                f"missing={sorted(REQUIRED_ARTIFACT_KEYS - set(relative_paths))}, "
                f"unexpected={sorted(set(relative_paths) - REQUIRED_ARTIFACT_KEYS)}"
            )
        artifacts = _absolute_artifact_paths(file_store, relative_paths)
        missing_artifacts = sorted(key for key, value in artifacts.items() if value is None)
        if missing_artifacts:
            raise RuntimeError(f"Large acceptance artifacts are missing: {missing_artifacts}")
        configuration = completed.configuration
        postprocess = configuration.resolved_postprocess
        morphometry = configuration.resolved_morphometry
        if postprocess is None or morphometry is None:
            raise RuntimeError("run did not freeze resolved scientific configuration")
        if postprocess != calibrated.postprocess_profile():
            raise RuntimeError("resolved postprocess differs from calibrated_analysis")
        if morphometry != calibrated.morphometry_config():
            raise RuntimeError("resolved morphometry differs from calibrated_analysis")
        run_config_path = Path(str(artifacts["run_config_path"]))
        execution_path = Path(str(artifacts["execution_provenance_path"]))
        _validate_frozen_run(
            configuration,
            completed.execution,
            run_config_path=run_config_path,
            execution_path=execution_path,
        )
        _validate_mask_bottom(Path(str(artifacts["pred_mask_path"])))
        bottom = _bottom_exclusion_evidence(
            configuration,
            width=image.width,
            height=image.height,
            bottom_crop_px=calibrated.bottom_crop_px,
        )
        if not bottom["matching_model_bottom_region_present"]:
            raise RuntimeError(f"bottom exclusion is missing for {filename}")
        if completed.summary.roi_area_px != bottom["expected_effective_roi_area_px"]:
            raise RuntimeError(f"scientific ROI area includes invalid bottom pixels: {filename}")
        particles_csv_value = artifacts.get("particles_csv_path")
        if particles_csv_value is None:
            raise RuntimeError(f"particles CSV is missing: {filename}")
        density = _density_evidence(
            particles_csv=Path(particles_csv_value),
            particle_count=completed.summary.particle_count,
            roi_area_px=completed.summary.roi_area_px,
            scale_nm_per_pixel=calibrated.scale_nm_per_pixel,
            coverage_ratio=completed.summary.coverage_ratio,
            number_density_um2=completed.summary.number_density_um2,
            perimeter_density_um=completed.summary.perimeter_density_um,
            valid_height=image.height - calibrated.bottom_crop_px,
        )
        canonical_artifact_identities = _validate_canonical_chain(
            artifacts=artifacts,
            configuration=configuration,
            width=image.width,
            height=image.height,
            sample_id=Path(filename).stem,
            run_id=completed.run_id,
            output_root=parameters.output_root,
            file_store=file_store,
        )
        current_identity = {
            "model_id": configuration.model_id,
            "version": configuration.model_version,
            "weight_sha256": configuration.weight_sha256,
            "config_sha256": configuration.config_sha256,
            "model_card_sha256": configuration.model_card_sha256,
            "adapter_sha256": configuration.adapter_sha256,
        }
        if model_identity is None:
            model_identity = current_identity
        elif current_identity != model_identity:
            raise RuntimeError("model artifact identity differs between test runs")
        run_results.append(
            {
                "filename": filename,
                "sample_id": Path(filename).stem,
                "job_id": completed.job_id,
                "image_id": completed.image_id,
                "run_id": completed.run_id,
                "final_status": completed.status.value,
                "status_history": [
                    event.model_dump(mode="json") for event in completed.status_history
                ],
                "frozen_inference": configuration.inference.model_dump(mode="json"),
                "resolved_postprocess": postprocess.model_dump(mode="json"),
                "resolved_morphometry": morphometry.model_dump(mode="json"),
                "scale_nm_per_pixel": configuration.scale_nm_per_pixel,
                "roi": {
                    "image_width_px": image.width,
                    "image_height_px": image.height,
                    "image_area_px": image.width * image.height,
                    "effective_roi_area_px": completed.summary.roi_area_px,
                    "analysis_roi": configuration.analysis_roi.model_dump(mode="json"),
                    "bottom_exclusion": bottom,
                    "density_consistency": density,
                },
                "scientific_results": {
                    "particle_count": completed.summary.particle_count,
                    "mean_equivalent_diameter_nm": (completed.summary.mean_equivalent_diameter_nm),
                    "number_density_um2": completed.summary.number_density_um2,
                    "perimeter_density_um": completed.summary.perimeter_density_um,
                    "coverage_ratio": completed.summary.coverage_ratio,
                },
                "quality": {
                    "status": completed.quality.status.value,
                    "reasons": completed.quality.reasons,
                    "metrics": completed.quality.metrics,
                    "recommendations": completed.quality.recommendations,
                    "warnings": {
                        "configuration": configuration.provenance_warnings,
                        "execution": completed.execution.warnings,
                    },
                },
                "artifacts": artifacts,
                "canonical_artifact_identities": canonical_artifact_identities,
            }
        )
    exported = ReportWriter(file_store).build_job_export(job.job.job_id, run_ids=set(run_ids))
    export_manifest = validate_export_zip(
        exported.path.read_bytes(),
        expected_job_id=job.job.job_id,
        expected_run_ids=set(run_ids),
    )
    _validate_export_canonical_artifacts(
        export_manifest,
        runs=run_results,
        file_store=file_store,
        job_id=job.job.job_id,
    )
    return {
        "job_id": job.job.job_id,
        "test_scope": "three held-out fields of view; no test masks were read",
        "test_images": list(TEST_FILENAMES),
        "model": model_identity,
        "calibrated_analysis": {
            **asdict(calibrated),
            "scale_expression": "100/184 nm_per_pixel",
        },
        "runs": run_results,
        "export_path": str(exported.path),
        "export_sha256": file_store.calculate_sha256(exported.path),
        "export_manifest": export_manifest,
    }


def run_smoke(parameters: SmokeParameters) -> dict[str, Any]:
    output_root = _validate_output_root(parameters.output_root)
    snapshot_root = output_root / "model-snapshots"
    registry = ModelRegistryService(parameters.registry, snapshot_root=snapshot_root)
    if registry.registry_error is not None:
        raise RuntimeError(f"private registry is invalid: {registry.registry_error}")
    registration = registry.get_registration(MODEL_ID)
    metadata = registration.metadata
    if metadata.status != ModelStatus.READY:
        raise RuntimeError(
            f"private registry model is not ready: {MODEL_ID} "
            f"({metadata.status.value}: {metadata.health_error})"
        )
    _validate_model_metadata(metadata)
    calibrated = load_calibrated_analysis(registration.config, metadata)

    output_root.mkdir(parents=True, exist_ok=False)
    artifact_root = output_root / "artifacts"
    database = Database(
        Settings(
            app_env="test",
            database_url=_sqlite_url(output_root / "analysis.sqlite3"),
            output_root=artifact_root,
            model_registry_path=parameters.registry,
            model_snapshot_root=snapshot_root,
            model_device=DEVICE.value,
        )
    )
    try:
        Base.metadata.create_all(database.engine)
        with database.session() as session:
            sync_model_registry(session, registry)
        maximum_upload = max(
            (parameters.image_dir / filename).stat().st_size for filename in TEST_FILENAMES
        )
        return execute_analyses(
            parameters,
            calibrated,
            database=database,
            file_store=LocalFileStore(
                StoragePaths(artifact_root),
                max_upload_bytes=max(1, maximum_upload),
            ),
            gateway=InferenceGateway(registry),
        )
    finally:
        database.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        parameters = _validated_parameters(build_parser().parse_args(argv))
        result = run_smoke(parameters)
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

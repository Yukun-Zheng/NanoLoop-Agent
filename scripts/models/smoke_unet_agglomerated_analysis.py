"""Run one fixed agglomerated U-Net validation view through the full Analysis chain."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml  # type: ignore[import-untyped]
from PIL import Image

from app.analysis.application import (
    AnalysisApplicationService,
    AnalysisCreationService,
    AnalysisUpload,
    InferenceGatewayProtocol,
)
from app.analysis.config import MorphometryConfig, PostprocessProfile
from app.contracts.analyses import (
    CreateAnalysisMetadata,
    CreateRunsRequest,
    ImageMetadataInput,
    InferenceOptions,
    ScaleInput,
)
from app.contracts.enums import (
    DevicePreference,
    JobStatus,
    ModelStatus,
    QualityStatus,
    RoiMode,
    ScaleMode,
)
from app.contracts.identity import AuthMode, PrincipalContext
from app.contracts.models import ModelMetadata
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
from scripts.models.calibrate_unet_agglomerated_min_area import (
    CHECKPOINT_SHA256,
    VALIDATION_FILENAMES,
    _load_threshold_evidence,
    _probability_paths,
)

MODEL_ID = "unet-agglomerated-specialized-v1"
TORCHSCRIPT_SHA256 = "d36cf627dc6d1eea83b769743b657e6bb3d9a1af39ddc6a00ae95acc3b20ffc9"
TEST_FILENAMES = ("BiCu-3.tif",)
INDEPENDENT_TEST_FILENAMES = ("YCu-1.tif", "YCu-2.tif", "YCu-3.tif")
EXPECTED_IMAGE_SIZE = (2048, 1536)
THRESHOLD = 0.25
MIN_AREA_PX = 1024
MIN_AREA_NM2 = 302.45746691871454
MIN_AREA_EQUIVALENT_DIAMETER_NM = 19.623985514704565
BOTTOM_CROP_PX = 130
SCALE_NM_PER_PIXEL = 100 / 184
DEVICE = DevicePreference.CPU
SEED = 2026
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CANONICAL_ARTIFACT_KEYS = (
    "pred_mask_path",
    "probability_path",
    "instances_path",
    "particles_csv_path",
    "overlay_path",
    "labeled_particles_path",
    "image_summary_path",
    "quality_report_path",
    "execution_provenance_path",
    "run_config_path",
    "transform_path",
)


@dataclass(frozen=True, slots=True)
class SmokeParameters:
    image_dir: Path
    registry: Path
    output_root: Path
    threshold_evidence: Path
    threshold_evidence_sha256: str
    min_area_evidence: Path
    min_area_evidence_sha256: str
    model_id: str = MODEL_ID


@dataclass(frozen=True, slots=True)
class PromotionEvidence:
    threshold_path: Path
    threshold_sha256: str
    threshold_payload: Mapping[str, Any]
    min_area_path: Path
    min_area_sha256: str
    min_area_payload: Mapping[str, Any]


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_argument(value: str) -> str:
    normalized = value.strip().lower()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return normalized


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _load_promotion_evidence(parameters: SmokeParameters) -> PromotionEvidence:
    if parameters.threshold_evidence.name != "threshold-calibration.json":
        raise ValueError("threshold evidence must be named threshold-calibration.json")
    if parameters.min_area_evidence.name != "min-area-calibration.json":
        raise ValueError("min-area evidence must be named min-area-calibration.json")
    for label, path, expected_digest in (
        (
            "threshold calibration evidence",
            parameters.threshold_evidence,
            parameters.threshold_evidence_sha256,
        ),
        (
            "min-area calibration evidence",
            parameters.min_area_evidence,
            parameters.min_area_evidence_sha256,
        ),
    ):
        if (
            _SHA256_RE.fullmatch(expected_digest) is None
            or _sha256(path) != expected_digest
        ):
            raise ValueError(f"{label} SHA-256 mismatch")

    threshold_payload = _load_threshold_evidence(
        parameters.threshold_evidence.parent,
        expected_sha256=parameters.threshold_evidence_sha256,
    )
    # Readiness requires the probability identities sealed by the threshold evidence
    # to still resolve to the reviewed bytes.
    _probability_paths(parameters.threshold_evidence.parent, threshold_payload)

    min_area_payload = _load_json(parameters.min_area_evidence, label="min-area evidence")
    contract_expected: dict[str, object] = {
        "schema_version": "1",
        "model_id": MODEL_ID,
        "torchscript_sha256": TORCHSCRIPT_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "validation_images": list(VALIDATION_FILENAMES),
    }
    mismatches: dict[str, dict[str, object]] = {
        key: {"expected": value, "observed": min_area_payload.get(key)}
        for key, value in contract_expected.items()
        if min_area_payload.get(key) != value
    }
    selected = min_area_payload.get("selected")
    if not isinstance(selected, Mapping) or selected.get("min_area_px") != MIN_AREA_PX:
        mismatches["selected.min_area_px"] = {
            "expected": MIN_AREA_PX,
            "observed": selected.get("min_area_px") if isinstance(selected, Mapping) else None,
        }
    fixed = min_area_payload.get("fixed_parameters")
    fixed_expected = {
        "threshold": THRESHOLD,
        "threshold_comparison": "gte",
        "bottom_crop_px": BOTTOM_CROP_PX,
        "watershed_enabled": False,
        "fill_holes": True,
        "exclude_border": True,
        "connectivity": 2,
        "perimeter_neighborhood": 8,
    }
    if not isinstance(fixed, Mapping):
        mismatches["fixed_parameters"] = {"expected": "mapping", "observed": fixed}
    else:
        for key, value in fixed_expected.items():
            if fixed.get(key) != value:
                mismatches[f"fixed_parameters.{key}"] = {
                    "expected": value,
                    "observed": fixed.get(key),
                }
    threshold_ref = min_area_payload.get("threshold_calibration_evidence")
    if (
        not isinstance(threshold_ref, Mapping)
        or threshold_ref.get("sha256") != parameters.threshold_evidence_sha256
        or threshold_ref.get("selected_threshold") != THRESHOLD
    ):
        mismatches["threshold_calibration_evidence"] = {
            "expected_sha256": parameters.threshold_evidence_sha256,
            "observed": threshold_ref,
        }
    if mismatches:
        raise ValueError(
            "min-area evidence does not match the frozen agglomerated contract: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    return PromotionEvidence(
        threshold_path=parameters.threshold_evidence,
        threshold_sha256=parameters.threshold_evidence_sha256,
        threshold_payload=threshold_payload,
        min_area_path=parameters.min_area_evidence,
        min_area_sha256=parameters.min_area_evidence_sha256,
        min_area_payload=min_area_payload,
    )


def _validate_output_root(output_root: Path, *, repository: Path | None = None) -> Path:
    resolved = output_root.expanduser().resolve(strict=False)
    repository = (repository or _repository_root()).resolve(strict=True)
    if resolved.exists():
        raise ValueError(f"output-root already exists: {resolved}")
    if resolved == repository or resolved.is_relative_to(repository):
        raise ValueError("output-root must be outside the repository")
    return resolved


def _validated_parameters(namespace: argparse.Namespace) -> SmokeParameters:
    if namespace.model_id != MODEL_ID:
        raise ValueError(f"model-id must be the frozen Agglomerated model: {MODEL_ID}")
    image_dir = namespace.image_dir.expanduser().resolve(strict=True)
    registry = namespace.registry.expanduser().resolve(strict=True)
    threshold_evidence = namespace.threshold_evidence.expanduser().resolve(strict=True)
    min_area_evidence = namespace.min_area_evidence.expanduser().resolve(strict=True)
    output_root = _validate_output_root(namespace.output_root)
    if not image_dir.is_dir():
        raise ValueError(f"image-dir is not a directory: {image_dir}")
    if not registry.is_file():
        raise ValueError(f"registry is not a file: {registry}")
    if not threshold_evidence.is_file() or not min_area_evidence.is_file():
        raise ValueError("threshold and min-area evidence must both be existing files")
    public_registry = (_repository_root() / "model_artifacts" / "registry.yaml").resolve()
    if registry == public_registry:
        raise ValueError("--registry must be a repository-external private registry")
    forbidden_parts = {"test_images", "test_mask_human"}
    if any(part.casefold() in forbidden_parts for part in image_dir.parts):
        raise ValueError("image-dir must not point to an independent test directory")
    for filename in TEST_FILENAMES:
        image_path = image_dir / filename
        if not image_path.is_file():
            raise ValueError(f"agglomerated validation image is missing: {filename}")
        with Image.open(image_path) as image:
            if image.size != EXPECTED_IMAGE_SIZE:
                raise ValueError(f"validation image must be 2048x1536: {filename}")
    return SmokeParameters(
        image_dir=image_dir,
        registry=registry,
        output_root=output_root,
        threshold_evidence=threshold_evidence,
        threshold_evidence_sha256=namespace.threshold_evidence_sha256,
        min_area_evidence=min_area_evidence,
        min_area_evidence_sha256=namespace.min_area_evidence_sha256,
        model_id=namespace.model_id,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--threshold-evidence", required=True, type=Path)
    parser.add_argument("--threshold-evidence-sha256", required=True, type=_sha256_argument)
    parser.add_argument("--min-area-evidence", required=True, type=Path)
    parser.add_argument("--min-area-evidence-sha256", required=True, type=_sha256_argument)
    parser.add_argument("--model-id", default=MODEL_ID)
    return parser


def _ready_smoke_registry_entry(
    registry_path: Path,
    evidence: PromotionEvidence,
) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError("private preflight registry is invalid") from error
    entries = raw.get("models") if isinstance(raw, Mapping) else None
    if not isinstance(entries, list):
        raise ValueError("private preflight registry must contain a models list")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, Mapping)
        and isinstance(entry.get("metadata"), Mapping)
        and entry["metadata"].get("model_id") == MODEL_ID
    ]
    if len(matches) != 1:
        raise ValueError("private preflight registry must contain exactly one agglomerated entry")
    source = deepcopy(dict(matches[0]))
    metadata_raw = source.get("metadata")
    if not isinstance(metadata_raw, dict) or metadata_raw.get("status") != "unavailable":
        raise ValueError("private preflight registry must remain unavailable before the smoke")

    preflight = ModelRegistryService(registry_path)
    if preflight.registry_error is not None:
        raise ValueError(f"private preflight registry is invalid: {preflight.registry_error}")
    registration = preflight.get_registration(MODEL_ID)
    metadata = registration.metadata
    if metadata.status != ModelStatus.UNAVAILABLE:
        raise ValueError("private preflight registry unexpectedly became ready")
    if registration.weight_sha256 != TORCHSCRIPT_SHA256:
        raise ValueError("private preflight registry has the wrong TorchScript SHA-256")
    required_assets = {
        "weight": registration.weight_path,
        "config": registration.config_path,
        "model_card": registration.model_card_path,
        "adapter_source": registration.adapter_source_path,
    }
    missing = [name for name, path in required_assets.items() if path is None or not path.is_file()]
    if missing or registration.config_sha256 is None or registration.model_card_sha256 is None:
        raise ValueError(f"private preflight registry is missing validated assets: {missing}")
    load_calibrated_analysis(registration.config, metadata)

    metadata_raw.update(
        {
            "status": "ready",
            "default_threshold": THRESHOLD,
            "inference_invalid_bottom_px": BOTTOM_CROP_PX,
            "health_error": None,
            "notes": "Temporary smoke-only execution manifest; persistent readiness pending.",
        }
    )
    source.update(
        {
            "weight_path": str(registration.weight_path),
            "weight_sha256": TORCHSCRIPT_SHA256,
            "config_path": str(registration.config_path),
            "model_card_path": str(registration.model_card_path),
            "promotion_evidence": {
                "schema_version": 1,
                "threshold_calibration": {
                    "path": str(evidence.threshold_path),
                    "sha256": evidence.threshold_sha256,
                },
                "min_area_calibration": {
                    "path": str(evidence.min_area_path),
                    "sha256": evidence.min_area_sha256,
                },
            },
        }
    )
    return source


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
        raise ValueError("agglomerated model config is missing calibrated_analysis")
    comparison = raw.get("threshold_comparison")
    if comparison != "gte":
        raise ValueError("calibrated_analysis requires threshold_comparison=gte")
    calibrated = CalibratedAnalysis(
        postprocess_profile_id=metadata.postprocess_profile,
        threshold=_require_float(raw, "threshold"),
        threshold_comparison=comparison,
        min_area_px=_require_int(raw, "min_area_px"),
        min_area_nm2=_require_float(raw, "min_area_nm2"),
        min_area_equivalent_diameter_nm=_require_float(
            raw, "min_area_equivalent_diameter_nm"
        ),
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
    expected_area = calibrated.min_area_px * calibrated.scale_nm_per_pixel**2
    expected_diameter = (
        2.0
        * math.sqrt(calibrated.min_area_px / math.pi)
        * calibrated.scale_nm_per_pixel
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
    expected = CalibratedAnalysis(
        postprocess_profile_id=metadata.postprocess_profile,
        threshold=THRESHOLD,
        threshold_comparison="gte",
        min_area_px=MIN_AREA_PX,
        min_area_nm2=MIN_AREA_NM2,
        min_area_equivalent_diameter_nm=MIN_AREA_EQUIVALENT_DIAMETER_NM,
        watershed_enabled=False,
        fill_holes=True,
        exclude_border=True,
        connectivity=2,
        perimeter_neighborhood=8,
        bottom_crop_px=BOTTOM_CROP_PX,
        scale_nm_per_pixel=SCALE_NM_PER_PIXEL,
    )
    if calibrated != expected:
        raise ValueError("calibrated_analysis differs from the frozen agglomerated contract")
    return calibrated


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


def _canonical_artifact_identities(
    artifacts: Mapping[str, str | None],
    *,
    output_root: Path,
) -> dict[str, dict[str, str]]:
    resolved_root = output_root.resolve(strict=True)
    identities: dict[str, dict[str, str]] = {}
    for key in CANONICAL_ARTIFACT_KEYS:
        raw_path = artifacts.get(key)
        if raw_path is None:
            raise RuntimeError(f"completed Analysis is missing canonical artifact: {key}")
        path = Path(raw_path).resolve(strict=True)
        try:
            relative = path.relative_to(resolved_root)
        except ValueError as error:
            raise RuntimeError(f"canonical artifact escaped the smoke output: {key}") from error
        identities[key] = {"path": relative.as_posix(), "sha256": _sha256(path)}
    return identities


def _validate_schema3_execution(
    completed: Any,
    artifacts: Mapping[str, str | None],
) -> None:
    if completed.status not in {JobStatus.COMPLETED, JobStatus.COMPLETED_WITH_WARNINGS}:
        raise RuntimeError(f"smoke run did not finish successfully: {completed.status.value}")
    if completed.quality is None or completed.quality.status not in {
        QualityStatus.PASS,
        QualityStatus.WARN,
    }:
        raise RuntimeError("smoke run did not produce an accepted quality status")
    configuration = completed.configuration
    bundle = configuration.model_bundle
    if (
        configuration.schema_version != 3
        or configuration.provenance_status != "complete"
        or configuration.provenance_warnings
        or bundle is None
        or configuration.adapter_sha256 != bundle.adapter_sha256
    ):
        raise RuntimeError("smoke run is missing complete schema-v3 model-bundle provenance")
    execution = completed.execution
    if (
        execution is None
        or not execution.build_identity_matches_contract
        or execution.executor_build != configuration.execution_build
        or execution.requested_device != DEVICE
        or execution.actual_device != DEVICE.value
        or execution.seed != SEED
        or not execution.python_random_seeded
        or not execution.numpy_random_seeded
        or not execution.torch_deterministic_algorithms
        or not execution.global_inference_serialized
        or not execution.backend.endswith(".UNetAdapter")
        or execution.model_bundle_id != bundle.bundle_id
        or execution.adapter_sha256 != configuration.adapter_sha256
        or execution.warnings
    ):
        raise RuntimeError("smoke run execution provenance is incomplete or inconsistent")

    run_config_path = artifacts.get("run_config_path")
    execution_path = artifacts.get("execution_provenance_path")
    if run_config_path is None or execution_path is None:
        raise RuntimeError("schema-v3 run evidence files are missing")
    run_payload = _load_json(Path(run_config_path), label="run configuration")
    execution_payload = _load_json(Path(execution_path), label="execution provenance")
    bundle_payload = run_payload.get("model_bundle")
    if (
        run_payload.get("contract_schema_version") != 3
        or run_payload.get("provenance_status") != "complete"
        or run_payload.get("provenance_warnings") != []
        or not isinstance(bundle_payload, Mapping)
        or bundle_payload.get("bundle_id") != bundle.bundle_id
        or run_payload.get("image_sha256") != configuration.image_sha256
    ):
        raise RuntimeError("persisted run configuration is not the frozen schema-v3 contract")
    if (
        execution_payload.get("contract_schema_version") != 1
        or execution_payload.get("model_bundle_id") != bundle.bundle_id
        or execution_payload.get("adapter_sha256") != configuration.adapter_sha256
        or execution_payload.get("warnings") != []
        or execution_payload.get("build_identity_matches_contract") is not True
    ):
        raise RuntimeError("persisted execution provenance differs from the schema-v3 contract")


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
    expected_perimeter_density = (
        perimeter_total_px * scale_nm_per_pixel / 1000.0 / roi_area_um2
    )
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


def _mask_bottom_evidence(
    mask_path: Path,
    *,
    width: int,
    height: int,
    bottom_crop_px: int,
) -> dict[str, Any]:
    with Image.open(mask_path) as image:
        if image.size != (width, height):
            raise RuntimeError("Analysis prediction mask dimensions differ from the input image")
        pixels = np.asarray(image)
    if pixels.ndim == 2:
        foreground = pixels != 0
    elif pixels.ndim == 3:
        foreground = np.any(pixels != 0, axis=2)
    else:
        raise RuntimeError("Analysis prediction mask has unsupported dimensions")
    bottom_nonzero = int(np.count_nonzero(foreground[-bottom_crop_px:]))
    if bottom_nonzero:
        raise RuntimeError(f"prediction mask contains {bottom_nonzero} bottom-bar pixels")
    return {
        "bottom_rows_checked": bottom_crop_px,
        "bottom_nonzero_pixels": bottom_nonzero,
        "bottom_prediction_is_zero": True,
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
    """Execute creation -> one FULL_IMAGE run -> execution through public services."""

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
                job_name="Agglomerated U-Net validation full-analysis smoke",
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
        raise RuntimeError("created Analysis image differs from the fixed validation input")
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
            model_ids=[parameters.model_id],
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
        raise RuntimeError(f"expected one run, got {len(run_ids)}")

    run_results: list[dict[str, Any]] = []
    model_identity: dict[str, Any] | None = None
    for filename, run_id in zip(TEST_FILENAMES, run_ids, strict=True):
        image = images_by_name[filename]
        completed = service.execute_run(run_id)
        if completed.summary is None or completed.quality is None or completed.execution is None:
            raise RuntimeError(f"completed run is missing acceptance results: {filename}")
        with uow_factory() as uow:
            relative_paths = uow.repositories.runs.get_artifact_paths(completed.run_id)
        artifacts = _absolute_artifact_paths(file_store, relative_paths)
        missing_artifacts = [
            name for name in CANONICAL_ARTIFACT_KEYS if artifacts.get(name) is None
        ]
        if missing_artifacts:
            raise RuntimeError(f"completed Analysis is missing artifacts: {missing_artifacts}")
        _validate_schema3_execution(completed, artifacts)
        artifact_identities = _canonical_artifact_identities(
            artifacts,
            output_root=parameters.output_root,
        )
        configuration = completed.configuration
        postprocess = configuration.resolved_postprocess
        morphometry = configuration.resolved_morphometry
        if postprocess is None or morphometry is None:
            raise RuntimeError("run did not freeze resolved scientific configuration")
        if postprocess != calibrated.postprocess_profile():
            raise RuntimeError("resolved postprocess differs from calibrated_analysis")
        if morphometry != calibrated.morphometry_config():
            raise RuntimeError("resolved morphometry differs from calibrated_analysis")
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
        mask_path_value = artifacts.get("pred_mask_path")
        if mask_path_value is None:
            raise RuntimeError(f"prediction mask is missing: {filename}")
        mask_bottom = _mask_bottom_evidence(
            Path(mask_path_value),
            width=image.width,
            height=image.height,
            bottom_crop_px=calibrated.bottom_crop_px,
        )
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
        bundle = configuration.model_bundle
        if bundle is None:
            raise RuntimeError("schema-v3 model bundle unexpectedly disappeared")
        current_identity = {
            "model_id": configuration.model_id,
            "version": configuration.model_version,
            "weight_sha256": configuration.weight_sha256,
            "config_sha256": configuration.config_sha256,
            "model_card_sha256": configuration.model_card_sha256,
            "adapter_sha256": configuration.adapter_sha256,
            "model_bundle_id": bundle.bundle_id,
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
                    "prediction_bottom_check": mask_bottom,
                    "density_consistency": density,
                },
                "scientific_results": {
                    "particle_count": completed.summary.particle_count,
                    "mean_equivalent_diameter_nm": (
                        completed.summary.mean_equivalent_diameter_nm
                    ),
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
                "artifacts": {
                    "mask": artifacts.get("pred_mask_path"),
                    "instances": artifacts.get("instances_path"),
                    "particles_csv": particles_csv_value,
                    "overlay": artifacts.get("overlay_path"),
                    "labeled_particles": artifacts.get("labeled_particles_path"),
                    "report": artifacts.get("image_summary_path"),
                    "quality_report": artifacts.get("quality_report_path"),
                    "execution_evidence": artifacts.get("execution_provenance_path"),
                    "run_configuration": artifacts.get("run_config_path"),
                    "transform": artifacts.get("transform_path"),
                    "probability": artifacts.get("probability_path"),
                },
                "canonical_artifact_identities": artifact_identities,
            }
        )
    return {
        "job_id": job.job.job_id,
        "test_scope": (
            "one calibrated validation field of view; no masks or YCu test data were read"
        ),
        "validation_images": list(TEST_FILENAMES),
        "model": model_identity,
        "calibrated_analysis": {
            **asdict(calibrated),
            "scale_expression": "100/184 nm_per_pixel",
        },
        "runs": run_results,
        "private_registry_ready_eligible": True,
        "readiness_scope": (
            "successful full Analysis smoke for the exact private asset; public registry unchanged"
        ),
    }


def run_smoke(parameters: SmokeParameters) -> dict[str, Any]:
    output_root = _validate_output_root(parameters.output_root)
    snapshot_root = output_root / "model-snapshots"
    promotion_evidence = _load_promotion_evidence(parameters)
    ready_entry = _ready_smoke_registry_entry(parameters.registry, promotion_evidence)
    ready_payload: dict[str, Any] = {"schema_version": "2.0", "models": [ready_entry]}
    with tempfile.TemporaryDirectory(prefix="nanoloop-agglomerated-smoke-") as temporary:
        smoke_registry_path = Path(temporary) / "smoke-registry.yaml"
        smoke_registry_path.write_text(
            yaml.safe_dump(ready_payload, sort_keys=False), encoding="utf-8"
        )
        registry = ModelRegistryService(smoke_registry_path, snapshot_root=snapshot_root)
        if registry.registry_error is not None:
            raise RuntimeError(f"smoke-only registry is invalid: {registry.registry_error}")
        registration = registry.get_registration(parameters.model_id)
        metadata = registration.metadata
        if metadata.status != ModelStatus.READY:
            raise RuntimeError(
                f"smoke-only registry model is not ready: {parameters.model_id} "
                f"({metadata.status.value}: {metadata.health_error})"
            )
        if metadata.weight_sha256 != TORCHSCRIPT_SHA256:
            raise RuntimeError("smoke-only registry has an unexpected TorchScript asset")
        calibrated = load_calibrated_analysis(registration.config, metadata)

        output_root.mkdir(parents=True, exist_ok=False)
        artifact_root = output_root / "artifacts"
        database = Database(
            Settings(
                app_env="test",
                database_url=_sqlite_url(output_root / "analysis.sqlite3"),
                output_root=artifact_root,
                model_registry_path=smoke_registry_path,
                model_snapshot_root=snapshot_root,
                model_device=DEVICE.value,
            )
        )
        try:
            Base.metadata.create_all(database.engine)
            with database.session() as session:
                sync_model_registry(session, registry)
            maximum_upload = max(
                (parameters.image_dir / filename).stat().st_size
                for filename in TEST_FILENAMES
            )
            result = execute_analyses(
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

    evidence_dir = output_root / "readiness-evidence"
    evidence_dir.mkdir(exist_ok=False)
    threshold_copy = evidence_dir / "threshold-calibration.json"
    min_area_copy = evidence_dir / "min-area-calibration.json"
    shutil.copyfile(promotion_evidence.threshold_path, threshold_copy)
    shutil.copyfile(promotion_evidence.min_area_path, min_area_copy)
    if (
        _sha256(threshold_copy) != promotion_evidence.threshold_sha256
        or _sha256(min_area_copy) != promotion_evidence.min_area_sha256
    ):
        raise RuntimeError("copied promotion evidence identity changed")
    smoke_evidence_path = evidence_dir / "analysis-smoke-evidence.json"
    smoke_evidence_payload = {
        "schema_version": "1",
        "model_id": MODEL_ID,
        "result": result,
    }
    smoke_evidence_path.write_text(
        json.dumps(smoke_evidence_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    persistent_payload: dict[str, Any] = deepcopy(ready_payload)
    persistent_payload["models"][0]["metadata"]["notes"] = (
        "Exact private asset passed evidence-gated Agglomerated schema-v3 Analysis smoke."
    )
    persistent_payload["models"][0]["promotion_evidence"] = {
        "schema_version": 1,
        "threshold_calibration": {
            "path": threshold_copy.relative_to(output_root).as_posix(),
            "sha256": promotion_evidence.threshold_sha256,
        },
        "min_area_calibration": {
            "path": min_area_copy.relative_to(output_root).as_posix(),
            "sha256": promotion_evidence.min_area_sha256,
        },
        "analysis_smoke": {
            "path": smoke_evidence_path.relative_to(output_root).as_posix(),
            "sha256": _sha256(smoke_evidence_path),
        },
    }
    ready_registry_path = output_root / "private-registry-ready.yaml"
    ready_registry_path.write_text(
        yaml.safe_dump(persistent_payload, sort_keys=False), encoding="utf-8"
    )
    result["private_registry_ready_manifest"] = str(ready_registry_path)
    result["promotion_evidence"] = persistent_payload["models"][0]["promotion_evidence"]
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        parameters = _validated_parameters(build_parser().parse_args(argv))
        result = run_smoke(parameters)
        report_path = parameters.output_root / "agglomerated-analysis-smoke.json"
        result["smoke_report"] = str(report_path)
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
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

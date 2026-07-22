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
from app.contracts.enums import DevicePreference, ModelStatus, RoiMode, ScaleMode
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

MODEL_ID = "unet-large-optimized-v1"
TORCHSCRIPT_SHA256 = "007d9a16bf31e5f960160c52eefa938b83feeac2e6c0d7dec9c8670a38626e05"
TEST_FILENAMES = ("SrZr-3.tif", "BaCu-2.tif", "PrCu-3.tif")
DEVICE = DevicePreference.CPU
SEED = 2026


@dataclass(frozen=True, slots=True)
class SmokeParameters:
    image_dir: Path
    registry: Path
    output_root: Path
    model_id: str = MODEL_ID


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
        model_id=namespace.model_id,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model-id", default=MODEL_ID)
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
        artifacts = _absolute_artifact_paths(file_store, relative_paths)
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
            }
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
    }


def run_smoke(parameters: SmokeParameters) -> dict[str, Any]:
    output_root = _validate_output_root(parameters.output_root)
    snapshot_root = output_root / "model-snapshots"
    registry = ModelRegistryService(parameters.registry, snapshot_root=snapshot_root)
    if registry.registry_error is not None:
        raise RuntimeError(f"private registry is invalid: {registry.registry_error}")
    registration = registry.get_registration(parameters.model_id)
    metadata = registration.metadata
    if metadata.status != ModelStatus.READY:
        raise RuntimeError(
            f"private registry model is not ready: {parameters.model_id} "
            f"({metadata.status.value}: {metadata.health_error})"
        )
    if metadata.weight_sha256 != TORCHSCRIPT_SHA256:
        raise RuntimeError("private registry points to an unexpected Large TorchScript asset")
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

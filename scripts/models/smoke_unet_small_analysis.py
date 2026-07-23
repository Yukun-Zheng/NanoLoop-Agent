"""Run a Small U-Net engineering acceptance smoke through the public application services.

The smoke verifies the frozen model bundle and Gateway-to-Analysis chain. It is engineering
readiness evidence only and must not be presented as Small-B scientific acceptance.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.analysis.application import (
    AnalysisApplicationService,
    AnalysisCreationService,
    AnalysisUpload,
    InferenceGatewayProtocol,
)
from app.contracts.analyses import (
    CreateAnalysisMetadata,
    CreateRunsRequest,
    ImageMetadataInput,
    InferenceOptions,
    ScaleInput,
)
from app.contracts.enums import DevicePreference, ModelStatus, RoiMode, ScaleMode
from app.contracts.identity import AuthMode, PrincipalContext
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

DEFAULT_MODEL_ID = "unet-small-balanced-v1"
BOTTOM_INFORMATION_BAR_PX = 130


@dataclass(frozen=True, slots=True)
class SmokeParameters:
    image: Path
    registry: Path
    output_root: Path
    scale_nm_per_pixel: float | None
    sample_id: str
    model_id: str = DEFAULT_MODEL_ID
    threshold: float = 0.30
    min_area_px: int = 64
    seed: int = 2026
    device: DevicePreference = DevicePreference.CPU


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("threshold must be between zero and one")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one real full-image small U-Net engineering acceptance smoke. The result is "
            "not scientific-acceptance evidence."
        )
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    scale = parser.add_mutually_exclusive_group(required=True)
    scale.add_argument("--scale-nm-per-pixel", type=_positive_float)
    scale.add_argument(
        "--pixel-only",
        action="store_true",
        help="Run engineering validation without unverified physical-scale claims",
    )
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--threshold", default=0.30, type=_unit_interval)
    parser.add_argument("--min-area-px", default=64, type=_nonnegative_int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument(
        "--device",
        choices=[item.value for item in DevicePreference],
        default=DevicePreference.CPU.value,
    )
    return parser


def _validated_parameters(namespace: argparse.Namespace) -> SmokeParameters:
    repository = _repository_root()
    image = namespace.image.expanduser().resolve(strict=True)
    registry = namespace.registry.expanduser().resolve(strict=True)
    output_root = namespace.output_root.expanduser().resolve(strict=False)
    if not image.is_file():
        raise ValueError(f"image is not a file: {image}")
    if not registry.is_file():
        raise ValueError(f"registry is not a file: {registry}")
    _validate_output_root(output_root, repository=repository)
    if not namespace.sample_id.strip():
        raise ValueError("sample-id must not be empty")
    return SmokeParameters(
        image=image,
        registry=registry,
        output_root=output_root,
        scale_nm_per_pixel=namespace.scale_nm_per_pixel,
        sample_id=namespace.sample_id,
        model_id=namespace.model_id,
        threshold=namespace.threshold,
        min_area_px=namespace.min_area_px,
        seed=namespace.seed,
        device=DevicePreference(namespace.device),
    )


def _validate_output_root(output_root: Path, *, repository: Path | None = None) -> None:
    resolved = output_root.expanduser().resolve(strict=False)
    repository = (repository or _repository_root()).resolve(strict=True)
    if resolved.exists():
        raise ValueError(f"output-root already exists: {resolved}")
    if resolved == repository or resolved.is_relative_to(repository):
        raise ValueError("output-root must be outside the repository")


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


def _bottom_exclusion_evidence(configuration: Any, *, width: int, height: int) -> dict[str, Any]:
    regions = [
        region
        for region in configuration.analysis_roi.invalid_rects
        if region.reason == "model_bottom_information_bar"
    ]
    expected_y1 = height - BOTTOM_INFORMATION_BAR_PX
    exact = [
        region
        for region in regions
        if (region.x1, region.y1, region.x2, region.y2)
        == (0, expected_y1, width, height)
    ]
    return {
        "expected_invalid_bottom_px": BOTTOM_INFORMATION_BAR_PX,
        "expected_bottom_area_px": width * BOTTOM_INFORMATION_BAR_PX,
        "expected_rect": {
            "x1": 0,
            "y1": expected_y1,
            "x2": width,
            "y2": height,
        },
        "matching_model_bottom_region_present": bool(exact),
        "model_bottom_regions": [region.model_dump(mode="json") for region in regions],
    }


def _ready_health(
    gateway: InferenceGatewayProtocol,
    *,
    model_id: str,
    stage: str,
) -> dict[str, Any]:
    health = next((item for item in gateway.health() if item.model_id == model_id), None)
    if health is None:
        raise RuntimeError(
            f"Small U-Net Gateway health is missing: model_id={model_id}, stage={stage}"
        )
    if health.status != ModelStatus.READY:
        raise RuntimeError(
            "Small U-Net Gateway health is not ready: "
            f"model_id={model_id}, stage={stage}, status={health.status.value}, "
            f"error={health.error_summary}"
        )
    return health.model_dump(mode="json")


def execute_analysis(
    parameters: SmokeParameters,
    *,
    database: Database,
    file_store: LocalFileStore,
    gateway: InferenceGatewayProtocol,
    principal: PrincipalContext | None = None,
) -> dict[str, Any]:
    """Execute the formal creation -> run creation -> execution service chain."""

    principal = principal or legacy_principal_context(AuthMode.DISABLED)

    def uow_factory() -> UnitOfWork:
        return SqlAlchemyUnitOfWork(database.session_factory)

    creation_service = AnalysisCreationService(
        uow_factory=uow_factory,
        file_store=file_store,
    )
    with parameters.image.open("rb") as stream:
        job = creation_service.create_analysis(
            CreateAnalysisMetadata(
                job_name=f"small U-Net smoke: {parameters.sample_id}",
                images=[
                    ImageMetadataInput(
                        filename=parameters.image.name,
                        sample_id=parameters.sample_id,
                        scale=ScaleInput(
                            mode=(
                                ScaleMode.NM_PER_PIXEL
                                if parameters.scale_nm_per_pixel is not None
                                else ScaleMode.PIXEL_ONLY
                            ),
                            value=parameters.scale_nm_per_pixel,
                        ),
                    )
                ],
            ),
            [AnalysisUpload(filename=parameters.image.name, stream=stream)],
            principal=principal,
        )
    image = job.images[0]
    service = AnalysisApplicationService(
        uow_factory=uow_factory,
        file_store=file_store,
        inference_gateway=gateway,
    )
    run_ids = service.create_runs(
        job.job.job_id,
        CreateRunsRequest(
            image_ids=[image.image_id],
            model_ids=[parameters.model_id],
            roi_mode=RoiMode.FULL_IMAGE,
            inference=InferenceOptions(
                threshold=parameters.threshold,
                min_area_px=parameters.min_area_px,
                watershed_enabled=False,
                exclude_border=True,
                device=parameters.device,
                seed=parameters.seed,
            ),
        ),
        principal=principal,
    )
    if len(run_ids) != 1:
        raise RuntimeError(f"expected exactly one run, got {len(run_ids)}")
    completed = service.execute_run(run_ids[0])
    if completed.summary is None or completed.quality is None or completed.execution is None:
        raise RuntimeError("completed run is missing required acceptance results")
    with uow_factory() as uow:
        relative_paths = uow.repositories.runs.get_artifact_paths(completed.run_id)
    paths = _absolute_artifact_paths(file_store, relative_paths)
    configuration = completed.configuration
    if configuration.schema_version != 3 or configuration.model_bundle is None:
        raise RuntimeError(
            "Small U-Net smoke requires a frozen schema-v3 model bundle: "
            f"model_id={parameters.model_id}, run_id={completed.run_id}, "
            f"schema_version={configuration.schema_version}"
        )
    postprocess = configuration.resolved_postprocess
    if postprocess is None:
        raise RuntimeError("run configuration did not freeze resolved_postprocess")
    bottom_evidence = _bottom_exclusion_evidence(
        configuration,
        width=image.width,
        height=image.height,
    )
    if not bottom_evidence["matching_model_bottom_region_present"]:
        raise RuntimeError("run configuration did not freeze the expected bottom 130 px exclusion")
    return {
        "evidence_class": "engineering_acceptance",
        "readiness_eligible": True,
        "scientific_acceptance_eligible": False,
        "limitations": [
            "Operator-supplied threshold, min_area_px, scale, and sample are not a frozen "
            "scientific acceptance contract.",
            "Small-B calibration and independent scientific evaluation have not started.",
        ],
        "job_id": completed.job_id,
        "image_id": completed.image_id,
        "run_id": completed.run_id,
        "final_status": completed.status.value,
        "status_history": [
            event.model_dump(mode="json") for event in completed.status_history
        ],
        "model": {
            "model_id": configuration.model_id,
            "version": configuration.model_version,
            "weight_sha256": configuration.weight_sha256,
            "config_sha256": configuration.config_sha256,
            "model_card_sha256": configuration.model_card_sha256,
            "adapter_sha256": configuration.adapter_sha256,
            "adapter_path": configuration.adapter_path,
            "bundle": configuration.model_bundle.model_dump(mode="json"),
        },
        "scale_nm_per_pixel": configuration.scale_nm_per_pixel,
        "frozen_inference": configuration.inference.model_dump(mode="json"),
        "resolved_postprocess": postprocess.model_dump(mode="json"),
        "roi": {
            "image_width_px": image.width,
            "image_height_px": image.height,
            "image_area_px": image.width * image.height,
            "effective_roi_area_px": completed.summary.roi_area_px,
            "analysis_roi": configuration.analysis_roi.model_dump(mode="json"),
            "bottom_exclusion": bottom_evidence,
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
            "mask": paths.get("pred_mask_path"),
            "instances": paths.get("instances_path"),
            "particles_csv": paths.get("particles_csv_path"),
            "overlay": paths.get("overlay_path"),
            "report": paths.get("image_summary_path"),
            "quality_report": paths.get("quality_report_path"),
            "execution_evidence": paths.get("execution_provenance_path"),
            "run_configuration": paths.get("run_config_path"),
            "transform": paths.get("transform_path"),
            "probability": paths.get("probability_path"),
            "labeled_particles": paths.get("labeled_particles_path"),
        },
    }


def run_smoke(parameters: SmokeParameters) -> dict[str, Any]:
    output_root = parameters.output_root.expanduser().resolve(strict=False)
    _validate_output_root(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    artifact_root = output_root / "artifacts"
    database_path = output_root / "analysis.sqlite3"
    snapshot_root = output_root / "model-snapshots"
    settings = Settings(
        app_env="test",
        database_url=_sqlite_url(database_path),
        output_root=artifact_root,
        model_registry_path=parameters.registry,
        model_snapshot_root=snapshot_root,
        model_device=parameters.device.value,
    )
    database = Database(settings)
    try:
        Base.metadata.create_all(database.engine)
        registry = ModelRegistryService(
            parameters.registry,
            snapshot_root=snapshot_root,
        )
        if registry.registry_error is not None:
            raise RuntimeError(f"private registry is invalid: {registry.registry_error}")
        metadata = {item.model_id: item for item in registry.list_models()}.get(
            parameters.model_id
        )
        if metadata is None:
            raise RuntimeError(f"model is absent from private registry: {parameters.model_id}")
        if metadata.status != ModelStatus.READY:
            raise RuntimeError(
                f"private registry model is not ready: {parameters.model_id} "
                f"({metadata.status.value}: {metadata.health_error})"
            )
        if metadata.inference_invalid_bottom_px != BOTTOM_INFORMATION_BAR_PX:
            raise RuntimeError(
                f"private registry must declare inference_invalid_bottom_px="
                f"{BOTTOM_INFORMATION_BAR_PX} for {parameters.model_id}"
            )
        with database.session() as session:
            sync_model_registry(session, registry)
        file_store = LocalFileStore(
            StoragePaths(artifact_root),
            max_upload_bytes=max(1, parameters.image.stat().st_size),
        )
        gateway = InferenceGateway(registry)
        health_before = _ready_health(
            gateway,
            model_id=parameters.model_id,
            stage="before_predict",
        )
        try:
            result = execute_analysis(
                parameters,
                database=database,
                file_store=file_store,
                gateway=gateway,
            )
            health_after_predict = _ready_health(
                gateway,
                model_id=parameters.model_id,
                stage="after_predict",
            )
            gateway.cache.unload(parameters.model_id)
            health_after_unload = _ready_health(
                gateway,
                model_id=parameters.model_id,
                stage="after_unload",
            )
            cached_after_unload = len(gateway.cache.loaded())
            if cached_after_unload != 0:
                raise RuntimeError(
                    "Small U-Net cache is not empty after unload: "
                    f"model_id={parameters.model_id}, cached={cached_after_unload}"
                )
            result["gateway_lifecycle"] = {
                "before_predict": health_before,
                "after_predict": health_after_predict,
                "after_unload": health_after_unload,
                "cached_adapter_count_after_unload": cached_after_unload,
            }
            return result
        finally:
            gateway.cache.unload(parameters.model_id)
    finally:
        database.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        parameters = _validated_parameters(parser.parse_args(argv))
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

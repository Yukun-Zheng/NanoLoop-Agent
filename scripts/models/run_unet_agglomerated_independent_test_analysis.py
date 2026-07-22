"""Generate YCu independent-test predictions through the formal Analysis workflow only.

This is deliberately an inference-generation tool, not an evaluator: it has no
mask/ground-truth input and does not calculate or inspect evaluation metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

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
from scripts.models.smoke_unet_agglomerated_analysis import (
    BOTTOM_CROP_PX,
    DEVICE,
    EXPECTED_IMAGE_SIZE,
    MIN_AREA_PX,
    MODEL_ID,
    PRIVATE_TORCHSCRIPT_PATH,
    SEED,
    THRESHOLD,
    TORCHSCRIPT_SHA256,
    CalibratedAnalysis,
    _mask_bottom_evidence,
    _sqlite_url,
    _validate_output_root,
    load_calibrated_analysis,
)

TEST_FILENAMES = ("YCu-1.tif", "YCu-2.tif", "YCu-3.tif")
EXPECTED_OUTPUT_ROOT = Path(
    "/share/home/tm1045914237210000/a1117256290/GJH/AI4S/"
    "smoke_results/agglomerated-unet-independent-test-v1"
)
REPORT_FILENAME = "agglomerated-independent-test-analysis.json"


@dataclass(frozen=True, slots=True)
class IndependentTestParameters:
    image_dir: Path
    registry: Path
    output_root: Path
    model_id: str = MODEL_ID


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _validate_inputs(image_dir: Path) -> Path:
    resolved = image_dir.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"image-dir is not a directory: {resolved}")
    for filename in TEST_FILENAMES:
        image_path = resolved / filename
        if not image_path.is_file():
            raise ValueError(f"required top-level independent image is missing: {filename}")
        with Image.open(image_path) as image:
            if image.size != EXPECTED_IMAGE_SIZE:
                raise ValueError(f"independent image must be 2048x1536: {filename}")
    return resolved


def _validated_parameters(
    namespace: argparse.Namespace,
    *,
    expected_output_root: Path = EXPECTED_OUTPUT_ROOT,
) -> IndependentTestParameters:
    if namespace.model_id != MODEL_ID:
        raise ValueError(f"model-id must be the frozen Agglomerated model: {MODEL_ID}")
    output_root = _validate_output_root(namespace.output_root)
    if output_root != expected_output_root.expanduser().resolve(strict=False):
        raise ValueError(
            f"output-root must be the fixed independent-test root: {expected_output_root}"
        )
    registry = namespace.registry.expanduser().resolve(strict=True)
    if not registry.is_file():
        raise ValueError(f"registry is not a file: {registry}")
    if registry == (_repository_root() / "model_artifacts" / "registry.yaml").resolve():
        raise ValueError(
            "--registry must be the generated repository-external private ready manifest"
        )
    return IndependentTestParameters(
        image_dir=_validate_inputs(namespace.image_dir),
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


def _ready_private_registry(
    parameters: IndependentTestParameters,
) -> tuple[ModelRegistryService, CalibratedAnalysis]:
    registry = ModelRegistryService(
        parameters.registry,
        snapshot_root=parameters.output_root / "model-snapshots",
    )
    if registry.registry_error is not None:
        raise ValueError(f"private ready registry is invalid: {registry.registry_error}")
    registration = registry.get_registration(parameters.model_id)
    if registration.metadata.status != ModelStatus.READY:
        raise ValueError("private registry model must already be ready after the BiCu-3 smoke")
    if registration.weight_sha256 != TORCHSCRIPT_SHA256:
        raise ValueError("private registry has the wrong TorchScript SHA-256")
    if registration.weight_path.resolve() != PRIVATE_TORCHSCRIPT_PATH.resolve():
        raise ValueError("private registry has the wrong TorchScript path")
    if not registration.weight_path.is_file():
        raise ValueError("private registry TorchScript asset is missing")
    calibrated = load_calibrated_analysis(registration.config, registration.metadata)
    return registry, calibrated


def _model_identity(configuration: Any) -> dict[str, Any]:
    return {
        "model_id": configuration.model_id,
        "version": configuration.model_version,
        "weight_sha256": configuration.weight_sha256,
        "config_sha256": configuration.config_sha256,
        "model_card_sha256": configuration.model_card_sha256,
        "adapter_sha256": configuration.adapter_sha256,
    }


def execute_analyses(
    parameters: IndependentTestParameters,
    calibrated: CalibratedAnalysis,
    *,
    database: Database,
    file_store: LocalFileStore,
    gateway: InferenceGatewayProtocol,
    principal: PrincipalContext | None = None,
) -> dict[str, Any]:
    """Use creation -> FULL_IMAGE runs -> gateway execution for the three YCu fields."""

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
                job_name="Agglomerated U-Net YCu independent-test inference generation",
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
        raise RuntimeError("created Analysis images differ from the fixed independent input")
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
            image_ids=[images_by_name[filename].image_id for filename in TEST_FILENAMES],
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
    if len(run_ids) != len(TEST_FILENAMES) or len(set(run_ids)) != len(TEST_FILENAMES):
        raise RuntimeError(f"expected exactly three distinct FULL_IMAGE runs, got {len(run_ids)}")

    model: dict[str, Any] | None = None
    runs: list[dict[str, Any]] = []
    for filename, run_id in zip(TEST_FILENAMES, run_ids, strict=True):
        image = images_by_name[filename]
        completed = service.execute_run(run_id)
        if completed.image_id != image.image_id or completed.job_id != job.job.job_id:
            raise RuntimeError(f"run-to-image mapping changed unexpectedly: {filename}")
        configuration = completed.configuration
        if (
            configuration.inference.threshold != THRESHOLD
            or configuration.inference.min_area_px != MIN_AREA_PX
        ):
            raise RuntimeError("run did not retain the frozen threshold/min-area parameters")
        if configuration.resolved_postprocess != calibrated.postprocess_profile():
            raise RuntimeError("run did not retain the frozen postprocess parameters")
        if configuration.resolved_morphometry != calibrated.morphometry_config():
            raise RuntimeError("run did not retain the frozen morphometry parameters")
        bottom_regions = [
            region
            for region in configuration.analysis_roi.invalid_rects
            if region.reason == "model_bottom_information_bar"
        ]
        if not any(
            (region.x1, region.y1, region.x2, region.y2)
            == (0, image.height - BOTTOM_CROP_PX, image.width, image.height)
            for region in bottom_regions
        ):
            raise RuntimeError(f"run is missing the frozen 130 px bottom exclusion: {filename}")
        with uow_factory() as uow:
            artifact_paths = uow.repositories.runs.get_artifact_paths(completed.run_id)
        relative_mask = artifact_paths.get("pred_mask_path")
        relative_config = artifact_paths.get("run_config_path")
        if relative_mask is None or relative_config is None:
            raise RuntimeError(f"required prediction metadata is missing: {filename}")
        mask_path = file_store.paths.require_managed(relative_mask, must_exist=True)
        run_config_path = file_store.paths.require_managed(relative_config, must_exist=True)
        metadata_path = file_store.paths.image_metadata(job.job.job_id, image.image_id)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("filename") != filename
            or metadata.get("sample_id") != Path(filename).stem
            or metadata.get("image_id") != image.image_id
        ):
            raise RuntimeError(
                f"image metadata does not strictly map UUID to YCu input: {filename}"
            )
        mask_bottom = _mask_bottom_evidence(
            mask_path,
            width=image.width,
            height=image.height,
            bottom_crop_px=BOTTOM_CROP_PX,
        )
        current_model = _model_identity(configuration)
        if (
            current_model["model_id"] != MODEL_ID
            or current_model["weight_sha256"] != TORCHSCRIPT_SHA256
        ):
            raise RuntimeError("run did not use the exact private TorchScript asset")
        if model is None:
            model = current_model
        elif model != current_model:
            raise RuntimeError("model artifact identity differs between independent runs")
        runs.append(
            {
                "filename": filename,
                "sample_id": Path(filename).stem,
                "job_id": completed.job_id,
                "image_id": completed.image_id,
                "run_id": completed.run_id,
                "final_status": completed.status.value,
                "image_metadata": str(metadata_path),
                "pred_mask": str(mask_path),
                "run_config": str(run_config_path),
                "frozen_inference": configuration.inference.model_dump(mode="json"),
                "resolved_postprocess": configuration.resolved_postprocess.model_dump(mode="json"),
                "resolved_morphometry": configuration.resolved_morphometry.model_dump(mode="json"),
                "prediction_bottom_check": mask_bottom,
            }
        )
    return {
        "job_id": job.job.job_id,
        "scope": (
            "YCu-1/2/3 independent-test inference generation only; no ground truth or "
            "evaluation metrics were read."
        ),
        "inputs": list(TEST_FILENAMES),
        "model": model,
        "calibrated_analysis": {**asdict(calibrated), "scale_expression": "100/184 nm_per_pixel"},
        "runs": runs,
        "public_registry_status": "unavailable (unchanged)",
    }


def run_independent_analysis(parameters: IndependentTestParameters) -> dict[str, Any]:
    output_root = _validate_output_root(parameters.output_root)
    if output_root != EXPECTED_OUTPUT_ROOT.resolve(strict=False):
        raise ValueError(
            f"output-root must be the fixed independent-test root: {EXPECTED_OUTPUT_ROOT}"
        )
    registry, calibrated = _ready_private_registry(parameters)
    output_root.mkdir(parents=True, exist_ok=False)
    database = Database(
        Settings(
            app_env="test",
            database_url=_sqlite_url(output_root / "analysis.sqlite3"),
            output_root=output_root / "artifacts",
            model_registry_path=parameters.registry,
            model_snapshot_root=output_root / "model-snapshots",
            model_device=DevicePreference.CPU.value,
        )
    )
    try:
        Base.metadata.create_all(database.engine)
        with database.session() as session:
            sync_model_registry(session, registry)
        maximum_upload = max(
            (parameters.image_dir / name).stat().st_size for name in TEST_FILENAMES
        )
        return execute_analyses(
            parameters,
            calibrated,
            database=database,
            file_store=LocalFileStore(
                StoragePaths(output_root / "artifacts"), max_upload_bytes=max(1, maximum_upload)
            ),
            gateway=InferenceGateway(registry),
        )
    finally:
        database.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        parameters = _validated_parameters(build_parser().parse_args(argv))
        result = run_independent_analysis(parameters)
        report_path = parameters.output_root / REPORT_FILENAME
        result["report"] = str(report_path)
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
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

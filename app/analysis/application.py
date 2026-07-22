"""Analysis use cases coordinating repositories, inference, files, and pure services."""

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Protocol
from uuid import uuid4

import numpy as np
from PIL import Image

from app.analysis.config import (
    ExecutionBuildProvenance,
    MorphometryConfig,
    PostprocessProfile,
    QualityGateConfig,
    capture_execution_build_provenance,
)
from app.analysis.instance_artifacts import canonical_instances_payload
from app.analysis.morphometry import measure
from app.analysis.postprocessing import (
    NormalizedInstance,
    PostprocessResult,
    normalize_native_instances_detailed,
    normalize_semantic_mask_detailed,
)
from app.analysis.preprocessing import build_analysis_roi, create_transform
from app.analysis.quality import QualityInputs, evaluate
from app.analysis.reporting import ReportWriter
from app.analysis.validation import infer_analysis_roi, validate_image
from app.analysis.visualization import write_review_visualizations
from app.contracts.analyses import (
    AnalysisJobDTO,
    CorrectedMaskUploadData,
    CreateAnalysisMetadata,
    CreateRunsRequest,
    ImageAssetDTO,
    InferenceOptions,
    JobDetailDTO,
    ReviewRunRequest,
    RunConfiguration,
    SegmentationRunDTO,
)
from app.contracts.common import utc_now
from app.contracts.enums import JobStatus, ModelStatus, QualityStatus, RoiMode
from app.contracts.execution import (
    ExecutionRuntimeProvenance,
    scientific_build_mismatches,
)
from app.contracts.inference import SegmentationOutput, SegmentationRequest
from app.contracts.models import ModelBundleReference, ModelHealth, ModelMetadata
from app.contracts.repositories import StoredImageAsset, UnitOfWork
from app.core.errors import (
    BoxRevisionConflictError,
    ExecutionBuildMismatchError,
    InferenceExecutionError,
    InputArtifactMismatchError,
    InvalidImageError,
    JobStateConflictError,
    ModelNotFoundError,
    ModelNotReadyError,
    NanoLoopError,
    PayloadTooLargeError,
    ResourceNotFoundError,
    ServiceUnavailableError,
)
from app.core.state_machine import aggregate_job_status
from app.storage.file_store import FileTokenError, LocalFileStore, UploadSizeExceededError
from app.storage.paths import StoragePathError


class InferenceGatewayProtocol(Protocol):
    def list_models(self, only_ready: bool = False) -> list[ModelMetadata]: ...

    def predict(
        self,
        model_id: str,
        request: SegmentationRequest,
        *,
        expected_model_version: str | None = None,
        expected_adapter_path: str | None = None,
        expected_weight_sha256: str | None = None,
        expected_config_sha256: str | None = None,
        expected_model_card_sha256: str | None = None,
        expected_adapter_sha256: str | None = None,
        model_bundle: ModelBundleReference | None = None,
    ) -> SegmentationOutput: ...

    def freeze_model_bundle(
        self,
        model_id: str,
        *,
        expected_model_version: str | None = None,
        expected_adapter_path: str | None = None,
        expected_weight_sha256: str | None = None,
        expected_config_sha256: str | None = None,
        expected_model_card_sha256: str | None = None,
        expected_adapter_sha256: str | None = None,
    ) -> ModelBundleReference: ...

    def health(self) -> list[ModelHealth]: ...


class DispatcherProtocol(Protocol):
    def submit(self, run_id: str) -> bool: ...


UnitOfWorkFactory = Callable[[], UnitOfWork]
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AnalysisUpload:
    filename: str
    stream: BinaryIO


@dataclass(frozen=True, slots=True)
class _ResolvedExecutionSettings:
    postprocess: PostprocessProfile
    morphometry: MorphometryConfig
    quality_gate: QualityGateConfig
    scale_nm_per_pixel: float | None
    warnings: tuple[str, ...] = ()


# Explicit compatibility values for schema-v1 rows.  Keeping every value here
# prevents a later code-default change from silently changing a legacy rerun.
_LEGACY_V1_POSTPROCESS = PostprocessProfile(
    profile_id="legacy-v1",
    min_area_px=8,
    fill_holes=True,
    watershed_enabled=False,
    exclude_border=True,
    connectivity=2,
    instance_iou_threshold=0.7,
)
_LEGACY_V1_MORPHOMETRY = MorphometryConfig(perimeter_neighborhood=8)
_LEGACY_V1_QUALITY_GATE = QualityGateConfig(
    foreground_ratio_review_low=0.0001,
    foreground_ratio_warn_high=0.30,
    foreground_ratio_review_high=0.40,
    confidence_warn_below=0.50,
    fragment_ratio_warn_above=0.30,
    edge_touch_ratio_warn_above=0.20,
)


class AnalysisCreationService:
    """Persist a validated multi-image job without trusting multipart filenames."""

    def __init__(self, *, uow_factory: UnitOfWorkFactory, file_store: LocalFileStore) -> None:
        self.uow_factory = uow_factory
        self.file_store = file_store

    def create_analysis(
        self,
        metadata: CreateAnalysisMetadata,
        uploads: list[AnalysisUpload],
    ) -> JobDetailDTO:
        if not isinstance(metadata, CreateAnalysisMetadata):
            raise TypeError("metadata must be CreateAnalysisMetadata")
        filenames = [upload.filename for upload in uploads]
        if len(filenames) != len(set(filenames)):
            raise InvalidImageError(details={"reason": "duplicate_upload_filename"})
        expected = {item.filename for item in metadata.images}
        if set(filenames) != expected:
            raise InvalidImageError(
                details={
                    "reason": "metadata_filename_mismatch",
                    "missing_uploads": sorted(expected - set(filenames)),
                    "missing_metadata": sorted(set(filenames) - expected),
                }
            )

        job_id = f"job_{uuid4().hex}"
        now = utc_now()
        with self.uow_factory() as uow:
            uow.repositories.jobs.create(
                AnalysisJobDTO(
                    job_id=job_id,
                    name=metadata.job_name,
                    status=JobStatus.CREATED,
                    config={"schema_version": "1.0"},
                    created_at=now,
                    updated_at=now,
                )
            )
            uow.commit()
        self._update_job(job_id, JobStatus.VALIDATING)

        metadata_by_name = {item.filename: item for item in metadata.images}
        stored_images: list[StoredImageAsset] = []
        stored_upload_paths: list[Path] = []
        manifest_images: list[dict[str, object]] = []
        seen_hashes: dict[str, str] = {}
        try:
            for upload in uploads:
                item = metadata_by_name[upload.filename]
                image_id = f"img_{uuid4().hex}"
                try:
                    stored = self.file_store.save_upload(
                        job_id,
                        upload.stream,
                        upload.filename,
                        image_id=image_id,
                    )
                except UploadSizeExceededError as exc:
                    raise PayloadTooLargeError(
                        details={"filename": upload.filename, "limit_bytes": exc.limit_bytes}
                    ) from exc
                except StoragePathError as exc:
                    raise InvalidImageError(
                        details={"filename": upload.filename, "reason": "unsafe_filename"}
                    ) from exc
                stored_upload_paths.append(stored.path)
                duplicate_of = seen_hashes.get(stored.sha256)
                if duplicate_of is not None:
                    raise InvalidImageError(
                        "同一任务内不能重复上传相同图像内容",
                        details={
                            "filename": upload.filename,
                            "duplicate_of": duplicate_of,
                            "reason": "duplicate_image_content",
                        },
                    )
                seen_hashes[stored.sha256] = upload.filename
                validated = validate_image(stored.path)
                scale = item.scale.value if item.scale.mode.value == "nm_per_pixel" else None
                analysis_roi = infer_analysis_roi(validated)
                asset = ImageAssetDTO(
                    image_id=image_id,
                    job_id=job_id,
                    filename=upload.filename,
                    sha256=stored.sha256,
                    width=validated.width,
                    height=validated.height,
                    bit_depth=validated.bit_depth,
                    sample_id=item.sample_id,
                    material_name=item.material_name,
                    material_formula=item.material_formula,
                    experiment_conditions=item.experiment_conditions,
                    scale_nm_per_pixel=scale,
                    analysis_roi=analysis_roi,
                    original_download_url=f"/api/v1/files/{stored.file_token}",
                )
                stored_images.append(
                    StoredImageAsset(asset=asset, storage_path=stored.relative_path)
                )
                manifest_images.append(
                    {
                        "image_id": image_id,
                        "filename": upload.filename,
                        "sha256": stored.sha256,
                        "size_bytes": stored.size_bytes,
                    }
                )

            with self.uow_factory() as uow:
                uow.repositories.images.add_many(stored_images)
                uow.commit()
            images = [stored.asset for stored in stored_images]
            for image in images:
                self.file_store.atomic_write_json(
                    self.file_store.paths.image_metadata(job_id, image.image_id),
                    image.model_copy(update={"original_download_url": None}).model_dump(
                        mode="json"
                    ),
                )
                self.file_store.atomic_write_json(
                    self.file_store.paths.boxes_revision(job_id, image.image_id, 0),
                    {"image_id": image.image_id, "revision": 0, "boxes": []},
                )
            self.file_store.atomic_write_json(
                self.file_store.paths.job_config(job_id),
                {
                    "job_id": job_id,
                    "job_name": metadata.job_name,
                    "images": [item.model_dump(mode="json") for item in metadata.images],
                },
            )
            self.file_store.atomic_write_json(
                self.file_store.paths.job_manifest(job_id),
                {"job_id": job_id, "images": manifest_images},
            )
            self._update_job(job_id, JobStatus.READY_FOR_CONFIGURATION)
            with self.uow_factory() as uow:
                job = uow.repositories.jobs.get(job_id)
            return JobDetailDTO(job=job, images=images, runs=[])
        except Exception as exc:
            self._cleanup_unreferenced_uploads(job_id, stored_upload_paths)
            code = exc.code if isinstance(exc, NanoLoopError) else "INVALID_IMAGE"
            self._update_job(job_id, JobStatus.FAILED, error_code=code)
            raise

    def _cleanup_unreferenced_uploads(
        self,
        job_id: str,
        candidates: list[Path],
    ) -> None:
        """Delete only upload files that did not become durable image originals."""

        if not candidates:
            return
        try:
            with self.uow_factory() as uow:
                images = uow.repositories.images.list_by_job(job_id)
                referenced = {
                    self.file_store.paths.require_managed(
                        uow.repositories.images.get_storage_path(image.image_id)
                    )
                    for image in images
                }
        except Exception:
            # A repository failure makes ownership ambiguous. Retaining bytes is
            # safer than deleting a possibly committed original image.
            logger.warning(
                "failed_upload_cleanup_skipped",
                extra={"job_id": job_id, "event": "failed_upload_cleanup_skipped"},
                exc_info=True,
            )
            return

        for candidate in candidates:
            try:
                managed = self.file_store.paths.require_managed(candidate)
                if managed not in referenced:
                    managed.unlink(missing_ok=True)
            except (OSError, StoragePathError):
                logger.warning(
                    "failed_upload_cleanup_error",
                    extra={
                        "job_id": job_id,
                        "event": "failed_upload_cleanup_error",
                    },
                    exc_info=True,
                )

    def _update_job(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error_code: str | None = None,
    ) -> None:
        with self.uow_factory() as uow:
            uow.repositories.jobs.update_status(job_id, status, error_code=error_code)
            uow.commit()


class AnalysisApplicationService:
    """Create immutable runs and execute one run through the auditable pipeline."""

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        file_store: LocalFileStore,
        inference_gateway: InferenceGatewayProtocol,
        dispatcher: DispatcherProtocol | None = None,
        postprocess_config: PostprocessProfile | None = None,
        morphometry_config: MorphometryConfig | None = None,
        quality_config: QualityGateConfig | None = None,
    ) -> None:
        self.uow_factory = uow_factory
        self.file_store = file_store
        self.inference_gateway = inference_gateway
        self.dispatcher = dispatcher
        self.postprocess_config = (postprocess_config or PostprocessProfile()).model_copy(
            deep=True
        )
        self.morphometry_config = (morphometry_config or MorphometryConfig()).model_copy(
            deep=True
        )
        self.quality_config = (quality_config or QualityGateConfig()).model_copy(deep=True)
        self.report_writer = ReportWriter(file_store)

    def create_runs(self, job_id: str, request: CreateRunsRequest) -> list[str]:
        models = {model.model_id: model for model in self.inference_gateway.list_models()}
        selected_models = [
            self._require_ready_model(model_id, models) for model_id in request.model_ids
        ]
        health = {item.model_id: item for item in self.inference_gateway.health()}
        freeze_bundle = getattr(self.inference_gateway, "freeze_model_bundle", None)
        if not callable(freeze_bundle):
            raise ModelNotReadyError(
                "模型网关不支持冻结完整 bundle，不能创建新的可复现运行",
                details={
                    "model_ids": [model.model_id for model in selected_models],
                    "reason": "model_bundle_freeze_unsupported",
                },
            )
        frozen_bundles: dict[str, ModelBundleReference] = {}
        for model in selected_models:
            model_health = health.get(model.model_id)
            weight_sha256 = model.weight_sha256 or (
                model_health.weight_sha256 if model_health else None
            )
            required = {
                "adapter_path": model.adapter_path,
                "weight_sha256": weight_sha256,
                "config_sha256": model.config_sha256,
                "model_card_sha256": model.model_card_sha256,
                "adapter_sha256": model.adapter_sha256,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ModelNotReadyError(
                    "模型缺少完整 bundle 来源，不能创建可复现运行",
                    details={
                        "model_id": model.model_id,
                        "reason": "artifact_provenance_incomplete",
                        "missing_fields": missing,
                    },
                )
            try:
                bundle = ModelBundleReference.model_validate(
                    freeze_bundle(
                        model.model_id,
                        expected_model_version=model.version,
                        expected_adapter_path=model.adapter_path,
                        expected_weight_sha256=weight_sha256,
                        expected_config_sha256=model.config_sha256,
                        expected_model_card_sha256=model.model_card_sha256,
                        expected_adapter_sha256=model.adapter_sha256,
                    )
                )
            except (TypeError, ValueError) as error:
                raise ModelNotReadyError(
                    "模型网关没有返回合法的冻结 bundle",
                    details={
                        "model_id": model.model_id,
                        "reason": "invalid_model_bundle_reference",
                    },
                ) from error
            if bundle.adapter_sha256 != model.adapter_sha256:
                raise ModelNotReadyError(
                    "冻结 bundle 与模型 Adapter 摘要不一致",
                    details={
                        "model_id": model.model_id,
                        "reason": "adapter_sha256_mismatch",
                    },
                )
            frozen_bundles[model.model_id] = bundle.model_copy(deep=True)
        created_at = utc_now()
        execution_build = capture_execution_build_provenance()
        runs: list[SegmentationRunDTO] = []

        with self.uow_factory() as uow:
            repositories = uow.repositories
            job = repositories.jobs.get(job_id)
            allowed_job_states = {
                JobStatus.READY_FOR_CONFIGURATION,
                JobStatus.COMPLETED,
                JobStatus.COMPLETED_WITH_WARNINGS,
                JobStatus.FAILED,
            }
            if job.status not in allowed_job_states:
                raise JobStateConflictError(
                    "任务尚未完成上传校验，不能创建运行",
                    details={"job_id": job_id, "status": job.status.value},
                )
            images = {image.image_id: image for image in repositories.images.list_by_job(job_id)}
            if not images:
                raise JobStateConflictError(
                    "任务没有可分析的已验证图像",
                    details={"job_id": job_id, "status": job.status.value},
                )
            selected_images: list[ImageAssetDTO] = []
            boxes_by_image = {}
            for image_id in request.image_ids:
                image = images.get(image_id)
                if image is None:
                    raise ResourceNotFoundError(
                        details={"resource": "image", "image_id": image_id, "job_id": job_id}
                    )
                selected_images.append(image)
                box_set = repositories.boxes.get_active(image_id)
                boxes_by_image[image_id] = box_set
                if request.roi_mode == RoiMode.BOXES:
                    expected = request.box_revisions[image_id]
                    if box_set.revision != expected:
                        raise BoxRevisionConflictError(
                            details={
                                "image_id": image_id,
                                "expected_revision": expected,
                                "current_revision": box_set.revision,
                            }
                        )
                    if not any(box.active for box in box_set.boxes):
                        raise BoxRevisionConflictError(
                            "选框模式需要至少一个已保存的活动框",
                            details={"image_id": image_id, "revision": box_set.revision},
                        )

            for image in selected_images:
                box_set = boxes_by_image[image.image_id]
                for model in selected_models:
                    effective_threshold = (
                        request.inference.threshold
                        if request.inference.threshold is not None
                        else model.default_threshold
                    )
                    inference = request.inference.model_copy(
                        update={"threshold": effective_threshold}
                    )
                    box_revision = (
                        box_set.revision if request.roi_mode == RoiMode.BOXES else None
                    )
                    model_health = health.get(model.model_id)
                    resolved_weight_sha256 = model.weight_sha256 or (
                        model_health.weight_sha256 if model_health else None
                    )
                    model_provenance = {
                        "adapter_path": model.adapter_path,
                        "weight_sha256": resolved_weight_sha256,
                        "config_sha256": model.config_sha256,
                        "model_card_sha256": model.model_card_sha256,
                        "adapter_sha256": model.adapter_sha256,
                    }
                    model_bundle = frozen_bundles[model.model_id]
                    missing_provenance = [
                        field for field, value in model_provenance.items() if value is None
                    ]
                    if missing_provenance:
                        raise ModelNotReadyError(
                            "模型缺少完整制品来源，不能创建可复现运行",
                            details={
                                "model_id": model.model_id,
                                "reason": "artifact_provenance_incomplete",
                                "missing_fields": missing_provenance,
                            },
                        )
                    configuration = RunConfiguration(
                        schema_version=3,
                        provenance_status="complete",
                        provenance_warnings=[],
                        model_id=model.model_id,
                        model_version=model.version,
                        adapter_path=model.adapter_path,
                        weight_sha256=resolved_weight_sha256,
                        config_sha256=model.config_sha256,
                        model_card_sha256=model.model_card_sha256,
                        adapter_sha256=model.adapter_sha256,
                        model_bundle=model_bundle,
                        roi_mode=request.roi_mode,
                        box_revision=box_revision,
                        boxes=box_set.boxes if request.roi_mode == RoiMode.BOXES else [],
                        analysis_roi=image.analysis_roi,
                        inference=inference,
                        preprocess_profile=model.preprocess_profile,
                        postprocess_profile=model.postprocess_profile,
                        image_sha256=image.sha256,
                        scale_nm_per_pixel=image.scale_nm_per_pixel,
                        resolved_postprocess=self._resolved_postprocess(
                            profile_id=model.postprocess_profile,
                            inference=inference,
                        ),
                        resolved_morphometry=self.morphometry_config.model_copy(deep=True),
                        resolved_quality_gate=self.quality_config.model_copy(deep=True),
                        execution_build=execution_build.model_copy(deep=True),
                        created_at=created_at,
                    )
                    runs.append(
                        SegmentationRunDTO(
                            run_id=f"run_{uuid4().hex}",
                            job_id=job_id,
                            image_id=image.image_id,
                            model_id=model.model_id,
                            status=JobStatus.QUEUED,
                            roi_mode=request.roi_mode,
                            box_revision=box_revision,
                            threshold=effective_threshold,
                            inference=inference,
                            configuration=configuration,
                            created_at=created_at,
                            updated_at=created_at,
                        )
                    )
            run_ids = repositories.runs.create_many(runs)
            repositories.jobs.update_status(job_id, JobStatus.QUEUED)
            uow.commit()

        self._dispatch_runs(run_ids)
        return run_ids

    def create_review_run(self, parent_run_id: str, request: ReviewRunRequest) -> str:
        """Create an immutable child run with reviewed inference parameters."""

        now = utc_now()
        with self.uow_factory() as uow:
            repositories = uow.repositories
            parent = repositories.runs.get(parent_run_id)
            image = repositories.images.get(parent.image_id)
            if parent.status not in {
                JobStatus.COMPLETED,
                JobStatus.COMPLETED_WITH_WARNINGS,
                JobStatus.FAILED,
            }:
                raise JobStateConflictError(
                    "运行尚未结束，不能创建复核子运行",
                    details={"run_id": parent_run_id, "status": parent.status.value},
                )

        legacy_parent = parent.configuration.provenance_status == "legacy_fallback"
        model_provenance_fields = (
            "adapter_path",
            "weight_sha256",
            "config_sha256",
            "model_card_sha256",
            "adapter_sha256",
        )
        needs_model_lookup = (
            request.corrected_mask_token is None
            and parent.configuration.model_bundle is None
        ) or (
            legacy_parent
            and any(
                getattr(parent.configuration, field) is None
                for field in model_provenance_fields
            )
        )
        current_model_provenance: dict[str, object] = {}
        if needs_model_lookup:
            models = {model.model_id: model for model in self.inference_gateway.list_models()}
            health = {item.model_id: item for item in self.inference_gateway.health()}
            model = self._require_ready_model(parent.model_id, models)
            if model.version != parent.configuration.model_version:
                raise ModelNotReadyError(
                    "当前模型版本与父运行不一致，无法保证复核可复现",
                    details={
                        "model_id": parent.model_id,
                        "parent_version": parent.configuration.model_version,
                        "current_version": model.version,
                    },
                )
            model_health = health.get(parent.model_id)
            parent_hash = parent.configuration.weight_sha256
            current_hash = model.weight_sha256 or (
                model_health.weight_sha256 if model_health is not None else None
            )
            if parent_hash is not None and current_hash != parent_hash:
                raise ModelNotReadyError(
                    "当前模型权重与父运行不一致，无法保证复核可复现",
                    details={"model_id": parent.model_id, "reason": "weight_sha256_mismatch"},
                )
            provenance_hashes = {
                "adapter_path": model.adapter_path,
                "config_sha256": model.config_sha256,
                "model_card_sha256": model.model_card_sha256,
                "adapter_sha256": model.adapter_sha256,
            }
            for field, current in provenance_hashes.items():
                expected = getattr(parent.configuration, field)
                if expected is not None and current != expected:
                    raise ModelNotReadyError(
                        "当前模型配置或模型卡与父运行不一致，无法保证复核可复现",
                        details={
                            "model_id": parent.model_id,
                            "reason": f"{field}_mismatch",
                        },
                    )
            current_model_provenance = {
                **provenance_hashes,
                "weight_sha256": current_hash,
            }
            supports_bundle_freeze = callable(
                getattr(self.inference_gateway, "freeze_model_bundle", None)
            )
            missing_current = [
                field
                for field, value in current_model_provenance.items()
                if value is None
                and (field != "adapter_sha256" or supports_bundle_freeze)
            ]
            if missing_current:
                raise ModelNotReadyError(
                    "当前模型缺少完整制品来源，无法创建可复现复核运行",
                    details={
                        "model_id": parent.model_id,
                        "reason": "artifact_provenance_incomplete",
                        "missing_fields": missing_current,
                    },
                )

        inference_updates = {
            field: value
            for field in (
                "threshold",
                "min_area_px",
                "watershed_enabled",
                "exclude_border",
            )
            if (value := getattr(request, field)) is not None
        }
        inference = parent.configuration.inference.model_copy(update=inference_updates)
        parent_settings = self._resolve_execution_settings(parent.configuration)
        resolved_postprocess = parent_settings.postprocess.model_copy(
            update={
                "min_area_px": inference.min_area_px,
                "watershed_enabled": inference.watershed_enabled,
                "exclude_border": inference.exclude_border,
            },
            deep=True,
        )
        child_run_id = f"run_{uuid4().hex}"
        configuration_updates: dict[str, object] = {
            "schema_version": (
                3 if parent.configuration.schema_version == 3 else 2
            ),
            "provenance_status": "complete",
            "provenance_warnings": (
                ["review_configuration_resolved_from_legacy_parent"]
                if legacy_parent
                else list(parent.configuration.provenance_warnings)
            ),
            "inference": inference,
            "image_sha256": (
                image.sha256 if legacy_parent else parent.configuration.image_sha256
            ),
            "scale_nm_per_pixel": (
                image.scale_nm_per_pixel
                if legacy_parent
                else parent.configuration.scale_nm_per_pixel
            ),
            "resolved_postprocess": resolved_postprocess,
            "resolved_morphometry": parent_settings.morphometry.model_copy(deep=True),
            "resolved_quality_gate": parent_settings.quality_gate.model_copy(deep=True),
            "execution_build": capture_execution_build_provenance(),
            "created_at": now,
            "review_source": "model_inference",
            "corrected_mask_sha256": None,
        }
        if legacy_parent:
            configuration_updates.update(current_model_provenance)
        if needs_model_lookup and request.corrected_mask_token is None:
            freeze_bundle = getattr(self.inference_gateway, "freeze_model_bundle", None)
            if callable(freeze_bundle):
                bundle_reference = freeze_bundle(
                    parent.model_id,
                    expected_model_version=parent.configuration.model_version,
                    expected_adapter_path=current_model_provenance["adapter_path"],
                    expected_weight_sha256=current_model_provenance["weight_sha256"],
                    expected_config_sha256=current_model_provenance["config_sha256"],
                    expected_model_card_sha256=current_model_provenance[
                        "model_card_sha256"
                    ],
                    expected_adapter_sha256=current_model_provenance["adapter_sha256"],
                )
                configuration_updates.update(
                    {
                        "schema_version": 3,
                        "adapter_sha256": bundle_reference.adapter_sha256,
                        "model_bundle": bundle_reference,
                    }
                )
        staged_corrected_path: Path | None = None
        if request.corrected_mask_token is not None:
            corrected_mask, staged_corrected_path = self._load_corrected_mask(
                request.corrected_mask_token,
                expected_shape=(image.height, image.width),
            )
            corrected_path = self.file_store.paths.run_artifact(
                parent.job_id,
                parent.image_id,
                child_run_id,
                "corrected_mask.png",
            )
            buffer = BytesIO()
            Image.fromarray(corrected_mask.astype(np.uint8) * 255).save(buffer, format="PNG")
            self.file_store.atomic_write_bytes(corrected_path, buffer.getvalue())
            configuration_updates.update(
                {
                    "review_source": "corrected_mask",
                    "corrected_mask_sha256": self.file_store.calculate_sha256(corrected_path),
                }
            )

        configuration = RunConfiguration.model_validate(
            parent.configuration.model_copy(
                update=configuration_updates,
                deep=True,
            ).model_dump(mode="python")
        )
        child = SegmentationRunDTO(
            run_id=child_run_id,
            job_id=parent.job_id,
            image_id=parent.image_id,
            model_id=parent.model_id,
            status=JobStatus.QUEUED,
            roi_mode=parent.roi_mode,
            box_revision=parent.box_revision,
            threshold=inference.threshold,
            inference=inference,
            configuration=configuration,
            parent_run_id=parent_run_id,
            created_at=now,
            updated_at=now,
        )
        with self.uow_factory() as uow:
            repositories = uow.repositories
            repositories.runs.create_many([child])
            repositories.jobs.update_status(parent.job_id, JobStatus.QUEUED)
            uow.commit()

        if staged_corrected_path is not None:
            self._delete_consumed_corrected_mask(
                job_id=parent.job_id,
                path=staged_corrected_path,
            )
        self._dispatch_runs([child.run_id])
        return child.run_id

    def stage_corrected_mask(
        self,
        run_id: str,
        stream: BinaryIO,
        filename: str,
    ) -> CorrectedMaskUploadData:
        """Store and validate a short-lived corrected-mask input for the review API."""

        with self.uow_factory() as uow:
            run = uow.repositories.runs.get(run_id)
            image = uow.repositories.images.get(run.image_id)
        try:
            stored = self.file_store.save_upload(
                run.job_id,
                stream,
                filename,
                image_id=f"review_mask_{uuid4().hex}",
            )
        except UploadSizeExceededError as error:
            raise PayloadTooLargeError(
                details={"filename": filename, "limit_bytes": error.limit_bytes}
            ) from error
        except StoragePathError as error:
            raise InvalidImageError(
                details={"filename": filename, "reason": "unsafe_filename"}
            ) from error
        try:
            self._load_corrected_mask(
                stored.file_token,
                expected_shape=(image.height, image.width),
            )
        except Exception:
            stored.path.unlink(missing_ok=True)
            raise
        return CorrectedMaskUploadData(
            corrected_mask_token=stored.file_token,
            sha256=stored.sha256,
            width=image.width,
            height=image.height,
        )

    def _dispatch_runs(self, run_ids: list[str]) -> None:
        if self.dispatcher is None:
            return
        for run_id in run_ids:
            try:
                self.dispatcher.submit(run_id)
            except ServiceUnavailableError:
                logger.warning(
                    "run_dispatch_deferred",
                    extra={"run_id": run_id, "event": "run_dispatch_deferred"},
                )

    def execute_run(self, run_id: str) -> SegmentationRunDTO:
        # Claim outside the failure-recording block. A duplicate delivery is a
        # scheduler race, not a scientific run failure, and must never overwrite
        # the state of the worker that already owns this immutable run.
        run, image, storage_path = self._claim_execution_inputs(run_id)
        try:
            image_path = self.file_store.paths.require_managed(
                storage_path,
                must_exist=True,
            )
            configuration = run.configuration
            image_bytes = image_path.read_bytes()
            self._verify_input_image(
                run_id=run_id,
                configuration=configuration,
                image=image,
                image_bytes=image_bytes,
            )
            executor_build = capture_execution_build_provenance()
            build_mismatches = (
                scientific_build_mismatches(configuration.execution_build, executor_build)
                if configuration.execution_build is not None
                else ["execution_build"]
            )
            if configuration.schema_version == 3 and build_mismatches:
                raise ExecutionBuildMismatchError(
                    details={
                        "run_id": run_id,
                        "mismatched_fields": build_mismatches,
                    }
                )
            settings = self._resolve_execution_settings(configuration)
            roi_mask = build_analysis_roi(
                width=image.width,
                height=image.height,
                analysis_roi=configuration.analysis_roi,
                roi_mode=configuration.roi_mode,
                boxes=configuration.boxes,
            )
            transform = create_transform(image.width, image.height, configuration.analysis_roi)
            run_dir = self.file_store.create_run_dir(run.job_id, run.image_id, run.run_id)
            self.file_store.atomic_write_json(
                run_dir / "run_config.json",
                self._file_contract_payload(configuration.model_dump(mode="json")),
            )
            self.file_store.atomic_write_json(
                run_dir / "transform.json",
                self._file_contract_payload(transform.model_dump(mode="json")),
            )

            self._set_run_status(run_id, JobStatus.SEGMENTING)
            if configuration.review_source == "corrected_mask":
                corrected_path = run_dir / "corrected_mask.png"
                observed_hash = self.file_store.calculate_sha256(corrected_path)
                if observed_hash != configuration.corrected_mask_sha256:
                    raise InferenceExecutionError(
                        details={"run_id": run_id, "reason": "corrected_mask_hash_mismatch"}
                    )
                output = SegmentationOutput(
                    width=image.width,
                    height=image.height,
                    binary_mask_path=corrected_path,
                    runtime_ms=0,
                    warnings=["manual_corrected_mask"],
                )
            else:
                segmentation_request = SegmentationRequest(
                    image_id=run.image_id,
                    # Every adapter receives the verified bytes. A deliberately absent path
                    # prevents legacy or handoff code from silently reopening mutable input.
                    image_path=run_dir / ".pinned-image-bytes",
                    image_bytes=image_bytes,
                    run_dir=run_dir,
                    roi_mode=configuration.roi_mode,
                    boxes=configuration.boxes,
                    threshold=configuration.inference.threshold,
                    min_area_px=configuration.inference.min_area_px,
                    roi_context_px=configuration.roi_context_px,
                    device=configuration.inference.device,
                    seed=configuration.inference.seed,
                )
                if configuration.model_bundle is None:
                    output = self.inference_gateway.predict(
                        run.model_id,
                        segmentation_request,
                        expected_model_version=configuration.model_version,
                        expected_adapter_path=configuration.adapter_path,
                        expected_weight_sha256=configuration.weight_sha256,
                        expected_config_sha256=configuration.config_sha256,
                        expected_model_card_sha256=configuration.model_card_sha256,
                    )
                else:
                    output = self.inference_gateway.predict(
                        run.model_id,
                        segmentation_request,
                        expected_model_version=configuration.model_version,
                        expected_adapter_path=configuration.adapter_path,
                        expected_weight_sha256=configuration.weight_sha256,
                        expected_config_sha256=configuration.config_sha256,
                        expected_model_card_sha256=configuration.model_card_sha256,
                        expected_adapter_sha256=configuration.adapter_sha256,
                        model_bundle=configuration.model_bundle,
                    )
            if output.width != image.width or output.height != image.height:
                raise InferenceExecutionError(
                    details={
                        "run_id": run_id,
                        "reason": "output_dimensions_mismatch",
                        "expected": [image.width, image.height],
                        "observed": [output.width, output.height],
                    }
                )
            execution = self._runtime_provenance(
                configuration=configuration,
                executor_build=executor_build,
                build_mismatches=build_mismatches,
                output=output,
            )

            self._set_run_status(run_id, JobStatus.POSTPROCESSING)
            binary = self._load_binary_mask(output.binary_mask_path, roi_mask.shape)
            postprocessed = self._normalize_output(
                output,
                binary,
                roi_mask,
                settings.postprocess,
            )
            instances = postprocessed.instances
            normalized_union = self._union(instances, roi_mask.shape)

            # The public mask must be the same postprocessed union used for
            # measurements and overlays, not an adapter's raw threshold output.
            normalized_mask_path = run_dir / "pred_mask.png"
            normalized_buffer = BytesIO()
            Image.fromarray(normalized_union.astype(np.uint8) * 255, mode="L").save(
                normalized_buffer,
                format="PNG",
            )
            self.file_store.atomic_write_bytes(normalized_mask_path, normalized_buffer.getvalue())
            normalized_instances_path = run_dir / "instances.json"
            self.file_store.atomic_write_json(
                normalized_instances_path,
                canonical_instances_payload(
                    instances,
                    width=image.width,
                    height=image.height,
                ),
            )

            self._set_run_status(run_id, JobStatus.QUALITY_CHECKING)
            quality = evaluate(
                QualityInputs(
                    roi_area_px=int(roi_mask.sum()),
                    foreground_area_px=int(normalized_union.sum()),
                    instances=instances,
                    minimum_area_px=settings.postprocess.min_area_px,
                    validation_warnings=[*output.warnings, *settings.warnings],
                    candidate_instance_count=postprocessed.candidate_count,
                    boundary_instance_count=postprocessed.boundary_candidate_count,
                ),
                settings.quality_gate,
            )

            self._set_run_status(run_id, JobStatus.ANALYZING)
            morphometry = measure(
                run_id=run_id,
                instances=instances,
                roi_mask=roi_mask,
                scale_nm_per_pixel=settings.scale_nm_per_pixel,
                config=settings.morphometry,
            )
            if morphometry.warnings and quality.status == QualityStatus.PASS:
                quality = quality.model_copy(
                    update={
                        "status": QualityStatus.WARN,
                        "reasons": [*quality.reasons, *morphometry.warnings],
                    }
                )
            summary = morphometry.image_summary.model_copy(
                update={"quality_status": quality.status}
            )
            morphometry = type(morphometry)(
                particles=morphometry.particles,
                image_summary=summary,
                warnings=morphometry.warnings,
            )
            overlay_path, labeled_path = write_review_visualizations(
                image_path=image_path,
                image_bytes=image_bytes,
                binary_mask=normalized_union,
                instances=instances,
                output_dir=run_dir,
            )
            reports = self.report_writer.write_run_reports(
                job_id=run.job_id,
                image_id=run.image_id,
                run_id=run.run_id,
                configuration=configuration,
                execution=execution,
                transform=transform,
                morphometry=morphometry,
                quality=quality,
            )
            paths = {
                **reports.as_paths_json(),
                "pred_mask_path": self._relative_managed(normalized_mask_path),
                "overlay_path": self._relative_managed(overlay_path),
                "labeled_particles_path": self._relative_managed(labeled_path),
                "probability_path": self._optional_relative(output.probability_path),
                "instances_path": self._relative_managed(normalized_instances_path),
            }

            with self.uow_factory() as uow:
                uow.repositories.runs.save_result(
                    run_id,
                    particles=morphometry.particles,
                    summary=summary,
                    quality=quality,
                    execution=execution,
                    runtime_ms=output.runtime_ms,
                    paths=paths,
                )
                uow.commit()
            self._set_run_status(run_id, JobStatus.AGGREGATING)
            terminal = (
                JobStatus.COMPLETED
                if quality.status == QualityStatus.PASS
                else JobStatus.COMPLETED_WITH_WARNINGS
            )
            self._set_run_status(run_id, terminal)
            with self.uow_factory() as uow:
                completed = uow.repositories.runs.get(run_id)
            return completed
        except Exception as exc:
            self._record_failure(run_id, exc)
            raise

    @staticmethod
    def _runtime_provenance(
        *,
        configuration: RunConfiguration,
        executor_build: ExecutionBuildProvenance,
        build_mismatches: list[str],
        output: SegmentationOutput,
    ) -> ExecutionRuntimeProvenance:
        warnings = [
            f"execution_build_mismatch:{field}" for field in build_mismatches
        ]
        evidence = output.execution
        if configuration.review_source == "corrected_mask":
            warnings.append("model_inference_not_applicable_corrected_mask")
            actual_device = "not_applicable"
            python_random_seeded = False
            numpy_random_seeded = False
            torch_deterministic_algorithms = False
            global_inference_serialized = False
            backend = "manual_corrected_mask"
        elif evidence is None:
            if configuration.schema_version == 3:
                raise InferenceExecutionError(
                    details={
                        "model_id": configuration.model_id,
                        "reason": "runtime_execution_evidence_missing",
                    }
                )
            warnings.append("runtime_execution_evidence_unavailable_legacy_adapter")
            actual_device = "not_applicable"
            python_random_seeded = False
            numpy_random_seeded = False
            torch_deterministic_algorithms = False
            global_inference_serialized = False
            backend = configuration.adapter_path or configuration.model_id
        else:
            actual_device = evidence.actual_device
            python_random_seeded = evidence.python_random_seeded
            numpy_random_seeded = evidence.numpy_random_seeded
            torch_deterministic_algorithms = evidence.torch_deterministic_algorithms
            global_inference_serialized = evidence.global_inference_serialized
            backend = evidence.backend
        return ExecutionRuntimeProvenance(
            executor_build=executor_build,
            build_identity_matches_contract=not build_mismatches,
            requested_device=configuration.inference.device,
            actual_device=actual_device,
            seed=configuration.inference.seed,
            python_random_seeded=python_random_seeded,
            numpy_random_seeded=numpy_random_seeded,
            torch_deterministic_algorithms=torch_deterministic_algorithms,
            global_inference_serialized=global_inference_serialized,
            backend=backend,
            model_bundle_id=(
                configuration.model_bundle.bundle_id
                if configuration.model_bundle is not None
                else None
            ),
            adapter_sha256=configuration.adapter_sha256,
            warnings=warnings,
            executed_at=utc_now(),
        )

    def _claim_execution_inputs(
        self, run_id: str
    ) -> tuple[SegmentationRunDTO, ImageAssetDTO, str]:
        with self.uow_factory() as uow:
            if not uow.repositories.runs.claim_queued(run_id):
                existing = uow.repositories.runs.get(run_id)
                raise JobStateConflictError(
                    "运行已被其他工作进程领取",
                    details={"run_id": run_id, "status": existing.status.value},
                )
            run = uow.repositories.runs.get(run_id)
            image = uow.repositories.images.get(run.image_id)
            storage_path = uow.repositories.images.get_storage_path(run.image_id)
            statuses = [item.status for item in uow.repositories.runs.list_by_job(run.job_id)]
            uow.repositories.jobs.update_status(run.job_id, aggregate_job_status(statuses))
            uow.commit()
        return run, image, storage_path

    def _set_run_status(self, run_id: str, status: JobStatus) -> None:
        with self.uow_factory() as uow:
            run = uow.repositories.runs.get(run_id)
            uow.repositories.runs.update_status(run_id, status)
            statuses = [item.status for item in uow.repositories.runs.list_by_job(run.job_id)]
            # list_by_job observes the flushed update from update_status.
            uow.repositories.jobs.update_status(run.job_id, aggregate_job_status(statuses))
            uow.commit()

    def _record_failure(self, run_id: str, exc: Exception) -> None:
        try:
            with self.uow_factory() as uow:
                run = uow.repositories.runs.get(run_id)
                if run.status not in {
                    JobStatus.COMPLETED,
                    JobStatus.COMPLETED_WITH_WARNINGS,
                    JobStatus.FAILED,
                }:
                    code = exc.code if isinstance(exc, NanoLoopError) else "INFERENCE_FAILED"
                    message = exc.message if isinstance(exc, NanoLoopError) else "分析运行失败"
                    uow.repositories.runs.update_status(
                        run_id,
                        JobStatus.FAILED,
                        error_code=code,
                        error_message=message,
                    )
                    statuses = [
                        item.status for item in uow.repositories.runs.list_by_job(run.job_id)
                    ]
                    uow.repositories.jobs.update_status(
                        run.job_id, aggregate_job_status(statuses), error_code=code
                    )
                    uow.commit()
        except Exception:
            # Preserve the original analysis failure; recovery can reconcile stuck states.
            return

    @staticmethod
    def _require_ready_model(
        model_id: str, models: dict[str, ModelMetadata]
    ) -> ModelMetadata:
        model = models.get(model_id)
        if model is None:
            raise ModelNotFoundError(details={"model_id": model_id})
        if model.status != ModelStatus.READY:
            raise ModelNotReadyError(
                details={
                    "model_id": model_id,
                    "status": model.status.value,
                    "reason": model.health_error,
                }
            )
        return model

    def _normalize_output(
        self,
        output: SegmentationOutput,
        binary: np.ndarray,
        roi_mask: np.ndarray,
        profile: PostprocessProfile,
    ) -> PostprocessResult:
        if output.instances_path is not None and output.instances_path.suffix.lower() == ".npz":
            instance_path = self.file_store.paths.require_managed(
                output.instances_path, must_exist=True
            )
            with np.load(instance_path, allow_pickle=False) as archive:
                masks = [np.asarray(mask, dtype=bool) for mask in archive["masks"]]
                scores = archive.get("confidences")
                if scores is None:
                    scores = archive.get("scores")
                confidences = (
                    [None if np.isnan(value) else float(value) for value in scores]
                    if scores is not None
                    else None
                )
            return normalize_native_instances_detailed(
                masks,
                roi_mask=np.asarray(roi_mask, dtype=bool),
                profile=profile,
                confidences=confidences,
            )
        probability = self._load_probability(output.probability_path, binary.shape)
        return normalize_semantic_mask_detailed(
            binary,
            roi_mask=np.asarray(roi_mask, dtype=bool),
            profile=profile,
            probability=probability,
        )

    def _resolved_postprocess(
        self,
        *,
        profile_id: str,
        inference: InferenceOptions,
    ) -> PostprocessProfile:
        return self.postprocess_config.model_copy(
            update={
                "profile_id": profile_id,
                "min_area_px": inference.min_area_px,
                "watershed_enabled": inference.watershed_enabled,
                "exclude_border": inference.exclude_border,
            },
            deep=True,
        )

    @staticmethod
    def _resolve_execution_settings(
        configuration: RunConfiguration,
    ) -> _ResolvedExecutionSettings:
        if configuration.provenance_status == "complete":
            if (
                configuration.resolved_postprocess is None
                or configuration.resolved_morphometry is None
                or configuration.resolved_quality_gate is None
            ):
                raise ValueError("complete run configuration is missing resolved settings")
            return _ResolvedExecutionSettings(
                postprocess=configuration.resolved_postprocess.model_copy(deep=True),
                morphometry=configuration.resolved_morphometry.model_copy(deep=True),
                quality_gate=configuration.resolved_quality_gate.model_copy(deep=True),
                scale_nm_per_pixel=configuration.scale_nm_per_pixel,
                warnings=tuple(configuration.provenance_warnings),
            )

        inference = configuration.inference
        postprocess = (
            configuration.resolved_postprocess.model_copy(deep=True)
            if configuration.resolved_postprocess is not None
            else _LEGACY_V1_POSTPROCESS.model_copy(
                update={
                    "profile_id": configuration.postprocess_profile,
                    "min_area_px": inference.min_area_px,
                    "watershed_enabled": inference.watershed_enabled,
                    "exclude_border": inference.exclude_border,
                },
                deep=True,
            )
        )
        warnings = list(configuration.provenance_warnings)
        warnings.append("legacy_run_configuration_fallback")
        if configuration.scale_nm_per_pixel is None:
            warnings.append("legacy_physical_scale_not_frozen_pixel_metrics_only")
        logger.warning(
            "legacy_run_configuration_fallback",
            extra={
                "event": "legacy_run_configuration_fallback",
                "model_id": configuration.model_id,
            },
        )
        return _ResolvedExecutionSettings(
            postprocess=postprocess,
            morphometry=(
                configuration.resolved_morphometry.model_copy(deep=True)
                if configuration.resolved_morphometry is not None
                else _LEGACY_V1_MORPHOMETRY.model_copy(deep=True)
            ),
            quality_gate=(
                configuration.resolved_quality_gate.model_copy(deep=True)
                if configuration.resolved_quality_gate is not None
                else _LEGACY_V1_QUALITY_GATE.model_copy(deep=True)
            ),
            scale_nm_per_pixel=configuration.scale_nm_per_pixel,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def _verify_input_image(
        self,
        *,
        run_id: str,
        configuration: RunConfiguration,
        image: ImageAssetDTO,
        image_bytes: bytes,
    ) -> None:
        observed = hashlib.sha256(image_bytes).hexdigest()
        expected = configuration.image_sha256
        mismatches: dict[str, str] = {}
        if observed != image.sha256:
            mismatches["database_sha256"] = image.sha256
        if expected is not None and observed != expected:
            mismatches["frozen_sha256"] = expected
        if expected is not None and expected != image.sha256:
            mismatches["database_vs_frozen_sha256"] = image.sha256
        if mismatches:
            raise InputArtifactMismatchError(
                details={
                    "run_id": run_id,
                    "image_id": image.image_id,
                    "observed_sha256": observed,
                    **mismatches,
                }
            )

    def _load_corrected_mask(
        self,
        token: str,
        *,
        expected_shape: tuple[int, int],
    ) -> tuple[np.ndarray, Path]:
        try:
            path = self.file_store.resolve_file_token(token)
        except (FileTokenError, FileNotFoundError, OSError, StoragePathError) as error:
            raise InvalidImageError(
                "人工修正掩膜令牌无效或已过期",
                details={"reason": "invalid_corrected_mask_token"},
            ) from error
        try:
            if path.suffix.casefold() == ".npy":
                data = np.load(path, allow_pickle=False)
            else:
                with Image.open(path) as image:
                    expected_size = (expected_shape[1], expected_shape[0])
                    if image.size != expected_size:
                        raise InvalidImageError(
                            "人工修正掩膜尺寸必须与原图一致且为单通道",
                            details={
                                "reason": "corrected_mask_shape_mismatch",
                                "expected_shape": expected_shape,
                                "observed_shape": (image.height, image.width),
                            },
                        )
                    data = np.asarray(image)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            raise InvalidImageError(
                "人工修正掩膜无法读取",
                details={"reason": "corrected_mask_decode_failed"},
            ) from error
        if data.ndim != 2 or data.shape != expected_shape:
            raise InvalidImageError(
                "人工修正掩膜尺寸必须与原图一致且为单通道",
                details={
                    "reason": "corrected_mask_shape_mismatch",
                    "expected_shape": expected_shape,
                    "observed_shape": data.shape,
                },
            )
        if not np.issubdtype(data.dtype, np.number) or not np.isfinite(data).all():
            raise InvalidImageError(details={"reason": "corrected_mask_invalid_values"})
        unique = np.unique(data)
        if unique.size > 2:
            raise InvalidImageError(
                "人工修正掩膜必须是二值图像",
                details={"reason": "corrected_mask_not_binary", "unique_values": unique.size},
            )
        return np.asarray(data > 0, dtype=bool), path

    def _delete_consumed_corrected_mask(self, *, job_id: str, path: Path) -> None:
        """Consume one staged review upload without touching durable image originals."""

        try:
            managed = self.file_store.paths.require_managed(path, must_exist=True)
            relative = managed.relative_to(self.file_store.paths.input_dir(job_id))
        except (FileNotFoundError, OSError, StoragePathError, ValueError):
            return
        if (
            len(relative.parts) != 2
            or not relative.parts[0].startswith("review_mask_")
            or (
                relative.parts[1] != "original"
                and not relative.parts[1].startswith("original.")
            )
        ):
            return

        try:
            with self.uow_factory() as uow:
                images = uow.repositories.images.list_by_job(job_id)
                referenced = {
                    self.file_store.paths.require_managed(
                        uow.repositories.images.get_storage_path(image.image_id)
                    )
                    for image in images
                }
            if managed not in referenced:
                managed.unlink(missing_ok=True)
        except Exception:
            logger.warning(
                "corrected_mask_cleanup_error",
                extra={"job_id": job_id, "event": "corrected_mask_cleanup_error"},
                exc_info=True,
            )

    def _load_binary_mask(self, path: Path, expected_shape: tuple[int, ...]) -> np.ndarray:
        managed = self.file_store.paths.require_managed(path, must_exist=True)
        if managed.suffix.lower() == ".npy":
            data = np.load(managed, allow_pickle=False)
        else:
            with Image.open(managed) as image:
                data = np.asarray(image)
        if data.shape != expected_shape:
            raise InferenceExecutionError(
                details={"reason": "binary_mask_shape_mismatch", "shape": data.shape}
            )
        return np.asarray(data > 0, dtype=bool)

    def _load_probability(
        self, path: Path | None, expected_shape: tuple[int, ...]
    ) -> np.ndarray | None:
        if path is None:
            return None
        managed = self.file_store.paths.require_managed(path, must_exist=True)
        if managed.suffix.lower() == ".npy":
            values = np.load(managed, allow_pickle=False)
        else:
            with Image.open(managed) as image:
                values = np.asarray(image, dtype=np.float32)
                if values.max(initial=0) > 1:
                    values = values / 255.0
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise InferenceExecutionError(details={"reason": "invalid_probability_map"})
        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def _union(instances: list[NormalizedInstance], shape: tuple[int, ...]) -> np.ndarray:
        if not instances:
            return np.zeros(shape, dtype=bool)
        union = np.zeros(shape, dtype=bool)
        for instance in instances:
            union |= instance.mask
        return union

    def _relative_managed(self, path: Path) -> str:
        managed = self.file_store.paths.require_managed(path, must_exist=True)
        return self.file_store.paths.relative_path(managed)

    def _optional_relative(self, path: Path | None) -> str | None:
        return self._relative_managed(path) if path is not None else None

    @staticmethod
    def _file_contract_payload(payload: dict[str, object]) -> dict[str, object]:
        prepared = dict(payload)
        contract_version = prepared.pop("schema_version", None)
        if contract_version is not None:
            prepared["contract_schema_version"] = contract_version
        return prepared

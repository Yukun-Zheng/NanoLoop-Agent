"""SQLAlchemy implementations of the service-facing repository protocols."""

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any, Self, cast
from uuid import uuid4

from sqlalchemy import delete, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session, selectinload

from app.contracts.analyses import (
    AnalysisJobDTO,
    AnalysisROI,
    BoxSetDTO,
    ImageAssetDTO,
    ImageSummaryDTO,
    InferenceOptions,
    ParticleRecordDTO,
    QualityReportDTO,
    ROIBox,
    RunArtifacts,
    RunConfiguration,
    RunStatusEventDTO,
    SegmentationRunDTO,
)
from app.contracts.enums import JobStatus, QualityStatus, RoiMode
from app.contracts.execution import ExecutionRuntimeProvenance
from app.contracts.identity import validate_principal_id, validate_tenant_id
from app.contracts.queries import QueryAuditRecordDTO, UnifiedQueryRequest, UnifiedQueryResponse
from app.contracts.repositories import AnalysisResourceScope, StoredImageAsset
from app.core.errors import BoxRevisionConflictError, InvalidBoxError, ResourceNotFoundError
from app.core.state_machine import ensure_transition
from app.db.models import (
    AnalysisJob,
    ImageAsset,
    ImageSummary,
    ParticleRecord,
    QueryLog,
    ROIBoxRecord,
    ROIBoxRevisionRecord,
    RunStatusEvent,
    SegmentationRun,
)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _job_dto(record: AnalysisJob) -> AnalysisJobDTO:
    return AnalysisJobDTO(
        job_id=record.job_id,
        name=record.name,
        status=JobStatus(record.status),
        config=record.config_json,
        created_at=_utc(record.created_at),
        updated_at=_utc(record.updated_at),
        error_code=record.error_code,
    )


def _analysis_scope(record: AnalysisJob) -> AnalysisResourceScope:
    return AnalysisResourceScope(
        job=_job_dto(record),
        tenant_id=record.tenant_id,
        owner_principal_id=record.owner_principal_id,
    )


def _require_scoped_job(
    session: Session,
    job_id: str,
    tenant_id: str,
) -> AnalysisJob:
    record = session.scalar(
        select(AnalysisJob).where(
            AnalysisJob.job_id == job_id,
            AnalysisJob.tenant_id == validate_tenant_id(tenant_id),
        )
    )
    if record is None:
        raise ResourceNotFoundError(details={"resource": "job", "job_id": job_id})
    return record


def _require_scoped_image(
    session: Session,
    job_id: str,
    image_id: str,
    tenant_id: str,
) -> ImageAsset:
    record = session.scalar(
        select(ImageAsset)
        .join(AnalysisJob, AnalysisJob.job_id == ImageAsset.job_id)
        .where(
            ImageAsset.image_id == image_id,
            ImageAsset.job_id == job_id,
            AnalysisJob.tenant_id == validate_tenant_id(tenant_id),
        )
    )
    if record is None:
        raise ResourceNotFoundError(
            details={"resource": "image", "job_id": job_id, "image_id": image_id}
        )
    return record


def _image_dto(record: ImageAsset) -> ImageAssetDTO:
    return ImageAssetDTO(
        image_id=record.image_id,
        job_id=record.job_id,
        filename=record.filename,
        sha256=record.sha256,
        width=record.width,
        height=record.height,
        bit_depth=record.bit_depth,
        sample_id=record.sample_id,
        material_name=record.material_name,
        material_formula=record.material_formula,
        experiment_conditions=record.experiment_conditions_json,
        scale_nm_per_pixel=record.scale_nm_per_pixel,
        analysis_roi=AnalysisROI.model_validate(record.analysis_roi_json),
    )


def _summary_dto(record: ImageSummary) -> ImageSummaryDTO:
    return ImageSummaryDTO(
        run_id=record.run_id,
        particle_count=record.particle_count,
        roi_area_px=record.roi_area_px,
        number_density_px2=record.number_density_px2,
        number_density_um2=record.number_density_um2,
        mean_equivalent_diameter_px=record.mean_equivalent_diameter_px,
        mean_equivalent_diameter_nm=record.mean_equivalent_diameter_nm,
        coverage_ratio=record.coverage_ratio,
        perimeter_density_px=record.perimeter_density_px,
        perimeter_density_um=record.perimeter_density_um,
        quality_status=QualityStatus(record.quality_status),
    )


def _quality_dto(record: ImageSummary) -> QualityReportDTO:
    payload = dict(record.quality_json)
    payload.setdefault("status", record.quality_status)
    return QualityReportDTO.model_validate(payload)


def _run_dto(record: SegmentationRun) -> SegmentationRunDTO:
    summary = _summary_dto(record.summary) if record.summary is not None else None
    quality = _quality_dto(record.summary) if record.summary is not None else None
    public_paths = {
        key: value for key, value in record.paths_json.items() if key in RunArtifacts.model_fields
    }
    return SegmentationRunDTO(
        run_id=record.run_id,
        job_id=record.job_id,
        image_id=record.image_id,
        model_id=record.model_id,
        status=JobStatus(record.status),
        roi_mode=RoiMode(record.roi_mode),
        box_revision=record.box_revision,
        threshold=record.threshold,
        inference=InferenceOptions.model_validate(record.inference_json),
        configuration=RunConfiguration.model_validate(record.run_config_json),
        parent_run_id=record.parent_run_id,
        artifacts=RunArtifacts.model_validate(public_paths),
        summary=summary,
        quality=quality,
        execution=(
            ExecutionRuntimeProvenance.model_validate(record.execution_json)
            if record.execution_json is not None
            else None
        ),
        runtime_ms=record.runtime_ms,
        error_code=record.error_code,
        error_message=record.error_message,
        status_history=[
            RunStatusEventDTO(
                event_id=event.event_id,
                from_status=(
                    JobStatus(event.from_status) if event.from_status is not None else None
                ),
                to_status=JobStatus(event.to_status),
                error_code=event.error_code,
                error_message=event.error_message,
                created_at=_utc(event.created_at),
            )
            for event in record.status_events
        ],
        created_at=_utc(record.created_at),
        updated_at=_utc(record.updated_at),
    )


class SqlAlchemyJobRepository:
    """Job persistence; HTTP callers use ``get_scope`` instead of unscoped reads."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        job: AnalysisJobDTO,
        *,
        tenant_id: str,
        owner_principal_id: str,
    ) -> AnalysisJobDTO:
        validated_tenant_id = validate_tenant_id(tenant_id)
        validated_owner_principal_id = validate_principal_id(owner_principal_id)
        record = AnalysisJob(
            job_id=job.job_id,
            tenant_id=validated_tenant_id,
            owner_principal_id=validated_owner_principal_id,
            name=job.name,
            status=job.status.value,
            config_json=job.config,
            error_code=job.error_code,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )
        self.session.add(record)
        self.session.flush()
        return _job_dto(record)

    def get(self, job_id: str) -> AnalysisJobDTO:
        """Return an unscoped job for trusted internal workers only."""

        record = self.session.get(AnalysisJob, job_id)
        if record is None:
            raise ResourceNotFoundError(details={"resource": "job", "job_id": job_id})
        return _job_dto(record)

    def get_scope(self, job_id: str, *, tenant_id: str) -> AnalysisResourceScope:
        """Resolve one aggregate only when it belongs to the authenticated tenant."""

        return _analysis_scope(_require_scoped_job(self.session, job_id, tenant_id))

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error_code: str | None = None,
    ) -> None:
        record = self.session.get(AnalysisJob, job_id)
        if record is None:
            raise ResourceNotFoundError(details={"resource": "job", "job_id": job_id})
        record.status = status.value
        record.error_code = error_code
        self.session.flush()


class SqlAlchemyImageRepository:
    """Image persistence with separate trusted-internal and tenant-scoped reads."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def add_many(self, images: list[StoredImageAsset]) -> list[ImageAssetDTO]:
        records: list[ImageAsset] = []
        for image in images:
            dto = image.asset
            record = ImageAsset(
                image_id=dto.image_id,
                job_id=dto.job_id,
                filename=dto.filename,
                storage_path=image.storage_path,
                sha256=dto.sha256,
                width=dto.width,
                height=dto.height,
                bit_depth=dto.bit_depth,
                sample_id=dto.sample_id,
                material_name=dto.material_name,
                material_formula=dto.material_formula,
                experiment_conditions_json=dto.experiment_conditions,
                analysis_roi_json=dto.analysis_roi.model_dump(mode="json"),
                scale_nm_per_pixel=dto.scale_nm_per_pixel,
                box_revision=0,
            )
            records.append(record)
        self.session.add_all(records)
        self.session.flush()
        self.session.add_all(
            [
                ROIBoxRevisionRecord(
                    image_id=record.image_id,
                    revision=0,
                    box_count=0,
                    created_at=record.created_at,
                )
                for record in records
            ]
        )
        self.session.flush()
        return [_image_dto(record) for record in records]

    def get(self, image_id: str) -> ImageAssetDTO:
        """Return an unscoped image for trusted internal workers only."""

        record = self.session.get(ImageAsset, image_id)
        if record is None:
            raise ResourceNotFoundError(details={"resource": "image", "image_id": image_id})
        return _image_dto(record)

    def list_by_job(self, job_id: str) -> list[ImageAssetDTO]:
        records = self.session.scalars(
            select(ImageAsset).where(ImageAsset.job_id == job_id).order_by(ImageAsset.created_at)
        ).all()
        return [_image_dto(record) for record in records]

    def get_storage_path(self, image_id: str) -> str:
        """Return an unscoped storage key for trusted internal workers only."""

        record = self.session.get(ImageAsset, image_id)
        if record is None:
            raise ResourceNotFoundError(details={"resource": "image", "image_id": image_id})
        return record.storage_path

    def get_scoped(
        self,
        job_id: str,
        image_id: str,
        *,
        tenant_id: str,
    ) -> ImageAssetDTO:
        return _image_dto(_require_scoped_image(self.session, job_id, image_id, tenant_id))

    def list_by_job_scoped(
        self,
        job_id: str,
        *,
        tenant_id: str,
    ) -> list[ImageAssetDTO]:
        _require_scoped_job(self.session, job_id, tenant_id)
        records = self.session.scalars(
            select(ImageAsset)
            .join(AnalysisJob, AnalysisJob.job_id == ImageAsset.job_id)
            .where(
                ImageAsset.job_id == job_id,
                AnalysisJob.tenant_id == validate_tenant_id(tenant_id),
            )
            .order_by(ImageAsset.created_at)
        ).all()
        return [_image_dto(record) for record in records]

    def get_storage_path_scoped(
        self,
        job_id: str,
        image_id: str,
        *,
        tenant_id: str,
    ) -> str:
        record = self.session.execute(
            select(ImageAsset.storage_path)
            .join(AnalysisJob, AnalysisJob.job_id == ImageAsset.job_id)
            .where(
                ImageAsset.image_id == image_id,
                ImageAsset.job_id == job_id,
                AnalysisJob.tenant_id == validate_tenant_id(tenant_id),
            )
        ).scalar_one_or_none()
        if record is None:
            raise ResourceNotFoundError(
                details={"resource": "image", "job_id": job_id, "image_id": image_id}
            )
        return record


def _intersects(a: ROIBox, b: dict[str, Any]) -> bool:
    x1, y1, x2, y2 = (int(b[key]) for key in ("x1", "y1", "x2", "y2"))
    return a.x1 < x2 and a.x2 > x1 and a.y1 < y2 and a.y2 > y1


class SqlAlchemyBoxRepository:
    """Box persistence with tenant-scoped HTTP entry points and internal primitives."""

    def __init__(self, session: Session, *, minimum_size_px: int = 32) -> None:
        self.session = session
        self.minimum_size_px = minimum_size_px

    def get_active(self, image_id: str) -> BoxSetDTO:
        """Return unscoped boxes for trusted internal workers only."""

        image = self.session.get(ImageAsset, image_id)
        if image is None:
            raise ResourceNotFoundError(details={"resource": "image", "image_id": image_id})
        return self._active_for_image(image)

    def get_active_scoped(
        self,
        job_id: str,
        image_id: str,
        *,
        tenant_id: str,
    ) -> BoxSetDTO:
        image = _require_scoped_image(self.session, job_id, image_id, tenant_id)
        return self._active_for_image(image)

    def _active_for_image(self, image: ImageAsset) -> BoxSetDTO:
        records = self.session.scalars(
            select(ROIBoxRecord)
            .where(
                ROIBoxRecord.image_id == image.image_id,
                ROIBoxRecord.revision == image.box_revision,
            )
            .order_by(ROIBoxRecord.row_id)
        ).all()
        return BoxSetDTO(
            image_id=image.image_id,
            revision=image.box_revision,
            boxes=[
                ROIBox(
                    box_id=record.box_id,
                    label=record.label,
                    x1=record.x1,
                    y1=record.y1,
                    x2=record.x2,
                    y2=record.y2,
                    active=record.active,
                )
                for record in records
            ],
        )

    def list_by_job(self, job_id: str) -> list[BoxSetDTO]:
        """Return unscoped revision history for trusted internal workers only."""

        revision_rows = self.session.scalars(
            select(ROIBoxRevisionRecord)
            .join(ImageAsset, ImageAsset.image_id == ROIBoxRevisionRecord.image_id)
            .where(ImageAsset.job_id == job_id)
            .order_by(ROIBoxRevisionRecord.image_id, ROIBoxRevisionRecord.revision)
        ).all()
        return self._box_sets(revision_rows)

    def list_by_job_scoped(
        self,
        job_id: str,
        *,
        tenant_id: str,
    ) -> list[BoxSetDTO]:
        _require_scoped_job(self.session, job_id, tenant_id)
        revision_rows = self.session.scalars(
            select(ROIBoxRevisionRecord)
            .join(ImageAsset, ImageAsset.image_id == ROIBoxRevisionRecord.image_id)
            .join(AnalysisJob, AnalysisJob.job_id == ImageAsset.job_id)
            .where(
                ImageAsset.job_id == job_id,
                AnalysisJob.tenant_id == validate_tenant_id(tenant_id),
            )
            .order_by(ROIBoxRevisionRecord.image_id, ROIBoxRevisionRecord.revision)
        ).all()
        return self._box_sets(revision_rows)

    def _box_sets(self, revision_rows: Sequence[ROIBoxRevisionRecord]) -> list[BoxSetDTO]:
        if not revision_rows:
            return []
        image_ids = {row.image_id for row in revision_rows}
        box_rows = self.session.scalars(
            select(ROIBoxRecord)
            .where(ROIBoxRecord.image_id.in_(image_ids))
            .order_by(ROIBoxRecord.image_id, ROIBoxRecord.revision, ROIBoxRecord.row_id)
        ).all()
        boxes_by_revision: dict[tuple[str, int], list[ROIBox]] = {}
        for record in box_rows:
            boxes_by_revision.setdefault((record.image_id, record.revision), []).append(
                ROIBox(
                    box_id=record.box_id,
                    label=record.label,
                    x1=record.x1,
                    y1=record.y1,
                    x2=record.x2,
                    y2=record.y2,
                    active=record.active,
                )
            )
        return [
            BoxSetDTO(
                image_id=row.image_id,
                revision=row.revision,
                boxes=boxes_by_revision.get((row.image_id, row.revision), []),
            )
            for row in revision_rows
        ]

    def replace(
        self,
        image_id: str,
        expected_revision: int,
        boxes: list[ROIBox],
    ) -> BoxSetDTO:
        """Replace unscoped boxes for trusted internal workers only."""

        image = self.session.get(ImageAsset, image_id)
        if image is None:
            raise ResourceNotFoundError(details={"resource": "image", "image_id": image_id})
        return self._replace_image(image, expected_revision, boxes)

    def replace_scoped(
        self,
        job_id: str,
        image_id: str,
        expected_revision: int,
        boxes: list[ROIBox],
        *,
        tenant_id: str,
    ) -> BoxSetDTO:
        image = _require_scoped_image(self.session, job_id, image_id, tenant_id)
        return self._replace_image(image, expected_revision, boxes)

    def _replace_image(
        self,
        image: ImageAsset,
        expected_revision: int,
        boxes: list[ROIBox],
    ) -> BoxSetDTO:
        image_id = image.image_id
        if image.box_revision != expected_revision:
            raise BoxRevisionConflictError(
                details={
                    "image_id": image_id,
                    "expected_revision": expected_revision,
                    "current_revision": image.box_revision,
                }
            )
        self._validate_boxes(image, boxes)
        revision = expected_revision + 1
        # SQLite ignores SELECT ... FOR UPDATE.  Claim the revision with a
        # compare-and-swap UPDATE instead so two writers cannot both publish
        # revision N+1.  A losing transaction has not inserted any box rows yet.
        claimed = cast(
            CursorResult[Any],
            self.session.execute(
                update(ImageAsset)
                .where(
                    ImageAsset.image_id == image_id,
                    ImageAsset.box_revision == expected_revision,
                )
                .values(box_revision=revision)
            ),
        )
        if claimed.rowcount != 1:
            self.session.expire_all()
            current = self.session.get(ImageAsset, image_id)
            raise BoxRevisionConflictError(
                details={
                    "image_id": image_id,
                    "expected_revision": expected_revision,
                    "current_revision": current.box_revision if current is not None else None,
                }
            )
        self.session.expire(image, ["box_revision"])
        assigned: list[ROIBox] = []
        seen_ids: set[str] = set()
        for position, box in enumerate(boxes, start=1):
            box_id = box.box_id or f"box_{uuid4().hex}"
            if box_id in seen_ids:
                raise InvalidBoxError(details={"box_id": box_id, "reason": "duplicate_box_id"})
            seen_ids.add(box_id)
            assigned_box = box.model_copy(update={"box_id": box_id})
            assigned.append(assigned_box)
            self.session.add(
                ROIBoxRecord(
                    box_id=box_id,
                    image_id=image_id,
                    x1=box.x1,
                    y1=box.y1,
                    x2=box.x2,
                    y2=box.y2,
                    label=box.label or f"区域 {position}",
                    active=box.active,
                    revision=revision,
                )
            )
        self.session.add(
            ROIBoxRevisionRecord(
                image_id=image_id,
                revision=revision,
                box_count=len(assigned),
            )
        )
        self.session.flush()
        return BoxSetDTO(image_id=image_id, revision=revision, boxes=assigned)

    def _validate_boxes(self, image: ImageAsset, boxes: list[ROIBox]) -> None:
        roi = AnalysisROI.model_validate(image.analysis_roi_json)
        valid = roi.valid_rect.model_dump()
        invalid = [rect.model_dump() for rect in roi.invalid_rects]
        for box in boxes:
            if box.x2 - box.x1 < self.minimum_size_px or box.y2 - box.y1 < self.minimum_size_px:
                raise InvalidBoxError(
                    details={"box_id": box.box_id, "reason": "minimum_size", "min_px": 32}
                )
            if (
                box.x1 < valid["x1"]
                or box.y1 < valid["y1"]
                or box.x2 > valid["x2"]
                or box.y2 > valid["y2"]
                or box.x2 > image.width
                or box.y2 > image.height
            ):
                raise InvalidBoxError(details={"box_id": box.box_id, "reason": "out_of_bounds"})
            if any(_intersects(box, region) for region in invalid):
                raise InvalidBoxError(
                    details={"box_id": box.box_id, "reason": "overlaps_invalid_region"}
                )


class SqlAlchemyRunRepository:
    """Run persistence; dispatcher/worker primitives remain deliberately unscoped."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_many(self, runs: list[SegmentationRunDTO]) -> list[str]:
        records = [
            SegmentationRun(
                run_id=run.run_id,
                job_id=run.job_id,
                image_id=run.image_id,
                model_id=run.model_id,
                roi_mode=run.roi_mode.value,
                box_revision=run.box_revision,
                threshold=run.threshold,
                status=run.status.value,
                inference_json=run.inference.model_dump(mode="json"),
                run_config_json=run.configuration.model_dump(mode="json"),
                execution_json=(
                    run.execution.model_dump(mode="json")
                    if run.execution is not None
                    else None
                ),
                paths_json=run.artifacts.model_dump(mode="json"),
                runtime_ms=run.runtime_ms,
                parent_run_id=run.parent_run_id,
                error_code=run.error_code,
                error_message=run.error_message,
                created_at=run.created_at,
                updated_at=run.updated_at,
            )
            for run in runs
        ]
        self.session.add_all(records)
        self.session.flush()
        self.session.add_all(
            [
                RunStatusEvent(
                    run_id=record.run_id,
                    from_status=None,
                    to_status=record.status,
                    created_at=record.created_at,
                )
                for record in records
            ]
        )
        self.session.flush()
        return [record.run_id for record in records]

    def get(self, run_id: str) -> SegmentationRunDTO:
        """Return an unscoped run for trusted dispatcher/worker paths only."""

        record = self.session.scalar(
            select(SegmentationRun)
            .where(SegmentationRun.run_id == run_id)
            .options(
                selectinload(SegmentationRun.summary),
                selectinload(SegmentationRun.status_events),
            )
        )
        if record is None:
            raise ResourceNotFoundError(details={"resource": "run", "run_id": run_id})
        return _run_dto(record)

    def get_with_scope(
        self,
        run_id: str,
        *,
        tenant_id: str,
    ) -> tuple[SegmentationRunDTO, AnalysisResourceScope]:
        row = self.session.execute(
            select(SegmentationRun, AnalysisJob)
            .join(AnalysisJob, AnalysisJob.job_id == SegmentationRun.job_id)
            .where(
                SegmentationRun.run_id == run_id,
                AnalysisJob.tenant_id == validate_tenant_id(tenant_id),
            )
            .options(
                selectinload(SegmentationRun.summary),
                selectinload(SegmentationRun.status_events),
            )
        ).one_or_none()
        if row is None:
            raise ResourceNotFoundError(details={"resource": "run", "run_id": run_id})
        run, job = row
        return _run_dto(run), _analysis_scope(job)

    def list_by_job(self, job_id: str) -> list[SegmentationRunDTO]:
        """Return unscoped runs for trusted internal workers only."""

        records = self.session.scalars(
            select(SegmentationRun)
            .where(SegmentationRun.job_id == job_id)
            .options(
                selectinload(SegmentationRun.summary),
                selectinload(SegmentationRun.status_events),
            )
            .order_by(SegmentationRun.created_at)
        ).all()
        return [_run_dto(record) for record in records]

    def list_by_job_scoped(
        self,
        job_id: str,
        *,
        tenant_id: str,
    ) -> list[SegmentationRunDTO]:
        _require_scoped_job(self.session, job_id, tenant_id)
        records = self.session.scalars(
            select(SegmentationRun)
            .join(AnalysisJob, AnalysisJob.job_id == SegmentationRun.job_id)
            .where(
                SegmentationRun.job_id == job_id,
                AnalysisJob.tenant_id == validate_tenant_id(tenant_id),
            )
            .options(
                selectinload(SegmentationRun.summary),
                selectinload(SegmentationRun.status_events),
            )
            .order_by(SegmentationRun.created_at)
        ).all()
        return [_run_dto(record) for record in records]

    def get_artifact_paths(self, run_id: str) -> dict[str, str | None]:
        """Return unscoped artifact keys for trusted dispatcher/worker paths only."""

        record = self.session.get(SegmentationRun, run_id)
        if record is None:
            raise ResourceNotFoundError(details={"resource": "run", "run_id": run_id})
        return dict(record.paths_json)

    def get_artifact_paths_scoped(
        self,
        run_id: str,
        *,
        tenant_id: str,
    ) -> dict[str, str | None]:
        paths = self.session.execute(
            select(SegmentationRun.paths_json)
            .join(AnalysisJob, AnalysisJob.job_id == SegmentationRun.job_id)
            .where(
                SegmentationRun.run_id == run_id,
                AnalysisJob.tenant_id == validate_tenant_id(tenant_id),
            )
        ).scalar_one_or_none()
        if paths is None:
            raise ResourceNotFoundError(details={"resource": "run", "run_id": run_id})
        return dict(paths)

    def claim_queued(self, run_id: str) -> bool:
        """Atomically claim a queued row for exactly one worker/process."""

        result = cast(
            CursorResult[Any],
            self.session.execute(
                update(SegmentationRun)
                .where(
                    SegmentationRun.run_id == run_id,
                    SegmentationRun.status == JobStatus.QUEUED.value,
                )
                .values(
                    status=JobStatus.PREPROCESSING.value,
                    error_code=None,
                    error_message=None,
                    updated_at=datetime.now(UTC),
                )
            ),
        )
        if result.rowcount == 1:
            self.session.add(
                RunStatusEvent(
                    run_id=run_id,
                    from_status=JobStatus.QUEUED.value,
                    to_status=JobStatus.PREPROCESSING.value,
                )
            )
        self.session.flush()
        return result.rowcount == 1

    def update_status(
        self,
        run_id: str,
        status: JobStatus,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        record = self.session.get(SegmentationRun, run_id)
        if record is None:
            raise ResourceNotFoundError(details={"resource": "run", "run_id": run_id})
        previous = JobStatus(record.status)
        ensure_transition(previous, status)
        record.status = status.value
        record.error_code = error_code
        record.error_message = error_message
        self.session.add(
            RunStatusEvent(
                run_id=run_id,
                from_status=previous.value,
                to_status=status.value,
                error_code=error_code,
                error_message=error_message,
            )
        )
        self.session.flush()

    def save_result(
        self,
        run_id: str,
        *,
        particles: list[ParticleRecordDTO],
        summary: ImageSummaryDTO,
        quality: QualityReportDTO,
        execution: ExecutionRuntimeProvenance,
        runtime_ms: int,
        paths: dict[str, str | None],
    ) -> None:
        record = self.session.get(SegmentationRun, run_id)
        if record is None:
            raise ResourceNotFoundError(details={"resource": "run", "run_id": run_id})
        self.session.execute(delete(ParticleRecord).where(ParticleRecord.run_id == run_id))
        self.session.add_all(
            [
                ParticleRecord(
                    particle_id=particle.particle_id,
                    run_id=run_id,
                    instance_index=particle.instance_index,
                    area_px=particle.area_px,
                    perimeter_px=particle.perimeter_px,
                    equivalent_diameter_px=particle.equivalent_diameter_px,
                    equivalent_diameter_nm=particle.equivalent_diameter_nm,
                    circularity=particle.circularity,
                    bbox_json=list(particle.bbox),
                    confidence=particle.confidence,
                )
                for particle in particles
            ]
        )
        existing_summary = self.session.get(ImageSummary, run_id)
        if existing_summary is not None:
            self.session.delete(existing_summary)
            self.session.flush()
        self.session.add(
            ImageSummary(
                run_id=run_id,
                particle_count=summary.particle_count,
                roi_area_px=summary.roi_area_px,
                number_density_px2=summary.number_density_px2,
                number_density_um2=summary.number_density_um2,
                mean_equivalent_diameter_px=summary.mean_equivalent_diameter_px,
                mean_equivalent_diameter_nm=summary.mean_equivalent_diameter_nm,
                coverage_ratio=summary.coverage_ratio,
                perimeter_density_px=summary.perimeter_density_px,
                perimeter_density_um=summary.perimeter_density_um,
                quality_status=quality.status.value,
                quality_json=quality.model_dump(mode="json"),
            )
        )
        record.runtime_ms = runtime_ms
        record.execution_json = execution.model_dump(mode="json")
        record.paths_json = paths
        self.session.flush()


class SqlAlchemyQueryRepository:
    """Query audit reads, including a tenant-scoped export snapshot boundary."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def list_by_job(self, job_id: str) -> list[QueryAuditRecordDTO]:
        """Return unscoped query audit rows for trusted internal workers only."""

        records = self.session.scalars(
            select(QueryLog)
            .where(QueryLog.job_id == job_id)
            .order_by(QueryLog.created_at, QueryLog.query_id)
        ).all()
        return self._query_dtos(records)

    def list_by_job_scoped(
        self,
        job_id: str,
        *,
        tenant_id: str,
    ) -> list[QueryAuditRecordDTO]:
        _require_scoped_job(self.session, job_id, tenant_id)
        records = self.session.scalars(
            select(QueryLog)
            .join(AnalysisJob, AnalysisJob.job_id == QueryLog.job_id)
            .where(
                QueryLog.job_id == job_id,
                AnalysisJob.tenant_id == validate_tenant_id(tenant_id),
            )
            .order_by(QueryLog.created_at, QueryLog.query_id)
        ).all()
        return self._query_dtos(records)

    @staticmethod
    def _query_dtos(records: Sequence[QueryLog]) -> list[QueryAuditRecordDTO]:
        results: list[QueryAuditRecordDTO] = []
        for record in records:
            request_payload = dict(record.request_json)
            request_payload.setdefault("question", record.question)
            request_payload.setdefault("query_type", record.query_type)
            request_payload.setdefault("image_id", record.image_id)
            results.append(
                QueryAuditRecordDTO(
                    query_id=record.query_id,
                    job_id=record.job_id,
                    image_id=record.image_id,
                    request=UnifiedQueryRequest.model_validate(request_payload),
                    response=UnifiedQueryResponse.model_validate(record.answer_json),
                    created_at=_utc(record.created_at),
                )
            )
        return results


class SqlAlchemyRepositorySet:
    def __init__(self, session: Session) -> None:
        self.jobs = SqlAlchemyJobRepository(session)
        self.images = SqlAlchemyImageRepository(session)
        self.boxes = SqlAlchemyBoxRepository(session)
        self.runs = SqlAlchemyRunRepository(session)
        self.queries = SqlAlchemyQueryRepository(session)


class SqlAlchemyUnitOfWork:
    """Short-lived explicit transaction boundary for application services."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self.session_factory = session_factory
        self.session: Session | None = None
        self._repositories: SqlAlchemyRepositorySet | None = None

    @property
    def repositories(self) -> SqlAlchemyRepositorySet:
        if self._repositories is None:
            raise RuntimeError("unit of work is not active")
        return self._repositories

    def __enter__(self) -> Self:
        self.session = self.session_factory()
        self._repositories = SqlAlchemyRepositorySet(self.session)
        return self

    def commit(self) -> None:
        if self.session is None:
            raise RuntimeError("unit of work is not active")
        self.session.commit()

    def rollback(self) -> None:
        if self.session is not None:
            self.session.rollback()

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self.session is not None:
            if exc_type is not None:
                self.session.rollback()
            self.session.close()
        self.session = None
        self._repositories = None

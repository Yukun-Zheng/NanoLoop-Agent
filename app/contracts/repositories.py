"""Persistence protocols used by services to avoid ORM coupling."""

from contextlib import AbstractContextManager
from typing import Protocol, Self

from app.contracts.analyses import (
    AnalysisJobDTO,
    BoxSetDTO,
    ImageAssetDTO,
    ImageSummaryDTO,
    ParticleRecordDTO,
    QualityReportDTO,
    ROIBox,
    SegmentationRunDTO,
)
from app.contracts.common import ContractModel
from app.contracts.enums import JobStatus
from app.contracts.execution import ExecutionRuntimeProvenance
from app.contracts.queries import QueryAuditRecordDTO


class StoredImageAsset(ContractModel):
    """Internal persistence input pairing a public asset DTO with its private storage key."""

    asset: ImageAssetDTO
    storage_path: str


class JobRepository(Protocol):
    def create(self, job: AnalysisJobDTO) -> AnalysisJobDTO: ...

    def get(self, job_id: str) -> AnalysisJobDTO: ...

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error_code: str | None = None,
    ) -> None: ...


class ImageRepository(Protocol):
    def add_many(self, images: list[StoredImageAsset]) -> list[ImageAssetDTO]: ...

    def get(self, image_id: str) -> ImageAssetDTO: ...

    def list_by_job(self, job_id: str) -> list[ImageAssetDTO]: ...

    def get_storage_path(self, image_id: str) -> str: ...


class BoxRepository(Protocol):
    def get_active(self, image_id: str) -> BoxSetDTO: ...

    def list_by_job(self, job_id: str) -> list[BoxSetDTO]: ...

    def replace(
        self,
        image_id: str,
        expected_revision: int,
        boxes: list[ROIBox],
    ) -> BoxSetDTO: ...


class RunRepository(Protocol):
    def create_many(self, runs: list[SegmentationRunDTO]) -> list[str]: ...

    def get(self, run_id: str) -> SegmentationRunDTO: ...

    def list_by_job(self, job_id: str) -> list[SegmentationRunDTO]: ...

    def get_artifact_paths(self, run_id: str) -> dict[str, str | None]: ...

    def claim_queued(self, run_id: str) -> bool:
        """Atomically move one durable run from QUEUED to PREPROCESSING."""
        ...

    def update_status(
        self,
        run_id: str,
        status: JobStatus,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None: ...

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
    ) -> None: ...


class QueryRepository(Protocol):
    def list_by_job(self, job_id: str) -> list[QueryAuditRecordDTO]: ...


class RepositorySet(Protocol):
    @property
    def jobs(self) -> JobRepository: ...

    @property
    def images(self) -> ImageRepository: ...

    @property
    def boxes(self) -> BoxRepository: ...

    @property
    def runs(self) -> RunRepository: ...

    @property
    def queries(self) -> QueryRepository: ...


class UnitOfWork(AbstractContextManager["UnitOfWork"], Protocol):
    @property
    def repositories(self) -> RepositorySet: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool | None: ...


class UnitOfWorkFactory(Protocol):
    def __call__(self) -> UnitOfWork: ...

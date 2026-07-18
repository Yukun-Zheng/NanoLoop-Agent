"""Audited ROI-box replacement over persistence and file artifacts."""

import logging
from collections.abc import Callable

from app.contracts.analyses import BoxSetDTO, ROIBox
from app.contracts.repositories import UnitOfWork
from app.core.errors import ResourceNotFoundError
from app.core.logging import log_context
from app.storage import LocalFileStore

logger = logging.getLogger(__name__)


class BoxApplicationService:
    def __init__(
        self,
        *,
        uow_factory: Callable[[], UnitOfWork],
        file_store: LocalFileStore,
    ) -> None:
        self.uow_factory = uow_factory
        self.file_store = file_store

    def replace(
        self,
        *,
        job_id: str,
        image_id: str,
        expected_revision: int,
        boxes: list[ROIBox],
    ) -> BoxSetDTO:
        with self.uow_factory() as uow:
            uow.repositories.jobs.get(job_id)
            image = uow.repositories.images.get(image_id)
            if image.job_id != job_id:
                raise ResourceNotFoundError(
                    details={"resource": "image", "job_id": job_id, "image_id": image_id}
                )
            result = uow.repositories.boxes.replace(image_id, expected_revision, boxes)
            uow.commit()

        # The relational revision is authoritative.  This JSON snapshot is a
        # rebuildable audit projection, so a filesystem outage after commit must
        # not turn a successful optimistic-lock update into an apparent failure.
        try:
            self.file_store.atomic_write_json(
                self.file_store.paths.boxes_revision(job_id, image_id, result.revision),
                {
                    **result.model_dump(mode="json"),
                    "coordinate_space": "original_px",
                },
            )
        except Exception:
            with log_context(job_id=job_id, image_id=image_id):
                logger.exception(
                    "boxes_revision_projection_failed",
                    extra={
                        "component": "boxes_revision_projection",
                        "detail": f"revision={result.revision}",
                        "event": "projection_write_failed",
                        "outcome": "degraded",
                    },
                )
        return result

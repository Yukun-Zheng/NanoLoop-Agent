"""Audited ROI-box replacement over persistence and file artifacts."""

import logging
from collections.abc import Callable

from app.analysis.authorization import require_mutation
from app.contracts.analyses import BoxSetDTO, ROIBox
from app.contracts.identity import PrincipalContext
from app.contracts.repositories import UnitOfWork
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
        principal: PrincipalContext,
    ) -> BoxSetDTO:
        tenant_id = principal.tenant_id
        if tenant_id is None:
            raise ValueError("principal must carry a tenant ID")
        with self.uow_factory() as uow:
            scope = uow.repositories.jobs.get_scope(job_id, tenant_id=tenant_id)
            require_mutation(principal, scope)
            result = uow.repositories.boxes.replace_scoped(
                job_id,
                image_id,
                expected_revision,
                boxes,
                tenant_id=tenant_id,
            )
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

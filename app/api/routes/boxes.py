"""Optimistically locked ROI box reads and full replacement."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool

from app.analysis.authorization import require_read
from app.analysis.boxes import BoxApplicationService
from app.api.deps import (
    get_database,
    get_file_store,
    get_repositories,
    require_api_key_contract,
)
from app.api.responses import success_response
from app.api.routing import COMMON_ERROR_RESPONSES
from app.contracts.analyses import BoxSetDTO, ReplaceBoxesRequest
from app.contracts.common import ApiResponse
from app.contracts.identity import PrincipalContext
from app.db.repositories import SqlAlchemyRepositorySet, SqlAlchemyUnitOfWork
from app.db.session import Database
from app.storage import LocalFileStore

router = APIRouter(tags=["boxes"], responses=COMMON_ERROR_RESPONSES)


@router.get(
    "/analyses/{job_id}/images/{image_id}/boxes",
    response_model=ApiResponse[BoxSetDTO],
    operation_id="getBoxes",
)
def get_boxes(
    job_id: str,
    image_id: str,
    request: Request,
    repositories: Annotated[SqlAlchemyRepositorySet, Depends(get_repositories)],
    principal: Annotated[PrincipalContext, Depends(require_api_key_contract)],
) -> ApiResponse[BoxSetDTO]:
    tenant_id = principal.tenant_id
    if tenant_id is None:
        raise ValueError("principal must carry a tenant ID")
    scope = repositories.jobs.get_scope(job_id, tenant_id=tenant_id)
    require_read(principal, scope)
    boxes = repositories.boxes.get_active_scoped(
        job_id,
        image_id,
        tenant_id=tenant_id,
    )
    return success_response(boxes, request=request)


@router.put(
    "/analyses/{job_id}/images/{image_id}/boxes",
    response_model=ApiResponse[BoxSetDTO],
    operation_id="replaceBoxes",
)
async def replace_boxes(
    job_id: str,
    image_id: str,
    payload: ReplaceBoxesRequest,
    request: Request,
    database: Annotated[Database, Depends(get_database)],
    file_store: Annotated[LocalFileStore, Depends(get_file_store)],
    principal: Annotated[PrincipalContext, Depends(require_api_key_contract)],
) -> ApiResponse[BoxSetDTO]:
    service = BoxApplicationService(
        uow_factory=lambda: SqlAlchemyUnitOfWork(database.session_factory),
        file_store=file_store,
    )
    result = await run_in_threadpool(
        service.replace,
        job_id=job_id,
        image_id=image_id,
        expected_revision=payload.expected_revision,
        boxes=payload.boxes,
        principal=principal,
    )
    return success_response(result, request=request)

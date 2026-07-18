"""Segmentation run submission, retrieval, and immutable review endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.analysis.application import AnalysisApplicationService
from app.analysis.authorization import require_read
from app.api.deps import (
    get_analysis_application_service,
    get_file_store,
    get_repositories,
    require_api_key_contract,
)
from app.api.downloads import decorate_run_downloads
from app.api.responses import success_response
from app.api.routing import COMMON_ERROR_RESPONSES, BoundedMultipartRoute
from app.contracts.analyses import (
    CorrectedMaskUploadData,
    CreateRunsData,
    CreateRunsRequest,
    ReviewRunData,
    ReviewRunRequest,
    SegmentationRunDTO,
)
from app.contracts.common import ApiResponse
from app.contracts.identity import PrincipalContext
from app.core.errors import InvalidImageError
from app.db.repositories import SqlAlchemyRepositorySet
from app.storage import LocalFileStore

router = APIRouter(
    tags=["runs"],
    responses=COMMON_ERROR_RESPONSES,
    route_class=BoundedMultipartRoute,
)


@router.post(
    "/runs/{run_id}/corrected-mask",
    response_model=ApiResponse[CorrectedMaskUploadData],
    status_code=status.HTTP_201_CREATED,
    operation_id="uploadCorrectedMask",
)
async def upload_corrected_mask(
    run_id: str,
    request: Request,
    file: Annotated[UploadFile, File()],
    service: Annotated[AnalysisApplicationService, Depends(get_analysis_application_service)],
    principal: Annotated[PrincipalContext, Depends(require_api_key_contract)],
) -> ApiResponse[CorrectedMaskUploadData]:
    if not file.filename:
        raise InvalidImageError(details={"reason": "missing_corrected_mask_filename"})
    data = await run_in_threadpool(
        service.stage_corrected_mask,
        run_id,
        file.file,
        file.filename,
        principal=principal,
    )
    return success_response(data, request=request)


@router.post(
    "/analyses/{job_id}/runs",
    response_model=ApiResponse[CreateRunsData],
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="createRuns",
)
async def create_runs(
    job_id: str,
    payload: CreateRunsRequest,
    request: Request,
    service: Annotated[AnalysisApplicationService, Depends(get_analysis_application_service)],
    principal: Annotated[PrincipalContext, Depends(require_api_key_contract)],
) -> ApiResponse[CreateRunsData]:
    run_ids = await run_in_threadpool(
        service.create_runs,
        job_id,
        payload,
        principal=principal,
    )
    return success_response(
        CreateRunsData(run_ids=run_ids),
        request=request,
        accepted=True,
    )


@router.get(
    "/runs/{run_id}",
    response_model=ApiResponse[SegmentationRunDTO],
    operation_id="getRun",
)
def get_run(
    run_id: str,
    request: Request,
    repositories: Annotated[SqlAlchemyRepositorySet, Depends(get_repositories)],
    file_store: Annotated[LocalFileStore, Depends(get_file_store)],
    principal: Annotated[PrincipalContext, Depends(require_api_key_contract)],
) -> ApiResponse[SegmentationRunDTO]:
    tenant_id = principal.tenant_id
    if tenant_id is None:
        raise ValueError("principal must carry a tenant ID")
    run, scope = repositories.runs.get_with_scope(run_id, tenant_id=tenant_id)
    require_read(principal, scope)
    decorated = decorate_run_downloads(
        run,
        private_paths=repositories.runs.get_artifact_paths_scoped(
            run_id,
            tenant_id=tenant_id,
        ),
        request=request,
        file_store=file_store,
    )
    return success_response(decorated, request=request)


@router.post(
    "/runs/{run_id}/review",
    response_model=ApiResponse[ReviewRunData],
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="reviewRun",
)
async def review_run(
    run_id: str,
    payload: ReviewRunRequest,
    request: Request,
    service: Annotated[AnalysisApplicationService, Depends(get_analysis_application_service)],
    principal: Annotated[PrincipalContext, Depends(require_api_key_contract)],
) -> ApiResponse[ReviewRunData]:
    child_run_id = await run_in_threadpool(
        service.create_review_run,
        run_id,
        payload,
        principal=principal,
    )
    return success_response(
        ReviewRunData(parent_run_id=run_id, run_id=child_run_id),
        request=request,
        accepted=True,
    )

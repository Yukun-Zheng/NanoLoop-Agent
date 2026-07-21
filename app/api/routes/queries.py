"""Unified data-and-knowledge query endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool

from app.agent.application import QueryApplicationService
from app.api.deps import get_query_application_service, require_api_key_contract
from app.api.responses import success_response
from app.api.routing import COMMON_ERROR_RESPONSES
from app.contracts.common import ApiResponse
from app.contracts.identity import PrincipalContext
from app.contracts.queries import UnifiedQueryRequest, UnifiedQueryResponse

router = APIRouter(tags=["queries"], responses=COMMON_ERROR_RESPONSES)


@router.post(
    "/analyses/{job_id}/query",
    response_model=ApiResponse[UnifiedQueryResponse],
    operation_id="queryAnalysis",
)
async def query_analysis(
    job_id: str,
    payload: UnifiedQueryRequest,
    request: Request,
    service: Annotated[QueryApplicationService, Depends(get_query_application_service)],
    principal: Annotated[PrincipalContext, Depends(require_api_key_contract)],
) -> ApiResponse[UnifiedQueryResponse]:
    response = await run_in_threadpool(
        service.answer,
        job_id,
        payload,
        principal=principal,
    )
    return success_response(response, request=request)

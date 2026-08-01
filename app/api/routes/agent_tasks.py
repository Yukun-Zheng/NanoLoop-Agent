"""Bounded scientific-agent task, approval, and continuation endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request, status
from fastapi.concurrency import run_in_threadpool

from app.agent.control import AgentControlService
from app.api.deps import get_agent_control_service, require_api_key_contract
from app.api.responses import success_response
from app.api.routing import COMMON_ERROR_RESPONSES
from app.contracts.agent_runtime import (
    AgentRunRequest,
    AgentTaskDTO,
    AgentTaskListData,
    CreateAgentTaskRequest,
    ResolveAgentApprovalRequest,
    SubmitAgentInputRequest,
)
from app.contracts.common import ApiResponse
from app.contracts.identity import PrincipalContext

router = APIRouter(tags=["agent-tasks"], responses=COMMON_ERROR_RESPONSES)


@router.post(
    "/analyses/{job_id}/agent-tasks",
    response_model=ApiResponse[AgentTaskDTO],
    status_code=status.HTTP_201_CREATED,
    operation_id="createAgentTask",
)
async def create_agent_task(
    job_id: str,
    payload: CreateAgentTaskRequest,
    request: Request,
    service: Annotated[AgentControlService, Depends(get_agent_control_service)],
    principal: Annotated[PrincipalContext, Depends(require_api_key_contract)],
) -> ApiResponse[AgentTaskDTO]:
    data = await run_in_threadpool(
        service.create,
        job_id,
        payload,
        principal=principal,
    )
    return success_response(data, request=request)


@router.get(
    "/analyses/{job_id}/agent-tasks",
    response_model=ApiResponse[AgentTaskListData],
    operation_id="listAgentTasks",
)
async def list_agent_tasks(
    job_id: str,
    request: Request,
    service: Annotated[AgentControlService, Depends(get_agent_control_service)],
    principal: Annotated[PrincipalContext, Depends(require_api_key_contract)],
) -> ApiResponse[AgentTaskListData]:
    data = await run_in_threadpool(service.list, job_id, principal=principal)
    return success_response(data, request=request)


@router.get(
    "/agent-tasks/{task_id}",
    response_model=ApiResponse[AgentTaskDTO],
    operation_id="getAgentTask",
)
async def get_agent_task(
    task_id: str,
    request: Request,
    service: Annotated[AgentControlService, Depends(get_agent_control_service)],
    principal: Annotated[PrincipalContext, Depends(require_api_key_contract)],
) -> ApiResponse[AgentTaskDTO]:
    data = await run_in_threadpool(service.get, task_id, principal=principal)
    return success_response(data, request=request)


@router.post(
    "/agent-tasks/{task_id}/run",
    response_model=ApiResponse[AgentTaskDTO],
    operation_id="runAgentTask",
)
async def run_agent_task(
    task_id: str,
    payload: AgentRunRequest,
    request: Request,
    service: Annotated[AgentControlService, Depends(get_agent_control_service)],
    principal: Annotated[PrincipalContext, Depends(require_api_key_contract)],
) -> ApiResponse[AgentTaskDTO]:
    data = await run_in_threadpool(
        service.run,
        task_id,
        principal=principal,
        max_steps=payload.max_steps,
    )
    return success_response(data, request=request)


@router.post(
    "/agent-tasks/{task_id}/input",
    response_model=ApiResponse[AgentTaskDTO],
    operation_id="submitAgentTaskInput",
)
async def submit_agent_task_input(
    task_id: str,
    payload: SubmitAgentInputRequest,
    request: Request,
    service: Annotated[AgentControlService, Depends(get_agent_control_service)],
    principal: Annotated[PrincipalContext, Depends(require_api_key_contract)],
) -> ApiResponse[AgentTaskDTO]:
    data = await run_in_threadpool(
        service.submit_input,
        task_id,
        payload,
        principal=principal,
    )
    return success_response(data, request=request)


@router.post(
    "/agent-tasks/{task_id}/approvals/{approval_id}",
    response_model=ApiResponse[AgentTaskDTO],
    operation_id="resolveAgentTaskApproval",
)
async def resolve_agent_task_approval(
    task_id: str,
    approval_id: str,
    payload: ResolveAgentApprovalRequest,
    request: Request,
    service: Annotated[AgentControlService, Depends(get_agent_control_service)],
    principal: Annotated[PrincipalContext, Depends(require_api_key_contract)],
) -> ApiResponse[AgentTaskDTO]:
    data = await run_in_threadpool(
        service.resolve_approval,
        task_id,
        approval_id,
        payload,
        principal=principal,
    )
    return success_response(data, request=request)


@router.post(
    "/agent-tasks/{task_id}/cancel",
    response_model=ApiResponse[AgentTaskDTO],
    operation_id="cancelAgentTask",
)
async def cancel_agent_task(
    task_id: str,
    request: Request,
    service: Annotated[AgentControlService, Depends(get_agent_control_service)],
    principal: Annotated[PrincipalContext, Depends(require_api_key_contract)],
) -> ApiResponse[AgentTaskDTO]:
    data = await run_in_threadpool(service.cancel, task_id, principal=principal)
    return success_response(data, request=request)

"""Persistent multi-turn conversation endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool

from app.agent.conversation import ConversationService
from app.api.deps import get_conversation_service, require_api_key_contract
from app.api.responses import success_response
from app.api.routing import COMMON_ERROR_RESPONSES
from app.contracts.common import ApiResponse
from app.contracts.conversations import (
    ConversationDetailDTO,
    ConversationListData,
    ConversationMessageRequest,
    CreateConversationRequest,
)
from app.contracts.identity import PrincipalContext

router = APIRouter(tags=["conversations"], responses=COMMON_ERROR_RESPONSES)


@router.post(
    "/analyses/{job_id}/conversations",
    response_model=ApiResponse[ConversationDetailDTO],
    operation_id="createConversation",
)
async def create_conversation(
    job_id: str,
    payload: CreateConversationRequest,
    request: Request,
    service: Annotated[ConversationService, Depends(get_conversation_service)],
    principal: Annotated[PrincipalContext, Depends(require_api_key_contract)],
) -> ApiResponse[ConversationDetailDTO]:
    result = await run_in_threadpool(service.create, job_id, payload, principal=principal)
    return success_response(result, request=request)


@router.get(
    "/analyses/{job_id}/conversations",
    response_model=ApiResponse[ConversationListData],
    operation_id="listConversations",
)
async def list_conversations(
    job_id: str,
    request: Request,
    service: Annotated[ConversationService, Depends(get_conversation_service)],
    principal: Annotated[PrincipalContext, Depends(require_api_key_contract)],
) -> ApiResponse[ConversationListData]:
    result = await run_in_threadpool(service.list, job_id, principal=principal)
    return success_response(result, request=request)


@router.get(
    "/analyses/{job_id}/conversations/{conversation_id}",
    response_model=ApiResponse[ConversationDetailDTO],
    operation_id="getConversation",
)
async def get_conversation(
    job_id: str,
    conversation_id: str,
    request: Request,
    service: Annotated[ConversationService, Depends(get_conversation_service)],
    principal: Annotated[PrincipalContext, Depends(require_api_key_contract)],
) -> ApiResponse[ConversationDetailDTO]:
    result = await run_in_threadpool(
        service.get,
        job_id,
        conversation_id,
        principal=principal,
    )
    return success_response(result, request=request)


@router.post(
    "/analyses/{job_id}/conversations/{conversation_id}/messages",
    response_model=ApiResponse[ConversationDetailDTO],
    operation_id="sendConversationMessage",
)
async def send_conversation_message(
    job_id: str,
    conversation_id: str,
    payload: ConversationMessageRequest,
    request: Request,
    service: Annotated[ConversationService, Depends(get_conversation_service)],
    principal: Annotated[PrincipalContext, Depends(require_api_key_contract)],
) -> ApiResponse[ConversationDetailDTO]:
    result = await run_in_threadpool(
        service.send,
        job_id,
        conversation_id,
        payload,
        principal=principal,
    )
    return success_response(result, request=request)

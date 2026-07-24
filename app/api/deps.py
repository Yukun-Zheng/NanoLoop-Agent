"""Request-scoped dependency injection for repositories and optional providers."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Annotated, Any

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from app.agent.application import QueryApplicationService
from app.agent.conversation import ConversationService
from app.analysis.application import AnalysisApplicationService, AnalysisCreationService
from app.authentication import AUTHENTICATION_VERIFIED_STATE_KEY, RequestAuthenticator
from app.contracts.identity import AuthMode, PrincipalContext
from app.core.errors import ApiNotImplementedError, ServiceUnavailableError
from app.core.logging import bind_log_context, reset_log_context
from app.db.repositories import SqlAlchemyRepositorySet
from app.db.session import Database
from app.files import FileArtifactAccessService
from app.rag.application import KnowledgeApplicationService
from app.storage.file_store import LocalFileStore
from app.storage.knowledge_store import KnowledgeSourceStore

api_key_header = APIKeyHeader(
    name="X-API-Key",
    scheme_name="ApiKeyAuth",
    description=(
        "NanoLoop API credential. The server may run in disabled, shared-key, or revocable "
        "principal mode."
    ),
    auto_error=False,
)


async def require_api_key_contract(
    request: Request,
    candidate: Annotated[str | None, Security(api_key_header)],
) -> PrincipalContext:
    """Expose the OpenAPI scheme and consume only middleware-verified identity state."""

    del candidate  # The middleware already performed the only credential verification/query.
    authenticator = getattr(request.app.state, "authenticator", None)
    principal = getattr(request.state, "principal", None)
    verified = getattr(request.state, AUTHENTICATION_VERIFIED_STATE_KEY, False) is True
    if (
        isinstance(authenticator, RequestAuthenticator)
        and isinstance(principal, PrincipalContext)
        and verified
        and principal.auth_mode is authenticator.mode
    ):
        return principal
    raise HTTPException(
        status_code=503,
        detail={
            "code": "AUTHENTICATION_UNAVAILABLE",
            "message": "认证服务暂时不可用",
        },
        headers={"Cache-Control": "no-store"},
    )


async def require_global_knowledge_access(
    principal: Annotated[PrincipalContext, Security(require_api_key_contract)],
) -> PrincipalContext:
    """Fail closed while the knowledge corpus has no tenant ownership model."""

    if principal.auth_mode is AuthMode.PRINCIPAL:
        raise ServiceUnavailableError(details={"component": "knowledge_tenant_scope"})
    return principal


async def bind_route_log_context(request: Request) -> AsyncIterator[None]:
    values = {
        key: value
        for key, value in request.path_params.items()
        if key in {"job_id", "image_id", "run_id", "model_id"} and isinstance(value, str)
    }
    token = bind_log_context(**values)
    try:
        yield
    finally:
        reset_log_context(token)


def get_database(request: Request) -> Database:
    database = getattr(request.app.state, "database", None)
    if not isinstance(database, Database):
        raise ServiceUnavailableError(details={"component": "database"})
    return database


def get_repositories(request: Request) -> Iterator[SqlAlchemyRepositorySet]:
    database = get_database(request)
    with database.session() as session:
        yield SqlAlchemyRepositorySet(session)


def get_file_store(request: Request) -> LocalFileStore:
    file_store = getattr(request.app.state, "file_store", None)
    if not isinstance(file_store, LocalFileStore):
        raise ServiceUnavailableError(details={"component": "file_store"})
    return file_store


def get_file_artifact_access_service(request: Request) -> FileArtifactAccessService:
    service = getattr(request.app.state, "file_artifact_access_service", None)
    if not isinstance(service, FileArtifactAccessService):
        raise ServiceUnavailableError(details={"component": "file_artifact_access"})
    return service


def get_inference_gateway(request: Request) -> Any:
    gateway = getattr(request.app.state, "inference_gateway", None)
    if gateway is None:
        raise ApiNotImplementedError(
            "模型推理网关尚未接入",
            details={"capability": "inference_gateway"},
        )
    return gateway


def get_analysis_creation_service(request: Request) -> AnalysisCreationService:
    service = getattr(request.app.state, "analysis_creation_service", None)
    if not isinstance(service, AnalysisCreationService):
        raise ServiceUnavailableError(details={"component": "analysis_creation_service"})
    return service


def get_analysis_application_service(request: Request) -> AnalysisApplicationService:
    service = getattr(request.app.state, "analysis_application_service", None)
    if not isinstance(service, AnalysisApplicationService):
        raise ServiceUnavailableError(details={"component": "analysis_application_service"})
    return service


def get_knowledge_application_service(request: Request) -> KnowledgeApplicationService:
    service = getattr(request.app.state, "knowledge_application_service", None)
    if not isinstance(service, KnowledgeApplicationService):
        raise ServiceUnavailableError(details={"component": "knowledge_application_service"})
    return service


def get_knowledge_source_store(request: Request) -> KnowledgeSourceStore:
    store = getattr(request.app.state, "knowledge_source_store", None)
    if not isinstance(store, KnowledgeSourceStore):
        raise ServiceUnavailableError(details={"component": "knowledge_source_store"})
    return store


def get_query_application_service(request: Request) -> QueryApplicationService:
    service = getattr(request.app.state, "query_application_service", None)
    if not isinstance(service, QueryApplicationService):
        raise ServiceUnavailableError(details={"component": "query_application_service"})
    return service


def get_conversation_service(request: Request) -> ConversationService:
    service = getattr(request.app.state, "conversation_service", None)
    if not isinstance(service, ConversationService):
        raise ServiceUnavailableError(details={"component": "conversation_service"})
    return service

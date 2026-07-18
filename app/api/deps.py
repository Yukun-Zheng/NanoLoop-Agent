"""Request-scoped dependency injection for repositories and optional providers."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Annotated, Any

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from app.agent.application import QueryApplicationService
from app.analysis.application import AnalysisApplicationService, AnalysisCreationService
from app.core.errors import ApiNotImplementedError, ServiceUnavailableError
from app.core.logging import bind_log_context, reset_log_context
from app.core.security import ApiKeyVerifier
from app.db.repositories import SqlAlchemyRepositorySet
from app.db.session import Database
from app.rag.application import KnowledgeApplicationService
from app.storage.file_store import LocalFileStore
from app.storage.knowledge_store import KnowledgeSourceStore

api_key_header = APIKeyHeader(
    name="X-API-Key",
    scheme_name="ApiKeyAuth",
    description=(
        "Optional shared deployment key. It is enforced when NANOLOOP_API_KEY is configured."
    ),
    auto_error=False,
)


async def require_api_key_contract(
    request: Request,
    candidate: Annotated[str | None, Security(api_key_header)],
) -> None:
    """Declare the OpenAPI scheme and fail closed if middleware wiring is bypassed."""

    verifier = getattr(request.app.state, "api_key_verifier", None)
    if not isinstance(verifier, ApiKeyVerifier):
        raise ServiceUnavailableError(details={"component": "api_key_verifier"})
    if not verifier.enabled:
        return
    authenticated = bool(getattr(request.state, "api_key_authenticated", False))
    if authenticated and verifier.matches([candidate] if candidate is not None else []):
        return
    raise HTTPException(
        status_code=401,
        detail={
            "code": "AUTHENTICATION_REQUIRED",
            "message": "需要有效的 API Key",
        },
        headers={"WWW-Authenticate": 'ApiKey realm="nanoloop"'},
    )


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

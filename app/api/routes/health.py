"""Service, database, model registry, and RAG health checks."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from alembic.runtime.migration import MigrationContext
from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import func, inspect, select, text

from app.api.interop import invoke
from app.api.responses import success_response
from app.api.routing import COMMON_ERROR_RESPONSES
from app.contracts.common import ApiResponse, HealthComponent, HealthData
from app.contracts.enums import ModelStatus
from app.contracts.models import ModelHealth
from app.core.config import Settings
from app.db.migration_state import expected_alembic_heads
from app.db.models import KnowledgeChunk
from app.db.session import Database

router = APIRouter(tags=["health"], responses=COMMON_ERROR_RESPONSES)


@router.get("/health", response_model=ApiResponse[HealthData], operation_id="getHealth")
async def health(request: Request) -> ApiResponse[HealthData]:
    settings: Settings = request.app.state.settings
    database: Database = request.app.state.database
    gateway = getattr(request.app.state, "inference_gateway", None)
    knowledge_service = getattr(request.app.state, "knowledge_service", None)
    conversation_service = getattr(request.app.state, "conversation_service", None)

    database_health = await run_in_threadpool(_database_health, database)
    model_health = await _model_registry_health(gateway, settings.model_registry_path)
    if knowledge_service is None:
        rag_health = await run_in_threadpool(_rag_index_health, database, settings.faiss_index_path)
    else:
        try:
            rag_health = HealthComponent.model_validate(await invoke(knowledge_service, "health"))
        except Exception as error:
            rag_health = HealthComponent(status="unavailable", detail=type(error).__name__)
    data = HealthData(
        service=HealthComponent(status="healthy"),
        database=database_health,
        model_registry=model_health,
        rag_index=rag_health,
        llm_provider=await run_in_threadpool(
            _llm_provider_health,
            conversation_service,
        ),
        version=_application_version(),
    )
    return success_response(data, request=request)


def _llm_provider_health(conversation_service: object | None) -> HealthComponent:
    provider = getattr(conversation_service, "llm_provider", None)
    if provider is None:
        return HealthComponent(
            status="degraded",
            detail="extractive fallback active; generative provider not configured",
        )
    try:
        return HealthComponent.model_validate(provider.health())
    except Exception as error:
        return HealthComponent(
            status="unavailable",
            detail=f"provider health probe failed: {type(error).__name__}",
        )


def _database_health(database: Database) -> HealthComponent:
    try:
        with database.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            inspector = inspect(connection)
            if not inspector.has_table("analysis_jobs"):
                return HealthComponent(
                    status="degraded",
                    detail="reachable; application schema is missing",
                )
            if not inspector.has_table("alembic_version"):
                return HealthComponent(
                    status="degraded",
                    detail="reachable; alembic_version table is missing",
                )
            current_heads = tuple(
                sorted(MigrationContext.configure(connection).get_current_heads())
            )
        if not current_heads:
            return HealthComponent(
                status="degraded",
                detail="reachable; alembic revision is missing",
            )
        expected_heads = expected_alembic_heads()
        if current_heads != expected_heads:
            return HealthComponent(
                status="degraded",
                detail=(
                    "reachable; database revision "
                    f"{','.join(current_heads)} does not match head {','.join(expected_heads)}"
                ),
            )
        return HealthComponent(status="healthy")
    except Exception as error:  # pragma: no cover - backend-specific failures
        return HealthComponent(status="unavailable", detail=type(error).__name__)


async def _model_registry_health(gateway: object | None, registry_path: Path) -> HealthComponent:
    if gateway is None:
        if registry_path.is_file():
            return HealthComponent(
                status="degraded", detail="registry present; gateway unavailable"
            )
        return HealthComponent(status="unavailable", detail="registry and gateway unavailable")
    try:
        result = await invoke(gateway, "health")
        records = _model_health_records(result)
    except Exception as error:  # provider errors are health data, not API failures
        return HealthComponent(status="unavailable", detail=type(error).__name__)
    if not records:
        return HealthComponent(status="degraded", detail="registry contains no models")
    if any(record.status == ModelStatus.READY for record in records):
        return HealthComponent(status="healthy")
    return HealthComponent(status="degraded", detail="no ready models")


def _model_health_records(value: Any) -> list[ModelHealth]:
    if isinstance(value, dict) and "models" in value:
        value = value["models"]
    if not isinstance(value, list):
        return []
    return [ModelHealth.model_validate(item) for item in value]


def _rag_index_health(database: Database, index_path: Path) -> HealthComponent:
    if index_path.is_file() and index_path.stat().st_size > 0:
        return HealthComponent(status="healthy")
    try:
        if inspect(database.engine).has_table("knowledge_chunks"):
            with database.session() as session:
                count = session.scalar(select(func.count()).select_from(KnowledgeChunk)) or 0
            if count:
                return HealthComponent(
                    status="degraded", detail="keyword data ready; vector index absent"
                )
    except Exception as error:  # pragma: no cover - backend-specific failures
        return HealthComponent(status="unavailable", detail=type(error).__name__)
    return HealthComponent(status="unavailable", detail="knowledge index not built")


def _application_version() -> str:
    try:
        return version("nanoloop-agent")
    except PackageNotFoundError:
        return "0.1.0"

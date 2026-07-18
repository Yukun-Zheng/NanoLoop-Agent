"""FastAPI application factory for the versioned NanoLoop HTTP contract."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect
from sqlalchemy.engine import make_url

from app.agent.application import QueryApplicationService
from app.agent.data_tools import SqlAlchemyDataToolService
from app.agent.router import QueryRouter
from app.agent.unified_query import UnifiedQueryService
from app.analysis.application import AnalysisApplicationService, AnalysisCreationService
from app.api.deps import bind_route_log_context
from app.api.errors import install_exception_handlers
from app.api.middleware import (
    ApiKeyAuthMiddleware,
    BrowserMutationGuardMiddleware,
    ErrorEnvelopeTrustedHostMiddleware,
    InMemoryRateLimitMiddleware,
    RequestBodyLimitMiddleware,
    RequestContextMiddleware,
)
from app.api.routes import api_router
from app.api.routes.health import health
from app.authentication import RequestAuthenticator
from app.contracts.common import ApiResponse, HealthData
from app.contracts.identity import AuthMode
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.core.rate_limit import BoundedKeyedTokenBucketLimiter, TokenBucketLimiter
from app.core.security import cors_origins, file_token_secret, trusted_hosts
from app.db.repositories import SqlAlchemyUnitOfWork
from app.db.session import Database
from app.operations.backup import StateDirectoryLock
from app.orchestration import (
    InProcessTaskDispatcher,
    QueuedRunScheduler,
    SqlAlchemyRunRecoveryStore,
    StaleRunPolicy,
    StartupRecovery,
    TaskDispatcher,
)
from app.rag.application import KnowledgeApplicationService
from app.rag.embeddings import SentenceTransformerEmbeddingProvider
from app.rag.ingestion import DocumentExtractor, IngestionPipeline
from app.rag.keyword_store import KeywordStore, SQLiteFTS5KeywordStore, UnavailableKeywordStore
from app.rag.providers import AnswerProvider, ExtractiveAnswerProvider, OpenAICompatibleProvider
from app.rag.retrieval import RetrievalService
from app.rag.service import KnowledgeService
from app.rag.vector_index import DatabaseVectorIndexPublisher, VectorIndexPublisher
from app.rag.vector_store import PersistentFaissVectorStore
from app.storage import KnowledgeSourceStore, LocalFileStore, StoragePaths

logger = logging.getLogger(__name__)


def create_app(
    *,
    settings: Settings | None = None,
    database: Database | None = None,
    file_store: LocalFileStore | None = None,
    inference_gateway: Any | None = None,
    analysis_creation_service: AnalysisCreationService | None = None,
    analysis_application_service: AnalysisApplicationService | None = None,
    dispatcher: TaskDispatcher | None = None,
    knowledge_application_service: KnowledgeApplicationService | None = None,
    knowledge_source_store: KnowledgeSourceStore | None = None,
    query_application_service: QueryApplicationService | None = None,
) -> FastAPI:
    """Build an app with injectable infrastructure and no schema mutation on startup."""

    configured = settings or get_settings()
    owned_database = database is None
    active_database = database or Database(configured)
    active_file_store = file_store or LocalFileStore(
        StoragePaths(configured.output_root),
        max_upload_bytes=configured.max_upload_mb * 1024 * 1024,
        token_secret=file_token_secret(),
    )

    @asynccontextmanager
    async def _service_lifespan(application: FastAPI) -> AsyncIterator[None]:
        configure_logging(configured.log_level)
        if application.state.inference_gateway is None:
            application.state.inference_gateway = _build_inference_gateway(configured)
        application.state.model_registry_sync = _sync_registry_projection(
            active_database,
            application.state.inference_gateway,
        )
        if application.state.analysis_creation_service is None:
            application.state.analysis_creation_service = AnalysisCreationService(
                uow_factory=lambda: SqlAlchemyUnitOfWork(active_database.session_factory),
                file_store=active_file_store,
            )
        if application.state.knowledge_source_store is None:
            application.state.knowledge_source_store = KnowledgeSourceStore(
                configured.knowledge_source_dir,
                max_upload_bytes=configured.max_upload_mb * 1024 * 1024,
            )
        vector_index_publisher: VectorIndexPublisher | None = None
        if application.state.query_application_service is None:
            query_service, knowledge_service, vector_index_publisher = _build_query_services(
                database=active_database,
                file_store=active_file_store,
                settings=configured,
            )
            application.state.query_application_service = query_service
            application.state.knowledge_service = knowledge_service
        if application.state.knowledge_application_service is None:
            application.state.knowledge_application_service = KnowledgeApplicationService(
                active_database,
                configured.knowledge_source_dir,
                pipeline=IngestionPipeline(
                    extractor=DocumentExtractor(
                        max_pdf_pages=configured.knowledge_max_pdf_pages,
                        max_extracted_chars=configured.knowledge_max_extracted_chars,
                    ),
                    max_chunks_per_document=(
                        configured.knowledge_max_chunks_per_document
                    ),
                ),
                vector_index_publisher=vector_index_publisher,
            )
        if (
            application.state.analysis_application_service is None
            and application.state.inference_gateway is not None
        ):
            if application.state.dispatcher is None:
                application.state.dispatcher = InProcessTaskDispatcher(
                    lambda run_id: application.state.analysis_application_service.execute_run(
                        run_id
                    ),
                    worker_count=configured.analysis_worker_count,
                    queue_capacity=configured.analysis_queue_capacity,
                )
                application.state.owns_dispatcher = True
            application.state.analysis_application_service = AnalysisApplicationService(
                uow_factory=lambda: SqlAlchemyUnitOfWork(active_database.session_factory),
                file_store=active_file_store,
                inference_gateway=application.state.inference_gateway,
                dispatcher=application.state.dispatcher,
            )
            if application.state.owns_dispatcher:
                if inspect(active_database.engine).has_table("segmentation_runs"):
                    recovery_store = SqlAlchemyRunRecoveryStore(active_database)
                    recovery = StartupRecovery(
                        recovery_store,
                        application.state.dispatcher,
                        stale_policy=StaleRunPolicy.FAIL,
                    )
                    report = await recovery.recover_async()
                    application.state.startup_recovery_report = report
                    if report.requires_attention:
                        logger.warning(
                            "startup_recovery_requires_attention",
                            extra={
                                "event": "startup_recovery_requires_attention",
                                "deferred_count": len(report.deferred_run_ids),
                                "error_count": len(report.errors),
                            },
                        )
                    scheduler = QueuedRunScheduler(
                        recovery_store,
                        application.state.dispatcher,
                        poll_interval_seconds=configured.analysis_scheduler_poll_seconds,
                    )
                    scheduler.start()
                    application.state.queued_run_scheduler = scheduler
                else:
                    application.state.dispatcher.start()
                    logger.warning(
                        "startup_recovery_skipped",
                        extra={
                            "event": "startup_recovery_skipped",
                            "detail": "segmentation_runs table is missing",
                        },
                    )
        logger.info("application_started", extra={"event": "application_started"})
        try:
            yield
        finally:
            scheduler = application.state.queued_run_scheduler
            if scheduler is not None:
                scheduler_stopped = await scheduler.astop(
                    timeout=configured.shutdown_timeout_seconds,
                )
                if not scheduler_stopped:
                    logger.warning(
                        "queued_run_scheduler_shutdown_incomplete",
                        extra={"event": "queued_run_scheduler_shutdown_incomplete"},
                    )
            if application.state.owns_dispatcher:
                stopped = await application.state.dispatcher.astop(
                    drain=True,
                    timeout=configured.shutdown_timeout_seconds,
                )
                if not stopped:
                    logger.warning(
                        "analysis_dispatcher_shutdown_incomplete",
                        extra={"event": "analysis_dispatcher_shutdown_incomplete"},
                    )
            if owned_database:
                active_database.dispose()
            logger.info("application_stopped", extra={"event": "application_stopped"})

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        state_lock = _state_directory_lock(configured)
        if state_lock is None:
            async with _service_lifespan(application):
                yield
            return
        with state_lock:
            async with _service_lifespan(application):
                yield

    application = FastAPI(
        title="NanoLoop Agent API",
        summary="Traceable SEM nanoparticle analysis and evidence-backed query API",
        version="0.1.0",
        lifespan=lifespan,
        dependencies=[Depends(bind_route_log_context)],
    )
    application.state.settings = configured
    application.state.database = active_database
    application.state.file_store = active_file_store
    application.state.inference_gateway = inference_gateway
    application.state.analysis_creation_service = analysis_creation_service
    application.state.analysis_application_service = analysis_application_service
    application.state.dispatcher = dispatcher
    application.state.owns_dispatcher = False
    application.state.startup_recovery_report = None
    application.state.queued_run_scheduler = None
    application.state.knowledge_application_service = knowledge_application_service
    application.state.knowledge_source_store = knowledge_source_store
    application.state.query_application_service = query_application_service
    application.state.knowledge_service = None

    allowed_origins = cors_origins(configured)
    authenticator = RequestAuthenticator.from_settings(configured, active_database)
    application.state.authenticator = authenticator
    # Transitional read-only state for extensions that introspect whether legacy shared-key mode
    # is active. Principal mode deliberately exposes a disabled verifier here.
    application.state.api_key_verifier = authenticator.shared_key_verifier
    principal_limiter = (
        BoundedKeyedTokenBucketLimiter(
            capacity=configured.api_rate_limit_requests,
            window_seconds=configured.api_rate_limit_window_seconds,
            max_buckets=configured.api_rate_limit_max_buckets,
        )
        if (
            authenticator.mode is AuthMode.PRINCIPAL
            and configured.api_rate_limit_requests > 0
        )
        else None
    )
    principal_preauth_limiter = (
        BoundedKeyedTokenBucketLimiter(
            capacity=configured.api_principal_preauth_rate_limit_requests,
            window_seconds=(
                configured.api_principal_preauth_rate_limit_window_seconds
            ),
            max_buckets=configured.api_rate_limit_max_buckets,
        )
        if (
            authenticator.mode is AuthMode.PRINCIPAL
            and configured.api_principal_preauth_rate_limit_requests > 0
        )
        else None
    )
    application.state.principal_rate_limiter = principal_limiter
    application.state.principal_preauth_rate_limiter = principal_preauth_limiter
    public_paths = tuple(
        path
        for path in (
            "/health",
            application.openapi_url,
            application.docs_url,
            application.swagger_ui_oauth2_redirect_url,
            application.redoc_url,
        )
        if path is not None
    )
    application.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_bytes=configured.max_request_mb * 1024 * 1024,
    )
    application.add_middleware(
        ApiKeyAuthMiddleware,
        authenticator=authenticator,
        principal_limiter=principal_limiter,
        public_paths=public_paths,
    )
    if principal_preauth_limiter is not None:
        application.add_middleware(
            InMemoryRateLimitMiddleware,
            authenticator=authenticator,
            principal_preauth_limiter=principal_preauth_limiter,
            prefer_downstream_rate_limit_headers=principal_limiter is not None,
            public_paths=public_paths,
        )
    elif (
        authenticator.mode is not AuthMode.PRINCIPAL
        and configured.api_rate_limit_requests > 0
    ):
        application.add_middleware(
            InMemoryRateLimitMiddleware,
            authenticator=authenticator,
            limiter=TokenBucketLimiter(
                capacity=configured.api_rate_limit_requests,
                window_seconds=configured.api_rate_limit_window_seconds,
            ),
            public_paths=public_paths,
        )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "OPTIONS"],
        allow_headers=["Accept", "Content-Type", "X-API-Key", "X-Request-ID"],
        expose_headers=[
            "Content-Disposition",
            "Retry-After",
            "WWW-Authenticate",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-Request-ID",
        ],
        max_age=600,
    )
    application.add_middleware(
        BrowserMutationGuardMiddleware,
        allowed_origins=allowed_origins,
    )
    application.add_middleware(
        ErrorEnvelopeTrustedHostMiddleware,
        allowed_hosts=trusted_hosts(configured),
    )
    application.add_middleware(RequestContextMiddleware)
    install_exception_handlers(application)

    application.include_router(api_router, prefix=configured.api_prefix)
    application.add_api_route(
        "/health",
        health,
        methods=["GET"],
        response_model=ApiResponse[HealthData],
        include_in_schema=False,
    )
    return application


def _state_directory_lock(settings: Settings) -> StateDirectoryLock | None:
    """Share-lock one ordinary SQLite state root for the complete API lifespan."""

    url = make_url(settings.database_url)
    database_path = url.database
    if (
        url.get_backend_name() != "sqlite"
        or database_path is None
        or database_path == ":memory:"
        or database_path.startswith("file:")
        or str(url.query.get("mode", "")).casefold() == "memory"
    ):
        return None
    data_root = Path(database_path).expanduser().absolute().parent
    return StateDirectoryLock(data_root, exclusive=False)


def _build_inference_gateway(settings: Settings) -> Any | None:
    """Delay model-registry imports until startup so API import remains lightweight."""

    try:
        from app.inference.gateway import InferenceGateway
        from app.inference.registry import ModelRegistryService

        registry = ModelRegistryService(
            settings.model_registry_path,
            snapshot_root=settings.model_snapshot_root,
        )
        return InferenceGateway(registry)
    except Exception as error:  # registry problems must not prevent health/error APIs from starting
        logger.warning(
            "inference_gateway_unavailable",
            extra={
                "component": "inference_gateway",
                "detail": type(error).__name__,
                "event": "component_unavailable",
            },
        )
        return None


def _sync_registry_projection(
    database: Database,
    gateway: object | None,
) -> dict[str, object] | None:
    """Mirror a real YAML registry after migrations without mutating the schema."""

    try:
        from app.inference.gateway import InferenceGateway
        from app.inference.model_sync import ModelRegistrySyncError, sync_model_registry

        if not isinstance(gateway, InferenceGateway):
            return None
        with database.session() as session:
            result = sync_model_registry(session, gateway.registry)
        return result.as_dict()
    except ModelRegistrySyncError as error:
        logger.warning(
            "model_registry_sync_unavailable",
            extra={
                "component": "model_registry_projection",
                "detail": error.code,
                "event": "component_unavailable",
            },
        )
        return error.as_dict()


def _build_query_services(
    *,
    database: Database,
    file_store: LocalFileStore,
    settings: Settings,
) -> tuple[QueryApplicationService, KnowledgeService, VectorIndexPublisher | None]:
    database_path = database.engine.url.database
    keyword_store: KeywordStore
    vector_index_publisher: VectorIndexPublisher | None = None
    if (
        database.engine.dialect.name == "sqlite"
        and database_path is not None
        and database_path != ":memory:"
    ):
        keyword_store = SQLiteFTS5KeywordStore(database_path)
        embedding_provider = SentenceTransformerEmbeddingProvider(
            settings.embedding_model,
            device=settings.model_device,
            revision=settings.embedding_model_revision,
        )
        vector_store = PersistentFaissVectorStore(
            settings.faiss_index_path,
            model_id=settings.embedding_model,
            model_fingerprint=lambda: embedding_provider.fingerprint,
            database_path=database_path,
        )
        vector_index_publisher = DatabaseVectorIndexPublisher(
            database,
            embedding_provider,
            vector_store,
            batch_size=settings.embedding_index_batch_size,
            max_chunks=settings.knowledge_max_vector_index_chunks,
        )
        retrieval = RetrievalService(
            keyword_store,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )
    else:
        keyword_store = UnavailableKeywordStore(
            "keyword retrieval requires a file-backed SQLite database"
        )
        retrieval = RetrievalService(keyword_store)
    fallback = ExtractiveAnswerProvider()
    provider: AnswerProvider
    if settings.llm_provider == "openai_compatible":
        provider = OpenAICompatibleProvider(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
    else:
        provider = fallback
    knowledge = KnowledgeService(
        retrieval,
        provider=provider,
        fallback_provider=fallback,
    )
    unified = UnifiedQueryService(
        router=QueryRouter(),
        knowledge_service=knowledge,
        data_tools=SqlAlchemyDataToolService(
            database.session_factory,
            distribution_evidence_limit=settings.data_distribution_evidence_limit,
        ),
    )
    return (
        QueryApplicationService(
            session_factory=database.session_factory,
            unified_query=unified,
            file_store=file_store,
        ),
        knowledge,
        vector_index_publisher,
    )


app = create_app()

"""Knowledge document ingestion, listing, and index maintenance contracts."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from app.api.deps import get_knowledge_application_service, get_knowledge_source_store
from app.api.responses import success_response
from app.api.routing import COMMON_ERROR_RESPONSES, BoundedMultipartRoute
from app.contracts.common import ApiResponse
from app.contracts.knowledge import (
    IngestDocumentMetadata,
    IngestReport,
    KnowledgeDocumentDTO,
    KnowledgeDocumentListData,
    ReindexReport,
    ReindexRequest,
    UpdateKnowledgeDocumentRequest,
)
from app.core.errors import (
    InvalidKnowledgeDocumentError,
    KnowledgeDocumentConflictError,
    KnowledgeDocumentStateConflictError,
    PayloadTooLargeError,
    RagIndexNotReadyError,
    ResourceNotFoundError,
    ServiceUnavailableError,
)
from app.rag.application import (
    DuplicateKnowledgeDocumentError,
    KnowledgeApplicationService,
    KnowledgeDocumentNotFoundError,
    KnowledgeDocumentStateError,
    KnowledgeIndexUnavailableError,
    KnowledgeSourcePathError,
)
from app.rag.ingestion import (
    DocumentExtractionError,
    DocumentExtractionUnavailableError,
)
from app.storage import KnowledgeSourceStore, UploadSizeExceededError

router = APIRouter(
    prefix="/knowledge",
    tags=["knowledge"],
    responses=COMMON_ERROR_RESPONSES,
    route_class=BoundedMultipartRoute,
)


@router.post(
    "/documents",
    response_model=ApiResponse[IngestReport],
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="ingestKnowledgeDocument",
)
async def ingest_document(
    request: Request,
    file: Annotated[UploadFile, File()],
    metadata_json: Annotated[str, Form(min_length=2)],
    service: Annotated[
        KnowledgeApplicationService, Depends(get_knowledge_application_service)
    ],
    source_store: Annotated[KnowledgeSourceStore, Depends(get_knowledge_source_store)],
) -> ApiResponse[IngestReport]:
    try:
        metadata = IngestDocumentMetadata.model_validate_json(metadata_json)
    except ValidationError as error:
        raise RequestValidationError(error.errors()) from error
    if not file.filename:
        raise InvalidKnowledgeDocumentError(details={"reason": "missing_filename"})
    try:
        stored = await run_in_threadpool(source_store.save, file.file, file.filename)
    except UploadSizeExceededError as error:
        raise PayloadTooLargeError(details={"limit_bytes": error.limit_bytes}) from error
    try:
        report = await run_in_threadpool(service.ingest_document, stored.path, metadata)
    except DuplicateKnowledgeDocumentError as error:
        raise KnowledgeDocumentConflictError(
            details={"sha256": error.sha256, "existing_doc_id": error.existing_doc_id}
        ) from error
    except KnowledgeIndexUnavailableError as error:
        raise RagIndexNotReadyError(details={"reason": str(error)}) from error
    except DocumentExtractionUnavailableError as error:
        raise ServiceUnavailableError(
            "PDF 文本提取依赖尚未安装",
            details={"component": "pymupdf"},
        ) from error
    except (DocumentExtractionError, KnowledgeSourcePathError) as error:
        raise InvalidKnowledgeDocumentError(details={"reason": str(error)}) from error
    # Content-addressed sources are intentionally not deleted here. Another
    # concurrent request may have committed a KnowledgeDocument referencing the
    # same digest after this request observed ``created=True``. Eager cleanup
    # would be a TOCTOU data-loss bug; orphan collection must use a grace period
    # and a database reference snapshot in an explicit maintenance operation.
    return success_response(report, request=request, accepted=True)


@router.get(
    "/documents",
    response_model=ApiResponse[KnowledgeDocumentListData],
    operation_id="listKnowledgeDocuments",
)
async def list_documents(
    request: Request,
    service: Annotated[
        KnowledgeApplicationService, Depends(get_knowledge_application_service)
    ],
) -> ApiResponse[KnowledgeDocumentListData]:
    documents = await run_in_threadpool(service.list_documents)
    return success_response(documents, request=request)


@router.patch(
    "/documents/{doc_id}",
    response_model=ApiResponse[KnowledgeDocumentDTO],
    operation_id="updateKnowledgeDocument",
)
async def update_document(
    doc_id: str,
    payload: UpdateKnowledgeDocumentRequest,
    request: Request,
    service: Annotated[
        KnowledgeApplicationService, Depends(get_knowledge_application_service)
    ],
) -> ApiResponse[KnowledgeDocumentDTO]:
    try:
        document = await run_in_threadpool(
            service.set_document_enabled,
            doc_id,
            enabled=payload.enabled,
        )
    except KnowledgeDocumentNotFoundError as error:
        raise ResourceNotFoundError(
            details={"resource": "knowledge_document", "doc_id": error.doc_id}
        ) from error
    except KnowledgeDocumentStateError as error:
        raise KnowledgeDocumentStateConflictError(
            details={
                "doc_id": error.doc_id,
                "current_status": error.current_status,
                "requested_status": error.requested_status.value,
                "reason": error.reason,
            }
        ) from error
    except KnowledgeIndexUnavailableError as error:
        raise RagIndexNotReadyError(details={"reason": str(error)}) from error
    return success_response(document, request=request)


@router.post(
    "/reindex",
    response_model=ApiResponse[ReindexReport],
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="reindexKnowledge",
)
async def reindex_knowledge(
    payload: ReindexRequest,
    request: Request,
    service: Annotated[
        KnowledgeApplicationService, Depends(get_knowledge_application_service)
    ],
) -> ApiResponse[ReindexReport]:
    try:
        report = await run_in_threadpool(service.reindex, payload)
    except KnowledgeIndexUnavailableError as error:
        raise RagIndexNotReadyError(details={"reason": str(error)}) from error
    return success_response(report, request=request, accepted=True)

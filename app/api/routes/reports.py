"""Previewable DOCX/PDF scientific-report generation."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool

from app.agent.conversation import ConversationService
from app.analysis.authorization import require_read
from app.analysis.reporting import JobExportSnapshot
from app.analysis.scientific_reports import ScientificReportBuilder
from app.api.deps import (
    get_conversation_service,
    get_file_artifact_access_service,
    get_file_store,
    get_repositories,
    require_api_key_contract,
)
from app.api.responses import success_response
from app.api.routing import COMMON_ERROR_RESPONSES
from app.contracts.common import ApiResponse
from app.contracts.enums import JobStatus
from app.contracts.file_artifacts import FileArtifactKind
from app.contracts.identity import PrincipalContext
from app.contracts.reports import (
    ReportArtifactData,
    ScientificReportData,
    ScientificReportRequest,
)
from app.core.errors import ExportNotReadyError, ResourceNotFoundError
from app.db.repositories import SqlAlchemyRepositorySet
from app.files import FileArtifactAccessService
from app.storage import LocalFileStore

router = APIRouter(tags=["reports"], responses=COMMON_ERROR_RESPONSES)

_DOCX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)


@router.post(
    "/analyses/{job_id}/report",
    response_model=ApiResponse[ScientificReportData],
    operation_id="generateScientificReport",
)
async def generate_scientific_report(
    job_id: str,
    payload: ScientificReportRequest,
    request: Request,
    repositories: Annotated[SqlAlchemyRepositorySet, Depends(get_repositories)],
    file_store: Annotated[LocalFileStore, Depends(get_file_store)],
    file_access: Annotated[
        FileArtifactAccessService,
        Depends(get_file_artifact_access_service),
    ],
    conversation: Annotated[ConversationService, Depends(get_conversation_service)],
    principal: Annotated[PrincipalContext, Depends(require_api_key_contract)],
) -> ApiResponse[ScientificReportData]:
    tenant_id = principal.tenant_id
    if tenant_id is None:
        raise ValueError("principal must carry a tenant ID")
    scope = repositories.jobs.get_scope(job_id, tenant_id=tenant_id)
    require_read(principal, scope)
    images = repositories.images.list_by_job_scoped(job_id, tenant_id=tenant_id)
    box_revisions = repositories.boxes.list_by_job_scoped(job_id, tenant_id=tenant_id)
    all_runs = {
        run.run_id: run
        for run in repositories.runs.list_by_job_scoped(job_id, tenant_id=tenant_id)
    }
    selected_ids = list(payload.run_ids)
    missing = [run_id for run_id in selected_ids if run_id not in all_runs]
    if missing:
        raise ResourceNotFoundError(
            details={"resource": "run", "job_id": job_id, "run_ids": missing}
        )
    terminal = {JobStatus.COMPLETED, JobStatus.COMPLETED_WITH_WARNINGS}
    not_ready = [
        run_id for run_id in selected_ids if all_runs[run_id].status not in terminal
    ]
    if not_ready:
        raise ExportNotReadyError(
            details={
                "job_id": job_id,
                "run_ids": not_ready,
                "reason": "runs_not_complete",
            }
        )
    selected = tuple(all_runs[run_id] for run_id in selected_ids)
    if any(run.summary is None or run.quality is None for run in selected):
        raise ExportNotReadyError(
            details={
                "job_id": job_id,
                "run_ids": selected_ids,
                "reason": "summary_or_quality_missing",
            }
        )
    snapshot = JobExportSnapshot(
        job=scope.job,
        images=tuple(images),
        runs=selected,
        box_revisions=tuple(box_revisions),
        image_storage_paths=tuple(
            sorted(repositories.images.get_storage_path(image.image_id) for image in images)
        ),
        run_artifact_paths=tuple(
            (run_id, artifact_path)
            for run_id in selected_ids
            for artifact_path in sorted(
                path
                for path in repositories.runs.get_artifact_paths(run_id).values()
                if path is not None
            )
        ),
    )
    preview, docx_file, pdf_file = await run_in_threadpool(
        ScientificReportBuilder(
            file_store=file_store,
            data_tools=conversation.data_tools,
            llm_provider=conversation.llm_provider,
        ).build,
        snapshot=snapshot,
        tenant_id=tenant_id,
        run_ids=selected_ids,
    )
    prefix = request.app.state.settings.api_prefix.rstrip("/")
    docx_token = file_access.issue_download_token(
        principal=principal,
        job_id=job_id,
        artifact_kind=FileArtifactKind.ANALYSIS_EXPORT,
        storage_path=docx_file.relative_path,
        filename=docx_file.filename,
        media_type=_DOCX_MEDIA_TYPE,
        expected_sha256=docx_file.sha256,
        expected_size_bytes=docx_file.size_bytes,
    )
    pdf_token = file_access.issue_download_token(
        principal=principal,
        job_id=job_id,
        artifact_kind=FileArtifactKind.ANALYSIS_EXPORT,
        storage_path=pdf_file.relative_path,
        filename=pdf_file.filename,
        media_type="application/pdf",
        expected_sha256=pdf_file.sha256,
        expected_size_bytes=pdf_file.size_bytes,
    )
    return success_response(
        ScientificReportData(
            **preview.model_dump(),
            docx=ReportArtifactData(
                filename=docx_file.filename,
                download_url=f"{prefix}/files/{docx_token}",
                sha256=docx_file.sha256,
                size_bytes=docx_file.size_bytes,
                media_type=_DOCX_MEDIA_TYPE,
            ),
            pdf=ReportArtifactData(
                filename=pdf_file.filename,
                download_url=f"{prefix}/files/{pdf_token}",
                sha256=pdf_file.sha256,
                size_bytes=pdf_file.size_bytes,
                media_type="application/pdf",
            ),
        ),
        request=request,
    )

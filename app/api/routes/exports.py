"""Analysis export contract."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from fastapi.concurrency import run_in_threadpool

from app.analysis.authorization import require_read
from app.analysis.reporting import JobExportSnapshot, ReportWriter
from app.api.deps import (
    get_file_artifact_access_service,
    get_file_store,
    get_repositories,
    require_api_key_contract,
)
from app.api.responses import success_response
from app.api.routing import COMMON_ERROR_RESPONSES
from app.contracts.analyses import ExportData
from app.contracts.common import ApiResponse
from app.contracts.enums import JobStatus
from app.contracts.file_artifacts import FileArtifactKind
from app.contracts.identity import PrincipalContext
from app.core.errors import ExportNotReadyError, ResourceNotFoundError
from app.db.repositories import SqlAlchemyRepositorySet
from app.files import FileArtifactAccessService
from app.storage import LocalFileStore

router = APIRouter(tags=["exports"], responses=COMMON_ERROR_RESPONSES)


@router.get(
    "/analyses/{job_id}/export",
    response_model=ApiResponse[ExportData],
    operation_id="exportAnalysis",
)
async def export_analysis(
    job_id: str,
    request: Request,
    repositories: Annotated[SqlAlchemyRepositorySet, Depends(get_repositories)],
    file_store: Annotated[LocalFileStore, Depends(get_file_store)],
    file_access: Annotated[
        FileArtifactAccessService,
        Depends(get_file_artifact_access_service),
    ],
    principal: Annotated[PrincipalContext, Depends(require_api_key_contract)],
    run_ids: Annotated[list[str] | None, Query()] = None,
) -> ApiResponse[ExportData]:
    tenant_id = principal.tenant_id
    if tenant_id is None:
        raise ValueError("principal must carry a tenant ID")
    scope = repositories.jobs.get_scope(job_id, tenant_id=tenant_id)
    require_read(principal, scope)
    job = scope.job
    images = repositories.images.list_by_job_scoped(job_id, tenant_id=tenant_id)
    box_revisions = repositories.boxes.list_by_job_scoped(job_id, tenant_id=tenant_id)
    all_runs = {
        run.run_id: run for run in repositories.runs.list_by_job_scoped(job_id, tenant_id=tenant_id)
    }
    queries = repositories.queries.list_by_job_scoped(job_id, tenant_id=tenant_id)
    selected_ids = list(dict.fromkeys(run_ids)) if run_ids else list(all_runs)
    if not selected_ids:
        raise ExportNotReadyError(details={"job_id": job_id, "reason": "no_runs"})
    missing = [run_id for run_id in selected_ids if run_id not in all_runs]
    if missing:
        raise ResourceNotFoundError(
            details={"resource": "run", "job_id": job_id, "run_ids": missing}
        )
    terminal = {JobStatus.COMPLETED, JobStatus.COMPLETED_WITH_WARNINGS}
    not_ready = [run_id for run_id in selected_ids if all_runs[run_id].status not in terminal]
    if not_ready:
        raise ExportNotReadyError(
            details={"job_id": job_id, "run_ids": not_ready, "reason": "runs_not_complete"}
        )

    exported = await run_in_threadpool(
        ReportWriter(file_store).build_job_export,
        job_id,
        run_ids=set(selected_ids),
        snapshot=JobExportSnapshot(
            job=job,
            images=tuple(images),
            runs=tuple(all_runs[run_id] for run_id in selected_ids),
            queries=tuple(queries),
            box_revisions=tuple(box_revisions),
        ),
    )
    token = file_access.issue_download_token(
        principal=principal,
        job_id=job_id,
        artifact_kind=FileArtifactKind.ANALYSIS_EXPORT,
        storage_path=exported.relative_path,
        filename=exported.filename,
        media_type="application/zip",
        expected_sha256=exported.sha256,
        expected_size_bytes=exported.size_bytes,
    )
    prefix = request.app.state.settings.api_prefix.rstrip("/")
    return success_response(
        ExportData(
            job_id=job_id,
            download_url=f"{prefix}/files/{token}",
            sha256=exported.sha256,
            filename=exported.filename,
        ),
        request=request,
    )

"""Analysis export contract."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from fastapi.concurrency import run_in_threadpool

from app.analysis.reporting import JobExportSnapshot, ReportWriter
from app.api.deps import get_file_store, get_repositories
from app.api.responses import success_response
from app.api.routing import COMMON_ERROR_RESPONSES
from app.contracts.analyses import ExportData
from app.contracts.common import ApiResponse
from app.contracts.enums import JobStatus
from app.core.errors import ExportNotReadyError, ResourceNotFoundError
from app.db.repositories import SqlAlchemyRepositorySet
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
    run_ids: Annotated[list[str] | None, Query()] = None,
) -> ApiResponse[ExportData]:
    job = repositories.jobs.get(job_id)
    images = repositories.images.list_by_job(job_id)
    box_revisions = repositories.boxes.list_by_job(job_id)
    all_runs = {run.run_id: run for run in repositories.runs.list_by_job(job_id)}
    queries = repositories.queries.list_by_job(job_id)
    # A run selection is a set by contract. Canonicalize it before building the
    # snapshot so query-string order cannot change report rows or ZIP bytes.
    selected_ids = sorted(set(run_ids)) if run_ids else sorted(all_runs)
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
        ),
    )
    prefix = request.app.state.settings.api_prefix.rstrip("/")
    return success_response(
        ExportData(
            job_id=job_id,
            download_url=f"{prefix}/files/{exported.file_token}",
            sha256=exported.sha256,
            filename=exported.filename,
        ),
        request=request,
    )

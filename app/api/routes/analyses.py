"""Analysis task creation signature and persisted job read model."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from app.analysis.application import AnalysisCreationService, AnalysisUpload
from app.analysis.authorization import require_read
from app.api.deps import (
    get_analysis_creation_service,
    get_file_artifact_access_service,
    get_repositories,
    require_api_key_contract,
)
from app.api.downloads import decorate_image_download, decorate_run_downloads
from app.api.responses import success_response
from app.api.routing import COMMON_ERROR_RESPONSES, BoundedMultipartRoute
from app.contracts.analyses import (
    CreateAnalysisMetadata,
    JobDetailDTO,
    RunFailureDTO,
)
from app.contracts.common import ApiResponse
from app.contracts.identity import PrincipalContext
from app.core.errors import InvalidImageError
from app.db.repositories import SqlAlchemyRepositorySet
from app.files import FileArtifactAccessService

router = APIRouter(
    prefix="/analyses",
    tags=["analyses"],
    responses=COMMON_ERROR_RESPONSES,
    route_class=BoundedMultipartRoute,
)


@router.post(
    "",
    response_model=ApiResponse[JobDetailDTO],
    status_code=status.HTTP_201_CREATED,
    operation_id="createAnalysis",
)
async def create_analysis(
    request: Request,
    files: Annotated[list[UploadFile], File(min_length=1, max_length=20)],
    metadata_json: Annotated[str, Form(min_length=2)],
    service: Annotated[AnalysisCreationService, Depends(get_analysis_creation_service)],
    repositories: Annotated[SqlAlchemyRepositorySet, Depends(get_repositories)],
    file_access: Annotated[
        FileArtifactAccessService,
        Depends(get_file_artifact_access_service),
    ],
    principal: Annotated[PrincipalContext, Depends(require_api_key_contract)],
) -> ApiResponse[JobDetailDTO]:
    try:
        metadata = CreateAnalysisMetadata.model_validate_json(metadata_json)
    except ValidationError as error:
        raise RequestValidationError(error.errors()) from error
    uploads: list[AnalysisUpload] = []
    for upload in files:
        if not upload.filename:
            raise InvalidImageError(details={"reason": "missing_upload_filename"})
        uploads.append(AnalysisUpload(filename=upload.filename, stream=upload.file))
    detail = await run_in_threadpool(
        service.create_analysis,
        metadata,
        uploads,
        principal=principal,
    )
    tenant_id = principal.tenant_id
    if tenant_id is None:
        raise ValueError("principal must carry a tenant ID")
    detail = detail.model_copy(
        update={
            "images": [
                decorate_image_download(
                    image,
                    storage_path=repositories.images.get_storage_path_scoped(
                        detail.job.job_id,
                        image.image_id,
                        tenant_id=tenant_id,
                    ),
                    request=request,
                    file_access=file_access,
                    principal=principal,
                )
                for image in detail.images
            ]
        }
    )
    return success_response(
        detail,
        request=request,
    )


@router.get(
    "/{job_id}",
    response_model=ApiResponse[JobDetailDTO],
    operation_id="getAnalysis",
)
def get_analysis(
    job_id: str,
    request: Request,
    repositories: Annotated[SqlAlchemyRepositorySet, Depends(get_repositories)],
    file_access: Annotated[
        FileArtifactAccessService,
        Depends(get_file_artifact_access_service),
    ],
    principal: Annotated[PrincipalContext, Depends(require_api_key_contract)],
) -> ApiResponse[JobDetailDTO]:
    tenant_id = principal.tenant_id
    if tenant_id is None:
        raise ValueError("principal must carry a tenant ID")
    scope = repositories.jobs.get_scope(job_id, tenant_id=tenant_id)
    require_read(principal, scope)
    job = scope.job
    images = [
        decorate_image_download(
            image,
            storage_path=repositories.images.get_storage_path_scoped(
                job_id,
                image.image_id,
                tenant_id=tenant_id,
            ),
            request=request,
            file_access=file_access,
            principal=principal,
        )
        for image in repositories.images.list_by_job_scoped(
            job_id,
            tenant_id=tenant_id,
        )
    ]
    runs = [
        decorate_run_downloads(
            run,
            private_paths=repositories.runs.get_artifact_paths_scoped(
                run.run_id,
                tenant_id=tenant_id,
            ),
            request=request,
            file_access=file_access,
            principal=principal,
        )
        for run in repositories.runs.list_by_job_scoped(
            job_id,
            tenant_id=tenant_id,
        )
    ]
    failures = [
        RunFailureDTO(
            run_id=run.run_id,
            image_id=run.image_id,
            model_id=run.model_id,
            error_code=run.error_code,
            error_message=run.error_message,
        )
        for run in runs
        if run.error_code is not None
    ]
    return success_response(
        JobDetailDTO(job=job, images=images, runs=runs, partial_failures=failures),
        request=request,
    )

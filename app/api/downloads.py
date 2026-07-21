"""Convert private storage keys into short-lived public download URLs."""

from __future__ import annotations

from fastapi import Request

from app.contracts.analyses import ImageAssetDTO, RunArtifacts, SegmentationRunDTO
from app.contracts.file_artifacts import FileArtifactKind
from app.contracts.identity import PrincipalContext
from app.files import FileArtifactAccessService, FileArtifactUnavailableError

_RUN_ARTIFACT_PATHS = {
    "mask_url": "pred_mask_path",
    "overlay_url": "overlay_path",
    "probability_url": "probability_path",
    "instances_url": "instances_path",
    "labeled_particles_url": "labeled_particles_path",
    "particles_csv_url": "particles_csv_path",
    "quality_report_url": "quality_report_path",
    "execution_provenance_url": "execution_provenance_path",
}


def download_url(
    request: Request,
    file_access: FileArtifactAccessService,
    *,
    principal: PrincipalContext,
    job_id: str,
    artifact_kind: FileArtifactKind,
    path: str,
    image_id: str | None = None,
    run_id: str | None = None,
    filename: str | None = None,
    expected_sha256: str | None = None,
) -> str | None:
    """Issue a v2 token only after authorization, pinning, and registration."""

    try:
        token = file_access.issue_download_token(
            principal=principal,
            job_id=job_id,
            artifact_kind=artifact_kind,
            storage_path=path,
            image_id=image_id,
            run_id=run_id,
            filename=filename,
            expected_sha256=expected_sha256,
        )
    except FileArtifactUnavailableError:
        return None
    prefix = request.app.state.settings.api_prefix.rstrip("/")
    return f"{prefix}/files/{token}"


def decorate_image_download(
    image: ImageAssetDTO,
    *,
    storage_path: str,
    request: Request,
    file_access: FileArtifactAccessService,
    principal: PrincipalContext,
) -> ImageAssetDTO:
    return image.model_copy(
        update={
            "original_download_url": download_url(
                request,
                file_access,
                principal=principal,
                job_id=image.job_id,
                image_id=image.image_id,
                artifact_kind=FileArtifactKind.ORIGINAL_IMAGE,
                path=storage_path,
                filename=image.filename,
                expected_sha256=image.sha256,
            )
        }
    )


def decorate_run_downloads(
    run: SegmentationRunDTO,
    *,
    private_paths: dict[str, str | None],
    request: Request,
    file_access: FileArtifactAccessService,
    principal: PrincipalContext,
) -> SegmentationRunDTO:
    urls = {
        public_name: download_url(
            request,
            file_access,
            principal=principal,
            job_id=run.job_id,
            image_id=run.image_id,
            run_id=run.run_id,
            artifact_kind=FileArtifactKind.RUN_ARTIFACT,
            path=path,
        )
        for public_name, private_name in _RUN_ARTIFACT_PATHS.items()
        if (path := private_paths.get(private_name)) is not None
    }
    return run.model_copy(update={"artifacts": RunArtifacts.model_validate(urls)})

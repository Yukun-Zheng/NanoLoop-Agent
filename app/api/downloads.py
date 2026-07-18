"""Convert private storage keys into short-lived public download URLs."""

from __future__ import annotations

from fastapi import Request

from app.contracts.analyses import ImageAssetDTO, RunArtifacts, SegmentationRunDTO
from app.storage import LocalFileStore
from app.storage.paths import StoragePathError

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


def download_url(request: Request, file_store: LocalFileStore, path: str) -> str | None:
    """Issue a token only when the referenced managed file still exists."""

    try:
        token = file_store.create_file_token(path)
    except (FileNotFoundError, OSError, StoragePathError):
        return None
    prefix = request.app.state.settings.api_prefix.rstrip("/")
    return f"{prefix}/files/{token}"


def decorate_image_download(
    image: ImageAssetDTO,
    *,
    storage_path: str,
    request: Request,
    file_store: LocalFileStore,
) -> ImageAssetDTO:
    return image.model_copy(
        update={"original_download_url": download_url(request, file_store, storage_path)}
    )


def decorate_run_downloads(
    run: SegmentationRunDTO,
    *,
    private_paths: dict[str, str | None],
    request: Request,
    file_store: LocalFileStore,
) -> SegmentationRunDTO:
    urls = {
        public_name: download_url(request, file_store, path)
        for public_name, private_name in _RUN_ARTIFACT_PATHS.items()
        if (path := private_paths.get(private_name)) is not None
    }
    return run.model_copy(update={"artifacts": RunArtifacts.model_validate(urls)})

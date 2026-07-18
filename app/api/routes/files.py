"""Opaque-token download endpoint for managed binary artifacts."""

import mimetypes
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app.api.deps import get_file_store
from app.api.routing import COMMON_ERROR_RESPONSES
from app.core.errors import ResourceNotFoundError
from app.storage import FileTokenError, LocalFileStore, StoragePathError

router = APIRouter(prefix="/files", tags=["files"], responses=COMMON_ERROR_RESPONSES)


@router.get(
    "/{token}",
    response_class=FileResponse,
    responses={
        200: {
            "description": "Managed artifact bytes",
            "content": {
                "application/octet-stream": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
        }
    },
    operation_id="downloadFile",
)
def download_file(
    token: str,
    file_store: Annotated[LocalFileStore, Depends(get_file_store)],
) -> FileResponse:
    try:
        path = file_store.resolve_file_token(token)
    except (FileTokenError, StoragePathError, FileNotFoundError, OSError) as error:
        raise ResourceNotFoundError(details={"resource": "file"}) from error
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(
        path,
        filename=path.name,
        media_type=media_type,
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )

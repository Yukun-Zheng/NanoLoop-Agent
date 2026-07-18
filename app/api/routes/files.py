"""Subject-bound download endpoint for registered managed artifacts."""

from collections.abc import Mapping
from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from starlette.types import Receive, Scope, Send

from app.api.deps import get_file_artifact_access_service, require_api_key_contract
from app.api.routing import COMMON_ERROR_RESPONSES
from app.contracts.identity import PrincipalContext
from app.core.errors import ResourceNotFoundError
from app.files import FileAccessTokenError, FileArtifactAccessService
from app.storage import PinnedFileChunkIterator

router = APIRouter(prefix="/files", tags=["files"], responses=COMMON_ERROR_RESPONSES)


class _PinnedStreamingResponse(StreamingResponse):
    """Close the pinned descriptor on success, cancellation, or send failure."""

    def __init__(
        self,
        chunks: PinnedFileChunkIterator,
        *,
        status_code: int = 200,
        headers: Mapping[str, str] | None = None,
        media_type: str | None = None,
        background: BackgroundTask | None = None,
    ) -> None:
        self._pinned_chunks = chunks
        super().__init__(
            chunks,
            status_code=status_code,
            headers=headers,
            media_type=media_type,
            background=background,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Starlette skips BackgroundTask when an ASGI 2.4 send raises after a
            # client disconnect. This lifecycle guard releases the fd immediately.
            self._pinned_chunks.close()


@router.get(
    "/{token}",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Managed artifact bytes",
            "content": {
                "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}
            },
        }
    },
    operation_id="downloadFile",
)
def download_file(
    token: str,
    file_access: Annotated[
        FileArtifactAccessService,
        Depends(get_file_artifact_access_service),
    ],
    principal: Annotated[PrincipalContext, Depends(require_api_key_contract)],
) -> StreamingResponse:
    try:
        resolved = file_access.resolve_download(token, principal=principal)
    except FileAccessTokenError as error:
        raise ResourceNotFoundError(details={"resource": "file"}) from error

    pinned = resolved.pinned_file
    chunks = pinned.iter_chunks()
    try:
        return _PinnedStreamingResponse(
            chunks,
            media_type=resolved.media_type,
            headers={
                "Cache-Control": "private, no-store",
                "Content-Disposition": _content_disposition(resolved.filename),
                "Content-Length": str(pinned.size_bytes),
                "Cross-Origin-Resource-Policy": "same-origin",
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
            },
            background=BackgroundTask(chunks.close),
        )
    except BaseException:
        chunks.close()
        raise


def _content_disposition(filename: str) -> str:
    """Encode the registry basename without reflecting it as raw header syntax."""

    encoded = quote(filename, safe="", encoding="utf-8", errors="strict")
    return f"attachment; filename=\"download\"; filename*=UTF-8''{encoded}"

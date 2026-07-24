"""Subject-bound download endpoint for registered managed artifacts."""

import math
import os
from collections.abc import Mapping
from io import BytesIO
from typing import Annotated, cast
from urllib.parse import quote

import numpy as np
from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response, StreamingResponse
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.background import BackgroundTask
from starlette.types import Receive, Scope, Send

from app.api.deps import get_file_artifact_access_service, require_api_key_contract
from app.api.routing import COMMON_ERROR_RESPONSES
from app.contracts.identity import PrincipalContext
from app.core.errors import InvalidImageError, ResourceNotFoundError
from app.files import (
    FileAccessTokenError,
    FileArtifactAccessService,
    ResolvedFileDownload,
)
from app.storage import PinnedFileChunkIterator

router = APIRouter(prefix="/files", tags=["files"], responses=COMMON_ERROR_RESPONSES)

_TIFF_MEDIA_TYPES = frozenset({"image/tiff", "image/x-tiff"})
_PREVIEW_MAX_DIMENSION = 4096
_PREVIEW_MAX_PIXELS = 64_000_000
_NUMERIC_GRAYSCALE_MODES = frozenset({"I", "F", "I;16", "I;16L", "I;16B", "I;16N"})
_PNG_NATIVE_MODES = frozenset({"1", "L", "LA", "P", "RGB", "RGBA"})
_PROBABILITY_COLORS = np.asarray(
    [
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37),
    ],
    dtype=np.float32,
)


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
                "application/octet-stream": {
                    "schema": {"type": "string", "format": "binary"}
                },
                "image/png": {"schema": {"type": "string", "format": "binary"}},
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
    preview: Annotated[
        bool,
        Query(
            description=(
                "Return a browser-native PNG when the artifact is TIFF "
                "or a numeric NumPy array."
            )
        ),
    ] = False,
) -> Response:
    try:
        resolved = file_access.resolve_download(token, principal=principal)
    except FileAccessTokenError as error:
        raise ResourceNotFoundError(details={"resource": "file"}) from error

    if preview and resolved.media_type.casefold() in _TIFF_MEDIA_TYPES:
        return _tiff_preview_response(resolved)
    if preview and resolved.filename.casefold().endswith(".npy"):
        return _probability_preview_response(resolved)

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


def _tiff_preview_response(resolved: ResolvedFileDownload) -> Response:
    """Render page one from the already verified descriptor without changing the raw file."""

    pinned = resolved.pinned_file
    try:
        with (
            os.fdopen(os.dup(pinned.fileno()), "rb") as source,
            Image.open(source) as opened,
        ):
            opened.seek(0)
            image = ImageOps.exif_transpose(opened)
            image.thumbnail(
                (_PREVIEW_MAX_DIMENSION, _PREVIEW_MAX_DIMENSION),
                Image.Resampling.LANCZOS,
                reducing_gap=3.0,
            )
            preview_image = _png_compatible(image)
            output = BytesIO()
            preview_image.save(output, format="PNG", optimize=True)
    except (
        Image.DecompressionBombError,
        OSError,
        UnidentifiedImageError,
        ValueError,
    ) as error:
        raise InvalidImageError(
            "该图像暂时无法生成浏览器预览，可下载原始制品审查",
            details={"reason": "preview_decode_failed"},
        ) from error
    finally:
        pinned.close()

    return _png_response(output)


def _probability_preview_response(resolved: ResolvedFileDownload) -> Response:
    """Render a model probability array with one fixed, interpretable 0-1 color scale."""

    pinned = resolved.pinned_file
    try:
        with os.fdopen(os.dup(pinned.fileno()), "rb") as source:
            loaded = np.load(source, allow_pickle=False, max_header_size=16_384)
            if not isinstance(loaded, np.ndarray):
                raise ValueError("NumPy preview must contain one array")
            preview_image = _probability_image(loaded)
            preview_image.thumbnail(
                (_PREVIEW_MAX_DIMENSION, _PREVIEW_MAX_DIMENSION),
                Image.Resampling.LANCZOS,
                reducing_gap=3.0,
            )
            output = BytesIO()
            preview_image.save(output, format="PNG", optimize=True)
    except (OSError, TypeError, ValueError) as error:
        raise InvalidImageError(
            "该数值制品暂时无法生成预览，可下载原始制品审查",
            details={"reason": "preview_decode_failed"},
        ) from error
    finally:
        pinned.close()

    return _png_response(output)


def _probability_image(array: NDArray[np.generic]) -> Image.Image:
    values = np.asarray(array)
    if values.ndim == 3 and 1 in (values.shape[0], values.shape[-1]):
        values = np.squeeze(values)
    if values.ndim != 2:
        raise ValueError("probability preview requires one 2D array")
    if values.size == 0 or values.size > _PREVIEW_MAX_PIXELS:
        raise ValueError("probability preview dimensions are unsupported")
    if not (
        np.issubdtype(values.dtype, np.bool_)
        or np.issubdtype(values.dtype, np.integer)
        or np.issubdtype(values.dtype, np.floating)
    ):
        raise TypeError("probability preview requires numeric values")

    normalized = values.astype(np.float32, copy=False)
    if not np.isfinite(normalized).all():
        raise ValueError("probability preview contains non-finite values")
    normalized = np.clip(normalized, 0.0, 1.0)
    scaled = normalized * float(len(_PROBABILITY_COLORS) - 1)
    lower = np.floor(scaled).astype(np.intp)
    upper = np.minimum(lower + 1, len(_PROBABILITY_COLORS) - 1)
    blend = (scaled - lower)[..., np.newaxis]
    rgb = _PROBABILITY_COLORS[lower] * (1.0 - blend) + _PROBABILITY_COLORS[upper] * blend
    return Image.fromarray(np.rint(rgb).astype(np.uint8), mode="RGB")


def _png_compatible(image: Image.Image) -> Image.Image:
    if image.mode in _NUMERIC_GRAYSCALE_MODES:
        low, high = cast(tuple[float, float], image.getextrema())
        low_value = float(low)
        high_value = float(high)
        if not math.isfinite(low_value) or not math.isfinite(high_value):
            raise ValueError("TIFF preview contains non-finite intensity bounds")
        if high_value <= low_value:
            fill = 0 if high_value <= 0 else 255
            return Image.new("L", image.size, color=fill)
        scale = 255.0 / (high_value - low_value)
        return image.point(lambda value: (float(value) - low_value) * scale).convert("L")
    if image.mode in _PNG_NATIVE_MODES:
        return image.copy()
    return image.convert("RGBA")


def _png_response(output: BytesIO) -> Response:
    return Response(
        content=output.getvalue(),
        media_type="image/png",
        headers={
            "Cache-Control": "private, no-store",
            "Content-Disposition": 'inline; filename="preview.png"',
            "Cross-Origin-Resource-Policy": "same-origin",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _content_disposition(filename: str) -> str:
    """Encode the registry basename without reflecting it as raw header syntax."""

    encoded = quote(filename, safe="", encoding="utf-8", errors="strict")
    return f"attachment; filename=\"download\"; filename*=UTF-8''{encoded}"

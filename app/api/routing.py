"""Shared API response declarations and bounded multipart parsing."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, ClassVar

from fastapi import Request, Response
from fastapi.routing import APIRoute
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException

from app.contracts.common import ApiResponse
from app.core.errors import InvalidMultipartError

ErrorEnvelope = ApiResponse[dict[str, object]]

COMMON_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorEnvelope, "description": "Bad request"},
    404: {"model": ErrorEnvelope, "description": "Resource not found"},
    409: {"model": ErrorEnvelope, "description": "State or revision conflict"},
    413: {"model": ErrorEnvelope, "description": "Payload too large"},
    415: {"model": ErrorEnvelope, "description": "Unsupported media type"},
    422: {"model": ErrorEnvelope, "description": "Validation error"},
    500: {"model": ErrorEnvelope, "description": "Internal server error"},
    501: {"model": ErrorEnvelope, "description": "Frozen route awaiting service integration"},
    503: {"model": ErrorEnvelope, "description": "Dependency unavailable"},
}


@dataclass(frozen=True, slots=True)
class _MultipartPolicy:
    max_files: int
    max_fields: int
    allowed_file_fields: dict[str, int]
    allowed_text_fields: dict[str, int]
    max_text_part_bytes: int = 256 * 1024


_MULTIPART_POLICIES: dict[str, _MultipartPolicy] = {
    "createAnalysis": _MultipartPolicy(
        max_files=20,
        max_fields=1,
        allowed_file_fields={"files": 20},
        allowed_text_fields={"metadata_json": 1},
    ),
    "ingestKnowledgeDocument": _MultipartPolicy(
        max_files=1,
        max_fields=1,
        allowed_file_fields={"file": 1},
        allowed_text_fields={"metadata_json": 1},
    ),
    "uploadCorrectedMask": _MultipartPolicy(
        max_files=1,
        max_fields=0,
        allowed_file_fields={"file": 1},
        allowed_text_fields={},
    ),
}


class BoundedMultipartRoute(APIRoute):
    """Apply operation-specific limits before FastAPI binds multipart fields.

    FastAPI otherwise calls ``request.form()`` with Starlette's permissive
    defaults (1,000 files and 1,000 fields). Parsing here uses the same Request
    object, so FastAPI reuses the cached, spooled ``FormData`` and remains
    responsible for closing it after the endpoint finishes.
    """

    policies: ClassVar[dict[str, _MultipartPolicy]] = _MULTIPART_POLICIES

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original = super().get_route_handler()
        policy = self.policies.get(self.operation_id or "")
        if policy is None:
            return original

        async def bounded_handler(request: Request) -> Response:
            if _is_multipart(request):
                try:
                    form = await request.form(
                        max_files=policy.max_files,
                        max_fields=policy.max_fields,
                        max_part_size=policy.max_text_part_bytes,
                    )
                except HTTPException as error:
                    raise InvalidMultipartError(
                        details=_policy_details(policy, reason="parser_rejected")
                    ) from error
                if not _parts_match_policy(form.multi_items(), policy):
                    await form.close()
                    raise InvalidMultipartError(
                        details=_policy_details(policy, reason="unexpected_part")
                    )
            return await original(request)

        return bounded_handler


def _is_multipart(request: Request) -> bool:
    content_type = request.headers.get("content-type", "")
    media_type = content_type.partition(";")[0].strip().casefold()
    return media_type == "multipart/form-data"


def _parts_match_policy(
    parts: list[tuple[str, str | UploadFile]],
    policy: _MultipartPolicy,
) -> bool:
    file_counts: Counter[str] = Counter()
    text_counts: Counter[str] = Counter()
    for name, value in parts:
        if isinstance(value, UploadFile):
            file_counts[name] += 1
        else:
            text_counts[name] += 1
    return _counts_within(file_counts, policy.allowed_file_fields) and _counts_within(
        text_counts,
        policy.allowed_text_fields,
    )


def _counts_within(counts: Counter[str], maxima: dict[str, int]) -> bool:
    return all(name in maxima and count <= maxima[name] for name, count in counts.items())


def _policy_details(policy: _MultipartPolicy, *, reason: str) -> dict[str, object]:
    return {
        "reason": reason,
        "max_files": policy.max_files,
        "max_fields": policy.max_fields,
        "max_text_part_bytes": policy.max_text_part_bytes,
    }

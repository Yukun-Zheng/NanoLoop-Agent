"""Helpers for constructing the one public JSON response envelope."""

from __future__ import annotations

from typing import TypeVar
from uuid import uuid4

from fastapi import Request

from app.contracts.common import ApiErrorPayload, ApiResponse
from app.contracts.enums import ApiStatus
from app.core.logging import current_request_id

T = TypeVar("T")


def request_id_for(request: Request | None = None) -> str:
    if request is not None:
        request_id = getattr(request.state, "request_id", None)
        if isinstance(request_id, str) and request_id:
            return request_id
    return current_request_id() or f"req_{uuid4().hex}"


def success_response(
    data: T,
    *,
    request: Request | None = None,
    accepted: bool = False,
) -> ApiResponse[T]:
    return ApiResponse[T](
        request_id=request_id_for(request),
        status=ApiStatus.ACCEPTED if accepted else ApiStatus.SUCCESS,
        data=data,
        error=None,
    )


def error_response(
    *,
    code: str,
    message: str,
    details: dict[str, object] | None = None,
    retryable: bool = False,
    request: Request | None = None,
) -> ApiResponse[dict[str, object]]:
    return ApiResponse[dict[str, object]](
        request_id=request_id_for(request),
        status=ApiStatus.ERROR,
        data=None,
        error=ApiErrorPayload(
            code=code,
            message=message,
            details=details or {},
            retryable=retryable,
        ),
    )

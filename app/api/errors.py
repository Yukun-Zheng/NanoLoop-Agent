"""FastAPI exception handlers that preserve the public error envelope."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.api.responses import error_response, request_id_for, success_response
from app.core.errors import InsufficientEvidenceError, NanoLoopError

logger = logging.getLogger(__name__)

_HTTP_CODES = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "RESOURCE_NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
    413: "PAYLOAD_TOO_LARGE",
    415: "UNSUPPORTED_MEDIA_TYPE",
    429: "RATE_LIMITED",
    500: "INTERNAL_ERROR",
    503: "SERVICE_UNAVAILABLE",
}


def install_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(NanoLoopError, _nanoloop_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(HTTPException, _http_error_handler)
    app.add_exception_handler(Exception, _unexpected_error_handler)


async def _nanoloop_error_handler(request: Request, exc: Exception) -> JSONResponse:
    error = _as_nanoloop_error(exc)
    if isinstance(error, InsufficientEvidenceError):
        insufficient_payload = success_response(
            {
                "outcome_code": error.code,
                "message": error.message,
                "limitations": [error.message],
                "details": error.details,
            },
            request=request,
        )
        return _json_response(insufficient_payload, status_code=200, request=request)

    logger.warning(
        "domain_error",
        extra={"event": "domain_error", "status_code": error.status_code},
    )
    error_payload = error_response(
        code=error.code,
        message=error.message,
        details=error.details,
        retryable=error.retryable,
        request=request,
    )
    return _json_response(error_payload, status_code=error.status_code, request=request)


async def _validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    validation_error = _as_validation_error(exc)
    issues = [
        {
            "location": [str(part) for part in issue.get("loc", ())],
            "message": str(issue.get("msg", "invalid value")),
            "type": str(issue.get("type", "value_error")),
        }
        for issue in validation_error.errors()
    ]
    payload = error_response(
        code="VALIDATION_ERROR",
        message="请求参数校验失败",
        details={"issues": issues},
        request=request,
    )
    return _json_response(payload, status_code=422, request=request)


async def _http_error_handler(request: Request, exc: Exception) -> JSONResponse:
    http_error = _as_http_error(exc)
    code = _HTTP_CODES.get(http_error.status_code, f"HTTP_{http_error.status_code}")
    message = http_error.detail if isinstance(http_error.detail, str) else "HTTP 请求失败"
    details: dict[str, object] = {}
    if isinstance(http_error.detail, dict):
        detail_code = http_error.detail.get("code")
        detail_message = http_error.detail.get("message")
        if isinstance(detail_code, str):
            code = detail_code
        if isinstance(detail_message, str):
            message = detail_message
        details = {
            str(key): value
            for key, value in http_error.detail.items()
            if key not in {"code", "message"}
        }
    payload = error_response(
        code=code,
        message=message,
        details=details,
        retryable=http_error.status_code in {429, 503},
        request=request,
    )
    return _json_response(
        payload,
        status_code=http_error.status_code,
        request=request,
        headers=http_error.headers,
    )


async def _unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "unhandled_api_error",
        exc_info=exc,
        extra={"event": "unhandled_api_error", "status_code": 500},
    )
    payload = error_response(
        code="INTERNAL_ERROR",
        message="服务器内部错误",
        retryable=False,
        request=request,
    )
    return _json_response(payload, status_code=500, request=request)


def _json_response(
    payload: Any,
    *,
    status_code: int,
    request: Request,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    response_headers = dict(headers or {})
    response_headers["X-Request-ID"] = request_id_for(request)
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
        headers=response_headers,
    )


def _as_nanoloop_error(exc: Exception) -> NanoLoopError:
    if not isinstance(exc, NanoLoopError):  # pragma: no cover - FastAPI dispatch invariant
        raise TypeError("expected NanoLoopError")
    return exc


def _as_validation_error(exc: Exception) -> RequestValidationError:
    if not isinstance(exc, RequestValidationError):  # pragma: no cover
        raise TypeError("expected RequestValidationError")
    return exc


def _as_http_error(exc: Exception) -> HTTPException:
    if not isinstance(exc, HTTPException):  # pragma: no cover
        raise TypeError("expected HTTPException")
    return exc

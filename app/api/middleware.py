"""ASGI request context and access logging middleware."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.api.responses import error_response
from app.core.logging import bind_log_context, log_context, reset_log_context
from app.core.security import normalize_http_origin, trusted_request_id

logger = logging.getLogger(__name__)
_FILE_TOKEN_SEGMENT = re.compile(r"(?<=/files/)[^/]+")

Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_TRUSTED_FETCH_SITES = frozenset({"same-origin", "same-site", "none"})


class _RequestBodyTooLarge(Exception):
    pass


class ErrorEnvelopeTrustedHostMiddleware(TrustedHostMiddleware):
    """Apply Starlette's trusted-host policy with the public JSON error envelope."""

    def __init__(
        self,
        app: Callable[..., Awaitable[None]],
        *,
        allowed_hosts: list[str],
    ) -> None:
        super().__init__(app, allowed_hosts=allowed_hosts, www_redirect=False)
        self.allowed_hosts = [host.casefold().rstrip(".") for host in self.allowed_hosts]

    async def __call__(self, scope: Message, receive: Receive, send: Send) -> None:
        if self.allow_any or scope.get("type") not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        host_headers = Headers(scope=scope).getlist("host")
        host = _host_header_name(host_headers[0]) if len(host_headers) == 1 else None
        valid = host is not None and any(
            host == pattern or (pattern.startswith("*") and host.endswith(pattern[1:]))
            for pattern in self.allowed_hosts
        )
        if valid:
            await self.app(scope, receive, send)
            return
        if scope.get("type") == "websocket":  # no WebSocket routes exist today
            await super().__call__(scope, receive, send)
            return
        await _send_security_rejection(
            scope,
            send,
            status_code=400,
            code="UNTRUSTED_HOST",
            message="Host 请求头不在允许列表中",
        )


class BrowserMutationGuardMiddleware:
    """Block browser cross-site mutations while retaining Origin-less API clients."""

    def __init__(
        self,
        app: Callable[..., Awaitable[None]],
        *,
        allowed_origins: list[str],
    ) -> None:
        self.app = app
        self.allowed_origins = frozenset(allowed_origins)

    async def __call__(self, scope: Message, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http" or str(scope.get("method", "GET")).upper() in _SAFE_METHODS:
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        origins = headers.getlist("origin")
        fetch_sites = headers.getlist("sec-fetch-site")
        origin = normalize_http_origin(origins[0]) if len(origins) == 1 else None
        same_origin = _request_origin(scope, headers)
        if origins:
            allowed = (
                len(origins) == 1
                and origin is not None
                and (origin == same_origin or origin in self.allowed_origins)
            )
        else:
            fetch_site = fetch_sites[0].casefold() if len(fetch_sites) == 1 else None
            allowed = not fetch_sites or (
                len(fetch_sites) == 1 and fetch_site in _TRUSTED_FETCH_SITES
            )
        if allowed:
            await self.app(scope, receive, send)
            return
        await _send_security_rejection(
            scope,
            send,
            status_code=403,
            code="CROSS_SITE_MUTATION_FORBIDDEN",
            message="浏览器跨站写请求已被拒绝",
        )


def _request_origin(scope: Message, headers: Headers) -> str | None:
    scheme = str(scope.get("scheme", "http"))
    host = headers.get("host", "")
    return normalize_http_origin(f"{scheme}://{host}")


def _host_header_name(value: str) -> str | None:
    try:
        parsed = urlsplit(f"//{value}")
        _ = parsed.port
    except ValueError:
        return None
    if (
        parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
        or value.endswith(":")
    ):
        return None
    return parsed.hostname.casefold().rstrip(".")


async def _send_security_rejection(
    scope: Message,
    send: Send,
    *,
    status_code: int,
    code: str,
    message: str,
) -> None:
    request = Request(scope)
    payload = error_response(code=code, message=message, request=request)
    response = JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )
    await response(scope, receive=lambda: _empty_request(), send=send)


class RequestBodyLimitMiddleware:
    """Reject oversized bodies before multipart parsing can spool them to disk."""

    def __init__(self, app: Callable[..., Awaitable[None]], *, max_body_bytes: int) -> None:
        if max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be positive")
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Message, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        content_length = Headers(scope=scope).get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                declared_length = 0
            if declared_length > self.max_body_bytes:
                await self._send_rejection(scope, send)
                return

        received_bytes = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received_bytes
            message = await receive()
            if message.get("type") == "http.request":
                body = message.get("body", b"")
                if isinstance(body, bytes):
                    received_bytes += len(body)
                    if received_bytes > self.max_body_bytes:
                        raise _RequestBodyTooLarge
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _RequestBodyTooLarge:
            if response_started:  # pragma: no cover - body parsing precedes route responses
                raise
            await self._send_rejection(scope, send)

    async def _send_rejection(self, scope: Message, send: Send) -> None:
        request = Request(scope)
        payload = error_response(
            code="PAYLOAD_TOO_LARGE",
            message="请求体超过允许大小",
            details={"limit_bytes": self.max_body_bytes},
            request=request,
        )
        response = JSONResponse(
            status_code=413,
            content=payload.model_dump(mode="json"),
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )
        await response(scope, receive=lambda: _empty_request(), send=send)


async def _empty_request() -> Message:
    return {"type": "http.request", "body": b"", "more_body": False}


class RequestContextMiddleware:
    """Attach one safe request ID to logs, response headers, and route responses."""

    def __init__(self, app: Callable[..., Awaitable[None]]) -> None:
        self.app = app

    async def __call__(self, scope: Message, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        request_id = trusted_request_id(headers.get("x-request-id")) or f"req_{uuid4().hex}"
        scope.setdefault("state", {})["request_id"] = request_id
        token = bind_log_context(request_id=request_id)
        started = time.perf_counter()
        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = int(message["status"])
                response_headers = MutableHeaders(scope=message)
                response_headers["X-Request-ID"] = request_id
                response_headers.setdefault("X-Content-Type-Options", "nosniff")
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        except Exception:
            logger.exception(
                "request_failed_before_response",
                extra={"event": "request_failed", "status_code": 500},
            )
            raise
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            path_params = scope.get("path_params", {})
            resource_context = {
                key: value
                for key, value in path_params.items()
                if key in {"job_id", "image_id", "run_id", "model_id"} and isinstance(value, str)
            }
            with log_context(**resource_context):
                logger.info(
                    "request_completed",
                    extra={
                        "duration_ms": duration_ms,
                        "event": "request_completed",
                        "method": scope.get("method"),
                        "path": _safe_log_path(scope.get("path")),
                        "status_code": status_code,
                    },
                )
            reset_log_context(token)


def _safe_log_path(value: object) -> object:
    """Keep route observability without writing bearer-like file tokens to logs."""

    if not isinstance(value, str):
        return value
    return _FILE_TOKEN_SEGMENT.sub("<redacted>", value, count=1)

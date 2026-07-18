"""ASGI request context and access logging middleware."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Awaitable, Callable, Mapping, MutableMapping, Sequence
from ipaddress import IPv6Address, ip_address
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.api.responses import error_response
from app.authentication import (
    AUTH_MODE_STATE_KEY,
    AUTH_OUTCOME_STATE_KEY,
    AUTH_REASON_STATE_KEY,
    AUTHENTICATION_VERIFIED_STATE_KEY,
    RequestAuthenticator,
)
from app.contracts.identity import AuthMode, PrincipalContext
from app.core.logging import bind_log_context, log_context, reset_log_context
from app.core.rate_limit import (
    BoundedKeyedTokenBucketLimiter,
    RateLimitBucket,
    RateLimitDecision,
    TokenBucketLimiter,
)
from app.core.security import ApiKeyVerifier, normalize_http_origin, trusted_request_id

logger = logging.getLogger(__name__)
_FILE_TOKEN_SEGMENT = re.compile(r"(?<=/files/)[^/]+")

Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_TRUSTED_FETCH_SITES = frozenset({"same-origin", "same-site", "none"})


class ApiKeyAuthMiddleware:
    """Authenticate protected HTTP routes before request bodies are parsed.

    ``verifier`` remains accepted for compatibility with integrations that constructed the old
    shared-key middleware directly.  Application assembly always supplies the unified
    ``authenticator``.
    """

    def __init__(
        self,
        app: Callable[..., Awaitable[None]],
        *,
        authenticator: RequestAuthenticator | None = None,
        verifier: ApiKeyVerifier | None = None,
        principal_limiter: BoundedKeyedTokenBucketLimiter | None = None,
        public_paths: Sequence[str],
    ) -> None:
        if (authenticator is None) == (verifier is None):
            raise ValueError("provide exactly one authenticator or legacy verifier")
        self.app = app
        if authenticator is not None:
            self.authenticator = authenticator
        else:
            assert verifier is not None
            self.authenticator = RequestAuthenticator.from_legacy_verifier(verifier)
        if principal_limiter is not None and self.authenticator.mode is not AuthMode.PRINCIPAL:
            raise ValueError("principal_limiter requires principal authentication mode")
        self.principal_limiter = principal_limiter
        self.public_paths = frozenset(public_paths)

    async def __call__(self, scope: Message, receive: Receive, send: Send) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("path") in self.public_paths
        ):
            await self.app(scope, receive, send)
            return

        values = Headers(scope=scope).getlist("x-api-key")
        decision = await self.authenticator.authenticate(values)
        state = scope.setdefault("state", {})
        state[AUTH_MODE_STATE_KEY] = self.authenticator.mode.value
        state[AUTH_OUTCOME_STATE_KEY] = decision.outcome
        state[AUTH_REASON_STATE_KEY] = decision.reason
        if decision.authenticated:
            principal = decision.principal
            if principal is None:  # Defensive narrowing; AuthenticationDecision enforces this.
                await self._send_unavailable(scope, send)
                return
            state["principal"] = principal
            state[AUTHENTICATION_VERIFIED_STATE_KEY] = True
            state["api_key_authenticated"] = True  # Transitional compatibility for extensions.
            principal_decision: RateLimitDecision | None = None
            if self.principal_limiter is not None:
                principal_id = principal.principal_id
                if not isinstance(principal_id, str) or not principal_id:
                    await self._send_unavailable(scope, send)
                    return
                principal_decision = self.principal_limiter.consume(
                    f"principal:{principal_id}"
                )
                if not principal_decision.allowed:
                    await _send_rate_limit_rejection(
                        scope,
                        send,
                        decision=principal_decision,
                        window_seconds=self.principal_limiter.window_seconds,
                    )
                    return
            identity_context = _principal_log_context(principal)
            token = bind_log_context(**identity_context)
            try:
                if principal_decision is None:
                    await self.app(scope, receive, send)
                else:
                    await self.app(
                        scope,
                        receive,
                        _send_with_rate_limit_headers(
                            send,
                            principal_decision,
                            authoritative=True,
                        ),
                    )
            finally:
                reset_log_context(token)
            return

        if decision.outcome == "unavailable":
            await self._send_unavailable(scope, send)
            return
        logger.warning(
            "api_key_authentication_failed",
            extra={
                "auth_mode": self.authenticator.mode.value,
                "auth_outcome": decision.outcome,
                "auth_reason": decision.reason,
                "event": "authentication_failed",
                "status_code": 401,
            },
        )
        await _send_security_rejection(
            scope,
            send,
            status_code=401,
            code="AUTHENTICATION_REQUIRED",
            message="需要有效的 API Key",
            headers={"WWW-Authenticate": 'ApiKey realm="nanoloop"'},
        )

    async def _send_unavailable(self, scope: Message, send: Send) -> None:
        logger.error(
            "authentication_backend_unavailable",
            extra={
                "auth_mode": self.authenticator.mode.value,
                "auth_outcome": "unavailable",
                "auth_reason": "backend_unavailable",
                "event": "authentication_unavailable",
                "status_code": 503,
            },
        )
        await _send_security_rejection(
            scope,
            send,
            status_code=503,
            code="AUTHENTICATION_UNAVAILABLE",
            message="认证服务暂时不可用",
            retryable=True,
        )


class InMemoryRateLimitMiddleware:
    """Apply one bounded token bucket per fixed authentication class."""

    def __init__(
        self,
        app: Callable[..., Awaitable[None]],
        *,
        authenticator: RequestAuthenticator | None = None,
        verifier: ApiKeyVerifier | None = None,
        limiter: TokenBucketLimiter | None = None,
        principal_preauth_limiter: BoundedKeyedTokenBucketLimiter | None = None,
        prefer_downstream_rate_limit_headers: bool = False,
        public_paths: Sequence[str],
    ) -> None:
        if (authenticator is None) == (verifier is None):
            raise ValueError("provide exactly one authenticator or legacy verifier")
        self.app = app
        if authenticator is not None:
            self.authenticator = authenticator
        else:
            assert verifier is not None
            self.authenticator = RequestAuthenticator.from_legacy_verifier(verifier)
        if self.authenticator.mode is AuthMode.PRINCIPAL:
            if limiter is not None or principal_preauth_limiter is None:
                raise ValueError(
                    "principal mode requires exactly one principal_preauth_limiter"
                )
        elif limiter is None or principal_preauth_limiter is not None:
            raise ValueError("compatibility modes require exactly one fixed limiter")
        if (
            prefer_downstream_rate_limit_headers
            and self.authenticator.mode is not AuthMode.PRINCIPAL
        ):
            raise ValueError(
                "downstream rate-limit headers are only valid for principal two-stage limiting"
            )
        self.limiter = limiter
        self.principal_preauth_limiter = principal_preauth_limiter
        self.prefer_downstream_rate_limit_headers = prefer_downstream_rate_limit_headers
        self.public_paths = frozenset(public_paths)

    async def __call__(self, scope: Message, receive: Receive, send: Send) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("path") in self.public_paths
        ):
            await self.app(scope, receive, send)
            return

        values = Headers(scope=scope).getlist("x-api-key")
        if self.authenticator.mode is AuthMode.PRINCIPAL:
            principal_limiter = self.principal_preauth_limiter
            if principal_limiter is None:  # Construction invariant.
                raise RuntimeError("principal pre-authentication limiter is unavailable")
            decision = principal_limiter.consume(_direct_peer_rate_limit_key(scope))
            window_seconds = principal_limiter.window_seconds
        else:
            limiter = self.limiter
            if limiter is None:  # Construction invariant.
                raise RuntimeError("fixed rate limiter is unavailable")
            bucket: RateLimitBucket = self.authenticator.rate_limit_bucket(values)
            decision = limiter.consume(bucket)
            window_seconds = limiter.window_seconds
        if not decision.allowed:
            await _send_rate_limit_rejection(
                scope,
                send,
                decision=decision,
                window_seconds=window_seconds,
            )
            return
        await self.app(
            scope,
            receive,
            _send_with_rate_limit_headers(
                send,
                decision,
                authoritative=not self.prefer_downstream_rate_limit_headers,
            ),
        )


def _direct_peer_rate_limit_key(scope: Mapping[str, Any]) -> str:
    """Return a bounded key from the socket peer, never from proxy-supplied headers."""

    client = scope.get("client")
    host = client[0] if isinstance(client, (list, tuple)) and client else None
    if not isinstance(host, str):
        return "peer:unknown"
    try:
        address = ip_address(host)
    except ValueError:
        return "peer:unknown"
    if isinstance(address, IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    normalized = address.compressed
    return f"peer:{normalized}"


def _rate_limit_headers(decision: RateLimitDecision) -> dict[str, str]:
    return {
        "X-RateLimit-Limit": str(decision.limit),
        "X-RateLimit-Remaining": str(decision.remaining),
    }


def _send_with_rate_limit_headers(
    send: Send,
    decision: RateLimitDecision,
    *,
    authoritative: bool = False,
) -> Send:
    async def send_with_rate_limit(message: Message) -> None:
        if message.get("type") == "http.response.start":
            response_headers = MutableHeaders(scope=message)
            for name, value in _rate_limit_headers(decision).items():
                if authoritative:
                    response_headers[name] = value
                else:
                    response_headers.setdefault(name, value)
        await send(message)

    return send_with_rate_limit


async def _send_rate_limit_rejection(
    scope: Message,
    send: Send,
    *,
    decision: RateLimitDecision,
    window_seconds: float,
) -> None:
    retry_after = str(decision.retry_after_seconds)
    logger.warning(
        "api_rate_limit_exceeded",
        extra={"event": "rate_limit_exceeded", "status_code": 429},
    )
    await _send_security_rejection(
        scope,
        send,
        status_code=429,
        code="RATE_LIMITED",
        message="请求过于频繁，请稍后重试",
        details={"limit": decision.limit, "window_seconds": window_seconds},
        retryable=True,
        headers={**_rate_limit_headers(decision), "Retry-After": retry_after},
    )


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
    details: dict[str, object] | None = None,
    retryable: bool = False,
    headers: Mapping[str, str] | None = None,
) -> None:
    request = Request(scope)
    payload = error_response(
        code=code,
        message=message,
        details=details,
        retryable=retryable,
        request=request,
    )
    response_headers = {
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
        **dict(headers or {}),
    }
    response = JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
        headers=response_headers,
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
            state = scope.get("state", {})
            principal = state.get("principal") if isinstance(state, Mapping) else None
            identity_context = (
                _principal_log_context(principal)
                if isinstance(principal, PrincipalContext)
                else {}
            )
            auth_fields = {
                key: value
                for key, state_key in (
                    ("auth_mode", AUTH_MODE_STATE_KEY),
                    ("auth_outcome", AUTH_OUTCOME_STATE_KEY),
                    ("auth_reason", AUTH_REASON_STATE_KEY),
                )
                if isinstance(state, Mapping)
                and isinstance(value := state.get(state_key), str)
                and value
            }
            with log_context(**resource_context, **identity_context):
                logger.info(
                    "request_completed",
                    extra={
                        **auth_fields,
                        **identity_context,
                        "duration_ms": duration_ms,
                        "event": "request_completed",
                        "method": scope.get("method"),
                        "path": _safe_log_path(scope.get("path")),
                        "status_code": status_code,
                    },
                )
            reset_log_context(token)


def _principal_log_context(principal: PrincipalContext) -> dict[str, str]:
    return {
        key: value
        for key, value in (
            ("tenant_id", principal.tenant_id),
            ("principal_id", principal.principal_id),
            ("credential_id", principal.credential_id),
        )
        if isinstance(value, str) and value
    }


def _safe_log_path(value: object) -> object:
    """Keep route observability without writing bearer-like file tokens to logs."""

    if not isinstance(value, str):
        return value
    return _FILE_TOKEN_SEGMENT.sub("<redacted>", value, count=1)

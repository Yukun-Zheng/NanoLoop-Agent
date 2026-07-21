from __future__ import annotations

import json
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any, cast

import pytest
from starlette.datastructures import Headers

from app.api.middleware import (
    ApiKeyAuthMiddleware,
    InMemoryRateLimitMiddleware,
    RequestBodyLimitMiddleware,
    _direct_peer_rate_limit_key,
    _safe_log_path,
)
from app.authentication import AuthenticationDecision, RequestAuthenticator
from app.contracts.identity import (
    AuthMode,
    PrincipalContext,
    PrincipalKind,
    PrincipalRole,
)
from app.core.rate_limit import BoundedKeyedTokenBucketLimiter, TokenBucketLimiter
from app.core.security import ApiKeyVerifier

_API_KEY = "k" * 32
_PRINCIPAL = PrincipalContext(
    tenant_id=f"tnt_{'a' * 32}",
    principal_id=f"prn_{'b' * 32}",
    credential_id=f"crd_{'c' * 32}",
    kind=PrincipalKind.USER,
    role=PrincipalRole.ANALYST,
    auth_mode=AuthMode.PRINCIPAL,
)
ASGIApp = Callable[
    [
        dict[str, Any],
        Callable[[], Awaitable[dict[str, Any]]],
        Callable[[dict[str, Any]], Awaitable[None]],
    ],
    Awaitable[None],
]


async def _request_middleware(
    factory: Callable[[ASGIApp], ASGIApp],
    *,
    path: str = "/api/v1/models",
    method: str = "GET",
    headers: list[tuple[bytes, bytes]] | None = None,
    client: tuple[str, int] = ("127.0.0.1", 1234),
    downstream_headers: list[tuple[bytes, bytes]] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers or [],
        "client": client,
        "server": ("test", 80),
        "state": {"request_id": "req_security"},
    }
    sent: list[dict[str, Any]] = []
    downstream_called = False

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def downstream(
        _scope: dict[str, Any],
        _receive: Callable[[], Awaitable[dict[str, Any]]],
        send_response: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        nonlocal downstream_called
        downstream_called = True
        await send_response(
            {
                "type": "http.response.start",
                "status": 204,
                "headers": downstream_headers or [],
            }
        )
        await send_response({"type": "http.response.body", "body": b""})

    target = factory(downstream)
    await target(scope, receive, send)
    return sent, downstream_called


class _StubPrincipalAuthenticator:
    mode = AuthMode.PRINCIPAL

    async def authenticate(self, values: list[str]) -> AuthenticationDecision:
        if values == ["valid"]:
            return AuthenticationDecision(
                outcome="authenticated",
                reason="credential_active",
                principal=_PRINCIPAL,
            )
        if values == ["unavailable"]:
            return AuthenticationDecision(outcome="unavailable", reason="backend_unavailable")
        return AuthenticationDecision(outcome="rejected", reason="credential_rejected")


def _principal_authenticator() -> RequestAuthenticator:
    return cast(RequestAuthenticator, _StubPrincipalAuthenticator())


async def _exercise(
    *,
    limit: int,
    chunks: list[bytes],
    content_length: int | None,
) -> tuple[list[dict[str, Any]], bool]:
    headers = [] if content_length is None else [(b"content-length", str(content_length).encode())]
    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/upload",
        "raw_path": b"/upload",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
        "state": {"request_id": "req_body_limit"},
    }
    queued = deque(
        {
            "type": "http.request",
            "body": chunk,
            "more_body": index < len(chunks) - 1,
        }
        for index, chunk in enumerate(chunks)
    )
    sent: list[dict[str, Any]] = []
    downstream_called = False

    async def receive() -> dict[str, Any]:
        if queued:
            return queued.popleft()
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def downstream(
        _scope: dict[str, Any],
        receive_body: Callable[[], Awaitable[dict[str, Any]]],
        send_response: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        nonlocal downstream_called
        downstream_called = True
        while True:
            message = await receive_body()
            if not message.get("more_body", False):
                break
        await send_response({"type": "http.response.start", "status": 204, "headers": []})
        await send_response({"type": "http.response.body", "body": b""})

    middleware = RequestBodyLimitMiddleware(downstream, max_body_bytes=limit)
    await middleware(scope, receive, send)
    return sent, downstream_called


@pytest.mark.asyncio
async def test_declared_oversize_is_rejected_before_downstream() -> None:
    sent, downstream_called = await _exercise(limit=3, chunks=[b"data"], content_length=4)

    assert downstream_called is False
    assert sent[0]["status"] == 413
    payload = json.loads(sent[1]["body"])
    assert payload["request_id"] == "req_body_limit"
    assert payload["error"]["code"] == "PAYLOAD_TOO_LARGE"
    assert payload["error"]["details"] == {"limit_bytes": 3}


@pytest.mark.asyncio
async def test_streamed_oversize_without_content_length_is_rejected() -> None:
    sent, downstream_called = await _exercise(
        limit=3,
        chunks=[b"ab", b"cd"],
        content_length=None,
    )

    assert downstream_called is True
    assert sent[0]["status"] == 413


@pytest.mark.asyncio
async def test_body_at_limit_reaches_downstream() -> None:
    sent, downstream_called = await _exercise(limit=4, chunks=[b"ab", b"cd"], content_length=4)

    assert downstream_called is True
    assert sent[0]["status"] == 204


def test_signed_file_token_is_redacted_from_access_log_path() -> None:
    assert _safe_log_path("/api/v1/files/header.payload.signature") == (
        "/api/v1/files/<redacted>"
    )
    assert _safe_log_path("/api/v1/analyses/job_1") == "/api/v1/analyses/job_1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        [],
        [(b"x-api-key", b"wrong")],
        [(b"x-api-key", _API_KEY.encode()), (b"x-api-key", _API_KEY.encode())],
    ],
)
async def test_api_key_failures_are_indistinguishable(
    headers: list[tuple[bytes, bytes]],
) -> None:
    sent, downstream_called = await _request_middleware(
        lambda app: ApiKeyAuthMiddleware(
            app,
            verifier=ApiKeyVerifier(_API_KEY),
            public_paths=("/health", "/docs", "/openapi.json"),
        ),
        headers=headers,
    )

    assert downstream_called is False
    assert sent[0]["status"] == 401
    response_headers = Headers(raw=sent[0]["headers"])
    assert response_headers["www-authenticate"] == 'ApiKey realm="nanoloop"'
    payload = json.loads(sent[1]["body"])
    assert payload["request_id"] == "req_security"
    assert payload["error"] == {
        "code": "AUTHENTICATION_REQUIRED",
        "message": "需要有效的 API Key",
        "details": {},
        "retryable": False,
    }


@pytest.mark.asyncio
async def test_valid_api_key_reaches_downstream() -> None:
    sent, downstream_called = await _request_middleware(
        lambda app: ApiKeyAuthMiddleware(
            app,
            verifier=ApiKeyVerifier(_API_KEY),
            public_paths=("/health",),
        ),
        headers=[(b"x-api-key", _API_KEY.encode())],
    )

    assert downstream_called is True
    assert sent[0]["status"] == 204


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/health", "/docs", "/openapi.json"])
async def test_exact_public_paths_do_not_require_api_key(path: str) -> None:
    sent, downstream_called = await _request_middleware(
        lambda app: ApiKeyAuthMiddleware(
            app,
            verifier=ApiKeyVerifier(_API_KEY),
            public_paths=("/health", "/docs", "/openapi.json"),
        ),
        path=path,
    )

    assert downstream_called is True
    assert sent[0]["status"] == 204


@pytest.mark.asyncio
async def test_similar_path_and_plain_options_are_not_public() -> None:
    rejected, rejected_downstream = await _request_middleware(
        lambda app: ApiKeyAuthMiddleware(
            app,
            verifier=ApiKeyVerifier(_API_KEY),
            public_paths=("/health",),
        ),
        path="/health/extra",
    )
    plain_options, options_downstream = await _request_middleware(
        lambda app: ApiKeyAuthMiddleware(
            app,
            verifier=ApiKeyVerifier(_API_KEY),
            public_paths=("/health",),
        ),
        method="OPTIONS",
    )

    assert rejected_downstream is False
    assert rejected[0]["status"] == 401
    assert options_downstream is False
    assert plain_options[0]["status"] == 401


@pytest.mark.asyncio
async def test_plain_options_consumes_the_anonymous_rate_limit() -> None:
    verifier = ApiKeyVerifier(_API_KEY)
    limiter = TokenBucketLimiter(1, 60, clock=lambda: 0.0)

    def chain(app: ASGIApp) -> ASGIApp:
        authenticated = ApiKeyAuthMiddleware(
            app,
            verifier=verifier,
            public_paths=("/health",),
        )
        return InMemoryRateLimitMiddleware(
            authenticated,
            verifier=verifier,
            limiter=limiter,
            public_paths=("/health",),
        )

    first, _ = await _request_middleware(chain, method="OPTIONS")
    second, _ = await _request_middleware(chain, method="OPTIONS")

    assert first[0]["status"] == 401
    assert second[0]["status"] == 429


@pytest.mark.asyncio
async def test_rate_limit_keeps_anonymous_and_authenticated_buckets_separate() -> None:
    verifier = ApiKeyVerifier(_API_KEY)
    limiter = TokenBucketLimiter(1, 60, clock=lambda: 0.0)

    def chain(app: ASGIApp) -> ASGIApp:
        authenticated = ApiKeyAuthMiddleware(
            app,
            verifier=verifier,
            public_paths=("/health",),
        )
        return InMemoryRateLimitMiddleware(
            authenticated,
            verifier=verifier,
            limiter=limiter,
            public_paths=("/health",),
        )

    first_wrong, _ = await _request_middleware(
        chain,
        headers=[(b"x-api-key", b"wrong")],
    )
    second_wrong, _ = await _request_middleware(
        chain,
        headers=[(b"x-api-key", b"wrong")],
    )
    valid, valid_downstream = await _request_middleware(
        chain,
        headers=[(b"x-api-key", _API_KEY.encode())],
    )

    assert first_wrong[0]["status"] == 401
    first_headers = Headers(raw=first_wrong[0]["headers"])
    assert first_headers["x-ratelimit-remaining"] == "0"
    assert second_wrong[0]["status"] == 429
    limited_headers = Headers(raw=second_wrong[0]["headers"])
    assert limited_headers["retry-after"] == "60"
    limited_payload = json.loads(second_wrong[1]["body"])
    assert limited_payload["error"]["code"] == "RATE_LIMITED"
    assert limited_payload["error"]["retryable"] is True
    assert limited_payload["error"]["details"] == {
        "limit": 1,
        "window_seconds": 60.0,
    }
    assert valid_downstream is True
    assert valid[0]["status"] == 204


@pytest.mark.asyncio
async def test_single_stage_fixed_limiter_overrides_downstream_rate_headers() -> None:
    verifier = ApiKeyVerifier(_API_KEY)
    limiter = TokenBucketLimiter(2, 60, clock=lambda: 0.0)

    sent, downstream_called = await _request_middleware(
        lambda app: InMemoryRateLimitMiddleware(
            app,
            verifier=verifier,
            limiter=limiter,
            public_paths=("/health",),
        ),
        headers=[(b"x-api-key", _API_KEY.encode())],
        downstream_headers=[
            (b"x-ratelimit-limit", b"999"),
            (b"x-ratelimit-remaining", b"999"),
        ],
    )
    response_headers = Headers(raw=sent[0]["headers"])

    assert downstream_called is True
    assert response_headers["x-ratelimit-limit"] == "2"
    assert response_headers["x-ratelimit-remaining"] == "1"


@pytest.mark.asyncio
async def test_principal_preauth_only_overrides_downstream_rate_headers() -> None:
    authenticator = _principal_authenticator()
    preauth = BoundedKeyedTokenBucketLimiter(3, 60, max_buckets=8, clock=lambda: 0.0)

    sent, downstream_called = await _request_middleware(
        lambda app: InMemoryRateLimitMiddleware(
            app,
            authenticator=authenticator,
            principal_preauth_limiter=preauth,
            public_paths=("/health",),
        ),
        headers=[(b"x-api-key", b"valid")],
        downstream_headers=[
            (b"x-ratelimit-limit", b"999"),
            (b"x-ratelimit-remaining", b"999"),
        ],
    )
    response_headers = Headers(raw=sent[0]["headers"])

    assert downstream_called is True
    assert response_headers["x-ratelimit-limit"] == "3"
    assert response_headers["x-ratelimit-remaining"] == "2"


@pytest.mark.asyncio
async def test_principal_preauth_isolates_direct_peers_and_ignores_forwarded_headers() -> None:
    authenticator = _principal_authenticator()
    preauth = BoundedKeyedTokenBucketLimiter(1, 60, max_buckets=8, clock=lambda: 0.0)
    postauth = BoundedKeyedTokenBucketLimiter(1, 60, max_buckets=8, clock=lambda: 0.0)

    def chain(app: ASGIApp) -> ASGIApp:
        authenticated = ApiKeyAuthMiddleware(
            app,
            authenticator=authenticator,
            principal_limiter=postauth,
            public_paths=("/health",),
        )
        return InMemoryRateLimitMiddleware(
            authenticated,
            authenticator=authenticator,
            principal_preauth_limiter=preauth,
            prefer_downstream_rate_limit_headers=True,
            public_paths=("/health",),
        )

    attacker_headers = [
        (b"x-api-key", b"wrong"),
        (b"x-forwarded-for", b"198.51.100.10"),
        (b"forwarded", b"for=198.51.100.11"),
    ]
    first, _ = await _request_middleware(
        chain,
        headers=attacker_headers,
        client=("192.0.2.1", 1000),
    )
    same_peer, same_peer_downstream = await _request_middleware(
        chain,
        headers=[
            (b"x-api-key", b"valid"),
            (b"x-forwarded-for", b"203.0.113.99"),
        ],
        client=("192.0.2.1", 2000),
    )
    other_peer, other_peer_downstream = await _request_middleware(
        chain,
        headers=[(b"x-api-key", b"valid")],
        client=("192.0.2.2", 3000),
    )
    same_principal_other_peer, principal_downstream = await _request_middleware(
        chain,
        headers=[(b"x-api-key", b"valid")],
        client=("192.0.2.3", 4000),
    )

    assert first[0]["status"] == 401
    assert same_peer[0]["status"] == 429
    assert same_peer_downstream is False
    assert other_peer[0]["status"] == 204
    assert other_peer_downstream is True
    assert same_principal_other_peer[0]["status"] == 429
    assert principal_downstream is False
    assert preauth.bucket_count == 3
    assert postauth.bucket_count == 1


@pytest.mark.asyncio
async def test_principal_rejections_and_unavailability_do_not_consume_postauth_bucket() -> None:
    authenticator = _principal_authenticator()
    preauth = BoundedKeyedTokenBucketLimiter(10, 60, max_buckets=8, clock=lambda: 0.0)
    postauth = BoundedKeyedTokenBucketLimiter(1, 60, max_buckets=8, clock=lambda: 0.0)

    def chain(app: ASGIApp) -> ASGIApp:
        authenticated = ApiKeyAuthMiddleware(
            app,
            authenticator=authenticator,
            principal_limiter=postauth,
            public_paths=("/health",),
        )
        return InMemoryRateLimitMiddleware(
            authenticated,
            authenticator=authenticator,
            principal_preauth_limiter=preauth,
            prefer_downstream_rate_limit_headers=True,
            public_paths=("/health",),
        )

    rejected, _ = await _request_middleware(
        chain,
        headers=[(b"x-api-key", b"wrong")],
        client=("192.0.2.10", 1000),
    )
    unavailable, _ = await _request_middleware(
        chain,
        headers=[(b"x-api-key", b"unavailable")],
        client=("192.0.2.11", 1000),
    )

    assert rejected[0]["status"] == 401
    assert unavailable[0]["status"] == 503
    assert postauth.bucket_count == 0

    accepted, accepted_downstream = await _request_middleware(
        chain,
        headers=[(b"x-api-key", b"valid")],
        client=("192.0.2.12", 1000),
    )
    limited, limited_downstream = await _request_middleware(
        chain,
        headers=[(b"x-api-key", b"valid")],
        client=("192.0.2.13", 1000),
    )

    assert accepted[0]["status"] == 204
    assert accepted_downstream is True
    assert limited[0]["status"] == 429
    assert limited_downstream is False
    assert postauth.bucket_count == 1


@pytest.mark.asyncio
async def test_principal_postauth_headers_override_preauth_headers() -> None:
    authenticator = _principal_authenticator()
    preauth = BoundedKeyedTokenBucketLimiter(5, 60, max_buckets=8, clock=lambda: 0.0)
    postauth = BoundedKeyedTokenBucketLimiter(2, 60, max_buckets=8, clock=lambda: 0.0)

    def chain(app: ASGIApp) -> ASGIApp:
        authenticated = ApiKeyAuthMiddleware(
            app,
            authenticator=authenticator,
            principal_limiter=postauth,
            public_paths=("/health",),
        )
        return InMemoryRateLimitMiddleware(
            authenticated,
            authenticator=authenticator,
            principal_preauth_limiter=preauth,
            prefer_downstream_rate_limit_headers=True,
            public_paths=("/health",),
        )

    sent, downstream_called = await _request_middleware(
        chain,
        headers=[(b"x-api-key", b"valid")],
        client=("2001:db8::1", 1000),
        downstream_headers=[
            (b"x-ratelimit-limit", b"999"),
            (b"x-ratelimit-remaining", b"999"),
        ],
    )
    response_headers = Headers(raw=sent[0]["headers"])

    assert downstream_called is True
    assert response_headers["x-ratelimit-limit"] == "2"
    assert response_headers["x-ratelimit-remaining"] == "1"
    assert _direct_peer_rate_limit_key({"client": ("2001:0db8::1", 1000)}) == (
        "peer:2001:db8::1"
    )
    assert _direct_peer_rate_limit_key({"client": ("::ffff:192.0.2.1", 1000)}) == (
        "peer:192.0.2.1"
    )
    assert _direct_peer_rate_limit_key({"client": ("::ffff:c000:201", 1000)}) == (
        "peer:192.0.2.1"
    )
    assert _direct_peer_rate_limit_key({"client": ("192.0.2.1", 1000)}) == (
        "peer:192.0.2.1"
    )


@pytest.mark.asyncio
async def test_authentication_precedes_request_body_parsing() -> None:
    def chain(app: ASGIApp) -> ASGIApp:
        limited = RequestBodyLimitMiddleware(app, max_body_bytes=3)
        return ApiKeyAuthMiddleware(
            limited,
            verifier=ApiKeyVerifier(_API_KEY),
            public_paths=("/health",),
        )

    missing, missing_downstream = await _request_middleware(
        chain,
        method="POST",
        headers=[(b"content-length", b"4")],
    )
    authenticated, authenticated_downstream = await _request_middleware(
        chain,
        method="POST",
        headers=[
            (b"content-length", b"4"),
            (b"x-api-key", _API_KEY.encode()),
        ],
    )

    assert missing_downstream is False
    assert missing[0]["status"] == 401
    assert authenticated_downstream is False
    assert authenticated[0]["status"] == 413

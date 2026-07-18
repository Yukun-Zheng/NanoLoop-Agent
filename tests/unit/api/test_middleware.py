from __future__ import annotations

import json
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from app.api.middleware import RequestBodyLimitMiddleware, _safe_log_path


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

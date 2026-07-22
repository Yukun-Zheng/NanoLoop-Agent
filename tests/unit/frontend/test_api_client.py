from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import frontend.app as frontend_app
from frontend.api_client import (
    ApiClientError,
    ApiResult,
    NanoLoopApiClient,
    UploadPart,
)


def _success_response(
    request: httpx.Request,
    data: dict[str, object] | None = None,
    *,
    status: str = "success",
    status_code: int = 200,
) -> httpx.Response:
    request_id = request.headers["x-request-id"]
    return httpx.Response(
        status_code,
        headers={"X-Request-ID": request_id},
        json={
            "request_id": request_id,
            "status": status,
            "data": data or {"ok": True},
            "error": None,
        },
    )


@pytest.fixture
def recorded_client() -> Iterator[tuple[NanoLoopApiClient, list[httpx.Request]]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        status = "accepted" if request.url.path.endswith(("/runs", "/reindex")) else "success"
        return _success_response(request, {"path": request.url.path}, status=status)

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    client = NanoLoopApiClient(
        "http://backend.test",
        client=http_client,
        request_id_factory=lambda: "web_test_request",
    )
    try:
        yield client, requests
    finally:
        http_client.close()


def test_public_json_methods_match_current_routes(
    recorded_client: tuple[NanoLoopApiClient, list[httpx.Request]],
) -> None:
    client, requests = recorded_client

    results = [
        client.health(),
        client.list_models(
            status="ready",
            family="unet",
            variant="general",
            quality_tier="balanced",
            material="TiO2",
        ),
        client.recommend_models({"image_id": "img_1", "roi_mode": "full_image"}),
        client.get_analysis("job_1"),
        client.get_boxes("job_1", "img_1"),
        client.replace_boxes(
            "job_1",
            "img_1",
            expected_revision=3,
            boxes=[{"x1": 0, "y1": 0, "x2": 64, "y2": 64}],
        ),
        client.create_runs(
            "job_1",
            {
                "image_ids": ["img_1"],
                "model_ids": ["model_1"],
                "roi_mode": "full_image",
            },
        ),
        client.get_run("run_1"),
        client.review_run("run_1", {"threshold": 0.6}),
        client.query_analysis(
            "job_1",
            {"question": "颗粒数是多少？", "query_type": "analysis_data"},
        ),
        client.list_knowledge_documents(),
        client.update_knowledge_document("doc_1", enabled=False),
        client.reindex_knowledge(force=True),
        client.export_analysis("job_1", run_ids=["run_2", "run_1", "run_2"]),
    ]

    assert all(isinstance(result, ApiResult) for result in results)
    assert all(result.request_id == "web_test_request" for result in results)
    assert results[6].status == "accepted"
    assert results[12].status == "accepted"
    assert [(request.method, request.url.path) for request in requests] == [
        ("GET", "/api/v1/health"),
        ("GET", "/api/v1/models"),
        ("POST", "/api/v1/models/recommend"),
        ("GET", "/api/v1/analyses/job_1"),
        ("GET", "/api/v1/analyses/job_1/images/img_1/boxes"),
        ("PUT", "/api/v1/analyses/job_1/images/img_1/boxes"),
        ("POST", "/api/v1/analyses/job_1/runs"),
        ("GET", "/api/v1/runs/run_1"),
        ("POST", "/api/v1/runs/run_1/review"),
        ("POST", "/api/v1/analyses/job_1/query"),
        ("GET", "/api/v1/knowledge/documents"),
        ("PATCH", "/api/v1/knowledge/documents/doc_1"),
        ("POST", "/api/v1/knowledge/reindex"),
        ("GET", "/api/v1/analyses/job_1/export"),
    ]
    assert dict(requests[1].url.params.multi_items()) == {
        "status": "ready",
        "family": "unet",
        "variant": "general",
        "quality_tier": "balanced",
        "material": "TiO2",
    }
    assert json.loads(requests[5].content) == {
        "expected_revision": 3,
        "boxes": [{"x1": 0, "y1": 0, "x2": 64, "y2": 64}],
    }
    assert json.loads(requests[11].content) == {"enabled": False}
    assert requests[13].url.params.get_list("run_ids") == ["run_2", "run_1"]
    assert all(request.headers["x-request-id"] == "web_test_request" for request in requests)


def test_list_models_omits_every_unset_optional_filter(
    recorded_client: tuple[NanoLoopApiClient, list[httpx.Request]],
) -> None:
    client, requests = recorded_client

    client.list_models()

    assert requests[0].url.path == "/api/v1/models"
    assert list(requests[0].url.params.multi_items()) == []


def test_api_key_is_sent_for_json_multipart_and_artifact_downloads() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.startswith("/api/v1/files/"):
            return httpx.Response(
                200,
                headers={"X-Request-ID": request.headers["x-request-id"]},
                content=b"artifact",
            )
        return _success_response(request)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    api_key = "client-test-key_123"
    client = NanoLoopApiClient(
        "https://backend.test",
        api_key=api_key,
        client=http_client,
        request_id_factory=lambda: "web_authenticated",
    )
    try:
        client.recommend_models({"image_id": "img_1"})
        client.create_analysis(
            [UploadPart("image.tif", b"image-bytes", "image/tiff")],
            {"job_name": "authenticated upload"},
        )
        client.download_artifact("/api/v1/files/signed.token")
        client._send(
            "GET",
            "https://backend.test/api/v1/health",
            request_id="web_internal_header",
            headers={"x-api-key": "must-not-win"},
            timeout=1.0,
        )
    finally:
        http_client.close()

    assert [request.headers["x-api-key"] for request in requests] == [api_key] * 4
    assert "multipart/form-data" in requests[1].headers["content-type"]
    assert requests[2].headers["accept"] == "application/octet-stream"


@pytest.mark.parametrize("api_key", [None, ""])
def test_absent_api_key_does_not_add_authentication_header(api_key: str | None) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _success_response(request)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = NanoLoopApiClient("http://backend.test", api_key=api_key, client=http_client)
    try:
        client.health()
    finally:
        http_client.close()

    assert "x-api-key" not in requests[0].headers


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://127.42.0.9:8000",
        "http://[::1]:8000",
    ],
)
def test_api_key_allows_plain_http_only_for_loopback_hosts(base_url: str) -> None:
    client = NanoLoopApiClient(base_url, api_key="loopback-development-key")
    client.close()


@pytest.mark.parametrize(
    "base_url",
    [
        "http://10.0.0.5:8000",
        "http://backend.test",
        "http://0.0.0.0:8000",
    ],
)
def test_api_key_rejects_plain_http_for_non_loopback_hosts(base_url: str) -> None:
    secret = "must-not-leak-remote-key"
    with pytest.raises(ValueError, match="require HTTPS") as exc_info:
        NanoLoopApiClient(base_url, api_key=secret)
    assert secret not in str(exc_info.value)


def test_api_key_allows_https_for_remote_host() -> None:
    client = NanoLoopApiClient("https://backend.test", api_key="remote-shared-key")
    client.close()


def test_api_key_never_appears_in_client_or_validation_error_text() -> None:
    api_key = "repr-secret-key"
    client = NanoLoopApiClient("https://backend.test", api_key=api_key)
    try:
        assert api_key not in str(client)
        assert api_key not in repr(client)
    finally:
        client.close()

    invalid_key = "invalid-secret\nvalue"
    with pytest.raises(ValueError) as exc_info:
        NanoLoopApiClient("https://backend.test", api_key=invalid_key)
    assert invalid_key not in str(exc_info.value)


def test_api_key_never_appears_in_transport_error_text_or_repr() -> None:
    api_key = "transport-secret-key"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = NanoLoopApiClient(
        "https://backend.test",
        api_key=api_key,
        client=http_client,
        request_id_factory=lambda: "web_secret_transport",
    )
    try:
        with pytest.raises(ApiClientError) as exc_info:
            client.health()
    finally:
        http_client.close()

    assert api_key not in str(exc_info.value)
    assert api_key not in repr(exc_info.value)


def test_streamlit_client_cache_uses_only_api_key_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(
            self,
            base_url: str,
            *,
            api_key: str | None = None,
            timeout: float,
            upload_timeout: float,
        ) -> None:
            self.base_url = base_url
            self.api_key = api_key
            self.timeout = timeout
            self.upload_timeout = upload_timeout
            self.closed = False
            created.append(self)

        def close(self) -> None:
            self.closed = True

    created: list[FakeClient] = []

    monkeypatch.setattr(
        frontend_app.importlib,
        "import_module",
        lambda _: SimpleNamespace(NanoLoopApiClient=FakeClient),
    )
    first_api_key = "first-cache-secret"
    monkeypatch.setenv("NANOLOOP_API_KEY", first_api_key)
    monkeypatch.setenv("NANOLOOP_API_BASE_URL", "https://backend.test")
    state: dict[str, Any] = {
        "api_base_url": "https://backend.test",
        "api_timeout_seconds": 17.0,
    }

    first = frontend_app._get_client(object(), state)
    cached = frontend_app._get_client(object(), state)

    assert first is cached
    assert len(created) == 1
    assert created[0].api_key == first_api_key
    fingerprint = hashlib.sha256(first_api_key.encode()).hexdigest()
    assert state["_api_client_key"] == ("https://backend.test", 17.0, fingerprint)
    assert first_api_key not in repr(state["_api_client_key"])

    second_api_key = "rotated-cache-secret"
    monkeypatch.setenv("NANOLOOP_API_KEY", second_api_key)
    second = frontend_app._get_client(object(), state)

    assert second is not first
    assert created[0].closed
    assert created[1].api_key == second_api_key
    assert state["_api_client_key"][2] == hashlib.sha256(second_api_key.encode()).hexdigest()
    assert second_api_key not in repr(state["_api_client_key"])


def test_streamlit_never_sends_process_api_key_to_session_controlled_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[object] = []

    class FakeClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            created.append(self)

    class FakeStreamlit:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def error(self, message: str) -> None:
            self.messages.append(message)

        def caption(self, message: str) -> None:
            self.messages.append(message)

    secret = "process-only-shared-secret"
    monkeypatch.setenv("NANOLOOP_API_KEY", secret)
    monkeypatch.setenv("NANOLOOP_API_BASE_URL", "https://trusted.example/backend")
    monkeypatch.setattr(
        frontend_app.importlib,
        "import_module",
        lambda _: SimpleNamespace(NanoLoopApiClient=FakeClient),
    )
    streamlit = FakeStreamlit()
    state: dict[str, Any] = {
        "api_base_url": "https://attacker.example/collect",
        "api_timeout_seconds": 17.0,
    }

    client = frontend_app._get_client(streamlit, state)

    assert client is None
    assert created == []
    assert state["_api_client"] is None
    assert state["_api_client_key"] is None
    assert secret not in " ".join(streamlit.messages)


def test_all_multipart_methods_use_exact_field_names_and_upload_timeout() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _success_response(request)

    stream = io.BytesIO(b"mask-bytes")
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = NanoLoopApiClient(
        "http://backend.test",
        client=http_client,
        timeout=httpx.Timeout(7.0),
        upload_timeout=httpx.Timeout(91.0),
        request_id_factory=lambda: "web_upload",
    )
    try:
        client.create_analysis(
            [
                UploadPart("one.tif", b"image-one", "image/tiff"),
                UploadPart("two.png", b"image-two"),
            ],
            {
                "job_name": "中文任务",
                "images": [
                    {"filename": "one.tif", "sample_id": "s1"},
                    {"filename": "two.png", "sample_id": "s2"},
                ],
            },
        )
        client.upload_corrected_mask(
            "run_1",
            UploadPart("corrected.png", stream, "image/png"),
        )
        client.ingest_knowledge_document(
            UploadPart("paper.pdf", b"pdf", "application/pdf"),
            {
                "title": "Paper",
                "source_type": "paper",
                "citation_text": "Citation",
                "license_note": "Internal demo",
            },
        )
    finally:
        http_client.close()

    analysis_body = requests[0].content
    assert b'name="files"; filename="one.tif"' in analysis_body
    assert b'name="files"; filename="two.png"' in analysis_body
    assert b'name="metadata_json"' in analysis_body
    assert "中文任务".encode() in analysis_body

    corrected_body = requests[1].content
    assert b'name="file"; filename="corrected.png"' in corrected_body
    assert b"mask-bytes" in corrected_body

    knowledge_body = requests[2].content
    assert b'name="file"; filename="paper.pdf"' in knowledge_body
    assert b'name="metadata_json"' in knowledge_body
    assert all(request.extensions["timeout"]["read"] == 91.0 for request in requests)
    assert not stream.closed


def test_json_error_preserves_server_code_details_and_request_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            headers={"X-Request-ID": "req_server_409"},
            json={
                "request_id": "req_server_409",
                "status": "error",
                "data": None,
                "error": {
                    "code": "BOX_REVISION_CONFLICT",
                    "message": "矩形框版本已更新",
                    "details": {"current_revision": 4},
                    "retryable": False,
                },
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = NanoLoopApiClient(
        "http://backend.test",
        client=http_client,
        request_id_factory=lambda: "web_conflict",
    )
    try:
        with pytest.raises(ApiClientError) as exc_info:
            client.replace_boxes(
                "job_1",
                "img_1",
                expected_revision=3,
                boxes=[],
            )
    finally:
        http_client.close()

    error = exc_info.value
    assert error.status_code == 409
    assert error.code == "BOX_REVISION_CONFLICT"
    assert error.message == "矩形框版本已更新"
    assert error.details == {"current_revision": 4}
    assert error.request_id == "req_server_409"
    assert not error.retryable
    assert "req_server_409" in str(error)


@pytest.mark.parametrize(
    ("response", "expected_code", "expected_request_id"),
    [
        (
            httpx.Response(500, headers={"X-Request-ID": "req_html"}, text="<html>bad</html>"),
            "HTTP_500",
            "req_html",
        ),
        (
            httpx.Response(
                200,
                headers={"X-Request-ID": "req_bad"},
                json={"request_id": "req_bad", "status": "success", "data": []},
            ),
            "API_PROTOCOL_ERROR",
            "req_bad",
        ),
    ],
)
def test_non_json_error_and_malformed_success_are_explicit_protocol_failures(
    response: httpx.Response,
    expected_code: str,
    expected_request_id: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return response

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = NanoLoopApiClient(
        "http://backend.test",
        client=http_client,
        request_id_factory=lambda: "web_protocol",
    )
    try:
        with pytest.raises(ApiClientError) as exc_info:
            client.health()
    finally:
        http_client.close()

    assert exc_info.value.code == expected_code
    assert exc_info.value.request_id == expected_request_id


def test_mismatched_header_and_body_request_ids_are_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            headers={"X-Request-ID": "req_header"},
            json={
                "request_id": "req_body",
                "status": "success",
                "data": {"ok": True},
                "error": None,
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = NanoLoopApiClient("http://backend.test", client=http_client)
    try:
        with pytest.raises(ApiClientError, match="do not match") as exc_info:
            client.health()
    finally:
        http_client.close()

    assert exc_info.value.code == "API_PROTOCOL_ERROR"


@pytest.mark.parametrize(
    ("exception_factory", "expected_code"),
    [
        (lambda request: httpx.ReadTimeout("slow", request=request), "REQUEST_TIMEOUT"),
        (lambda request: httpx.ConnectError("offline", request=request), "TRANSPORT_ERROR"),
    ],
)
def test_transport_failures_keep_the_outbound_correlation_id(
    exception_factory: Callable[[httpx.Request], httpx.RequestError],
    expected_code: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exception_factory(request)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = NanoLoopApiClient(
        "http://backend.test",
        client=http_client,
        request_id_factory=lambda: "web_transport",
    )
    try:
        with pytest.raises(ApiClientError) as exc_info:
            client.health()
    finally:
        http_client.close()

    assert exc_info.value.code == expected_code
    assert exc_info.value.status_code == 0
    assert exc_info.value.request_id == "web_transport"
    assert exc_info.value.retryable


def test_signed_artifact_download_is_binary_same_origin_and_path_limited() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={
                "X-Request-ID": "req_download",
                "Content-Type": "text/csv; charset=utf-8",
                "Content-Disposition": 'attachment; filename="particles.csv"',
            },
            content=b"particle_id,area\np1,12\n",
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = NanoLoopApiClient(
        "http://backend.test",
        client=http_client,
        request_id_factory=lambda: "web_download",
    )
    try:
        artifact = client.download_artifact("/api/v1/files/signed.token")
        with pytest.raises(ValueError, match="configured backend origin"):
            client.download_artifact("https://attacker.example/api/v1/files/token")
        with pytest.raises(ValueError, match="not a signed"):
            client.download_artifact("/api/v1/health")
    finally:
        http_client.close()

    assert artifact.content == b"particle_id,area\np1,12\n"
    assert artifact.filename == "particles.csv"
    assert artifact.content_type == "text/csv"
    assert artifact.request_id == "req_download"
    assert len(requests) == 1
    assert requests[0].headers["accept"] == "application/octet-stream"
    assert requests[0].headers["x-request-id"] == "web_download"


def test_download_surfaces_json_file_token_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            headers={"X-Request-ID": request.headers["x-request-id"]},
            json={
                "request_id": request.headers["x-request-id"],
                "status": "error",
                "data": None,
                "error": {
                    "code": "RESOURCE_NOT_FOUND",
                    "message": "找不到指定资源",
                    "details": {"resource": "file"},
                    "retryable": False,
                },
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = NanoLoopApiClient(
        "http://backend.test",
        client=http_client,
        request_id_factory=lambda: "web_missing_file",
    )
    try:
        with pytest.raises(ApiClientError) as exc_info:
            client.download_artifact("/api/v1/files/expired")
    finally:
        http_client.close()

    assert exc_info.value.code == "RESOURCE_NOT_FOUND"
    assert exc_info.value.details == {"resource": "file"}


def test_input_validation_happens_before_network_io() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _success_response(request)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = NanoLoopApiClient("http://backend.test", client=http_client)
    try:
        with pytest.raises(ValueError, match="at least one"):
            client.create_analysis([], {"job_name": "empty"})
        with pytest.raises(ValueError, match="single path components"):
            client.upload_corrected_mask("run_1", UploadPart("../mask.png", b"x"))
        with pytest.raises(ValueError, match="not JSON serializable"):
            client.ingest_knowledge_document(
                UploadPart("paper.pdf", b"x"),
                {"invalid": object()},
            )
        with pytest.raises(TypeError, match="enabled must be a bool"):
            client.update_knowledge_document("doc_1", enabled=1)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="doc_id cannot be empty"):
            client.update_knowledge_document("", enabled=True)
    finally:
        http_client.close()

    assert calls == 0


def test_injected_http_client_is_not_closed_by_wrapper() -> None:
    http_client = httpx.Client(transport=httpx.MockTransport(_success_response))
    wrapper = NanoLoopApiClient("http://backend.test", client=http_client)

    wrapper.close()

    assert not http_client.is_closed
    http_client.close()


# ---------------------------------------------------------------------------
# 429 Retry logic tests
#
# Note: The project uses httpx.MockTransport (built into httpx) rather than the
# ``responses`` library, because the HTTP client is httpx, not requests.
# MockTransport provides the same capability — intercepting HTTP requests and
# returning canned responses — without an external dependency.
# ---------------------------------------------------------------------------


def _rate_limited_response(
    request: httpx.Request,
    *,
    retry_after: str | None = "0",
) -> httpx.Response:
    """Build a 429 RATE_LIMITED error response matching the backend envelope."""
    request_id = request.headers["x-request-id"]
    headers: dict[str, str] = {"X-Request-ID": request_id}
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return httpx.Response(
        429,
        headers=headers,
        json={
            "request_id": request_id,
            "status": "error",
            "data": None,
            "error": {
                "code": "RATE_LIMITED",
                "message": "请求过于频繁，已被限流",
                "retryable": True,
            },
        },
    )


def test_429_response_triggers_retry_until_success() -> None:
    """429 twice then 200 — client should retry and succeed on third attempt."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return _rate_limited_response(request, retry_after="0")
        return _success_response(request)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = NanoLoopApiClient(
        "http://backend.test",
        client=http_client,
        max_retries=2,
        default_retry_delay=0.0,
        request_id_factory=lambda: "web_429_retry",
    )
    try:
        result = client.health()
    finally:
        http_client.close()

    assert result.status == "success"
    assert call_count == 3  # initial + 2 retries


def test_429_response_without_retry_after_header_uses_default_delay() -> None:
    """429 without Retry-After header should still retry using default delay."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _rate_limited_response(request, retry_after=None)
        return _success_response(request)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = NanoLoopApiClient(
        "http://backend.test",
        client=http_client,
        max_retries=2,
        default_retry_delay=0.0,
        request_id_factory=lambda: "web_429_no_header",
    )
    try:
        result = client.health()
    finally:
        http_client.close()

    assert result.status == "success"
    assert call_count == 2  # initial 429 + 1 retry success


def test_429_response_exhausts_retries_and_raises() -> None:
    """Always 429 — client should exhaust retries and raise ApiClientError."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return _rate_limited_response(request, retry_after="0")

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = NanoLoopApiClient(
        "http://backend.test",
        client=http_client,
        max_retries=2,
        default_retry_delay=0.0,
        request_id_factory=lambda: "web_429_exhaust",
    )
    try:
        with pytest.raises(ApiClientError) as exc_info:
            client.health()
    finally:
        http_client.close()

    assert exc_info.value.status_code == 429
    assert exc_info.value.code == "RATE_LIMITED"
    assert exc_info.value.retryable is True
    assert call_count == 3  # initial + 2 retries, all 429


def test_429_retry_respects_retry_after_header_value() -> None:
    """Retry-After supports delta seconds, HTTP dates, and a bounded delay."""
    from frontend.api_client import _parse_retry_after

    assert _parse_retry_after("5", default=3.0) == 5.0
    assert _parse_retry_after("0", default=3.0) == 0.0
    assert _parse_retry_after(None, default=3.0) == 3.0
    assert _parse_retry_after("not-a-number", default=3.0) == 3.0
    assert _parse_retry_after("", default=7.0) == 7.0
    assert _parse_retry_after("600", default=3.0) == 30.0
    assert _parse_retry_after("inf", default=3.0) == 3.0

    now = datetime(2026, 7, 23, 0, 0, tzinfo=UTC)
    retry_at = now + timedelta(seconds=12)
    assert (
        _parse_retry_after(
            retry_at.strftime("%a, %d %b %Y %H:%M:%S GMT"),
            default=3.0,
            now=now,
        )
        == 12.0
    )


def test_429_write_request_is_not_automatically_replayed() -> None:
    """A rate-limited upload is surfaced once instead of duplicating a write."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _rate_limited_response(request, retry_after="7")

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = NanoLoopApiClient(
        "http://backend.test",
        client=http_client,
        max_retries=2,
        default_retry_delay=0.0,
        request_id_factory=lambda: "web_429_write",
    )
    try:
        with pytest.raises(ApiClientError) as exc_info:
            client.create_analysis(
                [UploadPart("image.tif", b"image-bytes", "image/tiff")],
                {
                    "job_name": "do not replay",
                    "images": [{"filename": "image.tif", "sample_id": "s1"}],
                },
            )
    finally:
        http_client.close()

    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after_seconds == 7.0


def test_final_429_uses_current_retry_after_header() -> None:
    """The surfaced error reports the final response's current delay value."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return _rate_limited_response(request, retry_after=str(call_count))

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = NanoLoopApiClient(
        "http://backend.test",
        client=http_client,
        max_retries=1,
        default_retry_delay=0.0,
        request_id_factory=lambda: "web_429_final_header",
    )
    try:
        with pytest.raises(ApiClientError) as exc_info:
            client.health()
    finally:
        http_client.close()

    assert call_count == 2
    assert exc_info.value.retry_after_seconds == 2.0


def test_non_429_error_does_not_trigger_retry() -> None:
    """503 should not retry — only 429 triggers the retry loop."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        request_id = request.headers["x-request-id"]
        return httpx.Response(
            503,
            headers={"X-Request-ID": request_id},
            json={
                "request_id": request_id,
                "status": "error",
                "data": None,
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "服务不可用",
                    "retryable": True,
                },
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = NanoLoopApiClient(
        "http://backend.test",
        client=http_client,
        max_retries=2,
        default_retry_delay=0.0,
        request_id_factory=lambda: "web_503_no_retry",
    )
    try:
        with pytest.raises(ApiClientError) as exc_info:
            client.health()
    finally:
        http_client.close()

    assert exc_info.value.status_code == 503
    assert exc_info.value.code == "SERVICE_UNAVAILABLE"
    assert call_count == 1  # no retries for 503


def test_upload_timeout_default_is_300_seconds() -> None:
    """Verify the default upload timeout is 300s (upgraded from 120s)."""
    client = NanoLoopApiClient("http://backend.test")
    try:
        # httpx.Timeout stores the read timeout in .read
        assert client._upload_timeout.read == 300.0
    finally:
        client.close()


def test_create_analysis_accepts_timeout_override() -> None:
    """Verify create_analysis accepts a custom timeout parameter."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _success_response(request)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = NanoLoopApiClient(
        "http://backend.test",
        client=http_client,
        request_id_factory=lambda: "web_upload_override",
    )
    custom_timeout = httpx.Timeout(42.0)
    try:
        client.create_analysis(
            [UploadPart("image.tif", b"image-bytes", "image/tiff")],
            {"job_name": "test", "images": [{"filename": "image.tif", "sample_id": "s1"}]},
            timeout=custom_timeout,
        )
    finally:
        http_client.close()

    assert len(requests) == 1
    assert requests[0].extensions["timeout"]["read"] == 42.0


# ---------------------------------------------------------------------------
# page=null citation rendering tests
# ---------------------------------------------------------------------------


class _RecordingStreamlit:
    """Minimal mock that captures all streamlit method calls for renderer tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        def _method(*args: Any, **kwargs: Any) -> _RecordingStreamlit:
            self.calls.append((name, args, kwargs))
            return self  # enables ``with streamlit.expander(...)`` chaining

        return _method

    def __enter__(self) -> _RecordingStreamlit:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def test_render_query_response_with_null_page_citation_does_not_crash() -> None:
    """render_query_response must not crash when a citation has page=None."""
    from frontend.components import render_query_response

    st = _RecordingStreamlit()
    response = {
        "query_type": "mixed",
        "answer": "TiO2 平均粒径约 45 nm。",
        "data_evidence": [],
        "citations": [
            {
                "citation_id": "cit_001",
                "doc_id": "doc_001",
                "title": "Test Paper",
                "page": None,  # Core test case: page is null
                "chunk_id": "chunk_001",
                "excerpt": "Test excerpt",
                "retrieval_score": 0.92,
                "citation_text": "Author et al., 2023.",
            },
            {
                "citation_id": "cit_002",
                "doc_id": "doc_002",
                "title": "Another Paper",
                "page": 42,
                "chunk_id": "chunk_002",
                "excerpt": "Another excerpt",
                "retrieval_score": 0.85,
            },
        ],
        "tool_calls": [],
        "confidence": "high",
        "outcome_code": "OK",
    }

    # Must not raise
    render_query_response(st, response)

    # Verify expander headings: the page=None citation should show "全文引用"
    expander_headings = [args[0] for name, args, _ in st.calls if name == "expander" and args]
    assert any("全文引用" in h for h in expander_headings), (
        f"Expected '全文引用' in a citation expander heading, got: {expander_headings}"
    )
    # The page=42 citation should show "第 42 页"
    assert any("第 42 页" in h for h in expander_headings), (
        f"Expected '第 42 页' in a citation expander heading, got: {expander_headings}"
    )


def test_render_query_response_insufficient_evidence_empty_citations() -> None:
    """When outcome_code=INSUFFICIENT_EVIDENCE and citations is empty, show '未返回材料知识引用'."""
    from frontend.components import render_query_response

    st = _RecordingStreamlit()
    response = {
        "query_type": "auto",
        "answer": "证据不足，无法回答。",
        "data_evidence": [],
        "citations": [],
        "tool_calls": [],
        "confidence": "low",
        "outcome_code": "INSUFFICIENT_EVIDENCE",
        "limitations": ["未找到相关文献"],
    }

    # Must not raise
    render_query_response(st, response)

    # Verify "未返回材料知识引用" appears in an info call
    info_messages = [args[0] for name, args, _ in st.calls if name == "info" and args]
    assert any("未返回材料知识引用" in msg for msg in info_messages), (
        f"Expected '未返回材料知识引用' in info messages, got: {info_messages}"
    )

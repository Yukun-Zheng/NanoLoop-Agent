from __future__ import annotations

import io
import json
from collections.abc import Callable, Iterator

import httpx
import pytest

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

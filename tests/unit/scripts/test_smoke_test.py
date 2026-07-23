from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

import scripts.smoke_test as smoke_test_module
from scripts.nanoloop_api_client import NanoLoopApiClient
from scripts.smoke_test import (
    SmokeFixture,
    SmokeRunner,
    SmokeTestFailure,
    load_fixture,
    validate_export_zip,
)


@dataclass(slots=True)
class FakeClock:
    value: float = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def _envelope(
    request: httpx.Request,
    data: dict[str, object],
    *,
    status_code: int = 200,
    status: str = "success",
) -> httpx.Response:
    request_id = request.headers["x-request-id"]
    return httpx.Response(
        status_code,
        headers={"X-Request-ID": request_id},
        json={
            "request_id": request_id,
            "status": status,
            "data": data,
            "error": None,
        },
    )


def _error(
    request: httpx.Request,
    *,
    status_code: int,
    code: str,
    message: str,
) -> httpx.Response:
    request_id = request.headers["x-request-id"]
    return httpx.Response(
        status_code,
        headers={"X-Request-ID": request_id},
        json={
            "request_id": request_id,
            "status": "error",
            "data": None,
            "error": {
                "code": code,
                "message": message,
                "details": {},
                "retryable": status_code >= 500,
            },
        },
    )


def _fixture(tmp_path: Path, *, create_files: bool = True) -> SmokeFixture:
    image_path = tmp_path / "real-test-image.tif"
    knowledge_path = tmp_path / "licensed-note.md"
    if create_files:
        image_path.write_bytes(b"test-only-image-content")
        knowledge_path.write_text(
            "Traceable material evidence for a test fixture.", encoding="utf-8"
        )
    fixture_path = tmp_path / "smoke.json"
    fixture_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "job_name": "smoke fixture",
                "images": [
                    {
                        "path": image_path.name,
                        "sample_id": "sample_1",
                        "material_name": "Test material",
                        "material_formula": "T2",
                        "experiment_conditions": {"source": "unit-test"},
                        "scale": {"mode": "nm_per_pixel", "value": 1.0},
                    }
                ],
                "box": {
                    "image_index": 0,
                    "label": "ROI",
                    "x1": 10,
                    "y1": 10,
                    "x2": 80,
                    "y2": 80,
                },
                "inference": {"device": "cpu", "seed": 42},
                "knowledge_document": {
                    "path": knowledge_path.name,
                    "metadata": {
                        "title": "Licensed note",
                        "source_type": "material_note",
                        "citation_text": "Test citation",
                        "license_note": "Test-only content",
                        "allowed_for_demo": True,
                    },
                },
                "queries": {
                    "analysis_data": "颗粒数是多少？",
                    "material_knowledge": "材料有哪些性质？",
                    "material_context": {
                        "formula": "T2",
                        "name": "Test material",
                        "aliases": [],
                        "source": "request",
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return load_fixture(fixture_path, require_files=create_files)


def _export_zip(job_id: str = "job_1", run_id: str = "run_1") -> bytes:
    member_name = f"images/img_1/runs/{run_id}/image_summary.json"
    member_bytes = b'{"particle_count":2}\n'
    manifest = {
        "schema_version": "1.0",
        "job_id": job_id,
        "generated_at": "2026-07-18T00:00:00+00:00",
        "files": [
            {
                "path": member_name,
                "sha256": hashlib.sha256(member_bytes).hexdigest(),
                "size_bytes": len(member_bytes),
            }
        ],
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member_name, member_bytes)
        archive.writestr(
            "export_manifest.json",
            json.dumps(manifest, sort_keys=True).encode(),
        )
    return buffer.getvalue()


class SuccessBackend:
    def __init__(self, export_bytes: bytes, *, complete_after: int = 2) -> None:
        self.export_bytes = export_bytes
        self.export_sha256 = hashlib.sha256(export_bytes).hexdigest()
        self.complete_after = complete_after
        self.run_polls = 0
        self.requests: list[tuple[str, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.requests.append((request.method, path))
        if path == "/api/v1/health":
            return _envelope(
                request,
                {
                    "service": {"status": "healthy"},
                    "database": {"status": "healthy"},
                    "model_registry": {"status": "healthy"},
                    "rag_index": {"status": "healthy"},
                    "version": "test",
                },
            )
        if path == "/api/v1/analyses" and request.method == "POST":
            return _envelope(
                request,
                {
                    "job": {"job_id": "job_1"},
                    "images": [
                        {
                            "image_id": "img_1",
                            "filename": "real-test-image.tif",
                        }
                    ],
                    "runs": [],
                    "partial_failures": [],
                },
                status_code=201,
            )
        if path.endswith("/boxes") and request.method == "GET":
            return _envelope(
                request,
                {"image_id": "img_1", "revision": 0, "boxes": []},
            )
        if path.endswith("/boxes") and request.method == "PUT":
            return _envelope(
                request,
                {
                    "image_id": "img_1",
                    "revision": 1,
                    "boxes": [{"box_id": "box_1"}],
                },
            )
        if path == "/api/v1/models":
            return _envelope(
                request,
                {
                    "models": [
                        {
                            "model_id": "real-ready-model",
                            "status": "ready",
                        }
                    ]
                },
            )
        if path == "/api/v1/analyses/job_1/runs":
            return _envelope(
                request,
                {"run_ids": ["run_1"]},
                status_code=202,
                status="accepted",
            )
        if path == "/api/v1/runs/run_1":
            self.run_polls += 1
            if self.run_polls < self.complete_after:
                return _envelope(
                    request,
                    {"run_id": "run_1", "status": "SEGMENTING", "summary": None},
                )
            return _envelope(
                request,
                {
                    "run_id": "run_1",
                    "status": "COMPLETED",
                    "summary": {"particle_count": 2},
                },
            )
        if path == "/api/v1/knowledge/documents":
            return _envelope(
                request,
                {"doc_id": "doc_1", "chunks_created": 1},
                status_code=202,
                status="accepted",
            )
        if path == "/api/v1/analyses/job_1/query":
            payload = json.loads(request.content)
            if payload["query_type"] == "analysis_data":
                return _envelope(
                    request,
                    {
                        "outcome_code": "OK",
                        "answer": "2 particles",
                        "data_evidence": [
                            {"source_run_ids": ["run_1"], "rows": []}
                        ],
                        "citations": [],
                    },
                )
            return _envelope(
                request,
                {
                    "outcome_code": "OK",
                    "answer": "Evidence-backed answer",
                    "data_evidence": [],
                    "citations": [{"doc_id": "doc_1", "chunk_id": "chunk_1"}],
                },
            )
        if path == "/api/v1/analyses/job_1/export":
            return _envelope(
                request,
                {
                    "job_id": "job_1",
                    "download_url": "/api/v1/files/signed-export",
                    "sha256": self.export_sha256,
                    "filename": "nanoloop-export.zip",
                },
            )
        if path == "/api/v1/files/signed-export":
            return httpx.Response(
                200,
                headers={
                    "X-Request-ID": request.headers["x-request-id"],
                    "Content-Type": "application/zip",
                    "Content-Disposition": 'attachment; filename="nanoloop-export.zip"',
                },
                content=self.export_bytes,
            )
        raise AssertionError(f"unexpected request: {request.method} {path}")


class FailedRunBackend(SuccessBackend):
    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/runs/run_1":
            self.requests.append((request.method, request.url.path))
            return _envelope(
                request,
                {
                    "run_id": "run_1",
                    "status": "FAILED",
                    "error_code": "MODEL_UNAVAILABLE",
                    "summary": None,
                },
            )
        return super().__call__(request)


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[NanoLoopApiClient, httpx.Client]:
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = NanoLoopApiClient(
        "http://backend.test",
        client=http_client,
        request_id_factory=lambda: "req_smoke_test",
    )
    return client, http_client


def test_full_smoke_workflow_completes_and_validates_manifest(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    backend = SuccessBackend(_export_zip())
    client, http_client = _client(backend)
    clock = FakeClock()
    output: list[str] = []
    try:
        report = SmokeRunner(
            client,
            fixture,
            poll_timeout=10,
            poll_interval=1,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
            output=output.append,
        ).run()
    finally:
        http_client.close()

    assert report.mode == "full"
    assert report.job_id == "job_1"
    assert report.run_ids == ("run_1",)
    assert report.export_sha256 == backend.export_sha256
    assert report.manifest is not None
    assert report.manifest["job_id"] == "job_1"
    assert backend.run_polls == 2
    assert clock.value == 1
    assert any("SCIENTIFIC CLOSED LOOP COMPLETE" in line for line in output)
    assert ("POST", "/api/v1/knowledge/documents") in backend.requests


def test_poll_timeout_fails_with_step_and_request_id(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    backend = SuccessBackend(_export_zip(), complete_after=10_000)
    client, http_client = _client(backend)
    clock = FakeClock()
    try:
        with pytest.raises(SmokeTestFailure) as exc_info:
            SmokeRunner(
                client,
                fixture,
                poll_timeout=2,
                poll_interval=1,
                sleep=clock.sleep,
                monotonic=clock.monotonic,
                output=lambda _: None,
            ).run()
    finally:
        http_client.close()

    assert exc_info.value.step == "poll_runs"
    assert exc_info.value.request_id == "req_smoke_test"
    assert "timed out" in str(exc_info.value)
    assert ("POST", "/api/v1/knowledge/documents") not in backend.requests


def test_failed_run_fails_fast_before_knowledge_and_reports_request_id(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    backend = FailedRunBackend(_export_zip())
    client, http_client = _client(backend)
    try:
        with pytest.raises(SmokeTestFailure) as exc_info:
            SmokeRunner(client, fixture, output=lambda _: None).run()
    finally:
        http_client.close()

    assert exc_info.value.step == "poll_runs"
    assert exc_info.value.request_id == "req_smoke_test"
    assert "MODEL_UNAVAILABLE" in str(exc_info.value)
    assert ("POST", "/api/v1/knowledge/documents") not in backend.requests


def test_api_failure_is_wrapped_with_health_step_and_request_id(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return _error(
            request,
            status_code=503,
            code="SERVICE_UNAVAILABLE",
            message="database offline",
        )

    client, http_client = _client(handler)
    try:
        with pytest.raises(SmokeTestFailure) as exc_info:
            SmokeRunner(client, fixture, output=lambda _: None).run()
    finally:
        http_client.close()

    assert exc_info.value.step == "health"
    assert exc_info.value.request_id == "req_smoke_test"
    assert "SERVICE_UNAVAILABLE" in str(exc_info.value)


def test_malformed_success_response_keeps_health_request_id(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return _envelope(
            request,
            {
                "service": {},
                "database": {"status": "healthy"},
                "model_registry": {"status": "healthy"},
                "rag_index": {"status": "healthy"},
            },
        )

    client, http_client = _client(handler)
    try:
        with pytest.raises(SmokeTestFailure) as exc_info:
            SmokeRunner(client, fixture, output=lambda _: None).run()
    finally:
        http_client.close()

    assert exc_info.value.step == "health"
    assert exc_info.value.request_id == "req_smoke_test"
    assert "must be a non-empty string" in str(exc_info.value)


def test_allow_degraded_verifies_truthful_unavailability_and_skips_loop(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, create_files=False)
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path == "/api/v1/health":
            return _envelope(
                request,
                {
                    "service": {"status": "healthy"},
                    "database": {"status": "healthy"},
                    "model_registry": {"status": "degraded"},
                    "rag_index": {"status": "unavailable"},
                    "version": "test",
                },
            )
        if request.url.path == "/api/v1/models":
            return _envelope(
                request,
                {
                    "models": [
                        {"model_id": "declared-but-missing", "status": "unavailable"}
                    ]
                },
            )
        raise AssertionError("degraded mode must not enter the scientific loop")

    client, http_client = _client(handler)
    output: list[str] = []
    try:
        report = SmokeRunner(client, fixture, output=output.append).run(
            allow_degraded=True
        )
    finally:
        http_client.close()

    assert report.mode == "degraded"
    assert requests == ["/api/v1/health", "/api/v1/models"]
    assert any("scientific closed loop was explicitly skipped" in line for line in output)


def test_allow_degraded_accepts_explicitly_unimplemented_model_registry(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, create_files=False)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/health":
            return _envelope(
                request,
                {
                    "service": {"status": "healthy"},
                    "database": {"status": "healthy"},
                    "model_registry": {"status": "unavailable"},
                    "rag_index": {"status": "unavailable"},
                    "version": "test",
                },
            )
        if request.url.path == "/api/v1/models":
            return _error(
                request,
                status_code=501,
                code="NOT_IMPLEMENTED",
                message="model gateway is not configured",
            )
        raise AssertionError("degraded mode must not enter the scientific loop")

    client, http_client = _client(handler)
    output: list[str] = []
    try:
        report = SmokeRunner(client, fixture, output=output.append).run(
            allow_degraded=True
        )
    finally:
        http_client.close()

    assert report.mode == "degraded"
    assert any("unavailable=NOT_IMPLEMENTED" in line for line in output)
    assert any("scientific closed loop was explicitly skipped" in line for line in output)


def test_manifest_validation_rejects_tampered_member() -> None:
    valid = _export_zip()
    with zipfile.ZipFile(io.BytesIO(valid)) as source:
        manifest = source.read("export_manifest.json")
    tampered = io.BytesIO()
    with zipfile.ZipFile(tampered, "w") as archive:
        archive.writestr(
            "images/img_1/runs/run_1/image_summary.json",
            b"tampered",
        )
        archive.writestr("export_manifest.json", manifest)

    with pytest.raises(ValueError, match="size does not match"):
        validate_export_zip(
            tampered.getvalue(),
            expected_job_id="job_1",
            expected_run_ids={"run_1"},
        )


def test_manifest_validation_rejects_unrecorded_member() -> None:
    valid = _export_zip()
    with zipfile.ZipFile(io.BytesIO(valid)) as source:
        original_members = {
            name: source.read(name)
            for name in source.namelist()
        }
    extended = io.BytesIO()
    with zipfile.ZipFile(extended, "w") as archive:
        for name, payload in original_members.items():
            archive.writestr(name, payload)
        archive.writestr("unexpected.txt", b"not in manifest")

    with pytest.raises(ValueError, match="absent from the manifest"):
        validate_export_zip(
            extended.getvalue(),
            expected_job_id="job_1",
            expected_run_ids={"run_1"},
        )


def test_manifest_validation_accepts_content_addressed_manifest_without_wall_clock() -> None:
    valid = _export_zip()
    with zipfile.ZipFile(io.BytesIO(valid)) as source:
        members = {name: source.read(name) for name in source.namelist()}
    manifest = json.loads(members["export_manifest.json"])
    manifest.pop("generated_at")
    manifest["selection_sha256"] = hashlib.sha256(
        json.dumps(
            manifest["files"],
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    members["export_manifest.json"] = json.dumps(manifest).encode("utf-8")
    content_addressed = io.BytesIO()
    with zipfile.ZipFile(content_addressed, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)

    validated = validate_export_zip(
        content_addressed.getvalue(),
        expected_job_id="job_1",
        expected_run_ids={"run_1"},
    )

    assert validated["selection_sha256"] == manifest["selection_sha256"]
    assert "generated_at" not in validated


def test_repository_example_is_valid_but_requires_real_referenced_files() -> None:
    project_root = Path(__file__).parents[3]
    example = project_root / "demo_data" / "smoke_fixture.example.json"

    fixture = load_fixture(example, require_files=False)

    assert fixture.images[0].path.name == "sem_image_01.tif"
    assert not fixture.images[0].path.exists()
    with pytest.raises(ValueError, match="does not reference a file"):
        load_fixture(example, require_files=True)


@pytest.mark.parametrize(
    ("environment_api_key", "cli_api_key", "expected_api_key"),
    [
        ("environment-secret", None, "environment-secret"),
        ("environment-secret", "argument-secret", "argument-secret"),
    ],
)
def test_smoke_cli_passes_environment_or_explicit_api_key_to_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    environment_api_key: str,
    cli_api_key: str | None,
    expected_api_key: str,
) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, base_url: str, *, api_key: str | None = None) -> None:
            captured["base_url"] = base_url
            captured["api_key"] = api_key

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: object) -> None:
            del args

    class FakeRunner:
        def __init__(
            self,
            client: object,
            fixture: object,
            *,
            poll_timeout: float,
            poll_interval: float,
        ) -> None:
            captured["client"] = client
            captured["fixture"] = fixture
            captured["poll_timeout"] = poll_timeout
            captured["poll_interval"] = poll_interval

        def run(self, *, allow_degraded: bool = False) -> None:
            captured["allow_degraded"] = allow_degraded

    monkeypatch.setenv("NANOLOOP_API_KEY", environment_api_key)
    monkeypatch.setattr(smoke_test_module, "NanoLoopApiClient", FakeClient)
    monkeypatch.setattr(smoke_test_module, "SmokeRunner", FakeRunner)
    monkeypatch.setattr(smoke_test_module, "load_fixture", lambda *args, **kwargs: object())
    arguments = [
        "--base-url",
        "http://backend.test",
        "--fixture",
        str(tmp_path / "fixture.json"),
    ]
    if cli_api_key is not None:
        arguments.extend(["--api-key", cli_api_key])

    result = smoke_test_module.main(arguments)

    assert result == 0
    assert captured["base_url"] == "http://backend.test"
    assert captured["api_key"] == expected_api_key

from __future__ import annotations

import json
from pathlib import Path

from tests.integration.api_contract.conftest import ApiHarness


def test_openapi_contains_the_frozen_v1_surface(api_harness: ApiHarness) -> None:
    schema = api_harness.client.get("/openapi.json").json()
    expected_operations = {
        ("/api/v1/health", "get"),
        ("/api/v1/models", "get"),
        ("/api/v1/models/recommend", "post"),
        ("/api/v1/analyses", "post"),
        ("/api/v1/analyses/{job_id}", "get"),
        ("/api/v1/analyses/{job_id}/images/{image_id}/boxes", "get"),
        ("/api/v1/analyses/{job_id}/images/{image_id}/boxes", "put"),
        ("/api/v1/analyses/{job_id}/runs", "post"),
        ("/api/v1/runs/{run_id}", "get"),
        ("/api/v1/runs/{run_id}/corrected-mask", "post"),
        ("/api/v1/runs/{run_id}/review", "post"),
        ("/api/v1/analyses/{job_id}/query", "post"),
        ("/api/v1/knowledge/documents", "post"),
        ("/api/v1/knowledge/documents", "get"),
        ("/api/v1/knowledge/documents/{doc_id}", "patch"),
        ("/api/v1/knowledge/reindex", "post"),
        ("/api/v1/analyses/{job_id}/export", "get"),
        ("/api/v1/files/{token}", "get"),
    }
    observed = {
        (path, method)
        for path, path_item in schema["paths"].items()
        for method in path_item
        if method in {"get", "post", "put", "delete", "patch"}
    }
    assert expected_operations == observed
    assert "/health" not in schema["paths"]

    scheme = schema["components"]["securitySchemes"]["ApiKeyAuth"]
    assert scheme["type"] == "apiKey"
    assert scheme["in"] == "header"
    assert scheme["name"] == "X-API-Key"
    assert "principal" in scheme["description"]
    for path, method in expected_operations:
        assert schema["paths"][path][method]["security"] == [{"ApiKeyAuth": []}]


def test_checked_in_openapi_matches_the_application(api_harness: ApiHarness) -> None:
    live = api_harness.client.get("/openapi.json").json()
    path = Path(__file__).parents[3] / "docs" / "api" / "openapi-v1.json"

    assert json.loads(path.read_text(encoding="utf-8")) == live


def test_declared_errors_reference_the_shared_envelope(api_harness: ApiHarness) -> None:
    schema = api_harness.client.get("/openapi.json").json()
    operation = schema["paths"]["/api/v1/analyses/{job_id}/runs"]["post"]
    for status_code in ("401", "403", "404", "409", "422", "429", "500", "501", "503"):
        response = operation["responses"][status_code]
        media_schema = response["content"]["application/json"]["schema"]
        assert "ApiResponse" in media_schema["$ref"]


def test_file_success_is_binary_not_json_envelope(api_harness: ApiHarness) -> None:
    schema = api_harness.client.get("/openapi.json").json()
    response = schema["paths"]["/api/v1/files/{token}"]["get"]["responses"]["200"]
    binary = response["content"]["application/octet-stream"]["schema"]
    preview = response["content"]["image/png"]["schema"]
    assert binary == {"type": "string", "format": "binary"}
    assert preview == {"type": "string", "format": "binary"}

from __future__ import annotations

import json
import time
from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import text

from app.contracts.models import ModelCandidate, ModelRecommendationRequest
from app.core.config import Settings
from app.core.errors import StorageError
from app.db.models import ImageAsset, KnowledgeDocument
from app.main import create_app
from tests.integration.api_contract.conftest import ApiHarness

_API_KEY = "integration_key_0123456789abcdef"


def _security_client(
    api_harness: ApiHarness,
    *,
    rate_limit: int = 0,
) -> TestClient:
    app = create_app(
        settings=Settings(
            app_env="test",
            nanoloop_api_key=_API_KEY,
            api_rate_limit_requests=rate_limit,
            api_rate_limit_window_seconds=60,
            log_level="WARNING",
        ),
        database=api_harness.database,
        file_store=api_harness.file_store,
        inference_gateway=api_harness.gateway,
    )
    # Deliberately avoid entering the lifespan: this contract fixture reuses the already
    # initialized database and only exercises middleware/read-only routes.
    return TestClient(app, raise_server_exceptions=False)


def _assert_success_envelope(payload: dict[str, object]) -> None:
    assert payload["status"] == "success"
    assert payload["error"] is None
    assert isinstance(payload["request_id"], str)
    assert payload["data"] is not None


def _assert_error_envelope(payload: dict[str, object], code: str) -> None:
    assert payload["status"] == "error"
    assert payload["data"] is None
    assert isinstance(payload["request_id"], str)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["code"] == code


def _wait_for_terminal_run(client: TestClient, run_id: str) -> dict[str, object]:
    """Wait on real work instead of assuming a fast hosted runner finishes in one second."""

    deadline = time.monotonic() + 10.0
    run: dict[str, object] = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/runs/{run_id}")
        assert response.status_code == 200
        run = response.json()["data"]
        if run["status"] in {"COMPLETED", "COMPLETED_WITH_WARNINGS", "FAILED"}:
            return run
        time.sleep(0.02)
    pytest.fail(f"run did not reach a terminal state before the deadline: {run.get('status')}")


def test_health_alias_and_versioned_health_are_real(api_harness: ApiHarness) -> None:
    for path in ("/health", "/api/v1/health"):
        response = api_harness.client.get(path)
        assert response.status_code == 200
        payload = response.json()
        _assert_success_envelope(payload)
        assert response.headers["x-request-id"] == payload["request_id"]
        data = payload["data"]
        assert data["database"]["status"] == "healthy"
        assert data["model_registry"]["status"] == "healthy"
        assert data["rag_index"]["status"] == "degraded"


def test_optional_api_key_protects_versioned_api_and_downloads(
    api_harness: ApiHarness,
) -> None:
    client = _security_client(api_harness)
    try:
        assert client.get("/health").status_code == 200
        for headers in (
            {},
            {"X-API-Key": "wrong"},
        ):
            response = client.get("/api/v1/health", headers=headers)
            assert response.status_code == 401
            _assert_error_envelope(response.json(), "AUTHENTICATION_REQUIRED")
            assert response.headers["www-authenticate"] == 'ApiKey realm="nanoloop"'

        authorized = client.get(
            "/api/v1/health",
            headers={"X-API-Key": _API_KEY},
        )
        assert authorized.status_code == 200

        protected_file = client.get(f"/api/v1/files/{api_harness.download_token}")
        assert protected_file.status_code == 401
        downloaded = client.get(
            f"/api/v1/files/{api_harness.download_token}",
            headers={"X-API-Key": _API_KEY},
        )
        assert downloaded.status_code == 200
        assert downloaded.content.startswith(b"particle_id")
    finally:
        client.close()


def test_security_order_preserves_cors_host_and_origin_guards(
    api_harness: ApiHarness,
) -> None:
    client = _security_client(api_harness)
    allowed_origin = "http://localhost:8501"
    try:
        preflight = client.options(
            "/api/v1/models",
            headers={
                "Origin": allowed_origin,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "X-API-Key",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == allowed_origin

        plain_options = client.options("/api/v1/models")
        assert plain_options.status_code == 401
        _assert_error_envelope(plain_options.json(), "AUTHENTICATION_REQUIRED")

        unauthenticated = client.get(
            "/api/v1/models",
            headers={"Origin": allowed_origin},
        )
        assert unauthenticated.status_code == 401
        assert unauthenticated.headers["access-control-allow-origin"] == allowed_origin

        cross_site = client.post(
            "/api/v1/models/recommend",
            headers={"Origin": "https://attacker.example"},
            json={"image_id": "img_1", "roi_mode": "full_image"},
        )
        assert cross_site.status_code == 403
        _assert_error_envelope(cross_site.json(), "CROSS_SITE_MUTATION_FORBIDDEN")

        hostile_host = client.get(
            "/api/v1/models",
            headers={"Host": "attacker.example"},
        )
        assert hostile_host.status_code == 400
        _assert_error_envelope(hostile_host.json(), "UNTRUSTED_HOST")
    finally:
        client.close()


def test_root_health_is_not_rate_limited_and_auth_buckets_are_separate(
    api_harness: ApiHarness,
) -> None:
    client = _security_client(api_harness, rate_limit=1)
    try:
        assert client.get("/health").status_code == 200
        assert client.get("/health").status_code == 200

        first_anonymous = client.get("/api/v1/models")
        second_anonymous = client.get("/api/v1/models")
        first_authenticated = client.get(
            "/api/v1/models",
            headers={"X-API-Key": _API_KEY},
        )
        second_authenticated = client.get(
            "/api/v1/models",
            headers={"X-API-Key": _API_KEY},
        )

        assert first_anonymous.status_code == 401
        assert second_anonymous.status_code == 429
        assert second_anonymous.headers["retry-after"] == "60"
        assert first_authenticated.status_code == 200
        assert second_authenticated.status_code == 429
        assert second_authenticated.json()["error"]["retryable"] is True
    finally:
        client.close()


def test_health_reports_missing_or_stale_alembic_revision(api_harness: ApiHarness) -> None:
    with api_harness.database.engine.begin() as connection:
        connection.execute(text("DELETE FROM alembic_version"))

    missing_revision = api_harness.client.get("/api/v1/health").json()["data"]["database"]
    assert missing_revision == {
        "status": "degraded",
        "detail": "reachable; alembic revision is missing",
    }

    with api_harness.database.engine.begin() as connection:
        connection.execute(
            text("INSERT INTO alembic_version (version_num) VALUES (:revision)"),
            {"revision": "53eaa43adc19"},
        )

    stale_revision = api_harness.client.get("/api/v1/health").json()["data"]["database"]
    assert stale_revision["status"] == "degraded"
    assert "database revision 53eaa43adc19 does not match head" in stale_revision["detail"]

    with api_harness.database.engine.begin() as connection:
        connection.execute(text("DROP TABLE alembic_version"))

    missing_table = api_harness.client.get("/api/v1/health").json()["data"]["database"]
    assert missing_table == {
        "status": "degraded",
        "detail": "reachable; alembic_version table is missing",
    }


def test_request_id_is_safe_and_correlated(api_harness: ApiHarness) -> None:
    response = api_harness.client.get(
        "/api/v1/analyses/job_1",
        headers={"X-Request-ID": "client_request-123"},
    )
    payload = response.json()
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "client_request-123"
    assert payload["request_id"] == "client_request-123"

    rejected = api_harness.client.get(
        "/api/v1/analyses/job_1",
        headers={"X-Request-ID": "unsafe request id\n"},
    )
    assert rejected.headers["x-request-id"].startswith("req_")


def test_persisted_job_boxes_and_run_read_models(api_harness: ApiHarness) -> None:
    job_response = api_harness.client.get("/api/v1/analyses/job_1")
    assert job_response.status_code == 200
    job_data = job_response.json()["data"]
    assert job_data["job"]["job_id"] == "job_1"
    assert job_data["images"][0]["analysis_roi"]["coordinate_space"] == "original_px"
    assert job_data["runs"][0]["configuration"]["model_version"] == "1.0.0"

    run_response = api_harness.client.get("/api/v1/runs/run_1")
    assert run_response.status_code == 200
    run_data = run_response.json()["data"]
    assert run_data["run_id"] == "run_1"
    assert run_data["configuration"]["analysis_roi"]["revision"] == 1

    initial_boxes = api_harness.client.get("/api/v1/analyses/job_1/images/img_1/boxes").json()[
        "data"
    ]
    assert initial_boxes == {"image_id": "img_1", "revision": 0, "boxes": []}

    replaced = api_harness.client.put(
        "/api/v1/analyses/job_1/images/img_1/boxes",
        json={
            "expected_revision": 0,
            "boxes": [{"label": "ROI", "x1": 20, "y1": 20, "x2": 100, "y2": 100}],
        },
    )
    assert replaced.status_code == 200
    assert replaced.json()["data"]["revision"] == 1
    assert api_harness.file_store.paths.boxes_revision("job_1", "img_1", 1).is_file()

    conflict = api_harness.client.put(
        "/api/v1/analyses/job_1/images/img_1/boxes",
        json={"expected_revision": 0, "boxes": []},
    )
    assert conflict.status_code == 409
    _assert_error_envelope(conflict.json(), "BOX_REVISION_CONFLICT")


def test_committed_box_revision_survives_projection_failure(
    api_harness: ApiHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_projection(*_args: object, **_kwargs: object) -> None:
        raise StorageError("simulated projection outage")

    monkeypatch.setattr(api_harness.file_store, "atomic_write_json", fail_projection)
    with patch("app.analysis.boxes.logger.exception") as log_exception:
        response = api_harness.client.put(
            "/api/v1/analyses/job_1/images/img_1/boxes",
            json={
                "expected_revision": 0,
                "boxes": [{"label": "ROI", "x1": 20, "y1": 20, "x2": 100, "y2": 100}],
            },
        )

    assert response.status_code == 200
    assert response.json()["data"]["revision"] == 1
    current = api_harness.client.get("/api/v1/analyses/job_1/images/img_1/boxes").json()["data"]
    assert current["revision"] == 1
    assert len(current["boxes"]) == 1
    with api_harness.database.session() as session:
        image = session.get(ImageAsset, "img_1")
        assert image is not None
        assert image.box_revision == 1
    assert not api_harness.file_store.paths.boxes_revision("job_1", "img_1", 1).exists()
    log_exception.assert_called_once()
    assert log_exception.call_args.args == ("boxes_revision_projection_failed",)
    assert log_exception.call_args.kwargs["extra"] == {
        "component": "boxes_revision_projection",
        "detail": "revision=1",
        "event": "projection_write_failed",
        "outcome": "degraded",
    }


def test_models_and_recommendation_use_the_gateway(api_harness: ApiHarness) -> None:
    listed = api_harness.client.get(
        "/api/v1/models",
        params={"status": "ready", "material": "TiO2"},
    )
    assert listed.status_code == 200
    assert listed.json()["data"]["models"][0]["model_id"] == "unet-general-balanced-v1"

    recommended = api_harness.client.post(
        "/api/v1/models/recommend",
        json={
            "image_id": "img_1",
            "roi_mode": "full_image",
            "target_profile": "general",
            "prefer": "accuracy",
        },
    )
    assert recommended.status_code == 200
    assert recommended.json()["data"]["candidates"][0]["score"] == 0.91


def test_validation_not_found_and_run_submission_share_contract_shape(
    api_harness: ApiHarness,
) -> None:
    invalid = api_harness.client.post(
        "/api/v1/models/recommend",
        json={"image_id": "", "roi_mode": "unknown"},
    )
    assert invalid.status_code == 422
    invalid_payload = invalid.json()
    _assert_error_envelope(invalid_payload, "VALIDATION_ERROR")
    assert invalid_payload["error"]["details"]["issues"]

    missing = api_harness.client.get("/api/v1/runs/missing")
    assert missing.status_code == 404
    _assert_error_envelope(missing.json(), "RESOURCE_NOT_FOUND")

    unknown_route = api_harness.client.get("/api/v1/not-a-route")
    assert unknown_route.status_code == 404
    _assert_error_envelope(unknown_route.json(), "RESOURCE_NOT_FOUND")

    pending = api_harness.client.post(
        "/api/v1/analyses/job_1/runs",
        json={
            "image_ids": ["img_1"],
            "model_ids": ["unet-general-balanced-v1"],
            "roi_mode": "full_image",
        },
    )
    assert pending.status_code == 202
    pending_payload = pending.json()
    assert pending_payload["status"] == "accepted"
    assert len(pending_payload["data"]["run_ids"]) == 1


def test_analysis_upload_is_persisted_and_immediately_downloadable(
    api_harness: ApiHarness,
) -> None:
    image_buffer = BytesIO()
    Image.new("L", (23, 19), color=75).save(image_buffer, format="PNG")
    metadata = {
        "job_name": "HTTP upload",
        "images": [
            {
                "filename": "fresh.png",
                "sample_id": "sample_http",
                "material_formula": "SiO2",
                "scale": {"mode": "pixel_only"},
            }
        ],
    }

    created = api_harness.client.post(
        "/api/v1/analyses",
        files={"files": ("fresh.png", image_buffer.getvalue(), "image/png")},
        data={"metadata_json": json.dumps(metadata)},
    )

    assert created.status_code == 201
    payload = created.json()
    _assert_success_envelope(payload)
    assert payload["data"]["job"]["status"] == "READY_FOR_CONFIGURATION"
    image = payload["data"]["images"][0]
    assert (image["width"], image["height"], image["bit_depth"]) == (23, 19, 8)
    download = api_harness.client.get(image["original_download_url"])
    assert download.status_code == 200
    assert download.content == image_buffer.getvalue()

    submitted = api_harness.client.post(
        f"/api/v1/analyses/{payload['data']['job']['job_id']}/runs",
        json={
            "image_ids": [image["image_id"]],
            "model_ids": ["unet-general-balanced-v1"],
            "roi_mode": "full_image",
        },
    )
    assert submitted.status_code == 202
    run_id = submitted.json()["data"]["run_ids"][0]
    run = _wait_for_terminal_run(api_harness.client, run_id)
    assert run["status"] in {"COMPLETED", "COMPLETED_WITH_WARNINGS"}
    artifacts = run["artifacts"]
    assert isinstance(artifacts, dict)
    assert api_harness.client.get(artifacts["overlay_url"]).status_code == 200
    execution_response = api_harness.client.get(
        artifacts["execution_provenance_url"]
    )
    assert execution_response.status_code == 200
    execution_payload = execution_response.json()
    assert execution_payload["seed"] == 42
    assert execution_payload["executor_build"]["application_source_sha256"]

    data_query = api_harness.client.post(
        f"/api/v1/analyses/{payload['data']['job']['job_id']}/query",
        json={
            "question": "颗粒数是多少？",
            "query_type": "analysis_data",
            "run_ids": [run_id],
        },
    )
    assert data_query.status_code == 200
    query_data = data_query.json()["data"]
    assert query_data["outcome_code"] == "OK"
    assert query_data["data_evidence"][0]["source_run_ids"] == [run_id]

    exported = api_harness.client.get(
        f"/api/v1/analyses/{payload['data']['job']['job_id']}/export",
        params={"run_ids": run_id},
    )
    assert exported.status_code == 200
    export_data = exported.json()["data"]
    archive = api_harness.client.get(export_data["download_url"])
    assert archive.status_code == 200
    assert archive.content.startswith(b"PK")

    corrected_buffer = BytesIO()
    corrected = Image.new("L", (23, 19), color=0)
    for x in range(7, 13):
        for y in range(6, 12):
            corrected.putpixel((x, y), 255)
    corrected.save(corrected_buffer, format="PNG")
    staged_mask = api_harness.client.post(
        f"/api/v1/runs/{run_id}/corrected-mask",
        files={"file": ("corrected.png", corrected_buffer.getvalue(), "image/png")},
    )
    assert staged_mask.status_code == 201

    reviewed = api_harness.client.post(
        f"/api/v1/runs/{run_id}/review",
        json={
            "corrected_mask_token": staged_mask.json()["data"]["corrected_mask_token"],
            "threshold": 0.8,
            "min_area_px": 10,
        },
    )
    assert reviewed.status_code == 202
    review_data = reviewed.json()["data"]
    assert review_data["parent_run_id"] == run_id
    child = api_harness.client.get(f"/api/v1/runs/{review_data['run_id']}")
    assert child.status_code == 200
    assert child.json()["data"]["parent_run_id"] == run_id
    assert child.json()["data"]["threshold"] == 0.8
    assert child.json()["data"]["configuration"]["review_source"] == "corrected_mask"


def test_signed_file_download_and_forged_token(api_harness: ApiHarness) -> None:
    response = api_harness.client.get(f"/api/v1/files/{api_harness.download_token}")
    assert response.status_code == 200
    assert response.content.startswith(b"particle_id,area_px")
    assert response.headers["cache-control"] == "private, no-store"
    assert "particles.csv" in response.headers["content-disposition"]

    forged = api_harness.client.get(f"/api/v1/files/{api_harness.download_token}x")
    assert forged.status_code == 404
    _assert_error_envelope(forged.json(), "RESOURCE_NOT_FOUND")


def test_knowledge_ingestion_listing_idempotency_and_reindex(
    api_harness: ApiHarness,
) -> None:
    metadata = {
        "title": "TiO2 evidence note",
        "source_type": "material_note",
        "year": 2026,
        "citation_text": "Internal TiO2 evidence note, 2026.",
        "material_aliases": ["TiO2", "titanium dioxide"],
        "license_note": "Team-authored and allowed for this test.",
        "allowed_for_demo": True,
    }
    document = b"# Properties\n\nTiO2 catalyst evidence supports photocatalytic applications."

    ingested = api_harness.client.post(
        "/api/v1/knowledge/documents",
        files={"file": ("tio2.md", document, "text/markdown")},
        data={"metadata_json": json.dumps(metadata)},
    )
    assert ingested.status_code == 202
    ingest_payload = ingested.json()
    assert ingest_payload["status"] == "accepted"
    assert ingest_payload["data"]["chunks_created"] == 1

    listed = api_harness.client.get("/api/v1/knowledge/documents")
    assert listed.status_code == 200
    listed_document = listed.json()["data"]["documents"][0]
    assert listed_document["doc_id"] == ingest_payload["data"]["doc_id"]
    assert listed_document["allowed_for_demo"] is True

    queried = api_harness.client.post(
        "/api/v1/analyses/job_1/query",
        json={
            "question": "What catalyst applications are supported by the evidence?",
            "query_type": "material_knowledge",
        },
    )
    assert queried.status_code == 200
    query_payload = queried.json()["data"]
    assert query_payload["outcome_code"] == "OK"
    assert query_payload["material_context"] == {
        "formula": "TiO2",
        "name": "titanium dioxide",
        "aliases": [],
        "source": "image_metadata",
    }
    assert query_payload["citations"][0]["doc_id"] == ingest_payload["data"]["doc_id"]
    assert api_harness.file_store.paths.query_history("job_1").is_file()
    assert api_harness.file_store.paths.rag_citations("job_1").is_file()

    mismatched = api_harness.client.post(
        "/api/v1/analyses/job_1/query",
        json={
            "question": "What catalyst applications are supported by the evidence?",
            "query_type": "material_knowledge",
            "material_context": {"formula": "SrNi", "source": "request"},
        },
    )
    assert mismatched.status_code == 200
    mismatch_payload = mismatched.json()["data"]
    assert mismatch_payload["outcome_code"] == "INSUFFICIENT_EVIDENCE"
    assert mismatch_payload["citations"] == []

    with api_harness.database.session() as session:
        session.add(
            ImageAsset(
                image_id="img_2",
                job_id="job_1",
                filename="ycu.tif",
                storage_path="job_1/input/img_2/original.tif",
                sha256="b" * 64,
                width=128,
                height=128,
                bit_depth=16,
                sample_id="sample_2",
                material_name="yttrium copper",
                material_formula="YCu",
                experiment_conditions_json={},
                analysis_roi_json={},
                scale_nm_per_pixel=0.5,
                box_revision=0,
            )
        )

    ambiguous = api_harness.client.post(
        "/api/v1/analyses/job_1/query",
        json={
            "question": "What properties does this material have?",
            "query_type": "material_knowledge",
        },
    )
    assert ambiguous.status_code == 200
    ambiguous_payload = ambiguous.json()["data"]
    assert ambiguous_payload["needs_clarification"] is True
    assert ambiguous_payload["outcome_code"] == "INSUFFICIENT_EVIDENCE"
    assert "TiO2" in ambiguous_payload["answer"]
    assert "YCu" in ambiguous_payload["answer"]
    assert ambiguous_payload["citations"] == []

    contextual_auto = api_harness.client.post(
        "/api/v1/analyses/job_1/query",
        json={"question": "这个材料怎么样？"},
    )
    assert contextual_auto.status_code == 200
    contextual_payload = contextual_auto.json()["data"]
    assert contextual_payload["query_type"] == "auto"
    assert contextual_payload["needs_clarification"] is True
    assert "TiO2" in contextual_payload["answer"]
    assert "YCu" in contextual_payload["answer"]

    conflict = api_harness.client.post(
        "/api/v1/analyses/job_1/query",
        json={
            "question": "What properties does this material have?",
            "query_type": "material_knowledge",
            "image_id": "img_1",
            "material_context": {"formula": "YCu", "source": "request"},
        },
    )
    assert conflict.status_code == 200
    conflict_payload = conflict.json()["data"]
    assert conflict_payload["needs_clarification"] is True
    assert "source=user_confirmation" in conflict_payload["answer"]
    assert conflict_payload["citations"] == []

    confirmed = api_harness.client.post(
        "/api/v1/analyses/job_1/query",
        json={
            "question": "What properties does this material have?",
            "query_type": "material_knowledge",
            "image_id": "img_1",
            "material_context": {
                "formula": "YCu",
                "source": "user_confirmation",
            },
        },
    )
    assert confirmed.status_code == 200
    confirmed_payload = confirmed.json()["data"]
    assert confirmed_payload["needs_clarification"] is False
    assert confirmed_payload["outcome_code"] == "INSUFFICIENT_EVIDENCE"
    assert confirmed_payload["citations"] == []

    duplicate = api_harness.client.post(
        "/api/v1/knowledge/documents",
        files={"file": ("tio2.md", document, "text/markdown")},
        data={"metadata_json": json.dumps(metadata)},
    )
    assert duplicate.status_code == 202
    assert duplicate.json()["data"]["chunks_created"] == 0

    conflicting_metadata = {**metadata, "title": "Conflicting title"}
    conflict = api_harness.client.post(
        "/api/v1/knowledge/documents",
        files={"file": ("tio2.md", document, "text/markdown")},
        data={"metadata_json": json.dumps(conflicting_metadata)},
    )
    assert conflict.status_code == 409
    _assert_error_envelope(conflict.json(), "KNOWLEDGE_DOCUMENT_CONFLICT")

    reindexed = api_harness.client.post(
        "/api/v1/knowledge/reindex",
        json={"force": False},
    )
    assert reindexed.status_code == 202
    assert reindexed.json()["data"]["documents_indexed"] == 1


def test_knowledge_document_toggle_controls_retrieval_and_validates_requests(
    api_harness: ApiHarness,
) -> None:
    metadata = {
        "title": "Toggleable TiO2 note",
        "source_type": "material_note",
        "year": 2026,
        "citation_text": "Toggleable TiO2 fixture, 2026.",
        "material_aliases": ["TiO2", "titanium dioxide"],
        "license_note": "Team-authored test fixture.",
        "allowed_for_demo": True,
    }
    ingested = api_harness.client.post(
        "/api/v1/knowledge/documents",
        files={
            "file": (
                "toggle.md",
                b"TiO2 catalyst evidence supports a toggle retrieval test.",
                "text/markdown",
            )
        },
        data={"metadata_json": json.dumps(metadata)},
    )
    assert ingested.status_code == 202
    doc_id = ingested.json()["data"]["doc_id"]

    disabled = api_harness.client.patch(
        f"/api/v1/knowledge/documents/{doc_id}",
        json={"enabled": False},
    )
    assert disabled.status_code == 200
    assert disabled.json()["data"]["status"] == "disabled"
    disabled_again = api_harness.client.patch(
        f"/api/v1/knowledge/documents/{doc_id}",
        json={"enabled": False},
    )
    assert disabled_again.status_code == 200
    assert disabled_again.json()["data"]["status"] == "disabled"
    listed = api_harness.client.get("/api/v1/knowledge/documents")
    assert listed.status_code == 200
    assert listed.json()["data"]["documents"][0]["status"] == "disabled"

    hidden = api_harness.client.post(
        "/api/v1/analyses/job_1/query",
        json={
            "question": "What TiO2 catalyst evidence is available?",
            "query_type": "material_knowledge",
            "image_id": "img_1",
        },
    )
    assert hidden.status_code == 200
    assert hidden.json()["data"]["outcome_code"] == "INSUFFICIENT_EVIDENCE"
    assert hidden.json()["data"]["citations"] == []

    enabled = api_harness.client.patch(
        f"/api/v1/knowledge/documents/{doc_id}",
        json={"enabled": True},
    )
    assert enabled.status_code == 200
    assert enabled.json()["data"]["status"] == "ready"
    restored = api_harness.client.post(
        "/api/v1/analyses/job_1/query",
        json={
            "question": "What TiO2 catalyst evidence is available?",
            "query_type": "material_knowledge",
            "image_id": "img_1",
        },
    )
    assert restored.status_code == 200
    assert restored.json()["data"]["outcome_code"] == "OK"
    assert restored.json()["data"]["citations"][0]["doc_id"] == doc_id

    invalid = api_harness.client.patch(
        f"/api/v1/knowledge/documents/{doc_id}",
        json={"enabled": "false"},
    )
    assert invalid.status_code == 422
    _assert_error_envelope(invalid.json(), "VALIDATION_ERROR")

    missing = api_harness.client.patch(
        "/api/v1/knowledge/documents/doc_missing",
        json={"enabled": False},
    )
    assert missing.status_code == 404
    _assert_error_envelope(missing.json(), "RESOURCE_NOT_FOUND")

    with api_harness.database.session() as session:
        document = session.get(KnowledgeDocument, doc_id)
        assert document is not None
        document.status = "unavailable"
    invalid_transition = api_harness.client.patch(
        f"/api/v1/knowledge/documents/{doc_id}",
        json={"enabled": False},
    )
    assert invalid_transition.status_code == 409
    _assert_error_envelope(
        invalid_transition.json(),
        "KNOWLEDGE_DOCUMENT_STATE_CONFLICT",
    )


def test_cors_is_allowlisted(api_harness: ApiHarness) -> None:
    allowed = api_harness.client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "X-Request-ID",
        },
    )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:8501"

    patch_allowed = api_harness.client.options(
        "/api/v1/knowledge/documents/doc_1",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "PATCH",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert patch_allowed.status_code == 200
    assert patch_allowed.headers["access-control-allow-origin"] == "http://localhost:8501"

    denied = api_harness.client.options(
        "/api/v1/health",
        headers={
            "Origin": "https://attacker.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" not in denied.headers


def test_host_and_browser_origin_guard_mutations_before_handler(
    api_harness: ApiHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = api_harness.gateway.recommend

    def tracked_recommend(request: ModelRecommendationRequest) -> list[ModelCandidate]:
        nonlocal calls
        calls += 1
        return original(request)

    monkeypatch.setattr(api_harness.gateway, "recommend", tracked_recommend)
    payload = {
        "image_id": "img_1",
        "roi_mode": "full_image",
        "target_profile": "general",
        "prefer": "accuracy",
    }

    hostile_host = api_harness.client.post(
        "/api/v1/models/recommend",
        headers={"Host": "attacker.example"},
        json=payload,
    )
    assert hostile_host.status_code == 400
    _assert_error_envelope(hostile_host.json(), "UNTRUSTED_HOST")
    assert calls == 0

    malformed_trusted_host = api_harness.client.post(
        "/api/v1/models/recommend",
        headers={"Host": "testserver:80:evil"},
        json=payload,
    )
    assert malformed_trusted_host.status_code == 400
    _assert_error_envelope(malformed_trusted_host.json(), "UNTRUSTED_HOST")
    assert calls == 0

    hostile_origin = api_harness.client.post(
        "/api/v1/models/recommend",
        headers={
            "Origin": "https://attacker.example",
            "Sec-Fetch-Site": "cross-site",
        },
        json=payload,
    )
    assert hostile_origin.status_code == 403
    _assert_error_envelope(hostile_origin.json(), "CROSS_SITE_MUTATION_FORBIDDEN")
    assert calls == 0

    metadata_only_cross_site = api_harness.client.post(
        "/api/v1/models/recommend",
        headers={"Sec-Fetch-Site": "cross-site"},
        json=payload,
    )
    assert metadata_only_cross_site.status_code == 403
    _assert_error_envelope(
        metadata_only_cross_site.json(),
        "CROSS_SITE_MUTATION_FORBIDDEN",
    )
    assert calls == 0

    non_browser = api_harness.client.post("/api/v1/models/recommend", json=payload)
    assert non_browser.status_code == 200
    assert calls == 1

    allowlisted_browser = api_harness.client.post(
        "/api/v1/models/recommend",
        headers={
            "Origin": "http://localhost:8501",
            "Sec-Fetch-Site": "same-site",
        },
        json=payload,
    )
    assert allowlisted_browser.status_code == 200
    assert allowlisted_browser.headers["access-control-allow-origin"] == ("http://localhost:8501")
    assert calls == 2

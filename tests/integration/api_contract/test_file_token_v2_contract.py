from __future__ import annotations

import base64
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import pytest
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import func, select

from app.contracts.analyses import AnalysisJobDTO
from app.contracts.enums import (
    JobStatus,
    ModelFamily,
    ModelStatus,
    ModelVariant,
    QualityTier,
)
from app.contracts.identity import (
    LEGACY_TENANT_ID,
    PrincipalKind,
    PrincipalRole,
)
from app.core.config import Settings
from app.core.identity import IssuedCredential, issue_credential
from app.db.base import Base
from app.db.identity import IdentityService
from app.db.models import ModelRegistryRecord, SegmentationRun
from app.db.repositories import SqlAlchemyRepositorySet
from app.db.session import Database
from app.main import create_app
from app.storage import FileTokenV2KeyRing, LocalFileStore, StoragePaths
from tests.integration.api_contract.conftest import ApiHarness, FakeInferenceGateway

_NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
_PEPPER = "file-token-v2-http-contract-pepper-32-bytes"
_SHARED_KEY = "file_token_contract_shared_key_123456"
_TENANT_A = f"tnt_{'a' * 32}"
_TENANT_B = f"tnt_{'b' * 32}"
_OWNER_A = f"prn_{'1' * 32}"
_VIEWER_A = f"prn_{'2' * 32}"
_OWNER_B = f"prn_{'3' * 32}"

_V2_PAYLOAD_KEYS = {
    "v",
    "tid",
    "sub",
    "jid",
    "aid",
    "pur",
    "aud",
    "sha256",
    "iat",
    "nbf",
    "exp",
    "jti",
}


@dataclass(slots=True)
class PrincipalFileHarness:
    client: TestClient
    database: Database
    file_store: LocalFileStore
    credentials: dict[str, IssuedCredential]


@pytest.fixture
def principal_file_harness(tmp_path: Path) -> Iterator[PrincipalFileHarness]:
    settings = Settings(
        app_env="test",
        auth_mode="principal",
        credential_pepper=_PEPPER,
        database_url=f"sqlite:///{tmp_path / 'principal-file-contract.db'}",
        output_root=tmp_path / "outputs",
        file_token_v2_keyring_path=tmp_path / "file-token-v2-keyring.json",
        model_registry_path=tmp_path / "registry.yaml",
        model_snapshot_root=tmp_path / "model-snapshots",
        knowledge_source_dir=tmp_path / "knowledge-sources",
        faiss_index_path=tmp_path / "faiss.index",
        log_level="WARNING",
        api_rate_limit_requests=0,
        api_principal_preauth_rate_limit_requests=1000,
        analysis_worker_count=1,
    )
    database = Database(settings)
    Base.metadata.create_all(database.engine)
    file_store = LocalFileStore(
        StoragePaths(settings.output_root),
        max_upload_bytes=1024 * 1024,
        token_secret=b"legacy-file-token-contract-secret" * 2,
    )
    credentials = _seed_principal_identities(database)
    with database.session() as session:
        session.add(
            ModelRegistryRecord(
                model_id="unet-general-balanced-v1",
                family=ModelFamily.UNET.value,
                variant=ModelVariant.GENERAL.value,
                quality_tier=QualityTier.BALANCED.value,
                version="1.0.0",
                adapter="tests.fake:FakeAdapter",
                status=ModelStatus.READY.value,
            )
        )

    app = create_app(
        settings=settings,
        database=database,
        file_store=file_store,
        inference_gateway=FakeInferenceGateway(),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield PrincipalFileHarness(
            client=client,
            database=database,
            file_store=file_store,
            credentials=credentials,
        )
    database.dispose()


def _seed_principal_identities(database: Database) -> dict[str, IssuedCredential]:
    specifications = (
        ("owner", _OWNER_A, _TENANT_A, PrincipalRole.ANALYST),
        ("viewer", _VIEWER_A, _TENANT_A, PrincipalRole.VIEWER),
        ("foreign", _OWNER_B, _TENANT_B, PrincipalRole.ANALYST),
    )
    credentials: dict[str, IssuedCredential] = {}
    with database.session() as session:
        identities = IdentityService.from_session(session)
        identities.create_tenant(
            tenant_id=_TENANT_A,
            slug="file-contract-a",
            display_name="File contract tenant A",
            now=_NOW,
        )
        identities.create_tenant(
            tenant_id=_TENANT_B,
            slug="file-contract-b",
            display_name="File contract tenant B",
            now=_NOW,
        )
        for name, principal_id, tenant_id, role in specifications:
            identities.create_principal(
                principal_id=principal_id,
                tenant_id=tenant_id,
                handle=f"file-{name}",
                display_name=f"File contract {name}",
                kind=PrincipalKind.USER,
                role=role,
                now=_NOW,
            )
            issued = issue_credential(_PEPPER)
            identities.issue_credential(
                credential_id=issued.credential_id,
                principal_id=principal_id,
                token_digest=issued.digest,
                label=f"file contract {name}",
                now=_NOW,
            )
            credentials[name] = issued
    return credentials


def _headers(harness: PrincipalFileHarness, actor: str) -> dict[str, str]:
    return {"X-API-Key": harness.credentials[actor].token.get_secret_value()}


def _create_analysis(
    harness: PrincipalFileHarness,
    *,
    actor: str = "owner",
    filename: str = "principal-original.png",
    image_format: str = "PNG",
    image_mode: str = "L",
    color: int = 75,
) -> tuple[bytes, dict[str, object]]:
    image_buffer = BytesIO()
    Image.new(image_mode, (32, 24), color=color).save(image_buffer, format=image_format)
    image_bytes = image_buffer.getvalue()
    media_type = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "TIFF": "image/tiff",
    }[image_format]
    response = harness.client.post(
        "/api/v1/analyses",
        headers=_headers(harness, actor),
        files={"files": (filename, image_bytes, media_type)},
        data={
            "metadata_json": json.dumps(
                {
                    "job_name": "Principal file-token contract",
                    "images": [
                        {
                            "filename": filename,
                            "sample_id": "principal_file_contract",
                            "scale": {"mode": "pixel_only"},
                        }
                    ],
                }
            )
        },
    )
    assert response.status_code == 201, response.text
    data = response.json()["data"]
    assert isinstance(data, dict)
    return image_bytes, data


def _token_from_url(url: object) -> str:
    assert isinstance(url, str)
    token = url.rsplit("/", maxsplit=1)[-1]
    assert token.startswith("v2.")
    return token


def _token_payload(token: str) -> dict[str, object]:
    prefix, _kid, encoded, _signature = token.split(".")
    assert prefix == "v2"
    decoded = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    payload = json.loads(decoded)
    assert isinstance(payload, dict)
    return payload


def _wait_for_terminal_run(
    harness: PrincipalFileHarness,
    run_id: str,
    *,
    actor: str = "owner",
) -> dict[str, object]:
    deadline = time.monotonic() + 10.0
    last: dict[str, object] = {}
    while time.monotonic() < deadline:
        response = harness.client.get(
            f"/api/v1/runs/{run_id}",
            headers=_headers(harness, actor),
        )
        assert response.status_code == 200, response.text
        last = response.json()["data"]
        if last["status"] in {"COMPLETED", "COMPLETED_WITH_WARNINGS", "FAILED"}:
            return last
        time.sleep(0.02)
    pytest.fail(f"run did not become terminal: {last.get('status')}")


def _create_completed_run(
    harness: PrincipalFileHarness,
    *,
    job_id: str,
    image_id: str,
) -> str:
    submitted = harness.client.post(
        f"/api/v1/analyses/{job_id}/runs",
        headers=_headers(harness, "owner"),
        json={
            "image_ids": [image_id],
            "model_ids": ["unet-general-balanced-v1"],
            "roi_mode": "full_image",
        },
    )
    assert submitted.status_code == 202, submitted.text
    run_id = submitted.json()["data"]["run_ids"][0]
    assert isinstance(run_id, str)
    terminal = _wait_for_terminal_run(harness, run_id)
    assert terminal["status"] in {"COMPLETED", "COMPLETED_WITH_WARNINGS"}, terminal
    return run_id


def test_principal_original_url_is_v2_subject_bound_and_secret_free(
    principal_file_harness: PrincipalFileHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = principal_file_harness
    original_bytes, created = _create_analysis(harness)
    job = created["job"]
    images = created["images"]
    assert isinstance(job, dict)
    assert isinstance(images, list)
    image = images[0]
    assert isinstance(image, dict)
    job_id = job["job_id"]
    image_id = image["image_id"]
    assert isinstance(job_id, str)
    assert isinstance(image_id, str)

    created_token = _token_from_url(image["original_download_url"])
    payload = _token_payload(created_token)
    assert set(payload) == _V2_PAYLOAD_KEYS
    assert payload["v"] == 2
    assert payload["tid"] == _TENANT_A
    assert payload["sub"] == _OWNER_A
    assert payload["jid"] == job_id
    assert payload["pur"] == "download.original_image"
    assert payload["aud"] == "nanoloop-api:file-download"
    serialized_payload = json.dumps(payload, sort_keys=True)
    assert "path" not in serialized_payload.casefold()
    assert "credential" not in serialized_payload.casefold()
    assert harness.credentials["owner"].credential_id not in serialized_payload

    read = harness.client.get(
        f"/api/v1/analyses/{job_id}",
        headers=_headers(harness, "owner"),
    )
    assert read.status_code == 200
    read_token = _token_from_url(read.json()["data"]["images"][0]["original_download_url"])
    assert _token_payload(read_token)["sub"] == _OWNER_A

    viewer_read = harness.client.get(
        f"/api/v1/analyses/{job_id}",
        headers=_headers(harness, "viewer"),
    )
    assert viewer_read.status_code == 200
    viewer_token = _token_from_url(viewer_read.json()["data"]["images"][0]["original_download_url"])
    assert _token_payload(viewer_token)["sub"] == _VIEWER_A

    same_tenant_leak = harness.client.get(
        f"/api/v1/files/{created_token}",
        headers=_headers(harness, "viewer"),
    )
    cross_tenant_leak = harness.client.get(
        f"/api/v1/files/{created_token}",
        headers=_headers(harness, "foreign"),
    )
    for rejected in (same_tenant_leak, cross_tenant_leak):
        assert rejected.status_code == 404
        assert rejected.json()["error"]["code"] == "RESOURCE_NOT_FOUND"
        assert created_token not in rejected.text
        assert "principal-original.png" not in rejected.text

    downloaded = harness.client.get(
        f"/api/v1/files/{created_token}",
        headers=_headers(harness, "owner"),
    )
    assert downloaded.status_code == 200
    assert downloaded.content == original_bytes
    assert downloaded.headers["cache-control"] == "private, no-store"
    assert downloaded.headers["content-length"] == str(len(original_bytes))
    assert downloaded.headers["x-content-type-options"] == "nosniff"
    assert downloaded.headers["referrer-policy"] == "no-referrer"
    assert "attachment" in downloaded.headers["content-disposition"]
    assert "principal-original.png" in downloaded.headers["content-disposition"]

    native_preview = harness.client.get(
        f"/api/v1/files/{created_token}?preview=1",
        headers=_headers(harness, "owner"),
    )
    assert native_preview.status_code == 200
    assert native_preview.headers["content-type"] == "image/png"
    assert native_preview.content == original_bytes

    with harness.database.session() as session:
        storage_path = SqlAlchemyRepositorySet(session).images.get_storage_path_scoped(
            job_id,
            image_id,
            tenant_id=_TENANT_A,
        )
    legacy_token = harness.file_store.create_file_token(
        harness.file_store.paths.root / storage_path,
        ttl_seconds=3600,
    )
    calls = {"decode": 0, "open": 0}

    def fail_decode(_store: LocalFileStore, _token: str) -> str:
        calls["decode"] += 1
        raise AssertionError("principal v1 must be rejected before legacy decoding")

    def fail_open(*_args: object, **_kwargs: object) -> object:
        calls["open"] += 1
        raise AssertionError("principal v1 must be rejected before filesystem access")

    monkeypatch.setattr(LocalFileStore, "decode_file_token_path", fail_decode)
    monkeypatch.setattr("app.files.application.open_pinned_managed_file", fail_open)
    old_v1 = harness.client.get(
        f"/api/v1/files/{legacy_token}",
        headers=_headers(harness, "owner"),
    )
    assert old_v1.status_code == 404
    assert old_v1.json()["error"]["code"] == "RESOURCE_NOT_FOUND"
    assert legacy_token not in old_v1.text
    assert storage_path not in old_v1.text
    assert calls == {"decode": 0, "open": 0}


def test_tiff_preview_is_png_and_raw_download_remains_unchanged(
    principal_file_harness: PrincipalFileHarness,
) -> None:
    harness = principal_file_harness
    original_bytes, created = _create_analysis(
        harness,
        filename="principal-original.tif",
        image_format="TIFF",
        image_mode="I;16",
        color=4096,
    )
    images = created["images"]
    assert isinstance(images, list)
    image = images[0]
    assert isinstance(image, dict)
    token = _token_from_url(image["original_download_url"])

    preview = harness.client.get(
        f"/api/v1/files/{token}?preview=1",
        headers=_headers(harness, "owner"),
    )
    assert preview.status_code == 200, preview.text
    assert preview.headers["content-type"] == "image/png"
    assert preview.headers["cache-control"] == "private, no-store"
    assert preview.headers["content-disposition"] == 'inline; filename="preview.png"'
    assert preview.headers["x-content-type-options"] == "nosniff"
    with Image.open(BytesIO(preview.content)) as rendered:
        assert rendered.format == "PNG"
        assert rendered.mode == "L"
        assert rendered.size == (32, 24)

    downloaded = harness.client.get(
        f"/api/v1/files/{token}",
        headers=_headers(harness, "owner"),
    )
    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"] == "image/tiff"
    assert downloaded.content == original_bytes


def test_probability_array_preview_is_fixed_scale_png(
    principal_file_harness: PrincipalFileHarness,
) -> None:
    harness = principal_file_harness
    _original_bytes, created = _create_analysis(harness)
    job = created["job"]
    images = created["images"]
    assert isinstance(job, dict)
    assert isinstance(images, list)
    image = images[0]
    assert isinstance(image, dict)
    job_id = job["job_id"]
    image_id = image["image_id"]
    assert isinstance(job_id, str)
    assert isinstance(image_id, str)

    run_id = _create_completed_run(harness, job_id=job_id, image_id=image_id)
    run_response = harness.client.get(
        f"/api/v1/runs/{run_id}",
        headers=_headers(harness, "owner"),
    )
    assert run_response.status_code == 200, run_response.text
    artifacts = run_response.json()["data"]["artifacts"]
    assert isinstance(artifacts, dict)
    token = _token_from_url(artifacts["probability_url"])

    preview = harness.client.get(
        f"/api/v1/files/{token}?preview=1",
        headers=_headers(harness, "owner"),
    )
    assert preview.status_code == 200, preview.text
    assert preview.headers["content-type"] == "image/png"
    with Image.open(BytesIO(preview.content)) as rendered:
        assert rendered.format == "PNG"
        assert rendered.mode == "RGB"
        assert rendered.size == (32, 24)
        assert rendered.getpixel((0, 0)) == (68, 1, 84)
        assert rendered.getpixel((11, 9)) != rendered.getpixel((0, 0))

    downloaded = harness.client.get(
        f"/api/v1/files/{token}",
        headers=_headers(harness, "owner"),
    )
    assert downloaded.status_code == 200
    probability = np.load(BytesIO(downloaded.content), allow_pickle=False)
    assert probability.shape == (24, 32)
    assert probability[9, 11] == pytest.approx(0.9)


def test_corrected_mask_token_is_endpoint_bound_and_consumed_once(
    principal_file_harness: PrincipalFileHarness,
) -> None:
    harness = principal_file_harness
    _original_bytes, created = _create_analysis(harness)
    job = created["job"]
    assert isinstance(job, dict)
    images = created["images"]
    assert isinstance(images, list)
    image = images[0]
    assert isinstance(image, dict)
    job_id = job["job_id"]
    image_id = image["image_id"]
    assert isinstance(job_id, str)
    assert isinstance(image_id, str)
    original_token = _token_from_url(image["original_download_url"])
    parent_run_id = _create_completed_run(
        harness,
        job_id=job_id,
        image_id=image_id,
    )

    download_token_as_review = harness.client.post(
        f"/api/v1/runs/{parent_run_id}/review",
        headers=_headers(harness, "owner"),
        json={"corrected_mask_token": original_token},
    )
    assert download_token_as_review.status_code == 400
    assert download_token_as_review.json()["error"]["code"] == "INVALID_IMAGE"
    assert download_token_as_review.json()["error"]["details"] == {
        "reason": "invalid_corrected_mask_token"
    }

    corrected_buffer = BytesIO()
    corrected = Image.new("L", (32, 24), color=0)
    for x in range(9, 17):
        for y in range(7, 15):
            corrected.putpixel((x, y), 255)
    corrected.save(corrected_buffer, format="PNG")
    staged = harness.client.post(
        f"/api/v1/runs/{parent_run_id}/corrected-mask",
        headers=_headers(harness, "owner"),
        files={"file": ("review-mask.png", corrected_buffer.getvalue(), "image/png")},
    )
    assert staged.status_code == 201, staged.text
    corrected_token = staged.json()["data"]["corrected_mask_token"]
    assert isinstance(corrected_token, str)
    assert corrected_token.startswith("v2.")
    corrected_payload = _token_payload(corrected_token)
    assert corrected_payload["pur"] == "review.corrected_mask"
    assert corrected_payload["aud"] == "nanoloop-api:review-corrected-mask"
    assert corrected_payload["sub"] == _OWNER_A

    review_token_as_download = harness.client.get(
        f"/api/v1/files/{corrected_token}",
        headers=_headers(harness, "owner"),
    )
    assert review_token_as_download.status_code == 404
    assert review_token_as_download.json()["error"]["code"] == "RESOURCE_NOT_FOUND"
    assert corrected_token not in review_token_as_download.text

    first = harness.client.post(
        f"/api/v1/runs/{parent_run_id}/review",
        headers=_headers(harness, "owner"),
        json={"corrected_mask_token": corrected_token, "min_area_px": 4},
    )
    assert first.status_code == 202, first.text
    child_run_id = first.json()["data"]["run_id"]

    replay = harness.client.post(
        f"/api/v1/runs/{parent_run_id}/review",
        headers=_headers(harness, "owner"),
        json={"corrected_mask_token": corrected_token, "min_area_px": 4},
    )
    assert replay.status_code == 400
    assert replay.json()["error"]["code"] == "INVALID_IMAGE"
    assert replay.json()["error"]["details"] == {"reason": "invalid_corrected_mask_token"}
    assert corrected_token not in replay.text

    with harness.database.session() as session:
        child_count = session.scalar(
            select(func.count())
            .select_from(SegmentationRun)
            .where(SegmentationRun.parent_run_id == parent_run_id)
        )
    assert child_count == 1
    child = _wait_for_terminal_run(harness, child_run_id)
    assert child["status"] in {"COMPLETED", "COMPLETED_WITH_WARNINGS"}
    assert child["parent_run_id"] == parent_run_id
    configuration = child["configuration"]
    assert isinstance(configuration, dict)
    assert configuration["review_source"] == "corrected_mask"


def test_legacy_v1_download_is_limited_to_legacy_owned_jobs_in_compatibility_modes(
    api_harness: ApiHarness,
) -> None:
    nonlegacy_principal_id = f"prn_{'9' * 32}"
    with api_harness.database.session() as session:
        identities = IdentityService.from_session(session)
        identities.create_principal(
            principal_id=nonlegacy_principal_id,
            tenant_id=LEGACY_TENANT_ID,
            handle="nonlegacy-v1-owner",
            display_name="Nonlegacy v1 owner",
            kind=PrincipalKind.USER,
            role=PrincipalRole.ANALYST,
            now=_NOW,
        )
        SqlAlchemyRepositorySet(session).jobs.create(
            AnalysisJobDTO(
                job_id="job_nonlegacy_v1",
                name="nonlegacy v1 contract",
                status=JobStatus.READY_FOR_CONFIGURATION,
                created_at=_NOW,
                updated_at=_NOW,
            ),
            tenant_id=LEGACY_TENANT_ID,
            owner_principal_id=nonlegacy_principal_id,
        )
    nonlegacy_path = api_harness.file_store.paths.root / "job_nonlegacy_v1" / "private.txt"
    api_harness.file_store.atomic_write_bytes(nonlegacy_path, b"not a legacy-owned artifact")
    nonlegacy_token = api_harness.file_store.create_file_token(
        nonlegacy_path,
        ttl_seconds=3600,
    )

    shared_app = create_app(
        settings=Settings(
            app_env="test",
            auth_mode="shared_key",
            nanoloop_api_key=_SHARED_KEY,
            database_url=str(api_harness.database.engine.url),
            output_root=api_harness.file_store.paths.root,
            log_level="WARNING",
        ),
        database=api_harness.database,
        file_store=api_harness.file_store,
        file_token_v2_keyring=FileTokenV2KeyRing(
            {"contract": b"shared-mode-file-token-v2-key-material"},
            active_kid="contract",
        ),
        inference_gateway=api_harness.gateway,
    )
    shared_client = TestClient(shared_app, raise_server_exceptions=False)
    try:
        mode_requests: tuple[tuple[TestClient, dict[str, str]], ...] = (
            (api_harness.client, {}),
            (shared_client, {"X-API-Key": _SHARED_KEY}),
        )
        for client, headers in mode_requests:
            legacy = client.get(
                f"/api/v1/files/{api_harness.download_token}",
                headers=headers,
            )
            rejected = client.get(
                f"/api/v1/files/{nonlegacy_token}",
                headers=headers,
            )
            assert legacy.status_code == 200
            assert legacy.content.startswith(b"particle_id,area_px")
            assert rejected.status_code == 404
            assert rejected.json()["error"]["code"] == "RESOURCE_NOT_FOUND"
            assert nonlegacy_token not in rejected.text
            assert "job_nonlegacy_v1" not in rejected.text
    finally:
        shared_client.close()

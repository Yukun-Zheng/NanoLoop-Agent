from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import event, func, select

from app.contracts.analyses import (
    AnalysisJobDTO,
    AnalysisROI,
    ImageAssetDTO,
    InferenceOptions,
    PixelRect,
    RunConfiguration,
    SegmentationRunDTO,
)
from app.contracts.enums import (
    JobStatus,
    ModelFamily,
    ModelStatus,
    ModelVariant,
    QualityTier,
    RoiMode,
)
from app.contracts.identity import PrincipalKind, PrincipalRole
from app.contracts.repositories import StoredImageAsset
from app.core.config import Settings
from app.core.identity import IssuedCredential, issue_credential
from app.db.base import Base
from app.db.identity import IdentityService
from app.db.models import AnalysisJob, ModelRegistryRecord
from app.db.repositories import SqlAlchemyRepositorySet
from app.db.session import Database
from app.main import create_app
from app.storage import LocalFileStore, StoragePaths

_NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
_PEPPER = "analysis-authorization-contract-pepper-32-bytes"
_TENANT_A = f"tnt_{'a' * 32}"
_TENANT_B = f"tnt_{'b' * 32}"
_OWNER_A = f"prn_{'1' * 32}"
_PEER_A = f"prn_{'2' * 32}"
_VIEWER_A = f"prn_{'3' * 32}"
_ADMIN_A = f"prn_{'4' * 32}"
_OWNER_B = f"prn_{'5' * 32}"


class _Gateway:
    def health(self) -> list[object]:
        return []


@dataclass(slots=True)
class AuthorizationHarness:
    client: TestClient
    database: Database
    file_store: LocalFileStore
    credentials: dict[str, IssuedCredential]


@pytest.fixture
def authorization_harness(tmp_path: Path) -> Iterator[AuthorizationHarness]:
    settings = Settings(
        app_env="test",
        auth_mode="principal",
        credential_pepper=_PEPPER,
        database_url=f"sqlite:///{tmp_path / 'analysis-authorization.db'}",
        output_root=tmp_path / "outputs",
        model_registry_path=tmp_path / "registry.yaml",
        faiss_index_path=tmp_path / "faiss.index",
        log_level="WARNING",
        api_rate_limit_requests=0,
        api_principal_preauth_rate_limit_requests=1000,
    )
    database = Database(settings)
    Base.metadata.create_all(database.engine)
    file_store = LocalFileStore(
        StoragePaths(settings.output_root),
        max_upload_bytes=1024 * 1024,
        token_secret=b"f" * 32,
    )
    credentials = _seed_identities(database)
    _seed_analyses(database, file_store)
    app = create_app(
        settings=settings,
        database=database,
        file_store=file_store,
        inference_gateway=_Gateway(),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield AuthorizationHarness(
            client=client,
            database=database,
            file_store=file_store,
            credentials=credentials,
        )
    database.dispose()


def _seed_identities(database: Database) -> dict[str, IssuedCredential]:
    specifications = (
        ("owner", _OWNER_A, _TENANT_A, PrincipalRole.ANALYST),
        ("peer", _PEER_A, _TENANT_A, PrincipalRole.ANALYST),
        ("viewer", _VIEWER_A, _TENANT_A, PrincipalRole.VIEWER),
        ("admin", _ADMIN_A, _TENANT_A, PrincipalRole.TENANT_ADMIN),
        ("foreign", _OWNER_B, _TENANT_B, PrincipalRole.ANALYST),
    )
    issued_by_name: dict[str, IssuedCredential] = {}
    with database.session() as session:
        identities = IdentityService.from_session(session)
        identities.create_tenant(
            tenant_id=_TENANT_A,
            slug="authorization-a",
            display_name="Authorization tenant A",
            now=_NOW,
        )
        identities.create_tenant(
            tenant_id=_TENANT_B,
            slug="authorization-b",
            display_name="Authorization tenant B",
            now=_NOW,
        )
        for name, principal_id, tenant_id, role in specifications:
            identities.create_principal(
                principal_id=principal_id,
                tenant_id=tenant_id,
                handle=f"auth-{name}",
                display_name=f"Authorization {name}",
                kind=PrincipalKind.USER,
                role=role,
                now=_NOW,
            )
            issued = issue_credential(_PEPPER)
            identities.issue_credential(
                credential_id=issued.credential_id,
                principal_id=principal_id,
                token_digest=issued.digest,
                label=f"authorization {name}",
                now=_NOW,
            )
            issued_by_name[name] = issued
    return issued_by_name


def _seed_analyses(database: Database, file_store: LocalFileStore) -> None:
    roi = AnalysisROI(valid_rect=PixelRect(x1=0, y1=0, x2=64, y2=64))
    with database.session() as session:
        repositories = SqlAlchemyRepositorySet(session)
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
        for suffix, tenant_id, owner_id in (
            ("a", _TENANT_A, _OWNER_A),
            ("b", _TENANT_B, _OWNER_B),
        ):
            job_id = f"job_{suffix}"
            image_id = f"img_{suffix}"
            storage_path = f"{job_id}/input/{image_id}/original.png"
            file_store.atomic_write_bytes(
                file_store.paths.root / storage_path,
                f"managed image fixture {suffix}".encode(),
            )
            repositories.jobs.create(
                AnalysisJobDTO(
                    job_id=job_id,
                    name=f"authorization {suffix}",
                    status=JobStatus.READY_FOR_CONFIGURATION,
                    created_at=_NOW,
                    updated_at=_NOW,
                ),
                tenant_id=tenant_id,
                owner_principal_id=owner_id,
            )
            repositories.images.add_many(
                [
                    StoredImageAsset(
                        storage_path=storage_path,
                        asset=ImageAssetDTO(
                            image_id=image_id,
                            job_id=job_id,
                            filename=f"{suffix}.png",
                            sha256=suffix * 64,
                            width=64,
                            height=64,
                            bit_depth=8,
                            sample_id=f"sample_{suffix}",
                            analysis_roi=roi,
                        ),
                    )
                ]
            )
            repositories.runs.create_many(
                [
                    SegmentationRunDTO(
                        run_id=f"run_{suffix}",
                        job_id=job_id,
                        image_id=image_id,
                        model_id="unet-general-balanced-v1",
                        status=JobStatus.COMPLETED,
                        roi_mode=RoiMode.FULL_IMAGE,
                        inference=InferenceOptions(),
                        configuration=RunConfiguration(
                            model_id="unet-general-balanced-v1",
                            model_version="1.0.0",
                            roi_mode=RoiMode.FULL_IMAGE,
                            analysis_roi=roi,
                            inference=InferenceOptions(),
                            preprocess_profile="sem_gray_v1",
                            postprocess_profile="default_v1",
                            created_at=_NOW,
                        ),
                        created_at=_NOW,
                        updated_at=_NOW,
                    )
                ]
            )


def _headers(harness: AuthorizationHarness, name: str) -> dict[str, str]:
    return {"X-API-Key": harness.credentials[name].token.get_secret_value()}


def test_foreign_tenant_and_missing_analysis_share_the_not_found_contract(
    authorization_harness: AuthorizationHarness,
) -> None:
    foreign = authorization_harness.client.get(
        "/api/v1/analyses/job_b",
        headers=_headers(authorization_harness, "viewer"),
    )
    missing = authorization_harness.client.get(
        "/api/v1/analyses/job_missing",
        headers=_headers(authorization_harness, "viewer"),
    )
    foreign_run = authorization_harness.client.get(
        "/api/v1/runs/run_b",
        headers=_headers(authorization_harness, "viewer"),
    )
    foreign_export = authorization_harness.client.get(
        "/api/v1/analyses/job_b/export",
        headers=_headers(authorization_harness, "viewer"),
    )

    assert (foreign.status_code, foreign.json()["error"]["code"]) == (
        404,
        "RESOURCE_NOT_FOUND",
    )
    assert (missing.status_code, missing.json()["error"]["code"]) == (
        404,
        "RESOURCE_NOT_FOUND",
    )
    assert (foreign_run.status_code, foreign_run.json()["error"]["code"]) == (
        404,
        "RESOURCE_NOT_FOUND",
    )
    assert (foreign_export.status_code, foreign_export.json()["error"]["code"]) == (
        404,
        "RESOURCE_NOT_FOUND",
    )


def test_same_tenant_reads_and_mutation_role_matrix(
    authorization_harness: AuthorizationHarness,
) -> None:
    viewer_read = authorization_harness.client.get(
        "/api/v1/analyses/job_a",
        headers=_headers(authorization_harness, "viewer"),
    )
    viewer_run = authorization_harness.client.get(
        "/api/v1/runs/run_a",
        headers=_headers(authorization_harness, "viewer"),
    )
    viewer_boxes = authorization_harness.client.get(
        "/api/v1/analyses/job_a/images/img_a/boxes",
        headers=_headers(authorization_harness, "viewer"),
    )
    peer_write = authorization_harness.client.put(
        "/api/v1/analyses/job_a/images/img_a/boxes",
        headers=_headers(authorization_harness, "peer"),
        json={"expected_revision": 0, "boxes": []},
    )
    viewer_write = authorization_harness.client.put(
        "/api/v1/analyses/job_a/images/img_a/boxes",
        headers=_headers(authorization_harness, "viewer"),
        json={"expected_revision": 0, "boxes": []},
    )
    owner_write = authorization_harness.client.put(
        "/api/v1/analyses/job_a/images/img_a/boxes",
        headers=_headers(authorization_harness, "owner"),
        json={"expected_revision": 0, "boxes": []},
    )
    admin_write = authorization_harness.client.put(
        "/api/v1/analyses/job_a/images/img_a/boxes",
        headers=_headers(authorization_harness, "admin"),
        json={"expected_revision": 1, "boxes": []},
    )

    assert viewer_read.status_code == 200
    assert viewer_run.status_code == 200
    assert viewer_boxes.status_code == 200
    assert (peer_write.status_code, peer_write.json()["error"]["code"]) == (403, "FORBIDDEN")
    assert (viewer_write.status_code, viewer_write.json()["error"]["code"]) == (
        403,
        "FORBIDDEN",
    )
    assert owner_write.status_code == 200
    assert owner_write.json()["data"]["revision"] == 1
    assert admin_write.status_code == 200
    assert admin_write.json()["data"]["revision"] == 2


def test_viewer_create_has_no_database_or_managed_file_side_effect(
    authorization_harness: AuthorizationHarness,
) -> None:
    image = BytesIO()
    Image.new("L", (13, 11), color=42).save(image, format="PNG")
    before_paths = set(authorization_harness.file_store.paths.root.rglob("*"))
    with authorization_harness.database.session() as session:
        before_jobs = session.scalar(select(func.count()).select_from(AnalysisJob))

    response = authorization_harness.client.post(
        "/api/v1/analyses",
        headers=_headers(authorization_harness, "viewer"),
        files={"files": ("blocked.png", image.getvalue(), "image/png")},
        data={
            "metadata_json": json.dumps(
                {
                    "job_name": "viewer blocked",
                    "images": [
                        {
                            "filename": "blocked.png",
                            "sample_id": "blocked",
                            "scale": {"mode": "pixel_only"},
                        }
                    ],
                }
            )
        },
    )

    with authorization_harness.database.session() as session:
        after_jobs = session.scalar(select(func.count()).select_from(AnalysisJob))
    assert (response.status_code, response.json()["error"]["code"]) == (403, "FORBIDDEN")
    assert after_jobs == before_jobs
    assert set(authorization_harness.file_store.paths.root.rglob("*")) == before_paths


def test_scoped_analysis_request_reuses_the_single_identity_join(
    authorization_harness: AuthorizationHarness,
) -> None:
    statements: list[str] = []

    def capture_statement(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        statements.append(" ".join(statement.split()))

    event.listen(authorization_harness.database.engine, "before_cursor_execute", capture_statement)
    try:
        response = authorization_harness.client.get(
            "/api/v1/analyses/job_a",
            headers=_headers(authorization_harness, "owner"),
        )
    finally:
        event.remove(
            authorization_harness.database.engine,
            "before_cursor_execute",
            capture_statement,
        )

    assert response.status_code == 200
    identity_reads = [
        statement for statement in statements if "FROM api_credentials JOIN principals" in statement
    ]
    assert len(identity_reads) == 1

from __future__ import annotations

import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import event

from app.analysis.application import AnalysisCreationService
from app.contracts.identity import PrincipalKind, PrincipalRole
from app.core.config import Settings
from app.core.identity import issue_credential
from app.db.base import Base
from app.db.identity import IdentityService
from app.db.models import AnalysisJob
from app.db.repositories import SqlAlchemyUnitOfWork
from app.db.session import Database
from app.main import create_app
from app.storage import LocalFileStore, StoragePaths

_NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
_PEPPER = "analysis-owner-contract-pepper-material-32-bytes"
_TENANT_ID = f"tnt_{'c' * 32}"
_PRINCIPAL_ID = f"prn_{'d' * 32}"


class _Gateway:
    def health(self) -> list[object]:
        return []


def test_principal_analysis_creation_persists_verified_owner_with_one_identity_query(
    tmp_path: Path,
) -> None:
    settings = Settings(
        app_env="test",
        auth_mode="principal",
        credential_pepper=_PEPPER,
        database_url=f"sqlite:///{tmp_path / 'principal-analysis-owner.db'}",
        output_root=tmp_path / "outputs",
        model_registry_path=tmp_path / "registry.yaml",
        faiss_index_path=tmp_path / "faiss.index",
        log_level="WARNING",
        api_rate_limit_requests=0,
        api_principal_preauth_rate_limit_requests=20,
    )
    database = Database(settings)
    Base.metadata.create_all(database.engine)
    file_store = LocalFileStore(
        StoragePaths(settings.output_root),
        max_upload_bytes=1024 * 1024,
        token_secret=b"o" * 32,
    )
    issued = issue_credential(_PEPPER)
    with database.session() as session:
        identities = IdentityService.from_session(session)
        identities.create_tenant(
            tenant_id=_TENANT_ID,
            slug="analysis-owner-contract",
            display_name="Analysis owner contract tenant",
            now=_NOW,
        )
        identities.create_principal(
            principal_id=_PRINCIPAL_ID,
            tenant_id=_TENANT_ID,
            handle="analysis-owner",
            display_name="Analysis owner",
            kind=PrincipalKind.USER,
            role=PrincipalRole.ANALYST,
            now=_NOW,
        )
        identities.issue_credential(
            credential_id=issued.credential_id,
            principal_id=_PRINCIPAL_ID,
            token_digest=issued.digest,
            label="analysis ownership contract",
            now=_NOW,
        )

    creation_service = AnalysisCreationService(
        uow_factory=lambda: SqlAlchemyUnitOfWork(database.session_factory),
        file_store=file_store,
    )
    app = create_app(
        settings=settings,
        database=database,
        file_store=file_store,
        inference_gateway=_Gateway(),
        analysis_creation_service=creation_service,
    )
    image_buffer = BytesIO()
    Image.new("L", (13, 11), color=42).save(image_buffer, format="PNG")
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

    event.listen(database.engine, "before_cursor_execute", capture_statement)
    client = TestClient(app, raise_server_exceptions=False)
    try:
        response = client.post(
            "/api/v1/analyses",
            headers={"X-API-Key": issued.token.get_secret_value()},
            files={"files": ("owned.png", image_buffer.getvalue(), "image/png")},
            data={
                "metadata_json": json.dumps(
                    {
                        "job_name": "principal-owned analysis",
                        "images": [
                            {
                                "filename": "owned.png",
                                "sample_id": "sample_owned",
                                "scale": {"mode": "pixel_only"},
                            }
                        ],
                    }
                )
            },
        )
    finally:
        client.close()
        event.remove(database.engine, "before_cursor_execute", capture_statement)

    try:
        assert response.status_code == 201, response.text
        job_id = response.json()["data"]["job"]["job_id"]
        with database.session() as session:
            record = session.get(AnalysisJob, job_id)
            assert record is not None
            assert (record.tenant_id, record.owner_principal_id) == (
                _TENANT_ID,
                _PRINCIPAL_ID,
            )
        identity_reads = [
            statement
            for statement in statements
            if "FROM api_credentials JOIN principals" in statement
        ]
        assert len(identity_reads) == 1
    finally:
        database.dispose()

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from app.contracts.analyses import AnalysisJobDTO
from app.contracts.enums import JobStatus
from app.contracts.identity import LEGACY_PRINCIPAL_ID, LEGACY_TENANT_ID
from app.core.config import Settings
from app.db.base import Base
from app.db.models import AnalysisJob, Principal, Tenant
from app.db.repositories import SqlAlchemyRepositorySet
from app.db.session import Database


def test_create_all_legacy_job_path_has_ownership_bootstrap_parity(tmp_path: Path) -> None:
    database = Database(
        Settings(app_env="test", database_url=f"sqlite:///{tmp_path / 'ownership.db'}")
    )
    Base.metadata.create_all(database.engine)
    now = datetime.now(UTC)
    try:
        with database.session() as session:
            repositories = SqlAlchemyRepositorySet(session)
            repositories.jobs.create(
                AnalysisJobDTO(
                    job_id="job_legacy_default",
                    name="legacy compatibility",
                    status=JobStatus.CREATED,
                    created_at=now,
                    updated_at=now,
                ),
                tenant_id=LEGACY_TENANT_ID,
                owner_principal_id=LEGACY_PRINCIPAL_ID,
            )

        with database.session() as session:
            job = session.get(AnalysisJob, "job_legacy_default")
            tenant = session.get(Tenant, LEGACY_TENANT_ID)
            principal = session.get(Principal, LEGACY_PRINCIPAL_ID)
            assert job is not None
            assert job.tenant_id == LEGACY_TENANT_ID
            assert job.owner_principal_id == LEGACY_PRINCIPAL_ID
            assert tenant is not None
            assert (
                tenant.slug,
                tenant.display_name,
                tenant.enabled,
                tenant.version,
            ) == ("legacy-local", "Legacy local tenant", True, 1)
            assert tenant.created_at is not None and tenant.updated_at is not None
            assert principal is not None
            assert (
                principal.tenant_id,
                principal.handle,
                principal.display_name,
                principal.kind,
                principal.role,
                principal.enabled,
                principal.version,
            ) == (
                LEGACY_TENANT_ID,
                "legacy-local",
                "Legacy local service",
                "service",
                "tenant_admin",
                True,
                1,
            )
            assert principal.created_at is not None and principal.updated_at is not None
            assert session.scalar(select(AnalysisJob.job_id)) == "job_legacy_default"

        with (
            pytest.raises(IntegrityError, match="NOT NULL"),
            database.engine.begin() as connection,
        ):
            connection.execute(
                text(
                    """
                    INSERT INTO analysis_jobs
                        (job_id, name, status, config_json, created_at, updated_at)
                    VALUES
                        (:job_id, :name, :status, :config_json, :created_at, :updated_at)
                    """
                ),
                {
                    "job_id": "job_missing_explicit_ownership",
                    "name": "invalid raw insert",
                    "status": JobStatus.CREATED.value,
                    "config_json": "{}",
                    "created_at": now,
                    "updated_at": now,
                },
            )
    finally:
        database.dispose()


def test_job_repository_rejects_noncanonical_explicit_ownership(tmp_path: Path) -> None:
    database = Database(
        Settings(app_env="test", database_url=f"sqlite:///{tmp_path / 'invalid-owner.db'}")
    )
    Base.metadata.create_all(database.engine)
    now = datetime.now(UTC)
    try:
        with database.session() as session:
            repositories = SqlAlchemyRepositorySet(session)
            with pytest.raises(ValueError, match="tenant ID"):
                repositories.jobs.create(
                    AnalysisJobDTO(
                        job_id="job_invalid_owner",
                        name="invalid ownership",
                        status=JobStatus.CREATED,
                        created_at=now,
                        updated_at=now,
                    ),
                    tenant_id="not-a-tenant",
                    owner_principal_id=LEGACY_PRINCIPAL_ID,
                )
    finally:
        database.dispose()

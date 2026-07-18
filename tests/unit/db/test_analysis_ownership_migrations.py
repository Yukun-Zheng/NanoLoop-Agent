from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from app.contracts.identity import LEGACY_PRINCIPAL_ID, LEGACY_TENANT_ID
from app.core.config import get_settings
from app.db.migration_state import expected_alembic_heads

_IDENTITY_HEAD = "f5c1d8a4b2e9"
_OWNERSHIP_REVISION = "a6d2e9f4c7b1"
_OLD_JOB_ID = "job_before_ownership"
_OTHER_TENANT_ID = f"tnt_{'1' * 32}"
_OTHER_PRINCIPAL_ID = f"prn_{'2' * 32}"
_TIMESTAMP = "2026-07-18 09:00:00.000000"


@pytest.fixture
def migration_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[Path, Config]]:
    database_path = tmp_path / "analysis-ownership-migration.db"
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path}")
    get_settings.cache_clear()
    expected_alembic_heads.cache_clear()
    config = Config(str(Path(__file__).resolve().parents[3] / "alembic.ini"))
    try:
        yield database_path, config
    finally:
        get_settings.cache_clear()
        expected_alembic_heads.cache_clear()


def test_analysis_ownership_upgrade_backfills_and_enforces_composite_scope(
    migration_database: tuple[Path, Config],
) -> None:
    database_path, config = migration_database
    command.upgrade(config, _IDENTITY_HEAD)
    _insert_pre_ownership_job(database_path)

    command.upgrade(config, _OWNERSHIP_REVISION)

    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        assert connection.execute("PRAGMA foreign_keys").fetchone() == (1,)
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchall()
        ownership = connection.execute(
            """
            SELECT tenant_id, owner_principal_id
            FROM analysis_jobs WHERE job_id = ?
            """,
            (_OLD_JOB_ID,),
        ).fetchone()
        columns = {
            row[1]: {
                "type": row[2],
                "not_null": bool(row[3]),
                "server_default": row[4],
            }
            for row in connection.execute("PRAGMA table_info(analysis_jobs)").fetchall()
        }
        indexes = {
            row[1]: tuple(
                column[2]
                for column in connection.execute(f'PRAGMA index_info("{row[1]}")').fetchall()
            )
            for row in connection.execute("PRAGMA index_list(analysis_jobs)").fetchall()
        }

        assert revision == [(_OWNERSHIP_REVISION,)]
        assert ownership == (LEGACY_TENANT_ID, LEGACY_PRINCIPAL_ID)
        assert columns["tenant_id"] == {
            "type": "VARCHAR(36)",
            "not_null": True,
            "server_default": None,
        }
        assert columns["owner_principal_id"] == {
            "type": "VARCHAR(36)",
            "not_null": True,
            "server_default": None,
        }
        assert indexes["ix_analysis_jobs_tenant_created"] == (
            "tenant_id",
            "created_at",
        )
        assert indexes["ix_analysis_jobs_tenant_owner_created"] == (
            "tenant_id",
            "owner_principal_id",
            "created_at",
        )
        foreign_keys = connection.execute("PRAGMA foreign_key_list(analysis_jobs)").fetchall()
        assert {row[2] for row in foreign_keys} == {"principals", "tenants"}
        assert {row[6] for row in foreign_keys} == {"RESTRICT"}
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []

        _insert_other_identity(connection)
        _insert_owned_job(
            connection,
            job_id="job_matching_owner",
            tenant_id=_OTHER_TENANT_ID,
            owner_principal_id=_OTHER_PRINCIPAL_ID,
        )
        connection.commit()
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            _insert_owned_job(
                connection,
                job_id="job_mismatched_owner",
                tenant_id=LEGACY_TENANT_ID,
                owner_principal_id=_OTHER_PRINCIPAL_ID,
            )
        connection.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="NOT NULL"):
            _insert_owned_job(
                connection,
                job_id="job_missing_owner",
                tenant_id=LEGACY_TENANT_ID,
                owner_principal_id=None,
            )
        connection.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            connection.execute(
                "DELETE FROM principals WHERE principal_id = ?",
                (_OTHER_PRINCIPAL_ID,),
            )
        connection.rollback()
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_analysis_ownership_downgrade_upgrade_preserves_legacy_job(
    migration_database: tuple[Path, Config],
) -> None:
    database_path, config = migration_database
    command.upgrade(config, _IDENTITY_HEAD)
    _insert_pre_ownership_job(database_path)
    command.upgrade(config, _OWNERSHIP_REVISION)

    command.downgrade(config, _IDENTITY_HEAD)
    with sqlite3.connect(database_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(analysis_jobs)")}
        job = connection.execute(
            "SELECT job_id, name FROM analysis_jobs WHERE job_id = ?",
            (_OLD_JOB_ID,),
        ).fetchone()
    assert "tenant_id" not in columns
    assert "owner_principal_id" not in columns
    assert job == (_OLD_JOB_ID, "legacy analysis")

    command.upgrade(config, _OWNERSHIP_REVISION)
    with sqlite3.connect(database_path) as connection:
        ownership = connection.execute(
            """
            SELECT tenant_id, owner_principal_id
            FROM analysis_jobs WHERE job_id = ?
            """,
            (_OLD_JOB_ID,),
        ).fetchone()
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchall()
    assert ownership == (LEGACY_TENANT_ID, LEGACY_PRINCIPAL_ID)
    assert revision == [(_OWNERSHIP_REVISION,)]


def test_analysis_ownership_upgrade_rejects_wrong_legacy_principal_tenant_before_ddl(
    migration_database: tuple[Path, Config],
) -> None:
    database_path, config = migration_database
    command.upgrade(config, _IDENTITY_HEAD)
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            """
            INSERT INTO tenants
                (tenant_id, slug, display_name, enabled, version, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (_OTHER_TENANT_ID, "wrong-legacy", "Wrong legacy tenant", 1, 1, _TIMESTAMP, _TIMESTAMP),
        )
        connection.execute(
            "UPDATE principals SET tenant_id = ? WHERE principal_id = ?",
            (_OTHER_TENANT_ID, LEGACY_PRINCIPAL_ID),
        )
        connection.commit()

    with pytest.raises(RuntimeError, match="legacy ownership identity is invalid"):
        command.upgrade(config, _OWNERSHIP_REVISION)

    with sqlite3.connect(database_path) as connection:
        job_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(analysis_jobs)").fetchall()
        }
        principal_index_columns = {
            tuple(
                column[2]
                for column in connection.execute(f'PRAGMA index_info("{row[1]}")').fetchall()
            )
            for row in connection.execute("PRAGMA index_list(principals)").fetchall()
        }
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchall()
    assert "tenant_id" not in job_columns
    assert "owner_principal_id" not in job_columns
    assert ("principal_id", "tenant_id") not in principal_index_columns
    assert revision == [(_IDENTITY_HEAD,)]


def test_analysis_ownership_downgrade_refuses_nonlegacy_data_before_ddl(
    migration_database: tuple[Path, Config],
) -> None:
    database_path, config = migration_database
    command.upgrade(config, _IDENTITY_HEAD)
    command.upgrade(config, _OWNERSHIP_REVISION)
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        _insert_other_identity(connection)
        _insert_owned_job(
            connection,
            job_id="job_nonlegacy_downgrade_guard",
            tenant_id=_OTHER_TENANT_ID,
            owner_principal_id=_OTHER_PRINCIPAL_ID,
        )
        connection.commit()

    with pytest.raises(RuntimeError, match="downgrade refused"):
        command.downgrade(config, _IDENTITY_HEAD)

    with sqlite3.connect(database_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(analysis_jobs)")}
        indexes = {row[1] for row in connection.execute("PRAGMA index_list(analysis_jobs)")}
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchall()
        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    assert {"tenant_id", "owner_principal_id"} <= columns
    assert {
        "ix_analysis_jobs_tenant_created",
        "ix_analysis_jobs_tenant_owner_created",
    } <= indexes
    assert revision == [(_OWNERSHIP_REVISION,)]
    assert violations == []


def test_analysis_ownership_head_upgrade_smoke(
    migration_database: tuple[Path, Config],
) -> None:
    database_path, config = migration_database
    command.upgrade(config, "head")

    with sqlite3.connect(database_path) as connection:
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchall()
        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    assert revision == [(expected_alembic_heads()[0],)]
    assert violations == []


def _insert_pre_ownership_job(database_path: Path) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO analysis_jobs
                (job_id, name, status, config_json, error_code, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _OLD_JOB_ID,
                "legacy analysis",
                "READY_FOR_CONFIGURATION",
                "{}",
                None,
                _TIMESTAMP,
                _TIMESTAMP,
            ),
        )


def _insert_other_identity(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        INSERT INTO tenants
            (tenant_id, slug, display_name, enabled, version, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (_OTHER_TENANT_ID, "other-tenant", "Other tenant", 1, 1, _TIMESTAMP, _TIMESTAMP),
    )
    connection.execute(
        """
        INSERT INTO principals
            (principal_id, tenant_id, handle, display_name, kind, role,
             enabled, version, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _OTHER_PRINCIPAL_ID,
            _OTHER_TENANT_ID,
            "other-owner",
            "Other owner",
            "user",
            "analyst",
            1,
            1,
            _TIMESTAMP,
            _TIMESTAMP,
        ),
    )
    connection.commit()


def _insert_owned_job(
    connection: sqlite3.Connection,
    *,
    job_id: str,
    tenant_id: str,
    owner_principal_id: str | None,
) -> None:
    connection.execute(
        """
        INSERT INTO analysis_jobs
            (job_id, tenant_id, owner_principal_id, name, status, config_json,
             error_code, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job_id,
            tenant_id,
            owner_principal_id,
            "owned analysis",
            "CREATED",
            "{}",
            None,
            _TIMESTAMP,
            _TIMESTAMP,
        ),
    )

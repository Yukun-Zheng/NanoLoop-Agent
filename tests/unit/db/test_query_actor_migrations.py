from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine

from app.contracts.identity import LEGACY_PRINCIPAL_ID, LEGACY_TENANT_ID
from app.core.config import get_settings
from app.db import models as _models  # noqa: F401
from app.db.base import Base
from app.db.migration_state import expected_alembic_heads

_RELATIONSHIP_REVISION = "c9a4e7b2d6f1"
_ACTOR_REVISION = "e7b3c1d9a5f2"
_TENANT_A = f"tnt_{'1' * 32}"
_PRINCIPAL_A = f"prn_{'2' * 32}"
_CREDENTIAL_A = f"crd_{'3' * 32}"
_TENANT_B = f"tnt_{'4' * 32}"
_PRINCIPAL_B = f"prn_{'5' * 32}"
_CREDENTIAL_B = f"crd_{'6' * 32}"
_TIMESTAMP = "2026-07-18 15:00:00.000000"


@pytest.fixture
def migration_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[Path, Config]]:
    database_path = tmp_path / "query-actor-migration.db"
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


def test_upgrade_backfills_legacy_actor_and_enforces_frozen_identity(
    migration_database: tuple[Path, Config],
) -> None:
    database_path, config = migration_database
    command.upgrade(config, _RELATIONSHIP_REVISION)
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        _insert_job(
            connection,
            job_id="job_legacy",
            tenant_id=LEGACY_TENANT_ID,
            owner_principal_id=LEGACY_PRINCIPAL_ID,
        )
        _insert_historical_query(connection, query_id="query_legacy", job_id="job_legacy")
        connection.commit()

    command.upgrade(config, _ACTOR_REVISION)

    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        actor = connection.execute(
            """
            SELECT actor_tenant_id, actor_principal_id, actor_credential_id,
                   actor_role, actor_auth_mode
            FROM query_logs WHERE query_id = 'query_legacy'
            """
        ).fetchone()
        columns = {
            row[1]: {"not_null": bool(row[3]), "type": row[2]}
            for row in connection.execute("PRAGMA table_info(query_logs)").fetchall()
        }
        assert actor == (
            LEGACY_TENANT_ID,
            LEGACY_PRINCIPAL_ID,
            None,
            "tenant_admin",
            "legacy_unknown",
        )
        assert columns["actor_tenant_id"] == {"not_null": True, "type": "VARCHAR(36)"}
        assert columns["actor_principal_id"] == {"not_null": True, "type": "VARCHAR(36)"}
        assert columns["actor_credential_id"] == {
            "not_null": False,
            "type": "VARCHAR(36)",
        }
        assert columns["actor_role"] == {"not_null": True, "type": "VARCHAR(32)"}
        assert columns["actor_auth_mode"] == {"not_null": True, "type": "VARCHAR(24)"}
        assert ("job_id", "tenant_id") in _unique_column_sets(connection, "analysis_jobs")
        assert ("credential_id", "principal_id") in _unique_column_sets(
            connection, "api_credentials"
        )
        assert (
            "analysis_jobs",
            ("job_id", "actor_tenant_id"),
            ("job_id", "tenant_id"),
            "CASCADE",
        ) in _foreign_key_shapes(connection, "query_logs")
        assert (
            "principals",
            ("actor_principal_id", "actor_tenant_id"),
            ("principal_id", "tenant_id"),
            "RESTRICT",
        ) in _foreign_key_shapes(connection, "query_logs")
        assert (
            "api_credentials",
            ("actor_credential_id", "actor_principal_id"),
            ("credential_id", "principal_id"),
            "RESTRICT",
        ) in _foreign_key_shapes(connection, "query_logs")
        actor_index = next(
            index
            for index in connection.execute("PRAGMA index_list(query_logs)").fetchall()
            if index[1] == "ix_query_logs_actor_created"
        )
        assert tuple(
            row[2]
            for row in connection.execute(f'PRAGMA index_info("{actor_index[1]}")').fetchall()
        ) == ("actor_tenant_id", "actor_principal_id", "created_at")

        _insert_identity(
            connection,
            tenant_id=_TENANT_A,
            principal_id=_PRINCIPAL_A,
            credential_id=_CREDENTIAL_A,
            suffix="a",
        )
        _insert_identity(
            connection,
            tenant_id=_TENANT_B,
            principal_id=_PRINCIPAL_B,
            credential_id=_CREDENTIAL_B,
            suffix="b",
        )
        _insert_job(
            connection,
            job_id="job_a",
            tenant_id=_TENANT_A,
            owner_principal_id=_PRINCIPAL_A,
        )
        connection.commit()

        _insert_attributed_query(
            connection,
            query_id="query_principal",
            job_id="job_a",
            tenant_id=_TENANT_A,
            principal_id=_PRINCIPAL_A,
            credential_id=_CREDENTIAL_A,
            role="analyst",
            auth_mode="principal",
        )
        for auth_mode in ("disabled", "shared_key"):
            _insert_attributed_query(
                connection,
                query_id=f"query_{auth_mode}",
                job_id="job_legacy",
                tenant_id=LEGACY_TENANT_ID,
                principal_id=LEGACY_PRINCIPAL_ID,
                credential_id=None,
                role="tenant_admin",
                auth_mode=auth_mode,
            )
        connection.commit()

        _assert_attributed_query_rejected(
            connection,
            query_id="query_wrong_job_tenant",
            job_id="job_a",
            tenant_id=_TENANT_B,
            principal_id=_PRINCIPAL_B,
            credential_id=_CREDENTIAL_B,
            role="analyst",
            auth_mode="principal",
        )
        _assert_attributed_query_rejected(
            connection,
            query_id="query_wrong_principal_tenant",
            job_id="job_a",
            tenant_id=_TENANT_A,
            principal_id=_PRINCIPAL_B,
            credential_id=_CREDENTIAL_B,
            role="analyst",
            auth_mode="principal",
        )
        _assert_attributed_query_rejected(
            connection,
            query_id="query_wrong_credential_principal",
            job_id="job_a",
            tenant_id=_TENANT_A,
            principal_id=_PRINCIPAL_A,
            credential_id=_CREDENTIAL_B,
            role="analyst",
            auth_mode="principal",
        )
        _assert_attributed_query_rejected(
            connection,
            query_id="query_principal_without_credential",
            job_id="job_a",
            tenant_id=_TENANT_A,
            principal_id=_PRINCIPAL_A,
            credential_id=None,
            role="analyst",
            auth_mode="principal",
        )
        _assert_attributed_query_rejected(
            connection,
            query_id="query_forged_legacy",
            job_id="job_a",
            tenant_id=_TENANT_A,
            principal_id=_PRINCIPAL_A,
            credential_id=None,
            role="analyst",
            auth_mode="legacy_unknown",
        )
        _assert_attributed_query_rejected(
            connection,
            query_id="query_unknown_role",
            job_id="job_a",
            tenant_id=_TENANT_A,
            principal_id=_PRINCIPAL_A,
            credential_id=_CREDENTIAL_A,
            role="operator",
            auth_mode="principal",
        )
        _assert_attributed_query_rejected(
            connection,
            query_id="query_unknown_auth_mode",
            job_id="job_legacy",
            tenant_id=LEGACY_TENANT_ID,
            principal_id=LEGACY_PRINCIPAL_ID,
            credential_id=None,
            role="tenant_admin",
            auth_mode="unknown",
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_upgrade_rejects_nonlegacy_historical_query_before_ddl(
    migration_database: tuple[Path, Config],
) -> None:
    database_path, config = migration_database
    command.upgrade(config, _RELATIONSHIP_REVISION)
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        _insert_identity(
            connection,
            tenant_id=_TENANT_A,
            principal_id=_PRINCIPAL_A,
            credential_id=_CREDENTIAL_A,
            suffix="a",
        )
        _insert_job(
            connection,
            job_id="job_nonlegacy",
            tenant_id=_TENANT_A,
            owner_principal_id=_PRINCIPAL_A,
        )
        _insert_historical_query(
            connection,
            query_id="query_unattributable",
            job_id="job_nonlegacy",
        )
        connection.commit()

    with pytest.raises(RuntimeError, match="historical query actor cannot be reconstructed"):
        command.upgrade(config, _ACTOR_REVISION)

    with sqlite3.connect(database_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(query_logs)")}
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchall()
    assert "actor_tenant_id" not in columns
    assert ("job_id", "tenant_id") not in _unique_column_sets_from_path(
        database_path, "analysis_jobs"
    )
    assert revision == [(_RELATIONSHIP_REVISION,)]


def test_upgrade_rejects_invalid_legacy_sentinel_before_ddl(
    migration_database: tuple[Path, Config],
) -> None:
    database_path, config = migration_database
    command.upgrade(config, _RELATIONSHIP_REVISION)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE principals SET role = 'analyst' WHERE principal_id = ?",
            (LEGACY_PRINCIPAL_ID,),
        )
        connection.commit()

    with pytest.raises(RuntimeError, match="legacy query actor identity is invalid"):
        command.upgrade(config, _ACTOR_REVISION)

    with sqlite3.connect(database_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(query_logs)")}
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchall()
    assert "actor_auth_mode" not in columns
    assert revision == [(_RELATIONSHIP_REVISION,)]


def test_upgrade_rejects_existing_foreign_key_violation_before_ddl(
    migration_database: tuple[Path, Config],
) -> None:
    database_path, config = migration_database
    command.upgrade(config, _RELATIONSHIP_REVISION)
    with sqlite3.connect(database_path) as connection:
        _insert_orphan_credential(connection)
        connection.commit()
        assert connection.execute("PRAGMA foreign_key_check").fetchall()

    with pytest.raises(RuntimeError, match="failed before query actor migration"):
        command.upgrade(config, _ACTOR_REVISION)

    with sqlite3.connect(database_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(query_logs)")}
        indexes = {row[1] for row in connection.execute("PRAGMA index_list(query_logs)")}
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchall()
        job_unique_columns = _unique_column_sets(connection, "analysis_jobs")
        credential_unique_columns = _unique_column_sets(connection, "api_credentials")
    assert "actor_tenant_id" not in columns
    assert "ix_query_logs_actor_created" not in indexes
    assert ("job_id", "tenant_id") not in job_unique_columns
    assert ("credential_id", "principal_id") not in credential_unique_columns
    assert revision == [(_RELATIONSHIP_REVISION,)]


def test_downgrade_refuses_attributable_query_before_ddl(
    migration_database: tuple[Path, Config],
) -> None:
    database_path, config = migration_database
    command.upgrade(config, _ACTOR_REVISION)
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        _insert_identity(
            connection,
            tenant_id=_TENANT_A,
            principal_id=_PRINCIPAL_A,
            credential_id=_CREDENTIAL_A,
            suffix="a",
        )
        _insert_job(
            connection,
            job_id="job_a",
            tenant_id=_TENANT_A,
            owner_principal_id=_PRINCIPAL_A,
        )
        _insert_attributed_query(
            connection,
            query_id="query_attributable",
            job_id="job_a",
            tenant_id=_TENANT_A,
            principal_id=_PRINCIPAL_A,
            credential_id=_CREDENTIAL_A,
            role="analyst",
            auth_mode="principal",
        )
        connection.commit()

    with pytest.raises(RuntimeError, match="attributable query audit facts exist"):
        command.downgrade(config, _RELATIONSHIP_REVISION)

    with sqlite3.connect(database_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(query_logs)")}
        indexes = {row[1] for row in connection.execute("PRAGMA index_list(query_logs)")}
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchall()
    assert "actor_auth_mode" in columns
    assert "ix_query_logs_actor_created" in indexes
    assert revision == [(_ACTOR_REVISION,)]


def test_downgrade_rejects_existing_foreign_key_violation_before_ddl(
    migration_database: tuple[Path, Config],
) -> None:
    database_path, config = migration_database
    command.upgrade(config, _ACTOR_REVISION)
    with sqlite3.connect(database_path) as connection:
        _insert_orphan_credential(connection)
        connection.commit()
        assert connection.execute("PRAGMA foreign_key_check").fetchall()

    with pytest.raises(RuntimeError, match="failed before query actor migration"):
        command.downgrade(config, _RELATIONSHIP_REVISION)

    with sqlite3.connect(database_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(query_logs)")}
        indexes = {row[1] for row in connection.execute("PRAGMA index_list(query_logs)")}
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchall()
        job_unique_columns = _unique_column_sets(connection, "analysis_jobs")
        credential_unique_columns = _unique_column_sets(connection, "api_credentials")
    assert "actor_tenant_id" in columns
    assert "ix_query_logs_actor_created" in indexes
    assert ("job_id", "tenant_id") in job_unique_columns
    assert ("credential_id", "principal_id") in credential_unique_columns
    assert revision == [(_ACTOR_REVISION,)]


def test_legacy_query_downgrade_upgrade_roundtrip_preserves_audit_row(
    migration_database: tuple[Path, Config],
) -> None:
    database_path, config = migration_database
    command.upgrade(config, _RELATIONSHIP_REVISION)
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        _insert_job(
            connection,
            job_id="job_roundtrip",
            tenant_id=LEGACY_TENANT_ID,
            owner_principal_id=LEGACY_PRINCIPAL_ID,
        )
        _insert_historical_query(
            connection,
            query_id="query_roundtrip",
            job_id="job_roundtrip",
        )
        connection.commit()

    command.upgrade(config, _ACTOR_REVISION)
    command.downgrade(config, _RELATIONSHIP_REVISION)

    with sqlite3.connect(database_path) as connection:
        schema = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'query_logs'"
        ).fetchone()[0]
        columns = {row[1] for row in connection.execute("PRAGMA table_info(query_logs)")}
        row = connection.execute(
            "SELECT job_id, question FROM query_logs WHERE query_id = 'query_roundtrip'"
        ).fetchone()
        assert "fk_query_logs_job_id_analysis_jobs" in schema
        assert "fk_query_logs_job_actor_tenant" not in schema
        assert "actor_tenant_id" not in columns
        assert row == ("job_roundtrip", "historical question")
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []

    command.upgrade(config, _ACTOR_REVISION)
    with sqlite3.connect(database_path) as connection:
        actor_mode = connection.execute(
            "SELECT actor_auth_mode FROM query_logs WHERE query_id = 'query_roundtrip'"
        ).fetchone()
        assert actor_mode == ("legacy_unknown",)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_query_actor_head_has_no_model_metadata_drift(
    migration_database: tuple[Path, Config],
) -> None:
    database_path, config = migration_database
    command.upgrade(config, "head")

    engine = create_engine(f"sqlite:///{database_path}")
    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(
                connection,
                opts={
                    "compare_type": True,
                    "include_object": _exclude_fts_shadow_tables,
                },
            )
            differences = compare_metadata(context, Base.metadata)
            revision = connection.exec_driver_sql(
                "SELECT version_num FROM alembic_version"
            ).fetchall()
            violations = connection.exec_driver_sql("PRAGMA foreign_key_check").fetchall()
    finally:
        engine.dispose()

    assert [tuple(row) for row in revision] == [(expected_alembic_heads()[0],)]
    assert violations == []
    assert differences == []


def _insert_identity(
    connection: sqlite3.Connection,
    *,
    tenant_id: str,
    principal_id: str,
    credential_id: str,
    suffix: str,
) -> None:
    connection.execute(
        """
        INSERT INTO tenants
            (tenant_id, slug, display_name, enabled, version, created_at, updated_at)
        VALUES (?, ?, ?, 1, 1, ?, ?)
        """,
        (tenant_id, f"actor-{suffix}", f"Actor {suffix}", _TIMESTAMP, _TIMESTAMP),
    )
    connection.execute(
        """
        INSERT INTO principals
            (principal_id, tenant_id, handle, display_name, kind, role,
             enabled, version, created_at, updated_at)
        VALUES (?, ?, ?, ?, 'user', 'analyst', 1, 1, ?, ?)
        """,
        (
            principal_id,
            tenant_id,
            f"actor-{suffix}",
            f"Actor {suffix}",
            _TIMESTAMP,
            _TIMESTAMP,
        ),
    )
    connection.execute(
        """
        INSERT INTO api_credentials
            (credential_id, principal_id, label, token_digest, enabled, version,
             created_at, updated_at)
        VALUES (?, ?, ?, ?, 1, 1, ?, ?)
        """,
        (
            credential_id,
            principal_id,
            f"credential-{suffix}",
            bytes([int(suffix, 16)]) * 32,
            _TIMESTAMP,
            _TIMESTAMP,
        ),
    )


def _insert_job(
    connection: sqlite3.Connection,
    *,
    job_id: str,
    tenant_id: str,
    owner_principal_id: str,
) -> None:
    connection.execute(
        """
        INSERT INTO analysis_jobs
            (job_id, tenant_id, owner_principal_id, name, status, config_json,
             created_at, updated_at)
        VALUES (?, ?, ?, ?, 'CREATED', '{}', ?, ?)
        """,
        (job_id, tenant_id, owner_principal_id, job_id, _TIMESTAMP, _TIMESTAMP),
    )


def _insert_orphan_credential(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys=OFF")
    connection.execute(
        """
        INSERT INTO api_credentials
            (credential_id, principal_id, label, token_digest, enabled, version,
             created_at, updated_at)
        VALUES (?, ?, 'orphan credential', ?, 1, 1, ?, ?)
        """,
        (
            _CREDENTIAL_B,
            _PRINCIPAL_B,
            b"o" * 32,
            _TIMESTAMP,
            _TIMESTAMP,
        ),
    )


def _insert_historical_query(
    connection: sqlite3.Connection,
    *,
    query_id: str,
    job_id: str,
) -> None:
    connection.execute(
        """
        INSERT INTO query_logs
            (query_id, job_id, image_id, query_type, question, request_json,
             answer_json, created_at)
        VALUES (?, ?, NULL, 'auto', 'historical question', '{}', '{}', ?)
        """,
        (query_id, job_id, _TIMESTAMP),
    )


def _insert_attributed_query(
    connection: sqlite3.Connection,
    *,
    query_id: str,
    job_id: str,
    tenant_id: str,
    principal_id: str,
    credential_id: str | None,
    role: str,
    auth_mode: str,
) -> None:
    connection.execute(
        """
        INSERT INTO query_logs
            (query_id, job_id, image_id, query_type, question, request_json,
             answer_json, actor_tenant_id, actor_principal_id, actor_credential_id,
             actor_role, actor_auth_mode, created_at)
        VALUES (?, ?, NULL, 'auto', 'attributed question', '{}', '{}',
                ?, ?, ?, ?, ?, ?)
        """,
        (
            query_id,
            job_id,
            tenant_id,
            principal_id,
            credential_id,
            role,
            auth_mode,
            _TIMESTAMP,
        ),
    )


def _assert_attributed_query_rejected(
    connection: sqlite3.Connection,
    **values: str | None,
) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        _insert_attributed_query(connection, **values)  # type: ignore[arg-type]
    connection.rollback()


def _foreign_key_shapes(
    connection: sqlite3.Connection,
    table_name: str,
) -> set[tuple[str, tuple[str, ...], tuple[str, ...], str]]:
    grouped: dict[int, list[tuple[object, ...]]] = {}
    for row in connection.execute(f'PRAGMA foreign_key_list("{table_name}")').fetchall():
        grouped.setdefault(cast(int, row[0]), []).append(row)
    return {
        (
            str(rows[0][2]),
            tuple(
                str(row[3])
                for row in sorted(rows, key=lambda item: cast(int, item[1]))
            ),
            tuple(
                str(row[4])
                for row in sorted(rows, key=lambda item: cast(int, item[1]))
            ),
            str(rows[0][6]),
        )
        for rows in grouped.values()
    }


def _unique_column_sets(
    connection: sqlite3.Connection,
    table_name: str,
) -> set[tuple[str, ...]]:
    return {
        tuple(
            str(column[2])
            for column in connection.execute(f'PRAGMA index_info("{index[1]}")').fetchall()
        )
        for index in connection.execute(f'PRAGMA index_list("{table_name}")').fetchall()
        if bool(index[2])
    }


def _unique_column_sets_from_path(
    database_path: Path,
    table_name: str,
) -> set[tuple[str, ...]]:
    with sqlite3.connect(database_path) as connection:
        return _unique_column_sets(connection, table_name)


def _exclude_fts_shadow_tables(
    _object: object,
    name: str | None,
    type_: str,
    reflected: bool,
    _compare_to: object,
) -> bool:
    return not (
        reflected
        and type_ == "table"
        and name is not None
        and name.startswith("knowledge_chunks_fts")
    )

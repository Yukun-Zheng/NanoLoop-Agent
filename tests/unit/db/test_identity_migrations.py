from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from app.contracts.identity import LEGACY_PRINCIPAL_ID, LEGACY_TENANT_ID
from app.core.config import get_settings

_PREVIOUS_HEAD = "e2a7c4d8f1b3"
_IDENTITY_HEAD = "f5c1d8a4b2e9"


@pytest.fixture
def migration_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[Path, Config]]:
    database_path = tmp_path / "identity-migration.db"
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path}")
    get_settings.cache_clear()
    config = Config(str(Path(__file__).resolve().parents[3] / "alembic.ini"))
    try:
        yield database_path, config
    finally:
        get_settings.cache_clear()


def test_identity_upgrade_downgrade_upgrade_bootstraps_legacy(
    migration_database: tuple[Path, Config],
) -> None:
    database_path, config = migration_database
    command.upgrade(config, _PREVIOUS_HEAD)
    assert "tenants" not in _table_names(database_path)

    command.upgrade(config, _IDENTITY_HEAD)
    _assert_legacy_bootstrap(database_path)


def test_identity_sqlite_audit_triggers_reject_direct_update_and_delete(
    migration_database: tuple[Path, Config],
) -> None:
    database_path, config = migration_database
    command.upgrade(config, _IDENTITY_HEAD)

    with sqlite3.connect(database_path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE identity_audit_events SET event_type = 'tenant.disabled' WHERE event_id = 1"
            )
        connection.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM identity_audit_events WHERE event_id = 1")
        connection.rollback()
        assert connection.execute(
            "SELECT event_type FROM identity_audit_events ORDER BY event_id"
        ).fetchall() == [("tenant.created",), ("principal.created",)]

    command.downgrade(config, _PREVIOUS_HEAD)
    names_after_downgrade = _table_names(database_path)
    assert {
        "tenants",
        "principals",
        "api_credentials",
        "identity_audit_events",
    }.isdisjoint(names_after_downgrade)

    command.upgrade(config, _IDENTITY_HEAD)
    _assert_legacy_bootstrap(database_path)


def _assert_legacy_bootstrap(database_path: Path) -> None:
    with sqlite3.connect(database_path) as connection:
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchall()
        tenant = connection.execute(
            """
            SELECT tenant_id, slug, enabled, version
            FROM tenants WHERE tenant_id = ?
            """,
            (LEGACY_TENANT_ID,),
        ).fetchall()
        principal = connection.execute(
            """
            SELECT principal_id, tenant_id, handle, kind, role, enabled, version
            FROM principals WHERE principal_id = ?
            """,
            (LEGACY_PRINCIPAL_ID,),
        ).fetchall()
        audits = connection.execute(
            """
            SELECT event_type, tenant_id, principal_id, credential_id,
                   actor_principal_id, actor_kind
            FROM identity_audit_events ORDER BY event_id
            """
        ).fetchall()
        credential_columns = {
            row[1]: row[2]
            for row in connection.execute("PRAGMA table_info(api_credentials)").fetchall()
        }

    assert revision == [(_IDENTITY_HEAD,)]
    assert tenant == [(LEGACY_TENANT_ID, "legacy-local", 1, 1)]
    assert principal == [
        (
            LEGACY_PRINCIPAL_ID,
            LEGACY_TENANT_ID,
            "legacy-local",
            "service",
            "tenant_admin",
            1,
            1,
        )
    ]
    assert audits == [
        ("tenant.created", LEGACY_TENANT_ID, None, None, None, "migration"),
        (
            "principal.created",
            LEGACY_TENANT_ID,
            LEGACY_PRINCIPAL_ID,
            None,
            None,
            "migration",
        ),
    ]
    assert credential_columns["token_digest"] == "BLOB"
    assert not any("token" in name and name != "token_digest" for name in credential_columns)


def _table_names(database_path: Path) -> set[str]:
    with sqlite3.connect(database_path) as connection:
        return {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

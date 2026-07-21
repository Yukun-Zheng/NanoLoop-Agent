from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

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

_ACTOR_REVISION = "e7b3c1d9a5f2"
_ARTIFACT_REVISION = "b4f2e8c6a1d9"
_TENANT_B = f"tnt_{'b' * 32}"
_PRINCIPAL_B = f"prn_{'c' * 32}"
_TIMESTAMP = "2026-07-18 18:45:00.000000"


@pytest.fixture
def migration_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[Path, Config]]:
    database_path = tmp_path / "file-artifact-migration.db"
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


def test_upgrade_creates_empty_registry_without_interpreting_historical_paths(
    migration_database: tuple[Path, Config],
) -> None:
    database_path, config = migration_database
    command.upgrade(config, _ACTOR_REVISION)
    with sqlite3.connect(database_path) as connection:
        _seed_graph(connection)
        connection.commit()

    command.upgrade(config, _ARTIFACT_REVISION)

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT count(*) FROM file_artifacts").fetchone() == (0,)
        assert ("run_id", "image_id", "job_id") in _unique_column_sets(
            connection,
            "segmentation_runs",
        )
        assert _foreign_key_shapes(connection, "file_artifacts") >= {
            (
                "analysis_jobs",
                ("job_id",),
                ("job_id",),
                "CASCADE",
            ),
            (
                "image_assets",
                ("image_id", "job_id"),
                ("image_id", "job_id"),
                "CASCADE",
            ),
            (
                "segmentation_runs",
                ("run_id", "job_id"),
                ("run_id", "job_id"),
                "CASCADE",
            ),
            (
                "segmentation_runs",
                ("run_id", "image_id", "job_id"),
                ("run_id", "image_id", "job_id"),
                "CASCADE",
            ),
        }
        index = next(
            row
            for row in connection.execute("PRAGMA index_list(file_artifacts)").fetchall()
            if row[1] == "ix_file_artifacts_job_kind_state"
        )
        assert tuple(
            row[2] for row in connection.execute(f'PRAGMA index_info("{index[1]}")').fetchall()
        ) == ("job_id", "artifact_kind", "state")
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"artifact_id": f"art_{'g' * 32}"},
        {"job_id": "missing_job", "image_id": None, "artifact_kind": "analysis_export"},
        {"image_id": "missing_image"},
        {"image_id": "image_b"},
        {
            "artifact_kind": "run_artifact",
            "image_id": "image_a2",
            "run_id": "run_a",
        },
        {
            "artifact_kind": "run_artifact",
            "image_id": "image_a",
            "run_id": None,
        },
        {
            "artifact_kind": "corrected_mask_input",
            "image_id": None,
            "run_id": "run_a",
        },
        {
            "artifact_kind": "run_artifact",
            "image_id": "image_a",
            "run_id": "run_b",
        },
        {"artifact_kind": "unknown"},
        {"storage_path": "jobs/job_a/bad\x00path.tif"},
        {"filename": "unsafe\r\nname.tif"},
        {"media_type": "image/tiff\nunsafe"},
        {"sha256": "A" * 64},
        {"size_bytes": -1},
        {
            "state": "consumed",
            "consumed_at": _TIMESTAMP,
        },
        {
            "artifact_kind": "run_artifact",
            "image_id": "image_a",
            "run_id": "run_a",
            "state": "consumed",
            "consumed_at": _TIMESTAMP,
        },
        {
            "artifact_kind": "analysis_export",
            "image_id": None,
            "state": "consumed",
            "consumed_at": _TIMESTAMP,
        },
        {
            "artifact_kind": "corrected_mask_input",
            "image_id": "image_a",
            "run_id": "run_a",
            "state": "consumed",
            "consumed_at": "2026-01-01 00:00:00.000000",
        },
        {"consumed_at": _TIMESTAMP},
        {
            "state": "revoked",
            "revoked_at": None,
        },
        {
            "state": "revoked",
            "revoked_at": "2026-01-01 00:00:00.000000",
        },
    ],
)
def test_database_rejects_orphan_mismatch_unsafe_and_inconsistent_rows(
    migration_database: tuple[Path, Config],
    overrides: dict[str, object],
) -> None:
    database_path, config = migration_database
    command.upgrade(config, _ACTOR_REVISION)
    with sqlite3.connect(database_path) as connection:
        _seed_graph(connection)
        connection.commit()
    command.upgrade(config, _ARTIFACT_REVISION)

    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        values = _artifact_values()
        values.update(overrides)
        with pytest.raises(sqlite3.IntegrityError):
            _insert_artifact(connection, values)
        connection.rollback()
        assert connection.execute("SELECT count(*) FROM file_artifacts").fetchone() == (0,)


def test_database_locks_relationship_shape_immutable_facts_and_one_way_terminal_state(
    migration_database: tuple[Path, Config],
) -> None:
    database_path, config = migration_database
    command.upgrade(config, _ACTOR_REVISION)
    with sqlite3.connect(database_path) as connection:
        _seed_graph(connection)
        connection.commit()
    command.upgrade(config, _ARTIFACT_REVISION)

    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        original = _artifact_values()
        _insert_artifact(connection, original)
        corrected = _artifact_values(
            artifact_id=f"art_{'2' * 32}",
            artifact_kind="corrected_mask_input",
            image_id="image_a",
            run_id="run_a",
            storage_path="jobs/job_a/review/run_a/corrected-mask.png",
            filename="corrected-mask.png",
            media_type="image/png",
        )
        _insert_artifact(connection, corrected)
        connection.commit()

        connection.execute(
            """
            UPDATE file_artifacts
            SET state = 'consumed', consumed_at = ?
            WHERE artifact_id = ?
            """,
            (_TIMESTAMP, corrected["artifact_id"]),
        )
        connection.commit()
        assert connection.execute(
            "SELECT state FROM file_artifacts WHERE artifact_id = ?",
            (corrected["artifact_id"],),
        ).fetchone() == ("consumed",)

        with pytest.raises(sqlite3.IntegrityError, match="terminal file artifact state"):
            connection.execute(
                """
                UPDATE file_artifacts
                SET state = 'active', consumed_at = NULL
                WHERE artifact_id = ?
                """,
                (corrected["artifact_id"],),
            )
        connection.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="facts are immutable"):
            connection.execute(
                "UPDATE file_artifacts SET sha256 = ? WHERE artifact_id = ?",
                ("f" * 64, original["artifact_id"]),
            )
        connection.rollback()


def test_upgrade_refuses_existing_foreign_key_damage_before_first_ddl(
    migration_database: tuple[Path, Config],
) -> None:
    database_path, config = migration_database
    command.upgrade(config, _ACTOR_REVISION)
    with sqlite3.connect(database_path) as connection:
        _seed_graph(connection)
        connection.commit()
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute(
            "UPDATE segmentation_runs SET image_id = 'image_b' WHERE run_id = 'run_a'"
        )
        connection.commit()
        assert connection.execute("PRAGMA foreign_key_check").fetchall()

    with pytest.raises(RuntimeError, match="failed before file artifact migration"):
        command.upgrade(config, _ARTIFACT_REVISION)

    with sqlite3.connect(database_path) as connection:
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'file_artifacts'"
            ).fetchone()
            is None
        )
        assert ("run_id", "image_id", "job_id") not in _unique_column_sets(
            connection,
            "segmentation_runs",
        )
        assert connection.execute("SELECT version_num FROM alembic_version").fetchall() == [
            (_ACTOR_REVISION,)
        ]


def test_downgrade_refuses_data_loss_before_ddl_then_empty_roundtrip_succeeds(
    migration_database: tuple[Path, Config],
) -> None:
    database_path, config = migration_database
    command.upgrade(config, _ACTOR_REVISION)
    with sqlite3.connect(database_path) as connection:
        _seed_graph(connection)
        connection.commit()
    command.upgrade(config, _ARTIFACT_REVISION)
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        _insert_artifact(connection, _artifact_values())
        connection.commit()

    with pytest.raises(RuntimeError, match="registered artifact facts exist"):
        command.downgrade(config, _ACTOR_REVISION)

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT count(*) FROM file_artifacts").fetchone() == (1,)
        assert connection.execute("SELECT version_num FROM alembic_version").fetchall() == [
            (_ARTIFACT_REVISION,)
        ]
        connection.execute("DELETE FROM file_artifacts")
        connection.commit()

    command.downgrade(config, _ACTOR_REVISION)
    with sqlite3.connect(database_path) as connection:
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'file_artifacts'"
            ).fetchone()
            is None
        )
        assert ("run_id", "image_id", "job_id") not in _unique_column_sets(
            connection,
            "segmentation_runs",
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []

    command.upgrade(config, _ARTIFACT_REVISION)
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT count(*) FROM file_artifacts").fetchone() == (0,)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_file_artifact_head_has_no_model_metadata_drift(
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


def _seed_graph(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute(
        """
        INSERT INTO tenants
            (tenant_id, slug, display_name, enabled, version, created_at, updated_at)
        VALUES (?, 'tenant-b', 'Tenant B', 1, 1, ?, ?)
        """,
        (_TENANT_B, _TIMESTAMP, _TIMESTAMP),
    )
    connection.execute(
        """
        INSERT INTO principals
            (principal_id, tenant_id, handle, display_name, kind, role,
             enabled, version, created_at, updated_at)
        VALUES (?, ?, 'owner-b', 'Owner B', 'user', 'analyst', 1, 1, ?, ?)
        """,
        (_PRINCIPAL_B, _TENANT_B, _TIMESTAMP, _TIMESTAMP),
    )
    for job_id, tenant_id, principal_id in (
        ("job_a", LEGACY_TENANT_ID, LEGACY_PRINCIPAL_ID),
        ("job_b", _TENANT_B, _PRINCIPAL_B),
    ):
        connection.execute(
            """
            INSERT INTO analysis_jobs
                (job_id, tenant_id, owner_principal_id, name, status, config_json,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, 'CREATED', '{}', ?, ?)
            """,
            (job_id, tenant_id, principal_id, job_id, _TIMESTAMP, _TIMESTAMP),
        )
    connection.execute(
        """
        INSERT INTO model_registry
            (model_id, family, variant, quality_tier, version, adapter, status,
             metadata_json, created_at, updated_at)
        VALUES ('model_a', 'unet', 'general', 'balanced', '1',
                'tests.fake:FakeAdapter', 'ready', '{}', ?, ?)
        """,
        (_TIMESTAMP, _TIMESTAMP),
    )
    for image_id, job_id, digest in (
        ("image_a", "job_a", "a" * 64),
        ("image_a2", "job_a", "b" * 64),
        ("image_b", "job_b", "c" * 64),
    ):
        connection.execute(
            """
            INSERT INTO image_assets
                (image_id, job_id, filename, storage_path, sha256, width, height,
                 bit_depth, sample_id, experiment_conditions_json, analysis_roi_json,
                 box_revision, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 32, 32, 8, ?, '{}', '{}', 0, ?, ?)
            """,
            (
                image_id,
                job_id,
                f"{image_id}.tif",
                f"jobs/{job_id}/input/{image_id}/original.tif",
                digest,
                image_id,
                _TIMESTAMP,
                _TIMESTAMP,
            ),
        )
    for run_id, job_id, image_id in (
        ("run_a", "job_a", "image_a"),
        ("run_b", "job_b", "image_b"),
    ):
        connection.execute(
            """
            INSERT INTO segmentation_runs
                (run_id, job_id, image_id, model_id, roi_mode, status,
                 inference_json, run_config_json, paths_json, created_at, updated_at)
            VALUES (?, ?, ?, 'model_a', 'full_image', 'CREATED', '{}', '{}',
                    '{"mask_url":"legacy-path-only.png"}', ?, ?)
            """,
            (run_id, job_id, image_id, _TIMESTAMP, _TIMESTAMP),
        )


def _artifact_values(**updates: object) -> dict[str, object]:
    values: dict[str, object] = {
        "artifact_id": f"art_{'1' * 32}",
        "job_id": "job_a",
        "image_id": "image_a",
        "run_id": None,
        "artifact_kind": "original_image",
        "storage_path": "jobs/job_a/input/image_a/original.tif",
        "filename": "image_a.tif",
        "media_type": "image/tiff",
        "sha256": "a" * 64,
        "size_bytes": 512,
        "state": "active",
        "created_at": _TIMESTAMP,
        "consumed_at": None,
        "revoked_at": None,
    }
    values.update(updates)
    return values


def _insert_artifact(connection: sqlite3.Connection, values: dict[str, object]) -> None:
    columns = tuple(values)
    placeholders = ", ".join("?" for _ in columns)
    connection.execute(
        f"INSERT INTO file_artifacts ({', '.join(columns)}) VALUES ({placeholders})",
        tuple(values[column] for column in columns),
    )


def _unique_column_sets(
    connection: sqlite3.Connection,
    table_name: str,
) -> set[tuple[str, ...]]:
    return {
        tuple(row[2] for row in connection.execute(f'PRAGMA index_info("{index[1]}")').fetchall())
        for index in connection.execute(f'PRAGMA index_list("{table_name}")').fetchall()
        if index[2]
    }


def _foreign_key_shapes(
    connection: sqlite3.Connection,
    table_name: str,
) -> set[tuple[str, tuple[str, ...], tuple[str, ...], str]]:
    grouped: dict[int, list[tuple[object, ...]]] = {}
    for row in connection.execute(f'PRAGMA foreign_key_list("{table_name}")').fetchall():
        grouped.setdefault(int(row[0]), []).append(row)
    shapes: set[tuple[str, tuple[str, ...], tuple[str, ...], str]] = set()
    for rows in grouped.values():
        ordered = sorted(rows, key=lambda row: int(row[1]))
        shapes.add(
            (
                str(ordered[0][2]),
                tuple(str(row[3]) for row in ordered),
                tuple(str(row[4]) for row in ordered),
                str(ordered[0][6]),
            )
        )
    return shapes


def _exclude_fts_shadow_tables(
    _object: object,
    name: str | None,
    object_type: str,
    _reflected: bool,
    _compare_to: object,
) -> bool:
    if object_type != "table" or name is None:
        return True
    return name != "knowledge_chunks_fts" and not name.startswith("knowledge_chunks_fts_")

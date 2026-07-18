from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
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

_OWNERSHIP_REVISION = "a6d2e9f4c7b1"
_RELATIONSHIP_REVISION = "c9a4e7b2d6f1"
_TIMESTAMP = "2026-07-18 12:00:00.000000"


@pytest.fixture
def migration_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[Path, Config]]:
    database_path = tmp_path / "analysis-relationship-migration.db"
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


def _seed_run_image_mismatch(connection: sqlite3.Connection) -> None:
    _insert_run(
        connection,
        run_id="run_wrong_image_job",
        job_id="job_a",
        image_id="image_b",
    )


def _seed_query_image_mismatch(connection: sqlite3.Connection) -> None:
    _insert_query(
        connection,
        query_id="query_wrong_image_job",
        job_id="job_a",
        image_id="image_b",
    )


def _seed_parent_job_mismatch(connection: sqlite3.Connection) -> None:
    _insert_run(
        connection,
        run_id="run_parent_other_job",
        job_id="job_b",
        image_id="image_b",
    )
    _insert_run(
        connection,
        run_id="run_child_wrong_parent",
        job_id="job_a",
        image_id="image_a",
        parent_run_id="run_parent_other_job",
    )


@pytest.mark.parametrize(
    ("seed_mismatch", "message"),
    [
        (
            _seed_run_image_mismatch,
            "segmentation run image/job mismatch",
        ),
        (
            _seed_query_image_mismatch,
            "query image/job mismatch",
        ),
        (
            _seed_parent_job_mismatch,
            "review parent crosses jobs",
        ),
    ],
    ids=["run-image", "query-image", "review-parent"],
)
def test_upgrade_rejects_inconsistent_relationships_before_ddl(
    migration_database: tuple[Path, Config],
    seed_mismatch: Callable[[sqlite3.Connection], None],
    message: str,
) -> None:
    database_path, config = migration_database
    command.upgrade(config, _OWNERSHIP_REVISION)
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        _seed_analysis_graph(connection)
        seed_mismatch(connection)
        connection.commit()

    with pytest.raises(RuntimeError, match=message):
        command.upgrade(config, _RELATIONSHIP_REVISION)

    with sqlite3.connect(database_path) as connection:
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchall()
        run_foreign_keys = _foreign_key_shapes(connection, "segmentation_runs")
        image_unique_columns = _unique_column_sets(connection, "image_assets")
    assert revision == [(_OWNERSHIP_REVISION,)]
    assert ("image_assets", ("image_id", "job_id"), ("image_id", "job_id"), "CASCADE") not in (
        run_foreign_keys
    )
    assert ("image_id", "job_id") not in image_unique_columns


def test_upgrade_enforces_scope_and_preserves_set_null_semantics(
    migration_database: tuple[Path, Config],
) -> None:
    database_path, config = migration_database
    command.upgrade(config, _OWNERSHIP_REVISION)
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        _seed_analysis_graph(connection)
        _insert_run(
            connection,
            run_id="parent_a",
            job_id="job_a",
            image_id="image_a",
        )
        _insert_run(
            connection,
            run_id="parent_b",
            job_id="job_b",
            image_id="image_b",
        )
        connection.commit()

    command.upgrade(config, _RELATIONSHIP_REVISION)

    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        assert ("image_id", "job_id") in _unique_column_sets(connection, "image_assets")
        assert ("run_id", "job_id") in _unique_column_sets(connection, "segmentation_runs")
        assert (
            "image_assets",
            ("image_id", "job_id"),
            ("image_id", "job_id"),
            "CASCADE",
        ) in _foreign_key_shapes(connection, "segmentation_runs")
        assert (
            "segmentation_runs",
            ("parent_run_id", "job_id"),
            ("run_id", "job_id"),
            "NO ACTION",
        ) in _foreign_key_shapes(connection, "segmentation_runs")
        assert (
            "image_assets",
            ("image_id", "job_id"),
            ("image_id", "job_id"),
            "NO ACTION",
        ) in _foreign_key_shapes(connection, "query_logs")

        _assert_integrity_error(
            connection,
            lambda: _insert_run(
                connection,
                run_id="run_wrong_image_job",
                job_id="job_a",
                image_id="image_b",
            ),
        )
        _assert_integrity_error(
            connection,
            lambda: _insert_query(
                connection,
                query_id="query_wrong_image_job",
                job_id="job_a",
                image_id="image_b",
            ),
        )
        _assert_integrity_error(
            connection,
            lambda: _insert_run(
                connection,
                run_id="child_wrong_parent",
                job_id="job_a",
                image_id="image_a",
                parent_run_id="parent_b",
            ),
        )

        _insert_run(
            connection,
            run_id="child_a",
            job_id="job_a",
            image_id="image_a",
            parent_run_id="parent_a",
        )
        _insert_query(
            connection,
            query_id="query_a",
            job_id="job_a",
            image_id="image_a",
        )
        connection.commit()

        connection.execute("DELETE FROM segmentation_runs WHERE run_id = 'parent_a'")
        assert connection.execute(
            "SELECT parent_run_id FROM segmentation_runs WHERE run_id = 'child_a'"
        ).fetchone() == (None,)

        connection.execute("DELETE FROM image_assets WHERE image_id = 'image_a'")
        assert connection.execute(
            "SELECT image_id, job_id FROM query_logs WHERE query_id = 'query_a'"
        ).fetchone() == (None, "job_a")
        assert connection.execute(
            "SELECT count(*) FROM segmentation_runs WHERE run_id = 'child_a'"
        ).fetchone() == (0,)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_downgrade_upgrade_roundtrip_preserves_data_and_original_foreign_key(
    migration_database: tuple[Path, Config],
) -> None:
    database_path, config = migration_database
    command.upgrade(config, _OWNERSHIP_REVISION)
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        _seed_analysis_graph(connection)
        _insert_run(
            connection,
            run_id="run_roundtrip",
            job_id="job_a",
            image_id="image_a",
        )
        _insert_query(
            connection,
            query_id="query_roundtrip",
            job_id="job_a",
            image_id="image_a",
        )
        connection.commit()

    command.upgrade(config, _RELATIONSHIP_REVISION)
    command.downgrade(config, _OWNERSHIP_REVISION)

    with sqlite3.connect(database_path) as connection:
        schema = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'segmentation_runs'"
        ).fetchone()[0]
        assert "fk_segmentation_runs_image_id_image_assets" in schema
        assert "fk_segmentation_runs_image_job" not in schema
        assert ("run_id", "job_id") not in _unique_column_sets(
            connection, "segmentation_runs"
        )
        assert connection.execute(
            "SELECT job_id, image_id FROM segmentation_runs WHERE run_id = 'run_roundtrip'"
        ).fetchone() == ("job_a", "image_a")
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []

    command.upgrade(config, _RELATIONSHIP_REVISION)
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT image_id FROM query_logs WHERE query_id = 'query_roundtrip'"
        ).fetchone() == ("image_a",)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_head_upgrade_has_no_model_metadata_drift(
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


def _seed_analysis_graph(connection: sqlite3.Connection) -> None:
    for suffix in ("a", "b"):
        _insert_job(connection, job_id=f"job_{suffix}")
        _insert_image(
            connection,
            image_id=f"image_{suffix}",
            job_id=f"job_{suffix}",
            suffix=suffix,
        )
    connection.execute(
        """
        INSERT INTO model_registry
            (model_id, family, variant, quality_tier, version, adapter, status,
             metadata_json, created_at, updated_at)
        VALUES ('model_test', 'sam', 'base', 'draft', '1', 'stub', 'READY',
                '{}', ?, ?)
        """,
        (_TIMESTAMP, _TIMESTAMP),
    )


def _insert_job(connection: sqlite3.Connection, *, job_id: str) -> None:
    connection.execute(
        """
        INSERT INTO analysis_jobs
            (job_id, tenant_id, owner_principal_id, name, status, config_json,
             created_at, updated_at)
        VALUES (?, ?, ?, ?, 'CREATED', '{}', ?, ?)
        """,
        (job_id, LEGACY_TENANT_ID, LEGACY_PRINCIPAL_ID, job_id, _TIMESTAMP, _TIMESTAMP),
    )


def _insert_image(
    connection: sqlite3.Connection,
    *,
    image_id: str,
    job_id: str,
    suffix: str,
) -> None:
    connection.execute(
        """
        INSERT INTO image_assets
            (image_id, job_id, filename, storage_path, sha256, width, height,
             bit_depth, sample_id, experiment_conditions_json, analysis_roi_json,
             box_revision, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, 10, 10, 8, ?, '{}', '{}', 0, ?, ?)
        """,
        (
            image_id,
            job_id,
            f"{suffix}.png",
            f"images/{suffix}.png",
            suffix * 64,
            f"sample-{suffix}",
            _TIMESTAMP,
            _TIMESTAMP,
        ),
    )


def _insert_run(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    job_id: str,
    image_id: str,
    parent_run_id: str | None = None,
) -> None:
    connection.execute(
        """
        INSERT INTO segmentation_runs
            (run_id, job_id, image_id, model_id, roi_mode, status, inference_json,
             run_config_json, paths_json, parent_run_id, created_at, updated_at)
        VALUES (?, ?, ?, 'model_test', 'FULL_IMAGE', 'CREATED', '{}', '{}', '{}', ?, ?, ?)
        """,
        (run_id, job_id, image_id, parent_run_id, _TIMESTAMP, _TIMESTAMP),
    )


def _insert_query(
    connection: sqlite3.Connection,
    *,
    query_id: str,
    job_id: str,
    image_id: str | None,
) -> None:
    connection.execute(
        """
        INSERT INTO query_logs
            (query_id, job_id, image_id, query_type, question, request_json,
             answer_json, created_at)
        VALUES (?, ?, ?, 'analysis', 'test?', '{}', '{}', ?)
        """,
        (query_id, job_id, image_id, _TIMESTAMP),
    )


def _assert_integrity_error(
    connection: sqlite3.Connection,
    write: Callable[[], object],
) -> None:
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        write()
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
                for row in sorted(rows, key=lambda row: cast(int, row[1]))
            ),
            tuple(
                str(row[4])
                for row in sorted(rows, key=lambda row: cast(int, row[1]))
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

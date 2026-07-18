from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from app.core.config import get_settings
from app.db.migration_state import expected_alembic_heads

INITIAL_REVISION = "53eaa43adc19"
INTERMEDIATE_REVISION = "c4d7a1e6f2b9"
LEGACY_TIMESTAMP = "2026-07-01 08:09:10.000000"


@pytest.fixture
def migration_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[Path, Config]]:
    database_path = tmp_path / "legacy-upgrade.db"
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path}")
    get_settings.cache_clear()
    config = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
    try:
        yield database_path, config
    finally:
        get_settings.cache_clear()


@pytest.mark.integration
@pytest.mark.parametrize(
    ("start_revision", "request_json", "seed_status_snapshot"),
    [
        (INITIAL_REVISION, None, False),
        (
            INTERMEDIATE_REVISION,
            {"question": "保留已有请求", "query_type": "knowledge"},
            True,
        ),
    ],
)
def test_legacy_schema_upgrade_preserves_honest_audit_state(
    migration_database: tuple[Path, Config],
    start_revision: str,
    request_json: dict[str, str] | None,
    seed_status_snapshot: bool,
) -> None:
    database_path, config = migration_database
    application_logger = logging.getLogger("app.agent.application")
    application_logger.disabled = False
    command.upgrade(config, start_revision)
    _seed_legacy_rows(
        database_path,
        request_json=request_json,
        seed_status_snapshot=seed_status_snapshot,
    )

    command.upgrade(config, "head")

    with sqlite3.connect(database_path) as connection:
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchall()
        status_events = connection.execute(
            """
            SELECT from_status, to_status, error_code, error_message, created_at
            FROM run_status_events WHERE run_id = 'run-legacy' ORDER BY event_id
            """
        ).fetchall()
        stored_request = connection.execute(
            "SELECT request_json FROM query_logs WHERE query_id = 'query-legacy'"
        ).fetchone()
        revisions = connection.execute(
            """
            SELECT revision, box_count
            FROM roi_box_revisions WHERE image_id = 'image-legacy' ORDER BY revision
            """
        ).fetchall()
        execution_json = connection.execute(
            "SELECT execution_json FROM segmentation_runs WHERE run_id = 'run-legacy'"
        ).fetchone()

    assert revision == [(expected_alembic_heads()[0],)]
    assert status_events == [(None, "ANALYZING", None, None, LEGACY_TIMESTAMP)]
    assert stored_request is not None
    assert json.loads(stored_request[0]) == (request_json or {})
    # Revision 2 cannot be reconstructed from the old row model and must not be invented. The
    # current empty revision 3 is nevertheless known from image_assets.box_revision and retained.
    assert revisions == [(0, 0), (1, 1), (3, 0)]
    assert execution_json == (None,)
    assert application_logger.disabled is False


def _seed_legacy_rows(
    database_path: Path,
    *,
    request_json: dict[str, str] | None,
    seed_status_snapshot: bool,
) -> None:
    json_empty = json.dumps({})
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO analysis_jobs (
                job_id, name, status, config_json, error_code, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job-legacy",
                "legacy migration fixture",
                "ANALYZING",
                json_empty,
                None,
                LEGACY_TIMESTAMP,
                LEGACY_TIMESTAMP,
            ),
        )
        connection.execute(
            """
            INSERT INTO model_registry (
                model_id, family, variant, quality_tier, version, adapter,
                weight_path, config_path, model_card_path, status, metadata_json,
                health_error, weight_sha256, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "model-legacy",
                "unet",
                "general",
                "balanced",
                "legacy",
                "tests.fake:FakeAdapter",
                None,
                None,
                None,
                "ready",
                json_empty,
                None,
                None,
                LEGACY_TIMESTAMP,
                LEGACY_TIMESTAMP,
            ),
        )
        connection.execute(
            """
            INSERT INTO image_assets (
                image_id, job_id, filename, storage_path, sha256, width, height,
                bit_depth, sample_id, material_name, material_formula,
                experiment_conditions_json, analysis_roi_json, scale_nm_per_pixel,
                box_revision, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "image-legacy",
                "job-legacy",
                "legacy.tif",
                "job-legacy/input/image-legacy/original.tif",
                "a" * 64,
                64,
                48,
                16,
                "sample-legacy",
                None,
                None,
                json_empty,
                json_empty,
                None,
                3,
                LEGACY_TIMESTAMP,
                LEGACY_TIMESTAMP,
            ),
        )
        connection.execute(
            """
            INSERT INTO segmentation_runs (
                run_id, job_id, image_id, model_id, roi_mode, box_revision,
                threshold, status, inference_json, run_config_json, paths_json,
                runtime_ms, parent_run_id, error_code, error_message, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "run-legacy",
                "job-legacy",
                "image-legacy",
                "model-legacy",
                "boxes",
                3,
                0.5,
                "ANALYZING",
                json_empty,
                json_empty,
                json_empty,
                None,
                None,
                None,
                None,
                LEGACY_TIMESTAMP,
                LEGACY_TIMESTAMP,
            ),
        )
        connection.execute(
            """
            INSERT INTO roi_boxes (
                box_id, image_id, x1, y1, x2, y2, label, active,
                revision, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "box-revision-1",
                "image-legacy",
                1,
                2,
                20,
                22,
                "legacy ROI",
                1,
                1,
                LEGACY_TIMESTAMP,
                LEGACY_TIMESTAMP,
            ),
        )

        query_columns = "query_id, job_id, image_id, query_type, question, answer_json, created_at"
        query_values: tuple[object, ...] = (
            "query-legacy",
            "job-legacy",
            "image-legacy",
            "knowledge",
            "legacy question",
            json.dumps({"answer": "legacy"}),
            LEGACY_TIMESTAMP,
        )
        if request_json is not None:
            query_columns += ", request_json"
            query_values += (json.dumps(request_json, ensure_ascii=False),)
        placeholders = ", ".join("?" for _ in query_values)
        connection.execute(
            f"INSERT INTO query_logs ({query_columns}) VALUES ({placeholders})",
            query_values,
        )

        if seed_status_snapshot:
            connection.execute(
                """
                INSERT INTO run_status_events (
                    run_id, from_status, to_status, error_code, error_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("run-legacy", None, "ANALYZING", None, None, LEGACY_TIMESTAMP),
            )

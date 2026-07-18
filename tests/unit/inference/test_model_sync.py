from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from app.contracts.enums import ModelStatus
from app.core.config import Settings
from app.db.base import Base
from app.db.models import ModelRegistryRecord
from app.db.session import Database
from app.inference import model_sync
from app.inference.model_sync import (
    ModelRegistryTableMissingError,
    sync_model_registry,
)
from app.inference.registry import ModelRegistration, ModelRegistryService
from tests.unit.inference.helpers import build_registry, model_entry


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    instance = Database(Settings(database_url=f"sqlite:///{tmp_path / 'sync.db'}"))
    Base.metadata.create_all(instance.engine)
    try:
        yield instance
    finally:
        instance.dispose()


def test_sync_inserts_full_registration_and_is_idempotent(
    database: Database,
    tmp_path: Path,
) -> None:
    entry = model_entry(tmp_path, "ready-model")
    registry = build_registry(tmp_path, [entry])

    with database.session() as session:
        first = sync_model_registry(session, registry)
        assert first.inserted == ("ready-model",)
        assert first.changed_count == 1
        assert first.source_error is None

    with database.session() as session:
        record = session.get(ModelRegistryRecord, "ready-model")
        assert record is not None
        assert record.status == ModelStatus.READY.value
        assert record.adapter == entry["adapter_path"]
        assert record.weight_path == str((tmp_path / entry["weight_path"]).resolve())
        assert record.config_path == str((tmp_path / entry["config_path"]).resolve())
        assert record.model_card_path == str((tmp_path / entry["model_card_path"]).resolve())
        assert record.weight_sha256 == entry["weight_sha256"]
        assert record.metadata_json["model_id"] == "ready-model"
        assert record.metadata_json["status"] == ModelStatus.READY.value

        second = sync_model_registry(session, registry)
        assert second.inserted == ()
        assert second.updated == ()
        assert second.unchanged == ("ready-model",)
        assert second.marked_unavailable == ()
        assert second.changed_count == 0


def test_sync_updates_metadata_and_runtime_status(
    database: Database,
    tmp_path: Path,
) -> None:
    initial_entry = model_entry(tmp_path, "changing-model")
    initial = build_registry(tmp_path, [initial_entry])
    with database.session() as session:
        sync_model_registry(session, initial)

    changed_entry = model_entry(tmp_path, "changing-model", status="disabled")
    changed_entry["metadata"]["version"] = "test-2"
    changed_entry["metadata"]["notes"] = "disabled by operator"
    changed = build_registry(tmp_path, [changed_entry])

    with database.session() as session:
        result = sync_model_registry(session, changed)
        assert result.updated == ("changing-model",)

    with database.session() as session:
        record = session.get(ModelRegistryRecord, "changing-model")
        assert record is not None
        assert record.version == "test-2"
        assert record.status == ModelStatus.DISABLED.value
        assert record.metadata_json["notes"] == "disabled by operator"
        assert record.metadata_json["status"] == ModelStatus.DISABLED.value


def test_removed_declaration_is_retained_but_marked_unavailable(
    database: Database,
    tmp_path: Path,
) -> None:
    registry = build_registry(tmp_path, [model_entry(tmp_path, "removed-model")])
    with database.session() as session:
        sync_model_registry(session, registry)

    empty_registry = build_registry(tmp_path, [])
    with database.session() as session:
        first = sync_model_registry(session, empty_registry)
        assert first.marked_unavailable == ("removed-model",)

    with database.session() as session:
        record = session.get(ModelRegistryRecord, "removed-model")
        assert record is not None
        assert record.status == ModelStatus.UNAVAILABLE.value
        assert "absent from the YAML registry" in (record.health_error or "")
        assert record.metadata_json["status"] == ModelStatus.UNAVAILABLE.value

        second = sync_model_registry(session, empty_registry)
        assert second.marked_unavailable == ()
        assert second.unchanged == ("removed-model",)
        assert second.changed_count == 0


def test_invalid_registry_fails_closed_without_deleting_audit_rows(
    database: Database,
    tmp_path: Path,
) -> None:
    registry = build_registry(tmp_path, [model_entry(tmp_path, "audit-model")])
    with database.session() as session:
        sync_model_registry(session, registry)

    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text("models: not-a-list\n", encoding="utf-8")
    invalid_registry = ModelRegistryService(registry_path)
    assert invalid_registry.registry_error is not None

    with database.session() as session:
        result = sync_model_registry(session, invalid_registry)
        assert result.source_error == invalid_registry.registry_error
        assert result.marked_unavailable == ("audit-model",)

    with database.session() as session:
        record = session.get(ModelRegistryRecord, "audit-model")
        assert record is not None
        assert record.status == ModelStatus.UNAVAILABLE.value
        assert record.health_error == "YAML model registry is unavailable or invalid"


def test_missing_table_raises_structured_error_and_never_creates_schema(
    tmp_path: Path,
) -> None:
    database = Database(Settings(database_url=f"sqlite:///{tmp_path / 'unmigrated.db'}"))
    registry = build_registry(tmp_path, [model_entry(tmp_path, "model")])
    try:
        with database.session() as session, pytest.raises(
            ModelRegistryTableMissingError
        ) as exc_info:
            sync_model_registry(session, registry)

        assert exc_info.value.as_dict() == {
            "code": "MODEL_REGISTRY_TABLE_MISSING",
            "message": "database migrations have not created the model registry table",
            "details": {
                "table": "model_registry",
                "action": "run database migrations",
            },
        }
        assert inspect(database.engine).get_table_names() == []
    finally:
        database.dispose()


def test_reconciliation_rolls_back_partial_work_on_failure(
    database: Database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = build_registry(
        tmp_path,
        [model_entry(tmp_path, "model-a"), model_entry(tmp_path, "model-b")],
    )
    original = model_sync._registration_values
    calls = 0

    def fail_on_second(registration: ModelRegistration) -> model_sync._RecordValues:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected reconciliation failure")
        return original(registration)

    monkeypatch.setattr(model_sync, "_registration_values", fail_on_second)

    session: Session = database.session_factory()
    try:
        with pytest.raises(RuntimeError, match="injected reconciliation failure"):
            sync_model_registry(session, registry)
        assert session.scalars(select(ModelRegistryRecord)).all() == []
        assert session.is_active
    finally:
        session.close()

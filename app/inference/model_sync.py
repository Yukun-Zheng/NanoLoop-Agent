"""Transactional reconciliation from the YAML model registry into SQL metadata rows.

The YAML registry remains the readiness source of truth. Database rows are retained when a
declaration disappears because segmentation runs reference them, but such rows are marked
unavailable so they cannot masquerade as runnable models.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from sqlalchemy import inspect, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.contracts.enums import ModelStatus
from app.db.models import ModelRegistryRecord
from app.inference.registry import ModelRegistration, ModelRegistryService

_REMOVED_REASON = "model declaration is absent from the YAML registry"
_SOURCE_ERROR_REASON = "YAML model registry is unavailable or invalid"


class ModelRegistrySyncError(RuntimeError):
    """Base error with stable machine-readable fields for startup/CLI callers."""

    code: ClassVar[str] = "MODEL_REGISTRY_SYNC_FAILED"

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = dict(details or {})

    def as_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "details": self.details}


class ModelRegistryTableMissingError(ModelRegistrySyncError):
    """Raised when migrations have not created the required projection table."""

    code = "MODEL_REGISTRY_TABLE_MISSING"


class ModelRegistryPersistenceError(ModelRegistrySyncError):
    """Raised when database inspection or reconciliation fails."""

    code = "MODEL_REGISTRY_PERSISTENCE_FAILED"


@dataclass(frozen=True, slots=True)
class ModelRegistrySyncResult:
    """Stable reconciliation summary; tuple ordering is deterministic by model ID."""

    inserted: tuple[str, ...]
    updated: tuple[str, ...]
    unchanged: tuple[str, ...]
    marked_unavailable: tuple[str, ...]
    source_error: str | None = None

    @property
    def changed_count(self) -> int:
        return len(self.inserted) + len(self.updated) + len(self.marked_unavailable)

    def as_dict(self) -> dict[str, object]:
        return {
            "inserted": list(self.inserted),
            "updated": list(self.updated),
            "unchanged": list(self.unchanged),
            "marked_unavailable": list(self.marked_unavailable),
            "source_error": self.source_error,
            "changed_count": self.changed_count,
        }


@dataclass(frozen=True, slots=True)
class _RecordValues:
    family: str
    variant: str
    quality_tier: str
    version: str
    adapter: str
    weight_path: str
    config_path: str
    model_card_path: str
    status: str
    metadata_json: dict[str, Any]
    health_error: str | None
    weight_sha256: str | None


def sync_model_registry(
    session: Session,
    registry: ModelRegistryService,
) -> ModelRegistrySyncResult:
    """Reconcile one registry snapshot without committing the caller's transaction.

    Call this with a short-lived session after migrations have run. A nested transaction keeps
    partial inserts or updates from leaking if reconciliation fails. The caller remains responsible
    for committing the surrounding transaction, which makes startup and CLI composition explicit.

    A malformed or unreadable YAML source is fail-closed: existing rows are retained for run audit
    foreign keys but are marked unavailable, and ``source_error`` reports the registry failure.
    """

    _require_registry_table(session)
    source_error = registry.registry_error
    registrations = (
        []
        if source_error is not None
        else [
            registry.get_registration(metadata.model_id)
            for metadata in registry.list_models()
        ]
    )

    inserted: list[str] = []
    updated: list[str] = []
    unchanged: list[str] = []
    marked_unavailable: list[str] = []
    current_ids = {registration.metadata.model_id for registration in registrations}

    try:
        with session.begin_nested():
            persisted = {
                record.model_id: record
                for record in session.scalars(
                    select(ModelRegistryRecord).order_by(ModelRegistryRecord.model_id)
                ).all()
            }

            for registration in registrations:
                model_id = registration.metadata.model_id
                values = _registration_values(registration)
                record = persisted.get(model_id)
                if record is None:
                    session.add(_new_record(model_id, values))
                    inserted.append(model_id)
                elif _apply_values(record, values):
                    updated.append(model_id)
                else:
                    unchanged.append(model_id)

            stale_reason = _SOURCE_ERROR_REASON if source_error is not None else _REMOVED_REASON
            for model_id in sorted(set(persisted) - current_ids):
                record = persisted[model_id]
                if _mark_unavailable(record, stale_reason):
                    marked_unavailable.append(model_id)
                else:
                    unchanged.append(model_id)

            session.flush()
    except SQLAlchemyError as error:
        raise ModelRegistryPersistenceError(
            "model registry reconciliation failed",
            details={"error_type": type(error).__name__},
        ) from error

    return ModelRegistrySyncResult(
        inserted=tuple(sorted(inserted)),
        updated=tuple(sorted(updated)),
        unchanged=tuple(sorted(unchanged)),
        marked_unavailable=tuple(sorted(marked_unavailable)),
        source_error=source_error,
    )


def _require_registry_table(session: Session) -> None:
    table_name = ModelRegistryRecord.__tablename__
    try:
        table_exists = inspect(session.get_bind()).has_table(table_name)
    except SQLAlchemyError as error:
        raise ModelRegistryPersistenceError(
            "unable to inspect the model registry table",
            details={"table": table_name, "error_type": type(error).__name__},
        ) from error
    if not table_exists:
        raise ModelRegistryTableMissingError(
            "database migrations have not created the model registry table",
            details={"table": table_name, "action": "run database migrations"},
        )


def _registration_values(registration: ModelRegistration) -> _RecordValues:
    metadata = registration.metadata
    return _RecordValues(
        family=metadata.family.value,
        variant=metadata.variant.value,
        quality_tier=metadata.quality_tier.value,
        version=metadata.version,
        adapter=registration.adapter_path,
        weight_path=str(registration.weight_path),
        config_path=str(registration.config_path),
        model_card_path=str(registration.model_card_path),
        status=metadata.status.value,
        metadata_json=metadata.model_dump(mode="json"),
        health_error=metadata.health_error,
        weight_sha256=registration.weight_sha256,
    )


def _new_record(model_id: str, values: _RecordValues) -> ModelRegistryRecord:
    return ModelRegistryRecord(
        model_id=model_id,
        family=values.family,
        variant=values.variant,
        quality_tier=values.quality_tier,
        version=values.version,
        adapter=values.adapter,
        weight_path=values.weight_path,
        config_path=values.config_path,
        model_card_path=values.model_card_path,
        status=values.status,
        metadata_json=values.metadata_json,
        health_error=values.health_error,
        weight_sha256=values.weight_sha256,
    )


def _apply_values(record: ModelRegistryRecord, values: _RecordValues) -> bool:
    changed = False
    for field_name in _RecordValues.__dataclass_fields__:
        expected = getattr(values, field_name)
        if getattr(record, field_name) != expected:
            setattr(record, field_name, expected)
            changed = True
    return changed


def _mark_unavailable(record: ModelRegistryRecord, reason: str) -> bool:
    metadata_json = dict(record.metadata_json)
    metadata_json["status"] = ModelStatus.UNAVAILABLE.value
    metadata_json["health_error"] = reason
    if (
        record.status == ModelStatus.UNAVAILABLE.value
        and record.health_error == reason
        and record.metadata_json == metadata_json
    ):
        return False
    record.status = ModelStatus.UNAVAILABLE.value
    record.health_error = reason
    record.metadata_json = metadata_json
    return True

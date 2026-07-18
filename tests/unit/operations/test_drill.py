from __future__ import annotations

import json
import os
import sqlite3
import stat
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

import app.operations.drill as drill
from app.db.migration_state import expected_alembic_heads
from app.operations.backup import (
    BackupComponent,
    BackupFileRecord,
    BackupLayout,
    BackupManifest,
    BackupResult,
    BackupVerificationResult,
    RestoreResult,
)
from app.operations.drill import (
    DrillValidationError,
    OfflineFilesystemRestoreDrillReport,
    run_offline_filesystem_restore_drill,
)

_SNAPSHOT_AT = datetime(2026, 7, 18, 2, 3, 4, tzinfo=UTC)


def _state_layout(tmp_path: Path) -> BackupLayout:
    data = tmp_path / "source" / "data"
    outputs = tmp_path / "source" / "outputs"
    models = data / "model-snapshots"
    sources = tmp_path / "source" / "knowledge" / "sources"
    index = tmp_path / "source" / "knowledge" / "index"
    for directory in (data, outputs, models, sources, index):
        directory.mkdir(parents=True, exist_ok=True)

    database = data / "nanoloop.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        connection.execute(
            "INSERT INTO alembic_version (version_num) VALUES (?)",
            (expected_alembic_heads()[0],),
        )
        connection.execute("CREATE TABLE sentinel (value TEXT NOT NULL)")
        connection.execute("INSERT INTO sentinel (value) VALUES ('database-state')")
    (data / "runtime.json").write_text('{"ready": true}\n', encoding="utf-8")
    (outputs / "result.bin").write_bytes(b"result-state")
    (models / "model.bin").write_bytes(b"model-state")
    (sources / "source.md").write_text("knowledge-state\n", encoding="utf-8")
    (index / "index.bin").write_bytes(b"index-state")
    token = data / ".file_token_secret"
    token.write_text("secret-value-that-must-never-appear-in-a-report", encoding="utf-8")
    os.chmod(token, 0o600)
    return BackupLayout(
        database_path=database,
        data_root=data,
        output_root=outputs,
        model_snapshot_root=models,
        knowledge_source_root=sources,
        knowledge_index_root=index,
        file_token_secret_file=token,
    )


def _manifest() -> BackupManifest:
    records = (
        BackupFileRecord(
            path="data/nanoloop.db",
            component=BackupComponent.DATABASE,
            size=10,
            sha256="1" * 64,
            mode=0o600,
            mtime_ns=1,
        ),
        BackupFileRecord(
            path="data/runtime.json",
            component=BackupComponent.RUNTIME_DATA,
            size=20,
            sha256="2" * 64,
            mode=0o640,
            mtime_ns=2,
        ),
        BackupFileRecord(
            path="data/model-snapshots/model.bin",
            component=BackupComponent.MODEL_SNAPSHOTS,
            size=30,
            sha256="3" * 64,
            mode=0o440,
            mtime_ns=3,
        ),
        BackupFileRecord(
            path="outputs/result.bin",
            component=BackupComponent.OUTPUTS,
            size=40,
            sha256="4" * 64,
            mode=0o440,
            mtime_ns=4,
        ),
        BackupFileRecord(
            path="knowledge_base/sources/source.md",
            component=BackupComponent.KNOWLEDGE_SOURCES,
            size=50,
            sha256="5" * 64,
            mode=0o440,
            mtime_ns=5,
        ),
        BackupFileRecord(
            path="knowledge_base/index/index.bin",
            component=BackupComponent.KNOWLEDGE_INDEX,
            size=60,
            sha256="6" * 64,
            mode=0o440,
            mtime_ns=6,
        ),
    )
    return BackupManifest(
        created_at=_SNAPSHOT_AT,
        database_revision=expected_alembic_heads()[0],
        components=tuple(BackupComponent),
        files=records,
    )


def _fake_results(tmp_path: Path) -> tuple[BackupResult, BackupVerificationResult, RestoreResult]:
    manifest = _manifest()
    archive = tmp_path / "drill.zip"
    checksum = Path(f"{archive}.sha256")
    digest = "a" * 64
    return (
        BackupResult(
            archive_path=archive,
            checksum_path=checksum,
            archive_sha256=digest,
            manifest=manifest,
        ),
        BackupVerificationResult(
            archive_path=archive,
            checksum_path=checksum,
            archive_sha256=digest,
            manifest=manifest,
        ),
        RestoreResult(
            destination_root=tmp_path / "restored",
            archive_sha256=digest,
            manifest=manifest,
        ),
    )


def _sequence(values: list[object]) -> Callable[[], object]:
    iterator: Iterator[object] = iter(values)
    return lambda: next(iterator)


def test_real_drill_publishes_strict_private_report_without_sensitive_values(
    tmp_path: Path,
) -> None:
    layout = _state_layout(tmp_path)
    archive = tmp_path / "backup.zip"
    destination = tmp_path / "restored"
    report_path = tmp_path / "drill-report.json"

    result = run_offline_filesystem_restore_drill(
        layout,
        archive,
        destination,
        report_path,
        offline_confirmed=True,
    )

    assert result.report_path == report_path
    assert stat.S_IMODE(report_path.stat().st_mode) == 0o600
    parsed = OfflineFilesystemRestoreDrillReport.model_validate_json(report_path.read_bytes())
    assert parsed == result.report
    assert parsed.status == "success"
    assert parsed.scope == "offline_filesystem_restore"
    assert parsed.application_startup_verified is False
    assert parsed.rpo == parsed.rto == "not_measured"
    assert parsed.counts.component_count == len(BackupComponent)
    assert parsed.counts.file_count >= len(BackupComponent)
    assert parsed.counts.files_by_component[BackupComponent.DATABASE] == 1
    serialized = report_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized
    assert "secret-value-that-must-never-appear-in-a-report" not in serialized
    assert "file_token_secret" not in serialized
    assert "nanoloop.db" not in serialized


def test_injected_clocks_make_report_timestamps_and_durations_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created, verified, restored = _fake_results(tmp_path)
    monkeypatch.setattr(drill, "create_backup", lambda *_args, **_kwargs: created)
    monkeypatch.setattr(drill, "verify_backup", lambda *_args, **_kwargs: verified)
    monkeypatch.setattr(drill, "restore_backup", lambda *_args, **_kwargs: restored)
    started = datetime(2026, 7, 18, 2, 3, tzinfo=UTC)
    completed = started + timedelta(seconds=9)

    result = run_offline_filesystem_restore_drill(
        _state_layout(tmp_path),
        tmp_path / "drill.zip",
        tmp_path / "restored",
        tmp_path / "report.json",
        offline_confirmed=True,
        clock=_sequence([started, completed]),  # type: ignore[arg-type]
        monotonic=_sequence([100.0, 101.25, 101.75, 104.0]),  # type: ignore[arg-type]
    )

    report = result.report
    assert report.started_at == started
    assert report.snapshot_at == _SNAPSHOT_AT
    assert report.completed_at == completed
    assert report.durations.model_dump() == {
        "backup_seconds": 1.25,
        "verification_seconds": 0.5,
        "filesystem_restore_seconds": 2.25,
        "total_seconds": 4.0,
    }
    assert report.counts.file_count == 6
    assert report.counts.total_bytes == 210
    assert set(report.counts.files_by_component.values()) == {1}


@pytest.mark.parametrize("existing_target", ["archive", "checksum", "restore", "report"])
def test_all_drill_targets_are_new_only_before_any_operation_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_target: str,
) -> None:
    archive = tmp_path / "drill.zip"
    checksum = Path(f"{archive}.sha256")
    restore = tmp_path / "restored"
    report = tmp_path / "report.json"
    target = {
        "archive": archive,
        "checksum": checksum,
        "restore": restore,
        "report": report,
    }[existing_target]
    if existing_target == "restore":
        target.mkdir()
    else:
        target.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(
        drill,
        "create_backup",
        lambda *_args, **_kwargs: pytest.fail("no operation should run after preflight failure"),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        run_offline_filesystem_restore_drill(
            _state_layout(tmp_path),
            archive,
            restore,
            report,
            offline_confirmed=True,
        )

    if existing_target == "report":
        assert report.read_text(encoding="utf-8") == "keep"
    else:
        assert not report.exists()


@pytest.mark.parametrize("failing_stage", ["create", "verify", "restore"])
def test_failed_stage_never_publishes_a_success_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_stage: str,
) -> None:
    created, verified, restored = _fake_results(tmp_path)

    def fail() -> None:
        raise RuntimeError("sensitive failure at /private/state with secret-token")

    monkeypatch.setattr(
        drill,
        "create_backup",
        (lambda *_args, **_kwargs: fail())
        if failing_stage == "create"
        else (lambda *_args, **_kwargs: created),
    )
    monkeypatch.setattr(
        drill,
        "verify_backup",
        (lambda *_args, **_kwargs: fail())
        if failing_stage == "verify"
        else (lambda *_args, **_kwargs: verified),
    )
    monkeypatch.setattr(
        drill,
        "restore_backup",
        (lambda *_args, **_kwargs: fail())
        if failing_stage == "restore"
        else (lambda *_args, **_kwargs: restored),
    )
    report = tmp_path / "report.json"

    with pytest.raises(RuntimeError, match="sensitive failure"):
        run_offline_filesystem_restore_drill(
            _state_layout(tmp_path),
            tmp_path / "drill.zip",
            tmp_path / "restored",
            report,
            offline_confirmed=True,
        )

    assert not report.exists()


def test_report_schema_rejects_extra_fields_and_false_claims(tmp_path: Path) -> None:
    created, verified, restored = _fake_results(tmp_path)
    report = OfflineFilesystemRestoreDrillReport(
        started_at=_SNAPSHOT_AT,
        snapshot_at=_SNAPSHOT_AT,
        completed_at=_SNAPSHOT_AT,
        durations={
            "backup_seconds": 1.0,
            "verification_seconds": 1.0,
            "filesystem_restore_seconds": 1.0,
            "total_seconds": 3.0,
        },
        archive_sha256=verified.archive_sha256,
        database_revision=created.manifest.database_revision,
        counts={
            "file_count": 6,
            "total_bytes": 210,
            "component_count": 6,
            "files_by_component": {component: 1 for component in BackupComponent},
            "bytes_by_component": {
                component: record.size
                for component, record in zip(BackupComponent, restored.manifest.files, strict=True)
            },
        },
    )
    payload = json.loads(report.model_dump_json())
    payload["application_startup_verified"] = True
    payload["rpo"] = "1 minute"
    payload["host_path"] = str(tmp_path)

    with pytest.raises(ValidationError):
        OfflineFilesystemRestoreDrillReport.model_validate(payload)


def test_report_rejects_timestamps_outside_started_snapshot_completed_order() -> None:
    manifest = _manifest()
    files_by_component = {component: 0 for component in BackupComponent}
    bytes_by_component = {component: 0 for component in BackupComponent}
    for record in manifest.files:
        files_by_component[record.component] += 1
        bytes_by_component[record.component] += record.size

    with pytest.raises(ValidationError, match="started_at <= snapshot_at <= completed_at"):
        OfflineFilesystemRestoreDrillReport(
            started_at=_SNAPSHOT_AT + timedelta(seconds=1),
            snapshot_at=_SNAPSHOT_AT,
            completed_at=_SNAPSHOT_AT + timedelta(seconds=2),
            durations={
                "backup_seconds": 1.0,
                "verification_seconds": 0.0,
                "filesystem_restore_seconds": 1.0,
                "total_seconds": 2.0,
            },
            archive_sha256="a" * 64,
            database_revision=manifest.database_revision,
            counts={
                "file_count": len(manifest.files),
                "total_bytes": sum(record.size for record in manifest.files),
                "component_count": len(BackupComponent),
                "files_by_component": files_by_component,
                "bytes_by_component": bytes_by_component,
            },
        )


def test_offline_confirmation_is_exactly_required(tmp_path: Path) -> None:
    with pytest.raises(DrillValidationError, match="offline_confirmed"):
        run_offline_filesystem_restore_drill(
            _state_layout(tmp_path),
            tmp_path / "drill.zip",
            tmp_path / "restored",
            tmp_path / "report.json",
            offline_confirmed=False,
        )

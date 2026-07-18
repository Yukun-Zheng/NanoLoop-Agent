from __future__ import annotations

import os
import shutil
import sqlite3
import stat
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from app.db.migration_state import expected_alembic_heads
from app.operations.backup import (
    BackupComponent,
    BackupFileRecord,
    BackupLayout,
    BackupPreconditionError,
    BackupValidationError,
    StateDirectoryLock,
    create_backup,
    restore_backup,
    verify_backup,
)


@dataclass(frozen=True)
class _StateTree:
    layout: BackupLayout
    token: Path
    readonly_output: Path


def _make_state_tree(tmp_path: Path) -> _StateTree:
    data_root = tmp_path / "state" / "data"
    output_root = tmp_path / "state" / "outputs"
    model_root = data_root / "model-snapshots"
    source_root = tmp_path / "state" / "knowledge" / "sources"
    index_root = tmp_path / "state" / "knowledge" / "index"
    for directory in (data_root, output_root, model_root, source_root, index_root):
        directory.mkdir(parents=True, exist_ok=True)

    database = data_root / "nanoloop.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"
        )
        connection.execute(
            "INSERT INTO alembic_version (version_num) VALUES (?)",
            (expected_alembic_heads()[0],),
        )
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
        connection.execute(
            "CREATE TABLE child ("
            "id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parent(id))"
        )
        connection.execute("INSERT INTO parent (id) VALUES (1)")
        connection.execute("INSERT INTO child (id, parent_id) VALUES (1, 1)")

    (data_root / "runtime.json").write_text('{"ready": true}\n', encoding="utf-8")
    (data_root / "unrelated.db-wal").write_bytes(b"ordinary-runtime-data")
    ignored = data_root / "tmp" / "ignored.tmp"
    ignored.parent.mkdir()
    ignored.write_text("ephemeral", encoding="utf-8")
    (model_root / "模型.pt").write_bytes(b"model-snapshot")
    (source_root / "知识源.md").write_text("# 纳米颗粒\n", encoding="utf-8")
    (index_root / ".faiss-index").write_bytes(b"index")

    readonly_output = output_root / "样本 α" / ".只读结果"
    readonly_output.parent.mkdir()
    readonly_output.write_bytes(b"immutable-result")
    os.chmod(readonly_output, 0o444)
    os.utime(readonly_output, ns=(1_700_000_000_123_456_789,) * 2)

    token = data_root / ".file_token_secret"
    token.write_bytes(b"persisted-file-token-secret")
    os.chmod(token, 0o600)
    return _StateTree(
        layout=BackupLayout(
            database_path=database,
            data_root=data_root,
            output_root=output_root,
            model_snapshot_root=model_root,
            knowledge_source_root=source_root,
            knowledge_index_root=index_root,
            file_token_secret_file=None,
        ),
        token=token,
        readonly_output=readonly_output,
    )


def test_create_verify_restore_roundtrip_preserves_unicode_hidden_readonly_and_token(
    tmp_path: Path,
) -> None:
    state = _make_state_tree(tmp_path)
    archive = tmp_path / "state.zip"

    created = create_backup(state.layout, archive, offline_confirmed=True)
    verified = verify_backup(archive)
    destination = tmp_path / "restored"
    restored = restore_backup(archive, destination, offline_confirmed=True)

    assert created.archive_sha256 == verified.archive_sha256 == restored.archive_sha256
    assert stat.S_IMODE(archive.stat().st_mode) == 0o600
    assert stat.S_IMODE(created.checksum_path.stat().st_mode) == 0o600
    with zipfile.ZipFile(archive) as bundle:
        assert all(info.compress_type == zipfile.ZIP_STORED for info in bundle.infolist())
    assert (destination / "outputs" / "样本 α" / ".只读结果").read_bytes() == (
        b"immutable-result"
    )
    restored_readonly = destination / "outputs" / "样本 α" / ".只读结果"
    assert stat.S_IMODE(restored_readonly.stat().st_mode) == 0o444
    assert restored_readonly.stat().st_mtime_ns == state.readonly_output.stat().st_mtime_ns
    assert (destination / "data" / "model-snapshots" / "模型.pt").read_bytes() == (
        b"model-snapshot"
    )
    assert (destination / "knowledge_base" / "sources" / "知识源.md").exists()
    restored_token = destination / "data" / ".file_token_secret"
    assert restored_token.read_bytes() == state.token.read_bytes()
    assert stat.S_IMODE(restored_token.stat().st_mode) == 0o600
    assert not (destination / "data" / "tmp" / "ignored.tmp").exists()
    assert not (destination / "data" / StateDirectoryLock.filename).exists()
    assert (destination / "data" / "unrelated.db-wal").read_bytes() == (
        b"ordinary-runtime-data"
    )
    with sqlite3.connect(destination / "data" / "nanoloop.db") as connection:
        assert connection.execute("SELECT parent_id FROM child").fetchall() == [(1,)]


def test_backup_accepts_committed_wal_without_ephemeral_shm(
    tmp_path: Path,
) -> None:
    """A stopped container can leave durable WAL pages while SQLite recreates ``-shm``."""

    state = _make_state_tree(tmp_path)
    live_database = tmp_path / "live.db"
    writer = sqlite3.connect(live_database)
    try:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        writer.execute(
            "INSERT INTO alembic_version (version_num) VALUES (?)",
            (expected_alembic_heads()[0],),
        )
        writer.execute("CREATE TABLE committed_payload (value TEXT NOT NULL)")
        writer.executemany(
            "INSERT INTO committed_payload (value) VALUES (?)",
            [(f"row-{index}",) for index in range(100)],
        )
        writer.commit()
        shutil.copy2(live_database, state.layout.database_path)
        shutil.copy2(
            Path(f"{live_database}-wal"),
            Path(f"{state.layout.database_path}-wal"),
        )
    finally:
        writer.close()

    source_shm = Path(f"{state.layout.database_path}-shm")
    source_shm.unlink(missing_ok=True)
    archive = tmp_path / "wal-state.zip"
    destination = tmp_path / "wal-restored"

    create_backup(state.layout, archive, offline_confirmed=True)
    restore_backup(archive, destination, offline_confirmed=True)

    with sqlite3.connect(destination / "data" / "nanoloop.db") as connection:
        assert connection.execute("SELECT COUNT(*) FROM committed_payload").fetchone() == (100,)


@pytest.mark.parametrize("mutation", ["sidecar", "tamper", "truncate"])
def test_verify_rejects_archive_bytes_that_do_not_match_checksum(
    tmp_path: Path,
    mutation: str,
) -> None:
    state = _make_state_tree(tmp_path)
    archive = tmp_path / "state.zip"
    created = create_backup(state.layout, archive, offline_confirmed=True)
    content = bytearray(archive.read_bytes())
    if mutation == "sidecar":
        created.checksum_path.write_text(f"{'0' * 64}\n", encoding="ascii")
    elif mutation == "tamper":
        content[len(content) // 2] ^= 0x01
        archive.write_bytes(content)
    else:
        del content[-32:]
        archive.write_bytes(content)

    with pytest.raises(BackupValidationError, match="SHA-256"):
        verify_backup(archive)


def test_restore_refuses_existing_destination_without_modifying_it(tmp_path: Path) -> None:
    state = _make_state_tree(tmp_path)
    archive = tmp_path / "state.zip"
    create_backup(state.layout, archive, offline_confirmed=True)
    destination = tmp_path / "restored"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("untouched", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        restore_backup(archive, destination, offline_confirmed=True)

    assert sentinel.read_text(encoding="utf-8") == "untouched"


def test_shared_state_lock_blocks_backup_exclusive_lock(tmp_path: Path) -> None:
    state = _make_state_tree(tmp_path)
    archive = tmp_path / "state.zip"
    with StateDirectoryLock(state.layout.data_root, exclusive=False):
        with StateDirectoryLock(state.layout.data_root, exclusive=False):
            pass
        with pytest.raises(BackupPreconditionError, match="locked by another process"):
            create_backup(state.layout, archive, offline_confirmed=True)

    assert not archive.exists()
    assert not Path(f"{archive}.sha256").exists()
    lock_path = state.layout.data_root / StateDirectoryLock.filename
    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600


def test_runtime_file_cannot_collide_with_canonical_database_member(tmp_path: Path) -> None:
    state = _make_state_tree(tmp_path)
    custom_database = state.layout.data_root / "custom.db"
    state.layout.database_path.rename(custom_database)
    state.layout.database_path.write_bytes(b"must-not-be-silently-omitted")
    layout = replace(state.layout, database_path=custom_database)

    with pytest.raises(BackupPreconditionError, match="collides with canonical"):
        create_backup(layout, tmp_path / "state.zip", offline_confirmed=True)


@pytest.mark.parametrize(
    "path",
    [
        "data/model-snapshots",
        "data/tmp/partial-upload",
        "data/nanoloop.db-wal",
        "data/.nanoloop-state.lock",
    ],
)
def test_manifest_rejects_reserved_runtime_members(path: str) -> None:
    with pytest.raises(ValueError, match="reserved runtime state"):
        BackupFileRecord(
            path=path,
            component=BackupComponent.RUNTIME_DATA,
            size=0,
            sha256="0" * 64,
            mode=0o600,
            mtime_ns=0,
        )

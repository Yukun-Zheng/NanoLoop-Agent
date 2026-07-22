from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
import zipfile
from contextlib import closing
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
from app.storage.file_token_keyring_store import FileTokenV2KeyRingStore

_LOCK_PROBE = """
import sys
from pathlib import Path

from app.operations.backup import BackupPreconditionError, StateDirectoryLock

try:
    with StateDirectoryLock(Path(sys.argv[1]), exclusive=sys.argv[2] == "exclusive"):
        print("acquired")
except BackupPreconditionError:
    print("blocked")
"""


@dataclass(frozen=True)
class _StateTree:
    layout: BackupLayout
    token: Path
    v2_keyring: Path
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
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        connection.execute(
            "INSERT INTO alembic_version (version_num) VALUES (?)",
            (expected_alembic_heads()[0],),
        )
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
        connection.execute(
            "CREATE TABLE child (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parent(id))"
        )
        connection.execute("INSERT INTO parent (id) VALUES (1)")
        connection.execute("INSERT INTO child (id, parent_id) VALUES (1, 1)")
        connection.commit()

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
    v2_keyring = data_root / ".file_token_v2_keyring.json"
    FileTokenV2KeyRingStore(v2_keyring).initialize(
        active_kid="backup-test",
        key=b"v" * 32,
    )
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
        v2_keyring=v2_keyring,
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
    assert created.manifest.production_ready is True
    assert verified.manifest.missing_production_requirements == ()
    if os.name == "posix":
        assert stat.S_IMODE(archive.stat().st_mode) == 0o600
        assert stat.S_IMODE(created.checksum_path.stat().st_mode) == 0o600
    with zipfile.ZipFile(archive) as bundle:
        assert all(info.compress_type == zipfile.ZIP_STORED for info in bundle.infolist())
    assert (destination / "outputs" / "样本 α" / ".只读结果").read_bytes() == (b"immutable-result")
    restored_readonly = destination / "outputs" / "样本 α" / ".只读结果"
    assert stat.S_IMODE(restored_readonly.stat().st_mode) == 0o444
    assert restored_readonly.stat().st_mtime_ns == state.readonly_output.stat().st_mtime_ns
    assert (destination / "data" / "model-snapshots" / "模型.pt").read_bytes() == (
        b"model-snapshot"
    )
    assert (destination / "knowledge_base" / "sources" / "知识源.md").exists()
    restored_token = destination / "data" / ".file_token_secret"
    assert restored_token.read_bytes() == state.token.read_bytes()
    if os.name == "posix":
        assert stat.S_IMODE(restored_token.stat().st_mode) == 0o600
    else:
        assert restored_token.stat().st_mode & stat.S_IWRITE
    restored_v2_keyring = destination / "data" / ".file_token_v2_keyring.json"
    assert restored_v2_keyring.read_bytes() == state.v2_keyring.read_bytes()
    if os.name == "posix":
        assert stat.S_IMODE(restored_v2_keyring.stat().st_mode) == 0o600
    else:
        assert restored_v2_keyring.stat().st_mode & stat.S_IWRITE
    assert FileTokenV2KeyRingStore(restored_v2_keyring).load().active_kid == "backup-test"
    assert not (destination / "data" / "tmp" / "ignored.tmp").exists()
    assert not (destination / "data" / StateDirectoryLock.filename).exists()
    assert (destination / "data" / "unrelated.db-wal").read_bytes() == (b"ordinary-runtime-data")
    with sqlite3.connect(destination / "data" / "nanoloop.db") as connection:
        assert connection.execute("SELECT parent_id FROM child").fetchall() == [(1,)]


def test_old_archive_without_v2_keyring_remains_compatible_and_reports_gap(
    tmp_path: Path,
) -> None:
    state = _make_state_tree(tmp_path)
    state.v2_keyring.unlink()
    archive = tmp_path / "legacy-state.zip"

    created = create_backup(state.layout, archive, offline_confirmed=True)
    verified = verify_backup(archive)
    destination = tmp_path / "legacy-restored"
    restored = restore_backup(archive, destination, offline_confirmed=True)

    for manifest in (created.manifest, verified.manifest, restored.manifest):
        assert manifest.production_ready is False
        assert manifest.missing_production_requirements == ("file_token_v2_keyring",)
    assert (destination / "data" / ".file_token_secret").is_file()
    assert not (destination / "data" / ".file_token_v2_keyring.json").exists()


def test_restore_rejects_semantically_invalid_v2_keyring_before_publication(
    tmp_path: Path,
) -> None:
    state = _make_state_tree(tmp_path)
    archive = tmp_path / "invalid-restored-keyring.zip"
    created = create_backup(state.layout, archive, offline_confirmed=True)
    keyring_member = "data/.file_token_v2_keyring.json"
    invalid_keyring = b"not-a-valid-keyring"

    with zipfile.ZipFile(archive, mode="r") as source:
        infos = {info.filename: info for info in source.infolist()}
        members = {info.filename: source.read(info) for info in source.infolist()}
    manifest = json.loads(members["manifest.json"])
    for record in manifest["files"]:
        if record["path"] == keyring_member:
            record["size"] = len(invalid_keyring)
            record["sha256"] = hashlib.sha256(invalid_keyring).hexdigest()
            break
    members[keyring_member] = invalid_keyring
    members["manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    )
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_STORED) as target:
        for name, payload in members.items():
            target.writestr(infos[name], payload)
    created.checksum_path.write_text(
        f"{hashlib.sha256(archive.read_bytes()).hexdigest()}\n",
        encoding="ascii",
    )
    destination = tmp_path / "must-not-publish"

    with pytest.raises(BackupValidationError, match="v2 key ring is invalid"):
        restore_backup(archive, destination, offline_confirmed=True)

    assert not destination.exists()


def test_production_ready_backup_requires_a_valid_v2_keyring(tmp_path: Path) -> None:
    state = _make_state_tree(tmp_path)
    state.v2_keyring.unlink()
    required_layout = replace(state.layout, require_file_token_v2_keyring=True)

    with pytest.raises(BackupPreconditionError, match="production-ready backup"):
        create_backup(required_layout, tmp_path / "required.zip", offline_confirmed=True)


def test_explicit_v2_keyring_inside_data_root_maps_to_canonical_member(
    tmp_path: Path,
) -> None:
    state = _make_state_tree(tmp_path)
    explicit = state.layout.data_root / "operator" / "rotating-keyring.json"
    explicit.parent.mkdir()
    state.v2_keyring.rename(explicit)
    layout = replace(state.layout, file_token_v2_keyring_file=explicit)
    archive = tmp_path / "explicit.zip"

    created = create_backup(layout, archive, offline_confirmed=True)

    paths = {record.path for record in created.manifest.files}
    assert "data/.file_token_v2_keyring.json" in paths
    assert "data/operator/rotating-keyring.json" not in paths
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.read("data/.file_token_v2_keyring.json") == explicit.read_bytes()


def test_v2_keyring_must_be_protected_valid_and_inside_data_root(tmp_path: Path) -> None:
    state = _make_state_tree(tmp_path)
    outside = tmp_path / "outside-keyring.json"
    state.v2_keyring.rename(outside)
    outside_layout = replace(state.layout, file_token_v2_keyring_file=outside)

    with pytest.raises(BackupPreconditionError, match="inside data_root"):
        create_backup(outside_layout, tmp_path / "outside.zip", offline_confirmed=True)

    inside = state.layout.data_root / ".file_token_v2_keyring.json"
    outside.rename(inside)
    inside.chmod(0o640)
    with pytest.raises(BackupPreconditionError, match="mode 0600"):
        create_backup(state.layout, tmp_path / "mode.zip", offline_confirmed=True)

    inside.chmod(0o600)
    inside.write_text("not-a-keyring", encoding="utf-8")
    with pytest.raises(BackupPreconditionError, match="valid protected key ring"):
        create_backup(state.layout, tmp_path / "invalid.zip", offline_confirmed=True)


def test_v2_keyring_source_symlink_is_rejected_without_following_it(tmp_path: Path) -> None:
    state = _make_state_tree(tmp_path)
    target = state.layout.data_root / "target-keyring.json"
    state.v2_keyring.rename(target)
    state.v2_keyring.symlink_to(target)

    with pytest.raises(BackupPreconditionError, match="non-symlink regular file"):
        create_backup(state.layout, tmp_path / "symlink.zip", offline_confirmed=True)


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


def test_backup_accepts_reader_created_empty_wal_after_clean_checkpoint(
    tmp_path: Path,
) -> None:
    """A read-only backup may create an empty WAL for a clean WAL-mode database."""

    state = _make_state_tree(tmp_path)
    connection = sqlite3.connect(state.layout.database_path)
    try:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        connection.execute("CREATE TABLE clean_checkpoint_payload (value TEXT NOT NULL)")
        connection.execute("INSERT INTO clean_checkpoint_payload VALUES ('preserved')")
        connection.commit()
    finally:
        connection.close()

    source_wal = Path(f"{state.layout.database_path}-wal")
    assert not source_wal.exists()
    archive = tmp_path / "clean-wal-state.zip"
    destination = tmp_path / "clean-wal-restored"

    create_backup(state.layout, archive, offline_confirmed=True)
    restore_backup(archive, destination, offline_confirmed=True)

    # SQLite versions may either retain the reader-created empty WAL or remove it.  Neither
    # representation contains database pages, while the restored snapshot must retain the row.
    assert not source_wal.exists() or source_wal.stat().st_size == 0
    with sqlite3.connect(destination / "data" / "nanoloop.db") as connection:
        assert connection.execute("SELECT value FROM clean_checkpoint_payload").fetchone() == (
            "preserved",
        )


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
        created.checksum_path.write_bytes(f"{'0' * 64}\n".encode("ascii"))
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
    assert lock_path.is_file() and not lock_path.is_symlink()
    if os.name == "posix":
        assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600


def test_state_lock_has_cross_process_shared_and_exclusive_semantics(
    tmp_path: Path,
) -> None:
    with StateDirectoryLock(tmp_path, exclusive=False):
        assert _probe_state_lock(tmp_path, exclusive=False) == "acquired"
        assert _probe_state_lock(tmp_path, exclusive=True) == "blocked"

    with StateDirectoryLock(tmp_path, exclusive=True):
        assert _probe_state_lock(tmp_path, exclusive=False) == "blocked"
        assert _probe_state_lock(tmp_path, exclusive=True) == "blocked"

    assert _probe_state_lock(tmp_path, exclusive=True) == "acquired"


def _probe_state_lock(data_root: Path, *, exclusive: bool) -> str:
    project_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _LOCK_PROBE,
            str(data_root),
            "exclusive" if exclusive else "shared",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


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


def test_manifest_requires_0600_for_v2_keyring_member() -> None:
    with pytest.raises(ValueError, match="mode 0600"):
        BackupFileRecord(
            path="data/.file_token_v2_keyring.json",
            component=BackupComponent.RUNTIME_DATA,
            size=1,
            sha256="0" * 64,
            mode=0o640,
            mtime_ns=0,
        )

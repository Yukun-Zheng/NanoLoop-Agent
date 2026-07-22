"""Offline, manifest-verified backup and restore for local NanoLoop state."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import importlib
import os
import shutil
import sqlite3
import stat
import tempfile
import zipfile
from collections.abc import Callable, Iterable
from ctypes import wintypes
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from alembic.config import Config
from alembic.script import ScriptDirectory
from pydantic import BaseModel, ConfigDict, Field, model_validator

_BUFFER_BYTES = 1024 * 1024
_MANIFEST_PATH = "manifest.json"
_MANIFEST_LIMIT_BYTES = 16 * 1024 * 1024
_MAX_MEMBER_COUNT = 1_000_000
_DIGEST_PATTERN = r"^[0-9a-f]{64}$"
_O_BINARY = getattr(os, "O_BINARY", 0)
_CANONICAL_DATABASE_PATH = "data/nanoloop.db"
_CANONICAL_TOKEN_PATH = "data/.file_token_secret"
_CANONICAL_DIRECTORIES = (
    "data",
    "data/model-snapshots",
    "outputs",
    "knowledge_base",
    "knowledge_base/sources",
    "knowledge_base/index",
)

_WINDOWS_LOCKFILE_FAIL_IMMEDIATELY = 0x00000001
_WINDOWS_LOCKFILE_EXCLUSIVE_LOCK = 0x00000002
_WINDOWS_LOCK_CONFLICT_ERRORS = frozenset({32, 33})


class _WindowsOverlapped(ctypes.Structure):
    """Minimal OVERLAPPED layout used to lock the first byte of the lock file."""

    _fields_ = (
        ("Internal", ctypes.c_size_t),
        ("InternalHigh", ctypes.c_size_t),
        ("Offset", wintypes.DWORD),
        ("OffsetHigh", wintypes.DWORD),
        ("hEvent", wintypes.HANDLE),
    )


class BackupError(RuntimeError):
    """Base class for backup creation and validation failures."""


class BackupPreconditionError(BackupError, ValueError):
    """A required offline or filesystem precondition was not met."""


class BackupValidationError(BackupError, ValueError):
    """A source database or backup archive failed strict validation."""


class BackupSourceChangedError(BackupError):
    """A source file changed while an offline backup was being assembled."""


class BackupComponent(StrEnum):
    DATABASE = "database"
    RUNTIME_DATA = "runtime_data"
    MODEL_SNAPSHOTS = "model_snapshots"
    OUTPUTS = "outputs"
    KNOWLEDGE_SOURCES = "knowledge_sources"
    KNOWLEDGE_INDEX = "knowledge_index"


_ALL_COMPONENTS = tuple(BackupComponent)


@dataclass(frozen=True, slots=True)
class BackupLayout:
    """Physical source paths which are mapped to the canonical backup layout."""

    database_path: Path
    data_root: Path
    output_root: Path
    model_snapshot_root: Path
    knowledge_source_root: Path
    knowledge_index_root: Path
    file_token_secret_file: Path | None = None


class BackupFileRecord(BaseModel):
    """Integrity and restoration metadata for one regular archive member."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    path: str = Field(min_length=1, max_length=4096)
    component: BackupComponent
    size: int = Field(ge=0)
    sha256: str = Field(pattern=_DIGEST_PATTERN)
    mode: int = Field(ge=0, le=0o7777)
    mtime_ns: int = Field(ge=0)

    @model_validator(mode="after")
    def path_is_canonical(self) -> BackupFileRecord:
        _validate_member_path(self.path)
        if self.path == _MANIFEST_PATH:
            raise ValueError("manifest.json cannot describe itself")
        expected_component = _component_for_member_path(self.path)
        if self.component != expected_component:
            raise ValueError(
                f"member {self.path!r} must use component {expected_component.value!r}"
            )
        if self.mode & 0o7000:
            raise ValueError("setuid, setgid, and sticky permission bits are not allowed")
        if self.path == _CANONICAL_TOKEN_PATH and self.mode != 0o600:
            raise ValueError("the persisted file-token secret must use mode 0600")
        return self


class BackupManifest(BaseModel):
    """Strict, versioned description of every logical backup member."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1] = 1
    created_at: datetime
    database_revision: str = Field(min_length=1, max_length=255)
    components: tuple[BackupComponent, ...]
    files: tuple[BackupFileRecord, ...]

    @model_validator(mode="after")
    def manifest_is_complete_and_unambiguous(self) -> BackupManifest:
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must include a timezone")
        if self.components != _ALL_COMPONENTS:
            raise ValueError("components must list every logical component exactly once")
        paths = [record.path for record in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("manifest contains duplicate file paths")
        sorted_paths = sorted(paths)
        for parent, child in pairwise(sorted_paths):
            if child.startswith(f"{parent}/"):
                raise ValueError("manifest file paths cannot contain one another")
        database = [record for record in self.files if record.component == BackupComponent.DATABASE]
        if len(database) != 1 or database[0].path != _CANONICAL_DATABASE_PATH:
            raise ValueError("manifest must contain exactly one canonical database snapshot")
        return self


@dataclass(frozen=True, slots=True)
class BackupResult:
    archive_path: Path
    checksum_path: Path
    archive_sha256: str
    manifest: BackupManifest


@dataclass(frozen=True, slots=True)
class BackupVerificationResult:
    archive_path: Path
    checksum_path: Path
    archive_sha256: str
    manifest: BackupManifest


@dataclass(frozen=True, slots=True)
class RestoreResult:
    destination_root: Path
    archive_sha256: str
    manifest: BackupManifest


@dataclass(frozen=True, slots=True)
class _FileState:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    change_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _FileState:
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            mode=value.st_mode,
            size=value.st_size,
            mtime_ns=value.st_mtime_ns,
            change_ns=(
                value.st_ctime_ns
                if os.name == "posix"
                else int(getattr(value, "st_birthtime_ns", 0))
            ),
        )


@dataclass(frozen=True, slots=True)
class _SourceFile:
    source: Path
    archive_path: str
    component: BackupComponent
    state: _FileState


@dataclass(frozen=True, slots=True)
class _ResolvedLayout:
    database_path: Path
    data_root: Path
    output_root: Path
    model_snapshot_root: Path
    knowledge_source_root: Path
    knowledge_index_root: Path
    file_token_secret_file: Path | None

    @property
    def roots(self) -> tuple[Path, ...]:
        return (
            self.data_root,
            self.output_root,
            self.model_snapshot_root,
            self.knowledge_source_root,
            self.knowledge_index_root,
        )


def _acquire_descriptor_lock(descriptor: int, *, exclusive: bool) -> None:
    """Acquire a non-blocking advisory lock using the native platform API."""

    if os.name == "nt":
        _acquire_windows_descriptor_lock(descriptor, exclusive=exclusive)
        return
    if os.name == "posix":
        fcntl: Any = importlib.import_module("fcntl")
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
        return
    raise BackupPreconditionError(
        f"state directory locks are unsupported on platform {os.name!r}"
    )


def _release_descriptor_lock(descriptor: int) -> None:
    """Release a lock acquired by :func:`_acquire_descriptor_lock`."""

    if os.name == "nt":
        _release_windows_descriptor_lock(descriptor)
        return
    if os.name == "posix":
        fcntl: Any = importlib.import_module("fcntl")
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return
    raise BackupPreconditionError(
        f"state directory locks are unsupported on platform {os.name!r}"
    )


def _acquire_windows_descriptor_lock(descriptor: int, *, exclusive: bool) -> None:
    kernel32, handle = _windows_lock_api(descriptor)
    lock_file = kernel32.LockFileEx
    lock_file.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_WindowsOverlapped),
    )
    lock_file.restype = wintypes.BOOL
    flags = _WINDOWS_LOCKFILE_FAIL_IMMEDIATELY
    if exclusive:
        flags |= _WINDOWS_LOCKFILE_EXCLUSIVE_LOCK
    overlapped = _WindowsOverlapped()
    ctypes.set_last_error(0)
    if lock_file(handle, flags, 0, 1, 0, ctypes.byref(overlapped)):
        return
    error_code = ctypes.get_last_error()
    error = OSError(error_code, ctypes.FormatError(error_code))
    if error_code in _WINDOWS_LOCK_CONFLICT_ERRORS:
        raise BlockingIOError(errno.EACCES, "state directory lock is held") from error
    raise error


def _release_windows_descriptor_lock(descriptor: int) -> None:
    kernel32, handle = _windows_lock_api(descriptor)
    unlock_file = kernel32.UnlockFileEx
    unlock_file.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_WindowsOverlapped),
    )
    unlock_file.restype = wintypes.BOOL
    overlapped = _WindowsOverlapped()
    ctypes.set_last_error(0)
    if unlock_file(handle, 0, 1, 0, ctypes.byref(overlapped)):
        return
    error_code = ctypes.get_last_error()
    raise OSError(error_code, ctypes.FormatError(error_code))


def _windows_lock_api(descriptor: int) -> tuple[Any, wintypes.HANDLE]:
    """Return kernel32 and the Windows HANDLE owned by one CRT descriptor."""

    msvcrt: Any = importlib.import_module("msvcrt")
    raw_handle = int(msvcrt.get_osfhandle(descriptor))
    if raw_handle == -1:
        raise OSError(errno.EBADF, "invalid state lock descriptor")
    return ctypes.WinDLL("kernel32", use_last_error=True), wintypes.HANDLE(raw_handle)


def _set_open_file_mode(descriptor: int, path: Path, mode: int) -> None:
    """Apply Unix mode bits without requiring ``os.fchmod`` on Windows."""

    if os.name == "posix":
        platform_os: Any = os
        platform_os.fchmod(descriptor, mode)
        return
    # Windows chmod controls the read-only attribute rather than the DACL. The
    # file is already opened and proven to be the same non-symlink regular file.
    os.chmod(path, mode)


def _set_restored_file_time(path: Path, mtime_ns: int) -> None:
    timestamps = (mtime_ns, mtime_ns)
    if os.utime in os.supports_follow_symlinks:
        os.utime(path, ns=timestamps, follow_symlinks=False)
        return
    # The path was created with O_EXCL inside a private staging directory, so
    # Windows' lack of follow_symlinks support does not introduce a redirect.
    os.utime(path, ns=timestamps)


class StateDirectoryLock:
    """Non-blocking platform advisory lock for every state reader or writer process."""

    filename = ".nanoloop-state.lock"

    def __init__(self, data_root: str | Path, exclusive: bool) -> None:
        self.data_root = Path(data_root).expanduser()
        self.exclusive = exclusive
        self._descriptor: int | None = None
        self.lock_path: Path | None = None

    def acquire(self) -> StateDirectoryLock:
        if self._descriptor is not None:
            raise BackupPreconditionError("state directory lock is already acquired")
        root = _require_directory(self.data_root, "state lock data_root")
        lock_path = root / self.filename
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | _O_BINARY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as error:
            raise BackupPreconditionError(
                "state lock file cannot be opened safely"
            ) from error
        try:
            descriptor_state = os.fstat(descriptor)
            path_state = lock_path.lstat()
            if (
                not stat.S_ISREG(descriptor_state.st_mode)
                or stat.S_ISLNK(path_state.st_mode)
                or descriptor_state.st_nlink != 1
                or descriptor_state.st_dev != path_state.st_dev
                or descriptor_state.st_ino != path_state.st_ino
            ):
                raise BackupPreconditionError(
                    "state lock path must be one non-symlink regular file"
                )
            _set_open_file_mode(descriptor, lock_path, 0o600)
            try:
                _acquire_descriptor_lock(descriptor, exclusive=self.exclusive)
            except OSError as error:
                if error.errno in {errno.EACCES, errno.EAGAIN}:
                    raise BackupPreconditionError(
                        "NanoLoop state is locked by another process"
                    ) from error
                raise
        except BaseException:
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        self.lock_path = lock_path
        return self

    def release(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        try:
            _release_descriptor_lock(descriptor)
        finally:
            os.close(descriptor)

    def __enter__(self) -> StateDirectoryLock:
        return self.acquire()

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.release()


def create_backup(
    layout: BackupLayout,
    archive_path: str | Path,
    *,
    offline_confirmed: bool,
) -> BackupResult:
    """Create a no-overwrite offline backup and adjacent SHA-256 sidecar."""

    if offline_confirmed is not True:
        raise BackupPreconditionError("offline_confirmed=True is required")
    resolved = _resolve_layout(layout)
    destination = _resolve_new_archive_path(archive_path, resolved.roots)
    checksum_path = _checksum_path(destination)
    if os.path.lexists(destination) or os.path.lexists(checksum_path):
        raise FileExistsError("backup archive or checksum sidecar already exists")

    state_lock = StateDirectoryLock(resolved.data_root, exclusive=True)
    state_lock.acquire()
    try:
        inventory = _build_inventory(resolved)
        watched_database_files = _database_watch_states(resolved.database_path)
        archive_fd, archive_temp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(archive_fd)
        archive_temp = Path(archive_temp_name)
    except BaseException:
        state_lock.release()
        raise
    checksum_temp: Path | None = None
    database_temp: Path | None = None
    published_archive = False
    published_checksum = False
    try:
        os.chmod(archive_temp, 0o600)
        database_fd, database_temp_name = tempfile.mkstemp(
            prefix=".nanoloop-db-snapshot.", suffix=".db", dir=destination.parent
        )
        os.close(database_fd)
        database_temp = Path(database_temp_name)
        os.chmod(database_temp, 0o600)
        database_revision = _snapshot_database(resolved.database_path, database_temp)

        records: list[BackupFileRecord] = []
        with zipfile.ZipFile(
            archive_temp,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
            strict_timestamps=False,
        ) as archive:
            database_state = _regular_file_state(database_temp)
            records.append(
                _write_source_member(
                    archive,
                    _SourceFile(
                        source=database_temp,
                        archive_path=_CANONICAL_DATABASE_PATH,
                        component=BackupComponent.DATABASE,
                        state=database_state,
                    ),
                )
            )
            for source_file in inventory:
                records.append(_write_source_member(archive, source_file))

            _assert_inventory_unchanged(inventory)
            if _build_inventory(resolved) != inventory:
                raise BackupSourceChangedError(
                    "backup source membership changed while the archive was assembled"
                )
            _assert_database_watch_unchanged(
                resolved.database_path,
                watched_database_files,
            )
            manifest = BackupManifest(
                created_at=datetime.now(UTC),
                database_revision=database_revision,
                components=_ALL_COMPONENTS,
                files=tuple(sorted(records, key=lambda record: record.path)),
            )
            manifest_bytes = (
                manifest.model_dump_json(indent=2).encode("utf-8") + b"\n"
            )
            archive.writestr(_zip_info(_MANIFEST_PATH, 0o600), manifest_bytes)

        _fsync_file(archive_temp)
        archive_sha256 = _hash_regular_file(archive_temp)
        checksum_fd, checksum_temp_name = tempfile.mkstemp(
            prefix=f".{checksum_path.name}.", suffix=".tmp", dir=destination.parent
        )
        checksum_temp = Path(checksum_temp_name)
        with os.fdopen(checksum_fd, "wb") as checksum_stream:
            checksum_stream.write(f"{archive_sha256}\n".encode("ascii"))
            checksum_stream.flush()
            os.fsync(checksum_stream.fileno())
        os.chmod(checksum_temp, 0o600)

        published_archive, published_checksum = _publish_backup_pair(
            archive_temp,
            destination,
            checksum_temp,
            checksum_path,
        )
        return BackupResult(
            archive_path=destination,
            checksum_path=checksum_path,
            archive_sha256=archive_sha256,
            manifest=manifest,
        )
    finally:
        try:
            if published_archive != published_checksum:
                # _publish_backup_pair normally rolls back its own partial publication.
                # This defensive branch only removes an inode linked from our private temp.
                _rollback_partial_publication(
                    destination,
                    archive_temp,
                    checksum_path,
                    checksum_temp,
                )
            archive_temp.unlink(missing_ok=True)
            if checksum_temp is not None:
                checksum_temp.unlink(missing_ok=True)
            if database_temp is not None:
                database_temp.unlink(missing_ok=True)
        finally:
            state_lock.release()


def verify_backup(
    archive_path: str | Path,
    *,
    checksum_path: str | Path | None = None,
) -> BackupVerificationResult:
    """Verify the sidecar, archive structure, strict manifest, and every member byte."""

    archive = _require_regular_existing(Path(archive_path).expanduser(), "backup archive")
    checksum_candidate = (
        _checksum_path(archive) if checksum_path is None else Path(checksum_path).expanduser()
    )
    checksum = _require_regular_existing(checksum_candidate, "backup checksum")
    expected_sha256 = _read_checksum(checksum)
    observed_sha256 = _hash_regular_file(archive)
    if observed_sha256 != expected_sha256:
        raise BackupValidationError("backup archive SHA-256 does not match its sidecar")
    manifest = _verify_zip_archive(archive)
    return BackupVerificationResult(
        archive_path=archive,
        checksum_path=checksum,
        archive_sha256=observed_sha256,
        manifest=manifest,
    )


def restore_backup(
    archive_path: str | Path,
    destination_root: str | Path,
    *,
    offline_confirmed: bool,
    checksum_path: str | Path | None = None,
) -> RestoreResult:
    """Restore a verified archive through same-parent staging into a new root."""

    if offline_confirmed is not True:
        raise BackupPreconditionError("offline_confirmed=True is required")
    destination = Path(destination_root).expanduser().absolute()
    if os.path.lexists(destination):
        raise FileExistsError("restore destination already exists")
    parent = _require_directory(destination.parent, "restore destination parent")

    # No staging directory or destination file exists before this complete read-only pass.
    verification = verify_backup(archive_path, checksum_path=checksum_path)
    manifest = verification.manifest
    records = {record.path: record for record in manifest.files}

    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.restore-", dir=parent)
    )
    os.chmod(staging, 0o700)
    published = False
    try:
        for relative in _CANONICAL_DIRECTORIES:
            (staging / relative).mkdir(mode=0o750, parents=True, exist_ok=True)

        with zipfile.ZipFile(verification.archive_path, mode="r") as archive:
            database_record = records[_CANONICAL_DATABASE_PATH]
            _extract_member(archive, database_record, staging)
            restored_database = staging / _CANONICAL_DATABASE_PATH
            restored_revision = _validate_database_file(restored_database)
            if restored_revision != manifest.database_revision:
                raise BackupValidationError(
                    "restored database revision does not match the manifest"
                )

            for record in sorted(manifest.files, key=lambda item: item.path):
                if record.path != _CANONICAL_DATABASE_PATH:
                    _extract_member(archive, record, staging)

        restored_revision = _validate_database_file(staging / _CANONICAL_DATABASE_PATH)
        if restored_revision != manifest.database_revision:
            raise BackupValidationError("restored database revision changed during restore")
        if _hash_regular_file(verification.archive_path) != verification.archive_sha256:
            raise BackupSourceChangedError("backup archive changed while it was being restored")
        _fsync_tree(staging)
        if os.path.lexists(destination):
            raise FileExistsError("restore destination appeared during restore")
        os.rename(staging, destination)
        published = True
        _fsync_directory(parent)
        return RestoreResult(
            destination_root=destination,
            archive_sha256=verification.archive_sha256,
            manifest=manifest,
        )
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _resolve_layout(layout: BackupLayout) -> _ResolvedLayout:
    data_root = _require_directory(layout.data_root, "data_root")
    output_root = _require_directory(layout.output_root, "output_root")
    model_snapshot_root = _require_directory(
        layout.model_snapshot_root, "model_snapshot_root"
    )
    knowledge_source_root = _require_directory(
        layout.knowledge_source_root, "knowledge_source_root"
    )
    knowledge_index_root = _require_directory(
        layout.knowledge_index_root, "knowledge_index_root"
    )
    database_path = _require_regular_existing(layout.database_path, "database_path")
    configured_token_path = layout.file_token_secret_file
    default_token_path = data_root / _CANONICAL_TOKEN_PATH.removeprefix("data/")
    if configured_token_path is None and os.path.lexists(default_token_path):
        configured_token_path = default_token_path
    token_path = (
        _require_regular_existing(configured_token_path, "file_token_secret_file")
        if configured_token_path is not None
        else None
    )
    roots = (
        data_root,
        output_root,
        model_snapshot_root,
        knowledge_source_root,
        knowledge_index_root,
    )
    for index, first in enumerate(roots):
        for second in roots[index + 1 :]:
            if _paths_overlap(first, second):
                allowed_snapshot_nesting = (
                    first == data_root
                    and model_snapshot_root == second
                    and second.is_relative_to(first)
                ) or (
                    second == data_root
                    and model_snapshot_root == first
                    and first.is_relative_to(second)
                )
                if not allowed_snapshot_nesting:
                    raise BackupPreconditionError(
                        f"backup component roots overlap: {first} and {second}"
                    )
    if not database_path.is_relative_to(data_root):
        raise BackupPreconditionError("database_path must be located inside data_root")
    if token_path is not None and not token_path.is_relative_to(data_root):
        raise BackupPreconditionError(
            "file_token_secret_file must be located inside data_root"
        )
    _validate_source_permissions(database_path, "database_path")
    if token_path is not None:
        _validate_source_permissions(
            token_path,
            "file_token_secret_file",
            secret=True,
        )
    for special_file, label in (
        (database_path, "database_path"),
        *(([(token_path, "file_token_secret_file")]) if token_path is not None else []),
    ):
        for root in roots:
            if special_file.is_relative_to(root) and root != data_root:
                raise BackupPreconditionError(f"{label} overlaps component root {root}")
    if token_path is not None and token_path == database_path:
        raise BackupPreconditionError("database and token secret must be different files")
    return _ResolvedLayout(
        database_path=database_path,
        data_root=data_root,
        output_root=output_root,
        model_snapshot_root=model_snapshot_root,
        knowledge_source_root=knowledge_source_root,
        knowledge_index_root=knowledge_index_root,
        file_token_secret_file=token_path,
    )


def _build_inventory(layout: _ResolvedLayout) -> list[_SourceFile]:
    inventory: list[_SourceFile] = []
    canonical_database_name = _CANONICAL_DATABASE_PATH.removeprefix("data/")
    canonical_token_name = _CANONICAL_TOKEN_PATH.removeprefix("data/")
    database_sidecars = {
        Path(f"{layout.database_path}{suffix}")
        for suffix in ("-wal", "-shm", "-journal")
    }

    def exclude_runtime(path: Path, relative: PurePosixPath, is_directory: bool) -> bool:
        if path == layout.database_path or path in database_sidecars:
            return True
        if layout.file_token_secret_file is not None and path == layout.file_token_secret_file:
            return True
        relative_value = relative.as_posix()
        if relative_value in {canonical_database_name, canonical_token_name}:
            raise BackupPreconditionError(
                f"runtime data collides with canonical backup path: {relative_value}"
            )
        if relative_value == StateDirectoryLock.filename:
            if is_directory:
                raise BackupPreconditionError("state lock path must not be a directory")
            return True
        if relative.parts and relative.parts[0] in {"tmp", "model-snapshots"}:
            return True
        return path == layout.model_snapshot_root or path.is_relative_to(
            layout.model_snapshot_root
        )

    inventory.extend(
        _walk_component(
            layout.data_root,
            archive_prefix="data",
            component=BackupComponent.RUNTIME_DATA,
            exclude=exclude_runtime,
        )
    )
    inventory.extend(
        _walk_component(
            layout.model_snapshot_root,
            archive_prefix="data/model-snapshots",
            component=BackupComponent.MODEL_SNAPSHOTS,
        )
    )
    inventory.extend(
        _walk_component(
            layout.output_root,
            archive_prefix="outputs",
            component=BackupComponent.OUTPUTS,
        )
    )
    inventory.extend(
        _walk_component(
            layout.knowledge_source_root,
            archive_prefix="knowledge_base/sources",
            component=BackupComponent.KNOWLEDGE_SOURCES,
        )
    )
    inventory.extend(
        _walk_component(
            layout.knowledge_index_root,
            archive_prefix="knowledge_base/index",
            component=BackupComponent.KNOWLEDGE_INDEX,
        )
    )
    if layout.file_token_secret_file is not None:
        _validate_source_permissions(
            layout.file_token_secret_file,
            "file_token_secret_file",
            secret=True,
        )
        inventory.append(
            _SourceFile(
                source=layout.file_token_secret_file,
                archive_path=_CANONICAL_TOKEN_PATH,
                component=BackupComponent.RUNTIME_DATA,
                state=_regular_file_state(layout.file_token_secret_file),
            )
        )
    paths = [item.archive_path for item in inventory]
    if len(paths) != len(set(paths)):
        raise BackupPreconditionError("backup sources map to duplicate archive paths")
    return sorted(inventory, key=lambda item: item.archive_path)


def _walk_component(
    root: Path,
    *,
    archive_prefix: str,
    component: BackupComponent,
    exclude: Callable[[Path, PurePosixPath, bool], bool] | None = None,
) -> list[_SourceFile]:
    predicate = exclude or (lambda _path, _relative, _is_dir: False)
    result: list[_SourceFile] = []
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as error:
            raise BackupValidationError(f"cannot enumerate backup source {directory}") from error
        child_directories: list[Path] = []
        for entry in entries:
            path = Path(entry.path)
            try:
                # CPython's Windows DirEntry cache can report st_dev/st_ino as
                # zero, which cannot be compared with the later opened HANDLE.
                metadata = path.lstat()
            except OSError as error:
                raise BackupValidationError(f"cannot inspect backup source {path}") from error
            if stat.S_ISLNK(metadata.st_mode):
                raise BackupValidationError(f"symbolic links are not allowed: {path}")
            relative = PurePosixPath(path.relative_to(root).as_posix())
            if stat.S_ISDIR(metadata.st_mode):
                if not predicate(path, relative, True):
                    child_directories.append(path)
                continue
            if predicate(path, relative, False):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise BackupValidationError(f"special files are not allowed: {path}")
            if stat.S_IMODE(metadata.st_mode) & 0o7000:
                raise BackupValidationError(
                    f"special permission bits are not allowed: {path}"
                )
            archive_path = f"{archive_prefix}/{relative.as_posix()}"
            _validate_member_path(archive_path)
            result.append(
                _SourceFile(
                    source=path,
                    archive_path=archive_path,
                    component=component,
                    state=_FileState.from_stat(metadata),
                )
            )
        stack.extend(reversed(child_directories))
    return result


def _snapshot_database(source_path: Path, destination_path: Path) -> str:
    source_before = _database_watch_states(source_path)
    source_uri = f"{source_path.as_uri()}?mode=ro"
    try:
        source = sqlite3.connect(source_uri, uri=True)
        destination = sqlite3.connect(destination_path)
        try:
            source.execute("PRAGMA query_only=ON")
            data_version_before = int(source.execute("PRAGMA data_version").fetchone()[0])
            source.backup(destination)
            destination.commit()
            data_version_after = int(source.execute("PRAGMA data_version").fetchone()[0])
        finally:
            destination.close()
            source.close()
    except sqlite3.Error as error:
        raise BackupValidationError("SQLite backup API failed") from error
    if data_version_before != data_version_after:
        raise BackupSourceChangedError("database changed during SQLite backup")
    _assert_database_watch_unchanged(source_path, source_before)
    os.chmod(destination_path, stat.S_IMODE(source_path.stat().st_mode))
    _fsync_file(destination_path)
    return _validate_database_file(destination_path)


def _validate_database_file(path: Path) -> str:
    database = _require_regular_existing(path, "database snapshot")
    try:
        connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            integrity_rows = connection.execute("PRAGMA integrity_check").fetchall()
            if integrity_rows != [("ok",)]:
                raise BackupValidationError(
                    f"SQLite integrity_check failed: {integrity_rows!r}"
                )
            foreign_key_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
            if foreign_key_rows:
                raise BackupValidationError(
                    f"SQLite foreign_key_check failed: {foreign_key_rows!r}"
                )
            revision_rows = connection.execute(
                "SELECT version_num FROM alembic_version"
            ).fetchall()
        finally:
            connection.close()
    except BackupValidationError:
        raise
    except sqlite3.Error as error:
        raise BackupValidationError("database migration revision cannot be read") from error
    if len(revision_rows) != 1:
        raise BackupValidationError("database must contain exactly one Alembic revision")
    revision = revision_rows[0][0]
    if not isinstance(revision, str) or revision not in _known_alembic_revisions():
        raise BackupValidationError(f"unknown Alembic revision: {revision!r}")
    return revision


def _known_alembic_revisions() -> frozenset[str]:
    config = Config()
    config.set_main_option(
        "script_location",
        str(Path(__file__).resolve().parents[1] / "db" / "migrations"),
    )
    script = ScriptDirectory.from_config(config)
    return frozenset(revision.revision for revision in script.walk_revisions())


def _write_source_member(
    archive: zipfile.ZipFile,
    source_file: _SourceFile,
) -> BackupFileRecord:
    descriptor = _open_regular_no_follow(source_file.source)
    digest = hashlib.sha256()
    size = 0
    archive_mode = _archive_member_mode(source_file)
    try:
        opened_state = _FileState.from_stat(os.fstat(descriptor))
        if opened_state != source_file.state:
            raise BackupSourceChangedError(
                f"backup source changed before read: {source_file.source}"
            )
        with (
            os.fdopen(descriptor, "rb", closefd=False) as source,
            archive.open(
                _zip_info(
                    source_file.archive_path,
                    archive_mode,
                ),
                mode="w",
                force_zip64=True,
            ) as target,
        ):
            for block in iter(lambda: source.read(_BUFFER_BYTES), b""):
                digest.update(block)
                size += len(block)
                target.write(block)
        final_state = _FileState.from_stat(os.fstat(descriptor))
        if final_state != source_file.state or size != source_file.state.size:
            raise BackupSourceChangedError(
                f"backup source changed during read: {source_file.source}"
            )
    finally:
        os.close(descriptor)
    return BackupFileRecord(
        path=source_file.archive_path,
        component=source_file.component,
        size=size,
        sha256=digest.hexdigest(),
        mode=archive_mode,
        mtime_ns=source_file.state.mtime_ns,
    )


def _archive_member_mode(source_file: _SourceFile) -> int:
    mode = stat.S_IMODE(source_file.state.mode)
    if os.name == "nt" and source_file.archive_path == _CANONICAL_TOKEN_PATH:
        # Windows' stat mode exposes only its read-only attribute, not the DACL,
        # and reports a writable owner-only secret as 0666. Preserve the backup
        # contract's least-privilege Unix mode for restoration onto POSIX hosts.
        return 0o600
    return mode


def _verify_zip_archive(path: Path) -> BackupManifest:
    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            infos = archive.infolist()
            if len(infos) > _MAX_MEMBER_COUNT:
                raise BackupValidationError("backup archive contains too many members")
            names: set[str] = set()
            for info in infos:
                _validate_zip_info(info)
                if info.filename in names:
                    raise BackupValidationError(
                        f"backup archive contains duplicate member {info.filename!r}"
                    )
                names.add(info.filename)
            if _MANIFEST_PATH not in names:
                raise BackupValidationError("backup archive is missing manifest.json")
            manifest_info = archive.getinfo(_MANIFEST_PATH)
            if manifest_info.file_size > _MANIFEST_LIMIT_BYTES:
                raise BackupValidationError("backup manifest exceeds the size limit")
            try:
                manifest = BackupManifest.model_validate_json(
                    archive.read(manifest_info)
                )
            except Exception as error:
                raise BackupValidationError("backup manifest is invalid") from error
            if manifest.database_revision not in _known_alembic_revisions():
                raise BackupValidationError("manifest names an unknown Alembic revision")
            expected = {record.path: record for record in manifest.files}
            member_names = names - {_MANIFEST_PATH}
            if member_names != set(expected):
                missing = sorted(set(expected) - member_names)
                extra = sorted(member_names - set(expected))
                raise BackupValidationError(
                    f"backup members do not match manifest; missing={missing}, extra={extra}"
                )
            for info in infos:
                if info.filename == _MANIFEST_PATH:
                    continue
                record = expected[info.filename]
                if info.file_size != record.size:
                    raise BackupValidationError(
                        f"backup member size mismatch: {info.filename}"
                    )
                archived_mode = stat.S_IMODE(info.external_attr >> 16)
                if archived_mode != record.mode:
                    raise BackupValidationError(
                        f"backup member mode mismatch: {info.filename}"
                    )
                digest = hashlib.sha256()
                observed_size = 0
                with archive.open(info, mode="r") as source:
                    for block in iter(lambda: source.read(_BUFFER_BYTES), b""):
                        digest.update(block)
                        observed_size += len(block)
                if observed_size != record.size or digest.hexdigest() != record.sha256:
                    raise BackupValidationError(
                        f"backup member digest mismatch: {info.filename}"
                    )
            return manifest
    except BackupValidationError:
        raise
    except (OSError, EOFError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise BackupValidationError("backup ZIP cannot be read safely") from error


def _extract_member(
    archive: zipfile.ZipFile,
    record: BackupFileRecord,
    staging_root: Path,
) -> Path:
    info = archive.getinfo(record.path)
    _validate_zip_info(info)
    destination = staging_root.joinpath(*PurePosixPath(record.path).parts)
    destination.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | _O_BINARY
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(destination, flags, 0o600)
    digest = hashlib.sha256()
    observed_size = 0
    try:
        with archive.open(info, mode="r") as source:
            while True:
                block = source.read(_BUFFER_BYTES)
                if not block:
                    break
                digest.update(block)
                observed_size += len(block)
                view = memoryview(block)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("short write while restoring backup member")
                    view = view[written:]
        os.fsync(descriptor)
        _set_open_file_mode(descriptor, destination, record.mode)
    finally:
        os.close(descriptor)
    if observed_size != record.size or digest.hexdigest() != record.sha256:
        raise BackupValidationError(f"restored member digest mismatch: {record.path}")
    _set_restored_file_time(destination, record.mtime_ns)
    return destination


def _validate_zip_info(info: zipfile.ZipInfo) -> None:
    _validate_member_path(info.filename)
    if info.filename.endswith("/") or info.is_dir():
        raise BackupValidationError("directory entries are not allowed in backup ZIPs")
    if info.flag_bits & 0x1:
        raise BackupValidationError("encrypted backup members are not supported")
    if info.compress_type != zipfile.ZIP_STORED:
        raise BackupValidationError("backup members must use ZIP_STORED")
    if info.create_system != 3:
        raise BackupValidationError("backup members require Unix mode metadata")
    archived_mode = info.external_attr >> 16
    if not stat.S_ISREG(archived_mode):
        raise BackupValidationError("backup members must be regular files")
    if stat.S_IMODE(archived_mode) & 0o7000:
        raise BackupValidationError("backup members cannot use special permission bits")


def _validate_member_path(value: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise BackupValidationError("backup member path is invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise BackupValidationError(f"backup member path is unsafe: {value!r}")


def _component_for_member_path(value: str) -> BackupComponent:
    """Return the sole logical component allowed for one canonical member path."""

    if value == _CANONICAL_DATABASE_PATH:
        return BackupComponent.DATABASE
    if value in {
        "data/model-snapshots",
        "data/tmp",
        f"data/{StateDirectoryLock.filename}",
        *(
            f"{_CANONICAL_DATABASE_PATH}{suffix}"
            for suffix in ("-wal", "-shm", "-journal")
        ),
    } or value.startswith("data/tmp/"):
        raise BackupValidationError("backup member path is reserved runtime state")
    if value.startswith(f"{_CANONICAL_DATABASE_PATH}/") or value.startswith(
        f"{_CANONICAL_TOKEN_PATH}/"
    ):
        raise BackupValidationError("backup member path collides with a canonical file")
    if value.startswith("data/model-snapshots/"):
        return BackupComponent.MODEL_SNAPSHOTS
    if value.startswith("data/"):
        return BackupComponent.RUNTIME_DATA
    if value.startswith("outputs/"):
        return BackupComponent.OUTPUTS
    if value.startswith("knowledge_base/sources/"):
        return BackupComponent.KNOWLEDGE_SOURCES
    if value.startswith("knowledge_base/index/"):
        return BackupComponent.KNOWLEDGE_INDEX
    raise BackupValidationError(f"backup member path has no canonical component: {value!r}")


def _zip_info(filename: str, mode: int) -> zipfile.ZipInfo:
    _validate_member_path(filename)
    if mode & 0o7000:
        raise BackupValidationError("special permission bits are not allowed")
    info = zipfile.ZipInfo(filename=filename, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = (stat.S_IFREG | mode) << 16
    return info


def _resolve_new_archive_path(value: str | Path, roots: Iterable[Path]) -> Path:
    supplied = Path(value).expanduser()
    if supplied.name in {"", ".", ".."}:
        raise BackupPreconditionError("backup archive path must name a file")
    parent = _require_directory(supplied.absolute().parent, "backup destination parent")
    destination = parent / supplied.name
    for root in roots:
        if destination == root or destination.is_relative_to(root):
            raise BackupPreconditionError("backup archive cannot be created inside a source root")
    return destination


def _publish_backup_pair(
    archive_temp: Path,
    archive_path: Path,
    checksum_temp: Path,
    checksum_path: Path,
) -> tuple[bool, bool]:
    archive_published = False
    checksum_published = False
    publication_complete = False
    try:
        os.link(archive_temp, archive_path, follow_symlinks=False)
        archive_published = True
        os.chmod(archive_path, 0o600)
        os.link(checksum_temp, checksum_path, follow_symlinks=False)
        checksum_published = True
        os.chmod(checksum_path, 0o600)
        _fsync_directory(archive_path.parent)
        publication_complete = True
        return archive_published, checksum_published
    except FileExistsError as error:
        raise FileExistsError("backup archive or checksum sidecar already exists") from error
    finally:
        if not publication_complete:
            if archive_published and _same_inode(archive_path, archive_temp):
                archive_path.unlink(missing_ok=True)
            if checksum_published and _same_inode(checksum_path, checksum_temp):
                checksum_path.unlink(missing_ok=True)
            _fsync_directory(archive_path.parent)


def _rollback_partial_publication(
    archive_path: Path,
    archive_temp: Path,
    checksum_path: Path,
    checksum_temp: Path | None,
) -> None:
    if archive_temp.exists() and _same_inode(archive_path, archive_temp):
        archive_path.unlink(missing_ok=True)
    if checksum_temp is not None and checksum_temp.exists() and _same_inode(
        checksum_path, checksum_temp
    ):
        checksum_path.unlink(missing_ok=True)


def _read_checksum(path: Path) -> str:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise BackupValidationError("backup checksum cannot be read") from error
    if len(content) != 65 or content[-1:] != b"\n":
        raise BackupValidationError("backup checksum has an invalid format")
    try:
        digest = content[:-1].decode("ascii")
    except UnicodeError as error:
        raise BackupValidationError("backup checksum is not ASCII") from error
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise BackupValidationError("backup checksum is not a lowercase SHA-256 digest")
    return digest


def _checksum_path(archive_path: Path) -> Path:
    return Path(f"{archive_path}.sha256")


def _regular_file_state(path: Path) -> _FileState:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise BackupValidationError(f"cannot inspect regular file {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise BackupValidationError(f"backup source is not a regular file: {path}")
    return _FileState.from_stat(metadata)


def _require_regular_existing(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    try:
        metadata = path.lstat()
    except OSError as error:
        raise BackupPreconditionError(f"{label} does not exist: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise BackupPreconditionError(f"{label} must be a non-symlink regular file")
    return path.resolve(strict=True)


def _require_directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    try:
        metadata = path.lstat()
    except OSError as error:
        raise BackupPreconditionError(f"{label} does not exist: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise BackupPreconditionError(f"{label} must be a non-symlink directory")
    return path.resolve(strict=True)


def _validate_source_permissions(
    path: Path,
    label: str,
    *,
    secret: bool = False,
) -> None:
    mode = stat.S_IMODE(path.lstat().st_mode)
    if mode & 0o7000:
        raise BackupPreconditionError(f"{label} cannot use special permission bits")
    if secret and os.name == "posix" and mode != 0o600:
        raise BackupPreconditionError(f"{label} must use mode 0600")


def _open_regular_no_follow(path: Path, *, writable: bool = False) -> int:
    flags = (
        (os.O_RDWR if writable else os.O_RDONLY)
        | _O_BINARY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BackupValidationError(f"cannot open backup source safely: {path}") from error
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise BackupValidationError(f"backup source is not a regular file: {path}")
    return descriptor


def _hash_regular_file(path: Path) -> str:
    state = _regular_file_state(path)
    descriptor = _open_regular_no_follow(path)
    digest = hashlib.sha256()
    try:
        if _FileState.from_stat(os.fstat(descriptor)) != state:
            raise BackupSourceChangedError(f"file changed before hashing: {path}")
        while True:
            block = os.read(descriptor, _BUFFER_BYTES)
            if not block:
                break
            digest.update(block)
        if _FileState.from_stat(os.fstat(descriptor)) != state:
            raise BackupSourceChangedError(f"file changed while hashing: {path}")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _assert_inventory_unchanged(inventory: Iterable[_SourceFile]) -> None:
    for item in inventory:
        if _regular_file_state(item.source) != item.state:
            raise BackupSourceChangedError(f"backup source changed: {item.source}")


def _database_watch_states(path: Path) -> dict[Path, _FileState | None]:
    watched = [path, *(Path(f"{path}{suffix}") for suffix in ("-wal", "-shm", "-journal"))]
    result: dict[Path, _FileState | None] = {}
    for candidate in watched:
        if os.path.lexists(candidate):
            result[candidate] = _regular_file_state(candidate)
        else:
            result[candidate] = None
    return result


def _assert_database_watch_unchanged(
    database_path: Path,
    expected: dict[Path, _FileState | None],
) -> None:
    if _database_watch_states(database_path) != expected:
        raise BackupSourceChangedError("database or SQLite sidecar changed during backup")


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def _same_inode(first: Path, second: Path) -> bool:
    try:
        left = first.stat(follow_symlinks=False)
        right = second.stat(follow_symlinks=False)
    except OSError:
        return False
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _fsync_file(path: Path) -> None:
    # Windows rejects fsync on a read-only CRT descriptor even when the HANDLE
    # itself names a regular file. These are private writable publication temps.
    descriptor = _open_regular_no_follow(path, writable=True)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = (
        os.O_RDONLY
        | _O_BINARY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for current_directory, child_directories, _files in os.walk(root):
        base = Path(current_directory)
        directories.extend(base / name for name in child_directories)
    for current in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(current)


__all__ = [
    "BackupComponent",
    "BackupError",
    "BackupFileRecord",
    "BackupLayout",
    "BackupManifest",
    "BackupPreconditionError",
    "BackupResult",
    "BackupSourceChangedError",
    "BackupValidationError",
    "BackupVerificationResult",
    "RestoreResult",
    "StateDirectoryLock",
    "create_backup",
    "restore_backup",
    "verify_backup",
]

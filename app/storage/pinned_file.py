"""Race-resistant, descriptor-pinned reads for managed output files.

The path walk in this module deliberately does not use ``Path.resolve`` or reopen
the final path after validation.  Every component is opened relative to the
already-open parent directory, symbolic links are rejected by the kernel, and the
returned stream continues to refer to the exact inode that was hashed.

Descriptor pinning cannot prevent a different process from modifying that inode
after hashing finishes.  Callers therefore pair this primitive with the storage
contract that registered artifacts are published by atomic replacement and are
immutable after registration.
"""

from __future__ import annotations

import errno
import hashlib
import hmac
import os
import stat
import threading
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path
from types import TracebackType
from typing import Never, Self

from app.core.errors import StorageError
from app.storage.paths import StoragePathError

_DEFAULT_CHUNK_SIZE = 1024 * 1024
_MAX_CHUNK_SIZE = 16 * 1024 * 1024
_MAX_RELATIVE_PATH_BYTES = 4096
_MAX_PATH_COMPONENTS = 128
_MAX_COMPONENT_BYTES = 255

_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
# ``O_NONBLOCK`` is inert for regular files and prevents a hostile FIFO from
# hanging the process before ``fstat`` can reject it.
_FILE_OPEN_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
_SHA256_HEX_LENGTH = hashlib.sha256().digest_size * 2


class PinnedFileIntegrityError(StorageError):
    """Raised when a pinned file does not match trusted integrity metadata."""


class PinnedFileChangedError(PinnedFileIntegrityError):
    """Raised when a file changes while its descriptor is being hashed."""


class PinnedFileChunkIterator(Iterator[bytes]):
    """Single-use iterator that closes its owning pinned descriptor."""

    __slots__ = ("_owner",)

    def __init__(self, owner: PinnedManagedFile) -> None:
        self._owner: PinnedManagedFile | None = owner

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> bytes:
        owner = self._owner
        if owner is None:
            raise StopIteration
        try:
            chunk = owner._read_chunk()
        except BaseException:
            self._owner = None
            owner.close()
            raise
        if not chunk:
            self._owner = None
            raise StopIteration
        return chunk

    def close(self) -> None:
        """Close the underlying descriptor, whether or not iteration started."""

        owner = self._owner
        self._owner = None
        if owner is not None:
            owner.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(BaseException):
            self.close()


class PinnedManagedFile:
    """Own an already-verified file descriptor until its one stream is consumed."""

    __slots__ = (
        "_chunk_size",
        "_fd",
        "_iterator_claimed",
        "_lock",
        "filename",
        "relative_path",
        "sha256",
        "size_bytes",
    )

    def __init__(
        self,
        *,
        fd: int,
        relative_path: str,
        filename: str,
        size_bytes: int,
        sha256: str,
        chunk_size: int,
    ) -> None:
        self.filename = filename
        self.relative_path = relative_path
        self.size_bytes = size_bytes
        self.sha256 = sha256
        self._fd: int | None = fd
        self._chunk_size = chunk_size
        self._iterator_claimed = False
        self._lock = threading.Lock()

    @property
    def closed(self) -> bool:
        """Whether this object has released its descriptor."""

        with self._lock:
            return self._fd is None

    def fileno(self) -> int:
        """Return the pinned descriptor while it remains open."""

        with self._lock:
            if self._fd is None:
                raise ValueError("I/O operation on closed pinned file")
            return self._fd

    def iter_chunks(self) -> PinnedFileChunkIterator:
        """Claim and return the file's sole bounded-memory byte stream."""

        with self._lock:
            if self._fd is None:
                raise ValueError("I/O operation on closed pinned file")
            if self._iterator_claimed:
                raise RuntimeError("pinned file stream has already been claimed")
            self._iterator_claimed = True
        return PinnedFileChunkIterator(self)

    def __iter__(self) -> PinnedFileChunkIterator:
        return self.iter_chunks()

    def _read_chunk(self) -> bytes:
        with self._lock:
            file_descriptor = self._fd
            if file_descriptor is None:
                return b""
            try:
                chunk = os.read(file_descriptor, self._chunk_size)
            except BaseException:
                self._close_locked()
                raise
            if not chunk:
                self._close_locked()
            return chunk

    def close(self) -> None:
        """Idempotently release the pinned descriptor."""

        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        file_descriptor = self._fd
        if file_descriptor is None:
            return
        self._fd = None
        os.close(file_descriptor)

    def __enter__(self) -> Self:
        if self.closed:
            raise ValueError("I/O operation on closed pinned file")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(BaseException):
            self.close()


def open_pinned_managed_file(
    output_root: str | os.PathLike[str],
    relative_path: str,
    *,
    expected_size_bytes: int | None = None,
    expected_sha256: str | None = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> PinnedManagedFile:
    """Open, hash, rewind, and pin one regular file beneath ``output_root``.

    ``relative_path`` must be a canonical relative POSIX path.  Expected metadata
    is optional so callers can also use this primitive to obtain authoritative
    metadata, but when supplied it is checked before any bytes can be streamed.
    """

    components = _validate_relative_path(relative_path)
    checked_chunk_size = _validate_chunk_size(chunk_size)
    checked_expected_size = _validate_expected_size(expected_size_bytes)
    checked_expected_sha256 = _validate_expected_sha256(expected_sha256)

    directory_fd: int | None = None
    file_fd: int | None = None
    try:
        try:
            directory_fd = os.open(Path(output_root), _DIRECTORY_OPEN_FLAGS)
        except OSError as error:
            _raise_open_error(error, target="output root")
        assert directory_fd is not None

        for component in components[:-1]:
            try:
                next_directory_fd = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=directory_fd,
                )
            except OSError as error:
                _raise_open_error(error, target="managed parent directory")
            previous_directory_fd = directory_fd
            directory_fd = next_directory_fd
            os.close(previous_directory_fd)

        try:
            file_fd = os.open(
                components[-1],
                _FILE_OPEN_FLAGS,
                dir_fd=directory_fd,
            )
        except OSError as error:
            _raise_open_error(error, target="managed file")
        assert file_fd is not None

        os.close(directory_fd)
        directory_fd = None

        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise StoragePathError("managed file target must be a regular file")
        if checked_expected_size is not None and before.st_size != checked_expected_size:
            raise PinnedFileIntegrityError(
                "文件大小与可信元数据不一致",
                details={
                    "expected_size_bytes": checked_expected_size,
                    "observed_size_bytes": before.st_size,
                },
            )

        digest = hashlib.sha256()
        observed_size = 0
        while True:
            chunk = os.read(file_fd, checked_chunk_size)
            if not chunk:
                break
            observed_size += len(chunk)
            digest.update(chunk)

        after = os.fstat(file_fd)
        if _file_changed(before, after, observed_size=observed_size):
            raise PinnedFileChangedError(
                "文件在完整性校验期间发生变化",
                details={"relative_path": relative_path},
            )

        observed_sha256 = digest.hexdigest()
        if checked_expected_sha256 is not None and not hmac.compare_digest(
            observed_sha256, checked_expected_sha256
        ):
            raise PinnedFileIntegrityError(
                "文件哈希与可信元数据不一致",
                details={
                    "expected_sha256": checked_expected_sha256,
                    "observed_sha256": observed_sha256,
                },
            )

        os.lseek(file_fd, 0, os.SEEK_SET)
        pinned = PinnedManagedFile(
            fd=file_fd,
            relative_path=relative_path,
            filename=components[-1],
            size_bytes=observed_size,
            sha256=observed_sha256,
            chunk_size=checked_chunk_size,
        )
        file_fd = None
        return pinned
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def _validate_relative_path(value: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value:
        raise StoragePathError("managed file path must be a non-empty string")
    if "\\" in value or value.startswith("/"):
        raise StoragePathError("managed file path must be a relative POSIX path")
    if any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in value):
        raise StoragePathError("managed file path contains a control character")
    try:
        encoded_path = value.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise StoragePathError("managed file path is not valid UTF-8") from error
    if len(encoded_path) > _MAX_RELATIVE_PATH_BYTES:
        raise StoragePathError("managed file path is too long")

    components = tuple(value.split("/"))
    if len(components) > _MAX_PATH_COMPONENTS:
        raise StoragePathError("managed file path has too many components")
    if any(component in {"", ".", ".."} for component in components):
        raise StoragePathError("managed file path is not canonical")
    if any(len(component.encode("utf-8")) > _MAX_COMPONENT_BYTES for component in components):
        raise StoragePathError("managed file path component is too long")
    return components


def _validate_chunk_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= _MAX_CHUNK_SIZE:
        raise ValueError(f"chunk_size must be between 1 and {_MAX_CHUNK_SIZE} bytes")
    return value


def _validate_expected_size(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("expected_size_bytes must be a non-negative integer")
    return value


def _validate_expected_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
    return value


def _raise_open_error(error: OSError, *, target: str) -> Never:
    if error.errno in {errno.ELOOP, errno.ENOTDIR}:
        raise StoragePathError(f"{target} contains a symbolic link or non-directory") from error
    raise error


def _file_changed(
    before: os.stat_result,
    after: os.stat_result,
    *,
    observed_size: int,
) -> bool:
    stable_metadata_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    stable_metadata_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    return (
        stable_metadata_before != stable_metadata_after
        or observed_size != before.st_size
        or observed_size != after.st_size
    )

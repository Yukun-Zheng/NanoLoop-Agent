"""Content-addressed, immutable snapshots for model weight artifacts."""

from __future__ import annotations

import errno
import hashlib
import os
import re
import stat
import tempfile
from contextlib import suppress
from pathlib import Path, PurePosixPath

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COPY_BUFFER_BYTES = 1024 * 1024


class ModelArtifactSnapshotError(RuntimeError):
    """A weight source or an existing immutable snapshot failed validation."""


class ModelArtifactSnapshotStore:
    """Publish verified weights into a content-addressed local store.

    A caller never receives a destination name until the complete, fsynced file has been
    published without replacement. Existing destinations are opened without following symlinks
    and re-hashed before reuse.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def publish(self, source_path: str | Path, expected_sha256: str) -> Path:
        """Copy and verify ``source_path`` once, then atomically publish or reuse its snapshot."""

        expected = expected_sha256.lower()
        if not _SHA256_RE.fullmatch(expected):
            raise ValueError("expected_sha256 must be 64 lowercase hexadecimal characters")

        source = Path(source_path).expanduser().resolve()
        suffix = source.suffix
        digest_dir = self._prepare_directory(self.root / expected)

        destination = digest_dir / f"weights{suffix}"
        temp_fd, temp_name = tempfile.mkstemp(prefix=".weights-", suffix=".tmp", dir=digest_dir)
        temp_path = Path(temp_name)
        try:
            observed = self._copy_and_hash(source, temp_fd)
            if observed != expected:
                raise ModelArtifactSnapshotError(
                    f"weight sha256 mismatch: expected {expected}, observed {observed}"
                )
            os.fchmod(temp_fd, 0o444)
            os.fsync(temp_fd)
            os.close(temp_fd)
            temp_fd = -1

            try:
                # A same-directory hard link is an atomic no-replace publication. Concurrent
                # publishers either create the one complete destination or validate that winner.
                os.link(temp_path, destination, follow_symlinks=False)
                self._fsync_directory(digest_dir)
            except FileExistsError:
                self.verify(destination, expected)
            except OSError as error:
                if error.errno in {errno.EEXIST}:
                    self.verify(destination, expected)
                else:
                    raise ModelArtifactSnapshotError(
                        f"snapshot cannot be published atomically: {type(error).__name__}: {error}"
                    ) from error
            return self.verify(destination, expected)
        finally:
            if temp_fd >= 0:
                os.close(temp_fd)
            try:
                temp_path.unlink(missing_ok=True)
            finally:
                self._fsync_directory(digest_dir)

    def publish_bytes(
        self,
        artifact_name: str,
        content: bytes,
        expected_sha256: str,
    ) -> Path:
        """Publish already pinned bytes under their digest without replacement."""

        if Path(artifact_name).name != artifact_name or artifact_name in {"", ".", ".."}:
            raise ValueError("artifact_name must be one safe path component")
        expected = self._validate_digest(expected_sha256)
        observed = hashlib.sha256(content).hexdigest()
        if observed != expected:
            raise ModelArtifactSnapshotError(
                f"artifact sha256 mismatch: expected {expected}, observed {observed}"
            )
        digest_dir = self._prepare_directory(self.root / expected)
        return self._publish_complete_bytes(
            digest_dir / artifact_name,
            content,
            expected,
        )

    def publish_bundle_manifest(self, bundle_id: str, content: bytes) -> Path:
        """Publish a canonical manifest whose SHA-256 is the bundle identifier."""

        expected = self._validate_digest(bundle_id)
        observed = hashlib.sha256(content).hexdigest()
        if observed != expected:
            raise ModelArtifactSnapshotError(
                f"bundle manifest sha256 mismatch: expected {expected}, observed {observed}"
            )
        bundle_dir = self._prepare_directory(self.root / "bundles" / expected)
        return self._publish_complete_bytes(bundle_dir / "manifest.json", content, expected)

    def reference(self, snapshot_path: str | Path) -> str:
        """Return the normalized POSIX reference persisted in run configuration."""

        path = Path(snapshot_path).resolve()
        try:
            relative = path.relative_to(self.root)
        except ValueError as error:
            raise ModelArtifactSnapshotError("snapshot is outside the configured store") from error
        return relative.as_posix()

    def read_reference(self, reference: str, expected_sha256: str) -> bytes:
        """Read and hash one pinned descriptor; no verified filesystem path is reopened."""

        relative = PurePosixPath(reference)
        if relative.is_absolute() or ".." in relative.parts or str(relative) != reference:
            raise ModelArtifactSnapshotError("snapshot reference is not normalized and relative")
        current = self.root
        for part in relative.parts[:-1]:
            current = current / part
            try:
                metadata = current.lstat()
            except OSError as error:
                raise ModelArtifactSnapshotError(
                    f"snapshot reference parent cannot be inspected: {current}"
                ) from error
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise ModelArtifactSnapshotError(
                    f"snapshot reference parent is unsafe: {current}"
                )
        return self.read_verified(current / relative.name, expected_sha256)

    def read_verified(self, snapshot_path: str | Path, expected_sha256: str) -> bytes:
        """Return bytes read from the same descriptor that is type-checked and hashed."""

        expected = self._validate_digest(expected_sha256)
        path = Path(snapshot_path)
        descriptor = self._open_snapshot(path)
        try:
            content = bytearray()
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, _COPY_BUFFER_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                content.extend(chunk)
        finally:
            os.close(descriptor)
        observed = digest.hexdigest()
        if observed != expected:
            raise ModelArtifactSnapshotError(
                f"snapshot sha256 mismatch: expected {expected}, observed {observed}"
            )
        return bytes(content)

    def verify(self, snapshot_path: str | Path, expected_sha256: str) -> Path:
        """Fail closed unless an existing snapshot is a regular, read-only file with this digest."""

        expected = self._validate_digest(expected_sha256)
        path = Path(snapshot_path)
        descriptor = self._open_snapshot(path)
        try:
            observed = self._hash_descriptor(descriptor)
        finally:
            os.close(descriptor)
        if observed != expected:
            raise ModelArtifactSnapshotError(
                f"snapshot sha256 mismatch: expected {expected}, observed {observed}"
            )
        return path.resolve()

    def _publish_complete_bytes(
        self,
        destination: Path,
        content: bytes,
        expected_sha256: str,
    ) -> Path:
        temp_fd, temp_name = tempfile.mkstemp(
            prefix=f".{destination.name}-",
            suffix=".tmp",
            dir=destination.parent,
        )
        temp_path = Path(temp_name)
        try:
            view = memoryview(content)
            while view:
                written = os.write(temp_fd, view)
                if written <= 0:  # pragma: no cover - defensive OS contract guard
                    raise OSError("short write while creating model snapshot")
                view = view[written:]
            os.fchmod(temp_fd, 0o444)
            os.fsync(temp_fd)
            os.close(temp_fd)
            temp_fd = -1
            try:
                os.link(temp_path, destination, follow_symlinks=False)
                self._fsync_directory(destination.parent)
            except FileExistsError:
                self.verify(destination, expected_sha256)
            except OSError as error:
                if error.errno == errno.EEXIST:
                    self.verify(destination, expected_sha256)
                else:
                    raise ModelArtifactSnapshotError(
                        f"snapshot cannot be published atomically: {type(error).__name__}: {error}"
                    ) from error
            return self.verify(destination, expected_sha256)
        finally:
            if temp_fd >= 0:
                os.close(temp_fd)
            temp_path.unlink(missing_ok=True)
            self._fsync_directory(destination.parent)

    def _prepare_directory(self, directory: Path) -> Path:
        self.root.mkdir(mode=0o750, parents=True, exist_ok=True)
        try:
            relative = directory.relative_to(self.root)
        except ValueError as error:
            raise ModelArtifactSnapshotError(
                "snapshot directory escapes configured root"
            ) from error
        current = self.root
        for part in relative.parts:
            current = current / part
            with suppress(FileExistsError):
                current.mkdir(mode=0o750)
            metadata = current.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise ModelArtifactSnapshotError(
                    f"snapshot path is not a directory: {current}"
                )
        return directory

    @staticmethod
    def _validate_digest(value: str) -> str:
        normalized = value.lower()
        if not _SHA256_RE.fullmatch(normalized):
            raise ValueError("expected_sha256 must be 64 lowercase hexadecimal characters")
        return normalized

    @staticmethod
    def _open_snapshot(path: Path) -> int:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise ModelArtifactSnapshotError(
                f"snapshot cannot be opened safely: {type(error).__name__}: {error}"
            ) from error
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            os.close(descriptor)
            raise ModelArtifactSnapshotError(f"snapshot is not a regular file: {path}")
        if stat.S_IMODE(metadata.st_mode) & 0o222:
            os.close(descriptor)
            raise ModelArtifactSnapshotError(f"snapshot is writable: {path}")
        return descriptor

    @staticmethod
    def _copy_and_hash(source: Path, destination_fd: int) -> str:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            source_fd = os.open(source, flags)
        except OSError as error:
            raise ModelArtifactSnapshotError(
                f"weight source cannot be opened safely: {type(error).__name__}: {error}"
            ) from error
        digest = hashlib.sha256()
        try:
            metadata = os.fstat(source_fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise ModelArtifactSnapshotError(f"weight source is not a regular file: {source}")
            while True:
                chunk = os.read(source_fd, _COPY_BUFFER_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_fd, view)
                    if written <= 0:  # pragma: no cover - defensive guard around the OS contract
                        raise OSError("short write while creating model snapshot")
                    view = view[written:]
        finally:
            os.close(source_fd)
        return digest.hexdigest()

    @staticmethod
    def _hash_descriptor(descriptor: int) -> str:
        digest = hashlib.sha256()
        os.lseek(descriptor, 0, os.SEEK_SET)
        while True:
            chunk = os.read(descriptor, _COPY_BUFFER_BYTES)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(directory, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

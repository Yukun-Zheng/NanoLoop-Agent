"""Protected persistence for the file-token v2 HMAC key ring.

The on-disk document is exact canonical JSON with this schema::

    {"active_kid":"<kid>","keys":{"<kid>":"<base64url>"},"schema_version":1}

Each value in ``keys`` is the unpadded canonical base64url encoding of 32 to 64
raw random bytes.  The file is always mode ``0600`` and is opened without
following its final path component.  Initial publication is no-replace; key
rotation retains every old key and atomically replaces the complete document.
"""

from __future__ import annotations

import base64
import binascii
import errno
import json
import os
import secrets
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.storage.file_tokens_v2 import (
    FILE_TOKEN_V2_DEFAULT_CLOCK_SKEW_SECONDS,
    FILE_TOKEN_V2_DEFAULT_MAX_TTL_SECONDS,
    FileTokenV2KeyRing,
    validate_file_token_v2_kid,
)

FILE_TOKEN_V2_KEYRING_SCHEMA_VERSION = 1
FILE_TOKEN_V2_KEYRING_DEFAULT_MAX_KEYS = 8
FILE_TOKEN_V2_KEYRING_DEFAULT_MAX_FILE_BYTES = 8 * 1024
FILE_TOKEN_V2_KEYRING_ABSOLUTE_MAX_KEYS = 32
FILE_TOKEN_V2_KEYRING_ABSOLUTE_MAX_FILE_BYTES = 64 * 1024
FILE_TOKEN_V2_KEYRING_MINIMUM_KEY_BYTES = 32
FILE_TOKEN_V2_KEYRING_MAXIMUM_KEY_BYTES = 64

_DOCUMENT_KEYS = frozenset({"schema_version", "active_kid", "keys"})
_FILE_MODE = 0o600
_READ_CHUNK_BYTES = 4096
_LOAD_ATTEMPTS = 3
_TEMP_CREATE_ATTEMPTS = 32


class FileTokenV2KeyRingStoreError(RuntimeError):
    """One safe, diagnostic error type for all key-ring persistence failures."""

    __slots__ = ("code",)

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code!r}, message={str(self)!r})"


@dataclass(frozen=True, slots=True, repr=False)
class _KeyRingMaterial:
    active_kid: str
    keys: Mapping[str, bytes]


class _TransientKeyRingChange(RuntimeError):
    """The directory entry changed while a pinned descriptor was being read."""


class FileTokenV2KeyRingStore:
    """Load, initialize, and rotate one protected local file-token v2 key ring.

    ``load`` never creates material.  ``initialize`` publishes a complete file
    without replacement and, when another initializer wins, loads that winner.
    ``rotate`` requires a fresh key ID and preserves all retained verification
    keys up to ``max_keys``.
    """

    __slots__ = (
        "_clock_skew_seconds",
        "_filename",
        "_max_file_bytes",
        "_max_keys",
        "_max_ttl_seconds",
        "_parent",
    )

    def __init__(
        self,
        path: str | Path,
        *,
        max_keys: int = FILE_TOKEN_V2_KEYRING_DEFAULT_MAX_KEYS,
        max_file_bytes: int = FILE_TOKEN_V2_KEYRING_DEFAULT_MAX_FILE_BYTES,
        max_ttl_seconds: int = FILE_TOKEN_V2_DEFAULT_MAX_TTL_SECONDS,
        clock_skew_seconds: int = FILE_TOKEN_V2_DEFAULT_CLOCK_SKEW_SECONDS,
    ) -> None:
        try:
            supplied_path = Path(path).expanduser().absolute()
        except (TypeError, ValueError, OSError, RuntimeError):
            raise FileTokenV2KeyRingStoreError("invalid_path", "key-ring path is invalid") from None
        if not supplied_path.name or supplied_path.name in {".", ".."}:
            raise FileTokenV2KeyRingStoreError("invalid_path", "key-ring path must name one file")
        if (
            isinstance(max_keys, bool)
            or not isinstance(max_keys, int)
            or max_keys <= 0
            or max_keys > FILE_TOKEN_V2_KEYRING_ABSOLUTE_MAX_KEYS
        ):
            raise FileTokenV2KeyRingStoreError(
                "invalid_limits", "key-ring max_keys exceeds the supported bound"
            )
        if (
            isinstance(max_file_bytes, bool)
            or not isinstance(max_file_bytes, int)
            or max_file_bytes <= 0
            or max_file_bytes > FILE_TOKEN_V2_KEYRING_ABSOLUTE_MAX_FILE_BYTES
        ):
            raise FileTokenV2KeyRingStoreError(
                "invalid_limits", "key-ring max_file_bytes exceeds the supported bound"
            )
        self._parent = supplied_path.parent
        self._filename = supplied_path.name
        self._max_keys = max_keys
        self._max_file_bytes = max_file_bytes
        self._max_ttl_seconds = max_ttl_seconds
        self._clock_skew_seconds = clock_skew_seconds

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(path=<redacted>, max_keys={self._max_keys!r}, "
            f"max_file_bytes={self._max_file_bytes!r})"
        )

    def load(self) -> FileTokenV2KeyRing:
        """Load an existing protected key ring; never generate a missing one."""

        try:
            with self._open_parent() as parent_fd:
                material = self._load_material(parent_fd)
            return self._to_key_ring(material)
        except FileTokenV2KeyRingStoreError:
            raise
        except (TypeError, ValueError, OSError, UnicodeError, RecursionError):
            raise FileTokenV2KeyRingStoreError(
                "operation_failed", "key-ring load failed safely"
            ) from None

    def initialize(
        self,
        *,
        active_kid: str = "initial",
        key: bytes | None = None,
    ) -> FileTokenV2KeyRing:
        """Publish initial material once, or load the concurrent winner.

        A caller-supplied ``key`` is useful for controlled migration and tests;
        normal callers should omit it so 32 random bytes are generated locally.
        """

        try:
            return self._initialize(active_kid=active_kid, key=key)
        except FileTokenV2KeyRingStoreError:
            raise
        except (TypeError, ValueError, OSError, UnicodeError, RecursionError):
            raise FileTokenV2KeyRingStoreError(
                "operation_failed", "key-ring initialization failed safely"
            ) from None

    def _initialize(
        self,
        *,
        active_kid: str,
        key: bytes | None,
    ) -> FileTokenV2KeyRing:
        canonical_kid = self._validated_kid(active_kid)
        key_bytes = secrets.token_bytes(FILE_TOKEN_V2_KEYRING_MINIMUM_KEY_BYTES)
        if key is not None:
            key_bytes = self._validated_key(key)
        material = _KeyRingMaterial(active_kid=canonical_kid, keys={canonical_kid: key_bytes})
        payload = self._serialize(material)

        with self._open_parent() as parent_fd:
            temp_name = self._write_temporary(parent_fd, payload)
            try:
                try:
                    os.link(
                        temp_name,
                        self._filename,
                        src_dir_fd=parent_fd,
                        dst_dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    existing = self._load_material(parent_fd)
                    if existing.active_kid != canonical_kid:
                        raise FileTokenV2KeyRingStoreError(
                            "initialization_conflict",
                            "existing key ring has a different active key ID",
                        ) from None
                    return self._to_key_ring(existing)
                except OSError as error:
                    if error.errno == errno.EEXIST:
                        existing = self._load_material(parent_fd)
                        if existing.active_kid != canonical_kid:
                            raise FileTokenV2KeyRingStoreError(
                                "initialization_conflict",
                                "existing key ring has a different active key ID",
                            ) from None
                        return self._to_key_ring(existing)
                    raise FileTokenV2KeyRingStoreError(
                        "publish_failed", "key-ring initial publication failed"
                    ) from None
                self._fsync_directory(parent_fd)
                return self._to_key_ring(material)
            finally:
                self._remove_temporary(parent_fd, temp_name)

    def rotate(
        self,
        *,
        new_kid: str,
        key: bytes | None = None,
    ) -> FileTokenV2KeyRing:
        """Retain old keys, add one fresh key ID, and atomically activate it."""

        try:
            return self._rotate(new_kid=new_kid, key=key)
        except FileTokenV2KeyRingStoreError:
            raise
        except (TypeError, ValueError, OSError, UnicodeError, RecursionError):
            raise FileTokenV2KeyRingStoreError(
                "operation_failed", "key-ring rotation failed safely"
            ) from None

    def _rotate(
        self,
        *,
        new_kid: str,
        key: bytes | None,
    ) -> FileTokenV2KeyRing:
        canonical_kid = self._validated_kid(new_kid)
        key_bytes = secrets.token_bytes(FILE_TOKEN_V2_KEYRING_MINIMUM_KEY_BYTES)
        if key is not None:
            key_bytes = self._validated_key(key)

        with self._open_parent() as parent_fd:
            existing = self._load_material(parent_fd)
            if canonical_kid in existing.keys:
                raise FileTokenV2KeyRingStoreError(
                    "duplicate_key_id", "rotation requires a new key ID"
                )
            if len(existing.keys) >= self._max_keys:
                raise FileTokenV2KeyRingStoreError(
                    "key_limit", "key-ring retained-key limit has been reached"
                )
            rotated_keys = dict(existing.keys)
            rotated_keys[canonical_kid] = key_bytes
            rotated = _KeyRingMaterial(active_kid=canonical_kid, keys=rotated_keys)
            payload = self._serialize(rotated)
            temp_name = self._write_temporary(parent_fd, payload)
            try:
                try:
                    os.replace(
                        temp_name,
                        self._filename,
                        src_dir_fd=parent_fd,
                        dst_dir_fd=parent_fd,
                    )
                except OSError:
                    raise FileTokenV2KeyRingStoreError(
                        "rotate_failed", "key-ring rotation could not be published"
                    ) from None
                self._fsync_directory(parent_fd)
                temp_name = ""
            finally:
                if temp_name:
                    self._remove_temporary(parent_fd, temp_name)
        return self._to_key_ring(rotated)

    @contextmanager
    def _open_parent(self) -> Iterator[int]:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(self._parent, flags)
        except OSError:
            raise FileTokenV2KeyRingStoreError(
                "parent_unavailable", "key-ring parent directory is unavailable or unsafe"
            ) from None
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                raise FileTokenV2KeyRingStoreError(
                    "parent_unavailable", "key-ring parent is not a directory"
                )
            yield descriptor
        finally:
            os.close(descriptor)

    def _load_material(self, parent_fd: int) -> _KeyRingMaterial:
        for _ in range(_LOAD_ATTEMPTS):
            try:
                payload = self._read_stable_payload(parent_fd)
                return self._parse(payload)
            except _TransientKeyRingChange:
                continue
        raise FileTokenV2KeyRingStoreError(
            "unstable_file", "key-ring file changed while it was being loaded"
        )

    def _read_stable_payload(self, parent_fd: int) -> bytes:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self._filename, flags, dir_fd=parent_fd)
        except FileNotFoundError:
            raise FileTokenV2KeyRingStoreError("missing", "key-ring file does not exist") from None
        except OSError as error:
            if error.errno in {errno.ELOOP, errno.EISDIR}:
                raise FileTokenV2KeyRingStoreError(
                    "unsafe_type", "key-ring path must be a regular non-symlink file"
                ) from None
            raise FileTokenV2KeyRingStoreError(
                "open_failed", "key-ring file cannot be opened safely"
            ) from None

        try:
            before = os.fstat(descriptor)
            self._validate_file_metadata(before)
            payload = self._read_bounded(descriptor)
            after = os.fstat(descriptor)
            try:
                current = os.stat(self._filename, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                raise _TransientKeyRingChange from None
            except OSError:
                raise FileTokenV2KeyRingStoreError(
                    "inspect_failed", "key-ring file cannot be inspected safely"
                ) from None
            if stat.S_ISLNK(current.st_mode):
                raise FileTokenV2KeyRingStoreError(
                    "unsafe_type", "key-ring path must be a regular non-symlink file"
                )
            if self._metadata_fingerprint(before) != self._metadata_fingerprint(after):
                raise _TransientKeyRingChange
            if (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino):
                raise _TransientKeyRingChange
            if len(payload) != after.st_size:
                raise _TransientKeyRingChange
            return payload
        finally:
            os.close(descriptor)

    def _validate_file_metadata(self, metadata: os.stat_result) -> None:
        if not stat.S_ISREG(metadata.st_mode):
            raise FileTokenV2KeyRingStoreError(
                "unsafe_type", "key-ring path must be a regular non-symlink file"
            )
        if stat.S_IMODE(metadata.st_mode) != _FILE_MODE:
            raise FileTokenV2KeyRingStoreError(
                "unsafe_permissions", "key-ring file must use mode 0600"
            )
        getuid = getattr(os, "getuid", None)
        if getuid is not None and metadata.st_uid != getuid():
            raise FileTokenV2KeyRingStoreError(
                "unsafe_owner", "key-ring file must be owned by the current user"
            )
        if metadata.st_size <= 0:
            raise FileTokenV2KeyRingStoreError("truncated", "key-ring file is empty or truncated")
        if metadata.st_size > self._max_file_bytes:
            raise FileTokenV2KeyRingStoreError(
                "oversized", "key-ring file exceeds the configured size limit"
            )

    def _read_bounded(self, descriptor: int) -> bytes:
        content = bytearray()
        while len(content) <= self._max_file_bytes:
            chunk = os.read(
                descriptor,
                min(_READ_CHUNK_BYTES, self._max_file_bytes + 1 - len(content)),
            )
            if not chunk:
                break
            content.extend(chunk)
        if len(content) > self._max_file_bytes:
            raise FileTokenV2KeyRingStoreError(
                "oversized", "key-ring file exceeds the configured size limit"
            )
        return bytes(content)

    def _parse(self, payload: bytes) -> _KeyRingMaterial:
        try:
            document: Any = json.loads(
                payload.decode("utf-8", errors="strict"),
                object_pairs_hook=_json_object_without_duplicate_keys,
            )
            if not isinstance(document, dict) or set(document) != _DOCUMENT_KEYS:
                raise ValueError("invalid document shape")
            schema_version = document["schema_version"]
            if (
                isinstance(schema_version, bool)
                or not isinstance(schema_version, int)
                or schema_version != FILE_TOKEN_V2_KEYRING_SCHEMA_VERSION
            ):
                raise ValueError("invalid schema version")
            active_kid = self._validated_kid(document["active_kid"])
            supplied_keys = document["keys"]
            if not isinstance(supplied_keys, dict):
                raise ValueError("invalid keys mapping")
            if not supplied_keys or len(supplied_keys) > self._max_keys:
                raise ValueError("invalid key count")
            keys: dict[str, bytes] = {}
            for kid, encoded_key in supplied_keys.items():
                canonical_kid = self._validated_kid(kid)
                if not isinstance(encoded_key, str):
                    raise ValueError("invalid encoded key")
                keys[canonical_kid] = self._decode_key(encoded_key)
            if active_kid not in keys:
                raise ValueError("active key is absent")
            material = _KeyRingMaterial(active_kid=active_kid, keys=keys)
            if payload != self._serialize(material):
                raise ValueError("non-canonical document")
            return material
        except FileTokenV2KeyRingStoreError:
            raise
        except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError):
            raise FileTokenV2KeyRingStoreError(
                "invalid_payload", "key-ring file contains invalid canonical JSON"
            ) from None

    def _serialize(self, material: _KeyRingMaterial) -> bytes:
        document = {
            "schema_version": FILE_TOKEN_V2_KEYRING_SCHEMA_VERSION,
            "active_kid": material.active_kid,
            "keys": {kid: _base64url_encode(key) for kid, key in material.keys.items()},
        }
        payload = json.dumps(
            document,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        if len(payload) > self._max_file_bytes:
            raise FileTokenV2KeyRingStoreError(
                "oversized", "key-ring document exceeds the configured size limit"
            )
        return payload

    def _write_temporary(self, parent_fd: int, payload: bytes) -> str:
        descriptor = -1
        temp_name = ""
        for _ in range(_TEMP_CREATE_ATTEMPTS):
            candidate = f".{self._filename}.tmp-{secrets.token_hex(12)}"
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                descriptor = os.open(candidate, flags, _FILE_MODE, dir_fd=parent_fd)
                temp_name = candidate
                break
            except FileExistsError:
                continue
            except OSError:
                raise FileTokenV2KeyRingStoreError(
                    "temp_create_failed", "key-ring temporary file cannot be created"
                ) from None
        if descriptor < 0:
            raise FileTokenV2KeyRingStoreError(
                "temp_create_failed", "key-ring temporary name allocation failed"
            )

        try:
            os.fchmod(descriptor, _FILE_MODE)
            remaining = memoryview(payload)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:  # pragma: no cover - defensive OS contract guard
                    raise OSError("short key-ring write")
                remaining = remaining[written:]
            os.fsync(descriptor)
        except OSError:
            try:
                os.close(descriptor)
            finally:
                self._remove_temporary(parent_fd, temp_name)
            raise FileTokenV2KeyRingStoreError(
                "write_failed", "key-ring temporary file could not be persisted"
            ) from None
        os.close(descriptor)
        return temp_name

    def _remove_temporary(self, parent_fd: int, temp_name: str) -> None:
        try:
            os.unlink(temp_name, dir_fd=parent_fd)
        except FileNotFoundError:
            return
        except OSError:
            raise FileTokenV2KeyRingStoreError(
                "cleanup_failed", "key-ring temporary file could not be removed"
            ) from None
        self._fsync_directory(parent_fd)

    @staticmethod
    def _fsync_directory(parent_fd: int) -> None:
        try:
            os.fsync(parent_fd)
        except OSError:
            raise FileTokenV2KeyRingStoreError(
                "sync_failed", "key-ring directory could not be synchronized"
            ) from None

    def _validated_kid(self, value: object) -> str:
        try:
            if not isinstance(value, str):
                raise ValueError("invalid key ID type")
            return validate_file_token_v2_kid(value)
        except (TypeError, ValueError):
            raise FileTokenV2KeyRingStoreError(
                "invalid_key_id", "key-ring contains an invalid key ID"
            ) from None

    @staticmethod
    def _validated_key(value: bytes) -> bytes:
        if not isinstance(value, bytes):
            raise FileTokenV2KeyRingStoreError("invalid_key", "key-ring signing key must be bytes")
        key = bytes(value)
        if not (
            FILE_TOKEN_V2_KEYRING_MINIMUM_KEY_BYTES
            <= len(key)
            <= FILE_TOKEN_V2_KEYRING_MAXIMUM_KEY_BYTES
        ):
            raise FileTokenV2KeyRingStoreError(
                "invalid_key", "key-ring signing key has an invalid bounded length"
            )
        return key

    @classmethod
    def _decode_key(cls, value: str) -> bytes:
        try:
            decoded = _base64url_decode(value)
        except (TypeError, ValueError, UnicodeError):
            raise FileTokenV2KeyRingStoreError(
                "invalid_key", "key-ring contains invalid encoded key material"
            ) from None
        return cls._validated_key(decoded)

    def _to_key_ring(self, material: _KeyRingMaterial) -> FileTokenV2KeyRing:
        try:
            return FileTokenV2KeyRing(
                material.keys,
                active_kid=material.active_kid,
                max_ttl_seconds=self._max_ttl_seconds,
                clock_skew_seconds=self._clock_skew_seconds,
            )
        except (TypeError, ValueError):
            raise FileTokenV2KeyRingStoreError(
                "invalid_runtime_settings", "key-ring runtime settings are invalid"
            ) from None

    @staticmethod
    def _metadata_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_uid,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )


def _json_object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError("duplicate JSON object key")
        document[key] = value
    return document


def _base64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _base64url_decode(value: str) -> bytes:
    if not isinstance(value, str) or not value or "=" in value:
        raise ValueError("invalid base64url value")
    try:
        value.encode("ascii", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError("invalid base64url value") from error
    padding = "=" * (-len(value) % 4)
    try:
        decoded = base64.b64decode(value + padding, altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("invalid base64url value") from error
    if _base64url_encode(decoded) != value:
        raise ValueError("non-canonical base64url value")
    return decoded

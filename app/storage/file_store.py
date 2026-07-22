"""Streaming, atomic, and traversal-safe local file storage."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import stat
import tempfile
import time
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from app.core.errors import StorageError
from app.storage.paths import StoragePathError, StoragePaths

_CHUNK_SIZE = 1024 * 1024
_MAX_TOKEN_LENGTH = 4096


class UploadSizeExceededError(ValueError):
    """Raised after a streaming upload crosses its configured byte limit."""

    def __init__(self, limit_bytes: int) -> None:
        super().__init__(f"upload exceeds the configured limit of {limit_bytes} bytes")
        self.limit_bytes = limit_bytes


class FileTokenError(ValueError):
    """Raised for malformed, forged, expired, or stale file tokens."""


@dataclass(frozen=True, slots=True)
class StoredFile:
    """Metadata for one file persisted below the output root."""

    path: Path
    relative_path: str
    filename: str
    size_bytes: int
    sha256: str
    file_token: str


class LocalFileStore:
    """FileStore implementation backed by a single local output directory."""

    schema_version = "1.0"

    def __init__(
        self,
        paths: StoragePaths,
        *,
        max_upload_bytes: int,
        token_secret: str | bytes | None = None,
        default_token_ttl_seconds: int = 900,
    ) -> None:
        if max_upload_bytes <= 0:
            raise ValueError("max_upload_bytes must be positive")
        if default_token_ttl_seconds <= 0:
            raise ValueError("default_token_ttl_seconds must be positive")

        self.paths = paths
        self.max_upload_bytes = max_upload_bytes
        self.default_token_ttl_seconds = default_token_ttl_seconds
        if token_secret is None:
            self._token_secret = secrets.token_bytes(32)
        elif isinstance(token_secret, str):
            self._token_secret = token_secret.encode("utf-8")
        else:
            self._token_secret = bytes(token_secret)
        if len(self._token_secret) < 32:
            raise ValueError("token_secret must be at least 32 bytes")

    def save_upload(
        self,
        job_id: str,
        upload: BinaryIO,
        filename: str,
        *,
        image_id: str | None = None,
    ) -> StoredFile:
        """Stream an upload to a canonical input path while hashing and limiting it."""

        destination = self.paths.require_managed(
            self.paths.upload_file(job_id, filename, image_id=image_id)
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination = self.paths.require_managed(destination)

        digest = hashlib.sha256()
        size_bytes = 0
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=".nl-",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                while True:
                    chunk = upload.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    if not isinstance(chunk, bytes):
                        raise TypeError("upload.read() must return bytes")
                    size_bytes += len(chunk)
                    if size_bytes > self.max_upload_bytes:
                        raise UploadSizeExceededError(self.max_upload_bytes)
                    digest.update(chunk)
                    temporary.write(chunk)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, destination)
            temporary_path = None
        except (UploadSizeExceededError, TypeError):
            raise
        except OSError as error:
            raise StorageError(
                "上传文件写入失败",
                details={"filename": filename},
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        return self._stored_file(destination, sha256=digest.hexdigest(), size_bytes=size_bytes)

    def create_run_dir(self, job_id: str, image_id: str, run_id: str) -> Path:
        """Create and return the canonical immutable-run artifact directory."""

        directory = self.paths.require_managed(self.paths.run_dir(job_id, image_id, run_id))
        directory.mkdir(parents=True, exist_ok=True)
        return self.paths.require_managed(directory, must_exist=True)

    def atomic_write_json(
        self,
        path: str | Path,
        data: Mapping[str, Any],
        *,
        schema_version: str | None = None,
    ) -> None:
        """Serialize JSON with a mandatory schema version and replace atomically."""

        payload = dict(data)
        version = schema_version or self.schema_version
        existing_version = payload.setdefault("schema_version", version)
        if not isinstance(existing_version, str) or not existing_version.strip():
            raise ValueError("schema_version must be a non-empty string")
        serialized = (
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        self.atomic_write_bytes(path, serialized)

    def atomic_write_bytes(self, path: str | Path, data: bytes) -> None:
        """Atomically replace a managed file with *data*."""

        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        destination = self.paths.require_managed(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination = self.paths.require_managed(destination)

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=".nl-",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(data)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, destination)
            temporary_path = None
        except OSError as error:
            raise StorageError(
                "文件原子写入失败",
                details={"path": self._safe_detail_path(destination)},
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def calculate_sha256(self, path: str | Path) -> str:
        """Hash a managed regular file without following a final symlink."""

        managed = self._require_regular_file(path)
        digest = hashlib.sha256()
        with self._open_regular_file(managed) as source:
            for chunk in iter(lambda: source.read(_CHUNK_SIZE), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def build_zip(
        self,
        job_id: str,
        files: Iterable[str | Path],
        *,
        filename: str | None = "nanoloop-export.zip",
    ) -> StoredFile:
        """Build an atomic job export from an explicit file whitelist.

        Archive member names are always relative to the job directory. File hashes
        are calculated from the exact bytes written into the archive, and the same
        ``export_manifest.json`` bytes are stored beside and inside the ZIP. Passing
        ``filename=None`` publishes a deterministic, content-addressed archive. Such
        an archive is never replaced: an existing selection path is reused only
        after its exact ZIP bytes have been verified.
        """

        job_dir = self.paths.require_managed(self.paths.job_dir(job_id))
        job_dir.mkdir(parents=True, exist_ok=True)
        job_dir = self.paths.require_managed(job_dir, must_exist=True)
        content_addressed = filename is None
        destination: Path | None = None
        if filename is not None:
            destination = self.paths.require_managed(self.paths.export_zip(job_id, filename))

        selected: dict[str, Path] = {}
        for requested in files:
            requested_path = Path(requested)
            if requested_path.is_absolute():
                export_input = requested_path
            elif requested_path.parts and requested_path.parts[0] == job_id:
                export_input = self.paths.root / requested_path
            else:
                export_input = job_dir / requested_path
            managed = self._require_regular_file(export_input)
            try:
                archive_name = managed.relative_to(job_dir).as_posix()
            except ValueError as error:
                raise StoragePathError("export input is outside the requested job") from error
            generated_names = {"export_manifest.json"}
            if destination is not None:
                generated_names.add(destination.relative_to(job_dir).as_posix())
            if archive_name in generated_names:
                raise StoragePathError("export input collides with a generated export artifact")
            if archive_name in selected:
                raise StoragePathError(f"duplicate export member: {archive_name}")
            selected[archive_name] = managed

        export_dir = self.paths.require_managed(self.paths.export_dir(job_id))
        export_dir.mkdir(parents=True, exist_ok=True)
        export_dir = self.paths.require_managed(export_dir, must_exist=True)
        temporary_path: Path | None = None
        manifest_bytes: bytes | None = None
        selection_sha256: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=export_dir,
                prefix=".nanoloop-export.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                records: list[dict[str, Any]] = []
                with zipfile.ZipFile(
                    temporary,
                    mode="w",
                    compression=zipfile.ZIP_DEFLATED,
                    compresslevel=6,
                    allowZip64=True,
                ) as archive:
                    for archive_name, source_path in sorted(selected.items()):
                        digest = hashlib.sha256()
                        size_bytes = 0
                        with (
                            self._open_regular_file(source_path) as source,
                            archive.open(
                                self._deterministic_zip_info(archive_name),
                                mode="w",
                                force_zip64=True,
                            ) as target,
                        ):
                            for chunk in iter(lambda: source.read(_CHUNK_SIZE), b""):
                                digest.update(chunk)
                                size_bytes += len(chunk)
                                target.write(chunk)
                        records.append(
                            {
                                "path": archive_name,
                                "sha256": digest.hexdigest(),
                                "size_bytes": size_bytes,
                            }
                        )

                    selection_sha256 = self._selection_sha256(records)
                    manifest: dict[str, Any] = {
                        "schema_version": self.schema_version,
                        "job_id": job_id,
                        "selection_sha256": selection_sha256,
                        "files": records,
                    }
                    if not content_addressed:
                        manifest["generated_at"] = datetime.now(UTC).isoformat()
                    manifest_bytes = (
                        json.dumps(
                            manifest,
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                            allow_nan=False,
                        ).encode("utf-8")
                        + b"\n"
                    )
                    archive.writestr(
                        self._deterministic_zip_info("export_manifest.json"),
                        manifest_bytes,
                        compress_type=zipfile.ZIP_DEFLATED,
                        compresslevel=6,
                    )
                temporary.flush()
                os.fsync(temporary.fileno())

            if manifest_bytes is None or selection_sha256 is None:  # pragma: no cover
                raise StorageError("导出清单生成失败")
            if content_addressed:
                destination = self.paths.require_managed(
                    self.paths.export_zip(
                        job_id,
                        f"nanoloop-export-{selection_sha256}.zip",
                    )
                )
                manifest_path = self.paths.content_addressed_export_manifest(
                    job_id, selection_sha256
                )
                self._publish_no_replace(temporary_path, destination)
                temporary_path.unlink()
                self._publish_bytes_no_replace(manifest_path, manifest_bytes)
            else:
                if destination is None:  # pragma: no cover - narrowed above
                    raise StorageError("导出路径生成失败")
                manifest_path = self.paths.export_manifest(job_id)
                os.replace(temporary_path, destination)
                self.atomic_write_bytes(manifest_path, manifest_bytes)
            temporary_path = None
        except (StoragePathError, FileTokenError):
            raise
        except (OSError, zipfile.BadZipFile) as error:
            raise StorageError(
                "导出 ZIP 生成失败",
                details={"job_id": job_id},
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        if destination is None:  # pragma: no cover - defensive invariant
            raise StorageError("导出路径生成失败")
        return self._stored_file(destination)

    @staticmethod
    def _selection_sha256(records: list[dict[str, Any]]) -> str:
        canonical = json.dumps(
            records,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @staticmethod
    def _deterministic_zip_info(filename: str) -> zipfile.ZipInfo:
        info = zipfile.ZipInfo(filename, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.create_system = 3
        info.external_attr = (stat.S_IFREG | 0o644) << 16
        return info

    def _publish_no_replace(self, temporary_path: Path, destination: Path) -> None:
        """Atomically publish once, or prove an existing immutable archive is identical."""

        try:
            os.link(temporary_path, destination)
        except FileExistsError:
            existing = self._require_regular_file(destination)
            expected_sha256 = self.calculate_sha256(temporary_path)
            observed_sha256 = self.calculate_sha256(existing)
            if not hmac.compare_digest(observed_sha256, expected_sha256):
                raise StorageError(
                    "已有内容寻址导出文件校验失败",
                    details={
                        "path": self._safe_detail_path(destination),
                        "expected_sha256": expected_sha256,
                        "observed_sha256": observed_sha256,
                    },
                ) from None

    def _publish_bytes_no_replace(self, destination: Path, data: bytes) -> None:
        """Publish immutable bytes once without replacing a concurrently opened file.

        Windows does not allow ``os.replace`` while another worker has the
        destination open for hashing. Content-addressed manifests are immutable,
        so a hard-link publish plus byte-for-byte verification is both portable
        and stricter than replacing an existing file.
        """

        destination = self.paths.require_managed(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=".nl-",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(data)
                temporary.flush()
                os.fsync(temporary.fileno())
            self._publish_no_replace(temporary_path, destination)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def create_file_token(
        self,
        path: str | Path,
        *,
        ttl_seconds: int | None = None,
        now: int | None = None,
    ) -> str:
        """Issue a compact HMAC-signed token for one managed regular file."""

        managed = self._require_regular_file(path)
        ttl = ttl_seconds or self.default_token_ttl_seconds
        if ttl <= 0:
            raise ValueError("ttl_seconds must be positive")
        issued_at = int(time.time()) if now is None else now
        payload = {
            "exp": issued_at + ttl,
            "nonce": secrets.token_urlsafe(8),
            "path": self.paths.relative_path(managed),
            "v": 1,
        }
        encoded_payload = self._base64url_encode(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )
        signed_value = f"v1.{encoded_payload}".encode("ascii")
        signature = self._base64url_encode(
            hmac.new(self._token_secret, signed_value, hashlib.sha256).digest()
        )
        return f"v1.{encoded_payload}.{signature}"

    def resolve_file_token(self, token: str, *, now: int | None = None) -> Path:
        """Verify a file token and resolve it to an existing managed file."""

        if not token or len(token) > _MAX_TOKEN_LENGTH:
            raise FileTokenError("invalid file token")
        parts = token.split(".")
        if len(parts) != 3 or parts[0] != "v1":
            raise FileTokenError("invalid file token")

        try:
            signed_value = f"{parts[0]}.{parts[1]}".encode("ascii", errors="strict")
        except UnicodeEncodeError as error:
            raise FileTokenError("invalid file token") from error
        expected_signature = hmac.new(
            self._token_secret,
            signed_value,
            hashlib.sha256,
        ).digest()
        try:
            supplied_signature = self._base64url_decode(parts[2])
        except (ValueError, UnicodeError) as error:
            raise FileTokenError("invalid file token") from error
        if not hmac.compare_digest(supplied_signature, expected_signature):
            raise FileTokenError("invalid file token")

        try:
            payload = json.loads(self._base64url_decode(parts[1]))
            version = payload["v"]
            relative_path = payload["path"]
            expires_at = payload["exp"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, UnicodeError) as error:
            raise FileTokenError("invalid file token") from error
        if version != 1 or not isinstance(relative_path, str) or not isinstance(expires_at, int):
            raise FileTokenError("invalid file token")
        current_time = int(time.time()) if now is None else now
        if current_time >= expires_at:
            raise FileTokenError("file token has expired")
        try:
            return self._require_regular_file(relative_path)
        except (StoragePathError, FileNotFoundError, OSError) as error:
            raise FileTokenError("file token does not resolve to an available file") from error

    def _stored_file(
        self,
        path: Path,
        *,
        sha256: str | None = None,
        size_bytes: int | None = None,
    ) -> StoredFile:
        managed = self._require_regular_file(path)
        return StoredFile(
            path=managed,
            relative_path=self.paths.relative_path(managed),
            filename=managed.name,
            size_bytes=managed.stat().st_size if size_bytes is None else size_bytes,
            sha256=self.calculate_sha256(managed) if sha256 is None else sha256,
            file_token=self.create_file_token(managed),
        )

    def _require_regular_file(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_symlink():
            raise StoragePathError("symbolic links are not valid storage files")
        managed = self.paths.require_managed(candidate, must_exist=True)
        if not managed.is_file():
            raise FileNotFoundError(managed)
        return managed

    @staticmethod
    def _open_regular_file(path: Path) -> BinaryIO:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        file_status = os.fstat(descriptor)
        if not stat.S_ISREG(file_status.st_mode):
            os.close(descriptor)
            raise StoragePathError("storage path is not a regular file")
        return os.fdopen(descriptor, "rb")

    def _safe_detail_path(self, path: Path) -> str:
        try:
            return self.paths.relative_path(path)
        except StoragePathError:
            return "<unmanaged>"

    @staticmethod
    def _base64url_encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")

    @staticmethod
    def _base64url_decode(value: str) -> bytes:
        padding = "=" * (-len(value) % 4)
        return base64.b64decode(value + padding, altchars=b"-_", validate=True)

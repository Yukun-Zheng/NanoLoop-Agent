"""Content-addressed, streaming storage for uploaded knowledge sources."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from app.core.errors import StorageError, UnsupportedMediaTypeError
from app.storage.file_store import UploadSizeExceededError

_CHUNK_SIZE = 1024 * 1024
_SUPPORTED_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".pdf"})


@dataclass(frozen=True, slots=True)
class StoredKnowledgeSource:
    path: Path
    sha256: str
    size_bytes: int
    created: bool


class KnowledgeSourceStore:
    def __init__(self, root: str | Path, *, max_upload_bytes: int) -> None:
        if max_upload_bytes <= 0:
            raise ValueError("max_upload_bytes must be positive")
        configured = Path(root).expanduser()
        configured.mkdir(parents=True, exist_ok=True)
        self.root = configured.resolve(strict=True)
        if not self.root.is_dir() or self.root.is_symlink():
            raise ValueError("knowledge source root must be a regular directory")
        self.max_upload_bytes = max_upload_bytes

    def save(self, stream: BinaryIO, filename: str) -> StoredKnowledgeSource:
        suffix = Path(filename).suffix.casefold()
        if suffix not in _SUPPORTED_SUFFIXES:
            raise UnsupportedMediaTypeError(
                details={
                    "filename": Path(filename).name,
                    "supported_extensions": sorted(_SUPPORTED_SUFFIXES),
                }
            )

        digest = hashlib.sha256()
        size_bytes = 0
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.root,
                prefix=".knowledge-upload.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                while True:
                    chunk = stream.read(_CHUNK_SIZE)
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

            sha256 = digest.hexdigest()
            destination_dir = self.root / sha256[:2]
            destination_dir.mkdir(mode=0o750, parents=True, exist_ok=True)
            destination = destination_dir / f"{sha256}{suffix}"
            if destination.is_symlink():
                raise StorageError("知识源路径不允许使用符号链接")
            if destination.is_file():
                if self._hash_file(destination) != sha256:
                    raise StorageError("知识源内容寻址文件校验失败")
                temporary_path.unlink()
                temporary_path = None
                return StoredKnowledgeSource(
                    path=destination,
                    sha256=sha256,
                    size_bytes=size_bytes,
                    created=False,
                )
            os.replace(temporary_path, destination)
            temporary_path = None
            return StoredKnowledgeSource(
                path=destination,
                sha256=sha256,
                size_bytes=size_bytes,
                created=True,
            )
        except (TypeError, UploadSizeExceededError, UnsupportedMediaTypeError, StorageError):
            raise
        except OSError as error:
            raise StorageError("知识文档写入失败") from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(_CHUNK_SIZE), b""):
                digest.update(chunk)
        return digest.hexdigest()

"""Canonical, traversal-safe paths for the versioned job output layout."""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Self

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class StoragePathError(ValueError):
    """Raised when a caller attempts to address a path outside managed storage."""


def _validate_identifier(value: str, *, field: str) -> str:
    if not _IDENTIFIER_PATTERN.fullmatch(value) or value in {".", ".."}:
        raise StoragePathError(
            f"{field} must contain only ASCII letters, digits, dot, underscore, or hyphen"
        )
    return value


def _validate_filename(value: str, *, field: str = "filename") -> str:
    if not value or len(value) > 255 or value in {".", ".."}:
        raise StoragePathError(f"{field} is empty, reserved, or too long")
    if any(ord(character) < 32 for character in value):
        raise StoragePathError(f"{field} contains a control character")

    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or posix.name != value
        or windows.name != value
    ):
        raise StoragePathError(f"{field} must be a single relative path component")
    return value


class StoragePaths:
    """Build all runtime paths underneath one resolved output root.

    Identifiers and filenames are validated before joining. Every public path can
    also be passed through :meth:`require_managed` before file I/O so symlinks and
    caller-supplied absolute paths cannot escape the configured root.
    """

    def __init__(self, output_root: str | Path) -> None:
        root = Path(output_root).expanduser().resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir():
            raise StoragePathError("output_root must be a directory")
        self._root = root

    @classmethod
    def from_repo_root(
        cls,
        repo_root: str | Path,
        output_root: str | Path = "outputs",
    ) -> Self:
        """Resolve a configured relative output directory beneath a repository root."""

        repository = Path(repo_root).expanduser().resolve(strict=False)
        configured = Path(output_root)
        if configured.is_absolute():
            raise StoragePathError("repo-relative output_root cannot be absolute")
        candidate = (repository / configured).resolve(strict=False)
        try:
            candidate.relative_to(repository)
        except ValueError as error:
            raise StoragePathError("output_root escapes the repository root") from error
        return cls(candidate)

    @property
    def root(self) -> Path:
        return self._root

    def require_managed(
        self,
        path: str | Path,
        *,
        must_exist: bool = False,
        allow_root: bool = False,
    ) -> Path:
        """Resolve *path* and prove it remains within the managed output root."""

        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self._root / candidate
        lexical = Path(os.path.abspath(candidate))
        try:
            relative = lexical.relative_to(self._root)
        except ValueError as error:
            raise StoragePathError("path escapes the managed output root") from error

        current = self._root
        for component in relative.parts:
            current /= component
            if current.is_symlink():
                raise StoragePathError("symbolic links are not allowed in managed paths")
        try:
            resolved = lexical.resolve(strict=must_exist)
        except (OSError, RuntimeError) as error:
            raise StoragePathError("unable to resolve managed path") from error
        try:
            resolved.relative_to(self._root)
        except ValueError as error:
            raise StoragePathError("path escapes the managed output root") from error
        if not allow_root and resolved == self._root:
            raise StoragePathError("the storage root itself is not a file target")
        return resolved

    def relative_path(self, path: str | Path) -> str:
        """Return a portable POSIX path relative to the output root."""

        managed = self.require_managed(path)
        return managed.relative_to(self._root).as_posix()

    def job_dir(self, job_id: str) -> Path:
        return self._root / _validate_identifier(job_id, field="job_id")

    def job_manifest(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "manifest.json"

    def job_config(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job_config.json"

    def input_dir(self, job_id: str, image_id: str | None = None) -> Path:
        directory = self.job_dir(job_id) / "input"
        if image_id is not None:
            directory /= _validate_identifier(image_id, field="image_id")
        return directory

    def upload_file(
        self,
        job_id: str,
        filename: str,
        *,
        image_id: str | None = None,
    ) -> Path:
        """Return the canonical input location for an uploaded image.

        When the image ID is already allocated, the output contract uses
        ``input/{image_id}/original.<suffix>``. During initial multipart staging,
        callers may omit it and retain the validated original filename directly
        beneath ``input/``.
        """

        safe_name = _validate_filename(filename)
        if image_id is None:
            return self.input_dir(job_id) / safe_name
        suffix = Path(safe_name).suffix.lower()
        return self.input_dir(job_id, image_id) / f"original{suffix}"

    def image_dir(self, job_id: str, image_id: str) -> Path:
        return (
            self.job_dir(job_id)
            / "images"
            / _validate_identifier(image_id, field="image_id")
        )

    def image_metadata(self, job_id: str, image_id: str) -> Path:
        return self.image_dir(job_id, image_id) / "metadata.json"

    def boxes_revision(self, job_id: str, image_id: str, revision: int) -> Path:
        if revision < 0:
            raise StoragePathError("revision cannot be negative")
        return self.image_dir(job_id, image_id) / f"boxes_revision_{revision:03d}.json"

    def run_dir(self, job_id: str, image_id: str, run_id: str) -> Path:
        return (
            self.image_dir(job_id, image_id)
            / "runs"
            / _validate_identifier(run_id, field="run_id")
        )

    def run_artifact(
        self,
        job_id: str,
        image_id: str,
        run_id: str,
        filename: str,
    ) -> Path:
        return self.run_dir(job_id, image_id, run_id) / _validate_filename(filename)

    def charts_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "charts"

    def chart_file(self, job_id: str, filename: str) -> Path:
        return self.charts_dir(job_id) / _validate_filename(filename)

    def job_summary(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job_summary.json"

    def run_summary(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "run_summary.csv"

    def sample_summary(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "sample_summary.csv"

    def audit_summary(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "audit_summary.json"

    def software_manifest(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "software_manifest.json"

    def query_history(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "query_history.jsonl"

    def rag_citations(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "rag_citations.json"

    def export_manifest(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "export_manifest.json"

    def content_addressed_export_manifest(self, job_id: str, selection_sha256: str) -> Path:
        digest = _validate_identifier(selection_sha256, field="selection_sha256")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise StoragePathError("selection_sha256 must be a lowercase SHA-256 digest")
        return self.export_dir(job_id) / f"export-manifest-{digest}.json"

    def export_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "exports"

    def export_zip(self, job_id: str, filename: str = "nanoloop-export.zip") -> Path:
        safe_name = _validate_filename(filename)
        if Path(safe_name).suffix.lower() != ".zip":
            raise StoragePathError("export filename must use the .zip suffix")
        return self.export_dir(job_id) / safe_name

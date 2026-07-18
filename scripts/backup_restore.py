#!/usr/bin/env python3
"""Operator CLI for offline NanoLoop state backup verification and restore."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

from sqlalchemy.engine import make_url

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.core.config import Settings  # noqa: E402
from app.operations.backup import (  # noqa: E402
    BackupLayout,
    create_backup,
    restore_backup,
    verify_backup,
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CliUsageError(ValueError):
    """A safe-to-display CLI or configuration error."""


def _path(value: str | Path) -> Path:
    # Preserve the final path component so the core can reject symlinks with lstat/O_NOFOLLOW.
    return Path(value).expanduser().absolute()


def _file_sqlite_path(database_url: str) -> Path:
    """Resolve one ordinary file-backed SQLite URL without echoing its value."""

    try:
        url = make_url(database_url)
    except Exception as error:
        raise CliUsageError("DATABASE_URL is invalid") from error
    database = url.database
    if (
        url.get_backend_name() != "sqlite"
        or database is None
        or not database.strip()
        or database == ":memory:"
        or database.startswith("file:")
        or str(url.query.get("mode", "")).casefold() == "memory"
    ):
        raise CliUsageError("backup creation requires an ordinary file-backed SQLite database")
    return _path(database)


def _zip_path(value: str | Path) -> Path:
    path = _path(value)
    if path.suffix.casefold() != ".zip":
        raise CliUsageError("backup archives must use the .zip suffix")
    return path


def _value(result: object, names: Sequence[str]) -> object | None:
    for name in names:
        if isinstance(result, Mapping) and name in result:
            return result[name]
        if hasattr(result, name):
            return getattr(result, name)
    return None


def _safe_result_metrics(result: object) -> dict[str, object]:
    """Return only stable, non-sensitive counters and digests from a core report."""

    metrics: dict[str, object] = {}
    digest = _value(result, ("archive_sha256", "sha256"))
    if isinstance(digest, str) and _SHA256_PATTERN.fullmatch(digest):
        metrics["archive_sha256"] = digest

    for output_name, aliases in (
        (
            "file_count",
            (
                "file_count",
                "member_count",
                "verified_file_count",
                "restored_file_count",
            ),
        ),
        ("total_bytes", ("total_bytes", "size_bytes", "verified_bytes")),
    ):
        value = _value(result, aliases)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            metrics[output_name] = value

    manifest = _value(result, ("manifest",))
    files = getattr(manifest, "files", None)
    if isinstance(files, (list, tuple)):
        metrics.setdefault("file_count", len(files))
        sizes = [getattr(record, "size", None) for record in files]
        if all(
            isinstance(size, int) and not isinstance(size, bool) and size >= 0 for size in sizes
        ):
            metrics.setdefault("total_bytes", sum(sizes))
    return metrics


def _safe_checksum_path(result: object, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    value = _value(result, ("checksum_path",))
    if isinstance(value, (str, Path)):
        return _path(value)
    return None


def _emit(payload: Mapping[str, object], *, stream: TextIO | None = None) -> None:
    print(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        file=sys.stdout if stream is None else stream,
    )


def _layout_from_args(args: argparse.Namespace) -> BackupLayout:
    try:
        settings = Settings()
    except Exception as error:
        raise CliUsageError("NanoLoop configuration is invalid") from error

    configured_database = _file_sqlite_path(settings.database_url)
    database_path = _path(args.database_path) if args.database_path else configured_database
    data_root = _path(args.data_root) if args.data_root else database_path.parent
    output_root = _path(args.output_root) if args.output_root else _path(settings.output_root)
    model_snapshot_root = (
        _path(args.model_snapshot_root)
        if args.model_snapshot_root
        else _path(settings.model_snapshot_root)
    )
    knowledge_source_root = (
        _path(args.knowledge_source_root)
        if args.knowledge_source_root
        else _path(settings.knowledge_source_dir)
    )
    knowledge_index_root = (
        _path(args.knowledge_index_root)
        if args.knowledge_index_root
        else _path(settings.faiss_index_path).parent
    )
    configured_secret_file = os.environ.get("NANOLOOP_FILE_TOKEN_SECRET_FILE")
    selected_secret_file = args.file_token_secret_file or configured_secret_file
    if selected_secret_file:
        file_token_secret_file: Path | None = _path(selected_secret_file)
    else:
        default_secret_file = data_root / ".file_token_secret"
        file_token_secret_file = (
            default_secret_file if os.path.lexists(default_secret_file) else None
        )
    return BackupLayout(
        database_path=database_path,
        data_root=data_root,
        output_root=output_root,
        model_snapshot_root=model_snapshot_root,
        knowledge_source_root=knowledge_source_root,
        knowledge_index_root=knowledge_index_root,
        file_token_secret_file=file_token_secret_file,
    )


def _add_create_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("archive", help="destination .zip archive")
    parser.add_argument(
        "--offline-confirmed",
        action="store_true",
        required=True,
        help="confirm the API and all writers are stopped",
    )
    parser.add_argument("--database-path", help="override the SQLite file path")
    parser.add_argument("--data-root", help="override the managed data root")
    parser.add_argument("--output-root", help="override the output artifact root")
    parser.add_argument("--model-snapshot-root", help="override the model snapshot root")
    parser.add_argument("--knowledge-source-root", help="override the knowledge source root")
    parser.add_argument("--knowledge-index-root", help="override the knowledge index root")
    parser.add_argument(
        "--file-token-secret-file",
        help="override the persisted download-token secret file",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create, verify, or restore an offline NanoLoop state backup."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="create an offline backup")
    _add_create_arguments(create_parser)

    verify_parser = subparsers.add_parser("verify", help="verify an archive without restoring")
    verify_parser.add_argument("archive", help="source .zip archive")
    verify_parser.add_argument("--checksum-path", help="explicit SHA-256 sidecar path")

    restore_parser = subparsers.add_parser("restore", help="restore into a fresh destination root")
    restore_parser.add_argument("archive", help="source .zip archive")
    restore_parser.add_argument("destination_root", help="fresh restore destination")
    restore_parser.add_argument("--checksum-path", help="explicit SHA-256 sidecar path")
    restore_parser.add_argument(
        "--offline-confirmed",
        action="store_true",
        required=True,
        help="confirm the API and all writers are stopped",
    )
    return parser


def _run_create(args: argparse.Namespace) -> dict[str, object]:
    archive_path = _zip_path(args.archive)
    result = create_backup(
        _layout_from_args(args),
        archive_path,
        offline_confirmed=True,
    )
    payload: dict[str, object] = {
        "archive_path": str(archive_path),
        "command": "create",
        "status": "ok",
    }
    checksum_path = _safe_checksum_path(result, None)
    if checksum_path is not None:
        payload["checksum_path"] = str(checksum_path)
    payload.update(_safe_result_metrics(result))
    return payload


def _run_verify(args: argparse.Namespace) -> dict[str, object]:
    archive_path = _zip_path(args.archive)
    checksum_path = _path(args.checksum_path) if args.checksum_path else None
    result = verify_backup(archive_path, checksum_path=checksum_path)
    payload: dict[str, object] = {
        "archive_path": str(archive_path),
        "command": "verify",
        "status": "ok",
    }
    safe_checksum = _safe_checksum_path(result, checksum_path)
    if safe_checksum is not None:
        payload["checksum_path"] = str(safe_checksum)
    payload.update(_safe_result_metrics(result))
    return payload


def _run_restore(args: argparse.Namespace) -> dict[str, object]:
    archive_path = _zip_path(args.archive)
    destination_root = _path(args.destination_root)
    checksum_path = _path(args.checksum_path) if args.checksum_path else None
    result = restore_backup(
        archive_path,
        destination_root,
        offline_confirmed=True,
        checksum_path=checksum_path,
    )
    payload: dict[str, object] = {
        "archive_path": str(archive_path),
        "command": "restore",
        "destination_root": str(destination_root),
        "status": "ok",
    }
    safe_checksum = _safe_checksum_path(result, checksum_path)
    if safe_checksum is not None:
        payload["checksum_path"] = str(safe_checksum)
    payload.update(_safe_result_metrics(result))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "create":
            payload = _run_create(args)
        elif args.command == "verify":
            payload = _run_verify(args)
        else:
            payload = _run_restore(args)
    except CliUsageError as error:
        _emit(
            {"error": {"message": str(error), "type": type(error).__name__}, "status": "error"},
            stream=sys.stderr,
        )
        return 2
    except Exception as error:
        # Core errors can contain absolute managed paths. Keep CLI failure output deliberately
        # narrow; operators can enable application diagnostics without leaking source layout or
        # persisted secret material through automation logs.
        _emit(
            {
                "error": {"message": "backup operation failed", "type": type(error).__name__},
                "status": "error",
            },
            stream=sys.stderr,
        )
        return 1
    _emit(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Download, freeze, verify, and fingerprint the NanoLoop demo embedding snapshot."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import Any
from uuid import uuid4

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.embedding_snapshot_fingerprint import (
    directory_tree_sha256,
    total_size,
)

MODEL_ID = "BAAI/bge-small-zh-v1.5"
MODEL_REVISION = "13942ee7a1615d20a84b41e800c63775e174f97f"
LICENSE_NAME = "MIT"
LICENSE_URL = "https://huggingface.co/BAAI/bge-small-zh-v1.5/blob/main/LICENSE"


class SnapshotRecoveryRequiredError(RuntimeError):
    """A failed promotion left explicitly named paths for manual recovery."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verified-by", required=True)
    parser.add_argument("--verified-at", required=True, help="YYYY-MM-DD")
    return parser


def _resolved_cli_path(path: Path, *, field: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"{field} must not be a symbolic link: {expanded}")
    return expanded.resolve(strict=False)


def _validate_targets(output: Path, manifest_path: Path) -> None:
    anchor = Path(output.anchor)
    home = Path.home().resolve()
    working_directory = Path.cwd().resolve()
    temporary_root = Path(tempfile.gettempdir()).resolve()
    protected = {
        anchor,
        home,
        working_directory,
        temporary_root,
    }
    if output in protected:
        raise ValueError(f"refusing dangerous output directory: {output}")
    if output in home.parents or output in working_directory.parents:
        raise ValueError(f"refusing broad output directory: {output}")
    if output.parent == anchor:
        raise ValueError(f"refusing top-level output directory: {output}")
    if manifest_path == output or output in manifest_path.parents:
        raise ValueError("manifest must be outside the embedding output directory")
    if output.exists() and not output.is_dir():
        raise ValueError(f"output target exists but is not a directory: {output}")
    if manifest_path.exists() and not manifest_path.is_file():
        raise ValueError(f"manifest target exists but is not a file: {manifest_path}")


def _validate_operator_fields(*, verified_by: str, verified_at: str, device: str) -> None:
    if not verified_by.strip():
        raise ValueError("verified-by must not be blank")
    try:
        date.fromisoformat(verified_at)
    except ValueError as error:
        raise ValueError("verified-at must be YYYY-MM-DD") from error
    if not device.strip():
        raise ValueError("device must not be blank")


def _new_sibling_path(path: Path, *, purpose: str, suffix: str = "") -> Path:
    return path.with_name(f".{path.name}.{purpose}-{uuid4().hex}{suffix}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_staged_manifest(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    staged = _new_sibling_path(manifest_path, purpose="staging", suffix=".tmp")
    data = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        with staged.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(manifest_path.parent)
        return staged
    except Exception:
        staged.unlink(missing_ok=True)
        raise


def _remove_internal_path(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _restore_after_failed_promotion(
    *,
    output: Path,
    manifest_path: Path,
    staged_snapshot: Path,
    staged_manifest: Path,
    output_backup: Path | None,
    manifest_backup: Path | None,
    output_promoted: bool,
    manifest_promoted: bool,
) -> list[str]:
    errors: list[str] = []

    try:
        if manifest_backup is not None and manifest_backup.exists():
            os.replace(manifest_backup, manifest_path)
        elif manifest_promoted and manifest_path.exists():
            os.replace(manifest_path, staged_manifest)
    except Exception as error:
        errors.append(f"manifest rollback failed: {type(error).__name__}: {error}")

    try:
        if output_backup is not None and output_backup.exists():
            if output_promoted and output.exists():
                os.replace(output, staged_snapshot)
            os.replace(output_backup, output)
        elif output_promoted and output.exists():
            os.replace(output, staged_snapshot)
    except Exception as error:
        errors.append(f"snapshot rollback failed: {type(error).__name__}: {error}")

    for parent in {output.parent, manifest_path.parent}:
        try:
            _fsync_directory(parent)
        except Exception as error:
            errors.append(
                f"rollback fsync failed for {parent}: {type(error).__name__}: {error}"
            )
    return errors


def _promote_snapshot(
    *,
    staged_snapshot: Path,
    output: Path,
    staged_manifest: Path,
    manifest_path: Path,
    replace_existing: bool,
) -> None:
    if not replace_existing and (output.exists() or manifest_path.exists()):
        existing = output if output.exists() else manifest_path
        raise FileExistsError(f"snapshot asset appeared during preparation: {existing}")
    output_backup = (
        _new_sibling_path(output, purpose="backup") if output.exists() else None
    )
    manifest_backup = (
        _new_sibling_path(manifest_path, purpose="backup")
        if manifest_path.exists()
        else None
    )
    output_promoted = False
    manifest_promoted = False
    try:
        if output_backup is not None:
            os.replace(output, output_backup)
        os.replace(staged_snapshot, output)
        output_promoted = True

        if manifest_backup is not None:
            os.replace(manifest_path, manifest_backup)
        os.replace(staged_manifest, manifest_path)
        manifest_promoted = True

        for parent in {output.parent, manifest_path.parent}:
            _fsync_directory(parent)
    except Exception as promotion_error:
        rollback_errors = _restore_after_failed_promotion(
            output=output,
            manifest_path=manifest_path,
            staged_snapshot=staged_snapshot,
            staged_manifest=staged_manifest,
            output_backup=output_backup,
            manifest_backup=manifest_backup,
            output_promoted=output_promoted,
            manifest_promoted=manifest_promoted,
        )
        if rollback_errors:
            retained = sorted(
                str(path)
                for path in (
                    output,
                    output_backup,
                    staged_snapshot,
                    manifest_path,
                    manifest_backup,
                    staged_manifest,
                )
                if path is not None and path.exists()
            )
            details = "; ".join(rollback_errors)
            raise SnapshotRecoveryRequiredError(
                "snapshot promotion failed and automatic rollback was incomplete; "
                f"retained paths={retained}; {details}"
            ) from promotion_error
        raise

    for backup in (output_backup, manifest_backup):
        try:
            _remove_internal_path(backup)
        except OSError as error:
            print(
                f"warning: committed snapshot backup could not be removed: {backup}: {error}",
                file=sys.stderr,
            )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    staged_snapshot: Path | None = None
    staged_manifest: Path | None = None
    preserve_recovery_paths = False
    try:
        output = _resolved_cli_path(args.output_dir, field="output-dir")
        manifest_path = _resolved_cli_path(args.manifest, field="manifest")
        _validate_targets(output, manifest_path)
        _validate_operator_fields(
            verified_by=args.verified_by,
            verified_at=args.verified_at,
            device=args.device,
        )
        if (output.exists() or manifest_path.exists()) and not args.force:
            existing = output if output.exists() else manifest_path
            raise ValueError(f"snapshot asset already exists; use --force to replace: {existing}")

        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer

        output.parent.mkdir(parents=True, exist_ok=True)
        staged_snapshot = Path(
            tempfile.mkdtemp(
                prefix=f".{output.name}.staging-",
                dir=output.parent,
            )
        ).resolve()

        started = time.perf_counter()
        resolved = snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=str(staged_snapshot),
        )
        if Path(resolved).resolve() != staged_snapshot:
            raise RuntimeError(f"snapshot resolved to unexpected directory: {resolved}")
        downloaded_seconds = time.perf_counter() - started

        load_started = time.perf_counter()
        model = SentenceTransformer(
            str(staged_snapshot),
            device=args.device,
            local_files_only=True,
        )
        vector = model.encode(
            ["钙钛矿氧化物中的纳米颗粒析出"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        cold_start_seconds = time.perf_counter() - load_started
        if getattr(vector, "shape", None) is None or vector.shape[0] != 1:
            raise RuntimeError("embedding smoke test returned an invalid batch")
        dimension = int(vector.shape[1])
        if dimension != 512:
            raise RuntimeError(f"unexpected embedding dimension: {dimension}")

        tree_sha = directory_tree_sha256(staged_snapshot)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "verified",
            "verified": True,
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "dimension": dimension,
            "normalize": True,
            "max_length": int(getattr(model, "max_seq_length", 512)),
            "pooling": "model_defined",
            "license": LICENSE_NAME,
            "license_url": LICENSE_URL,
            "local_dir": str(output),
            "tree_sha256": tree_sha,
            "size_bytes": total_size(staged_snapshot),
            "resource": {
                "device": args.device,
                "download_seconds": round(downloaded_seconds, 3),
                "cold_start_seconds": round(cold_start_seconds, 3),
            },
            "verified_by": args.verified_by.strip(),
            "verified_at": args.verified_at,
            "notes": (
                "Immutable local snapshot prepared on a network-enabled asset machine; "
                "NanoLoop runtime must load this directory with local_files_only=True."
            ),
        }
        staged_manifest = _write_staged_manifest(manifest_path, manifest)
        try:
            _promote_snapshot(
                staged_snapshot=staged_snapshot,
                output=output,
                staged_manifest=staged_manifest,
                manifest_path=manifest_path,
                replace_existing=args.force,
            )
        except SnapshotRecoveryRequiredError:
            preserve_recovery_paths = True
            raise
        staged_snapshot = None
        staged_manifest = None
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        error_payload = {
            "status": "error",
            "error_type": type(error).__name__,
            "message": str(error),
        }
        print(
            json.dumps(error_payload, ensure_ascii=False, indent=2),
            file=sys.stderr,
        )
        return 1
    finally:
        if not preserve_recovery_paths:
            for temporary_path in (staged_snapshot, staged_manifest):
                try:
                    _remove_internal_path(temporary_path)
                except OSError as error:
                    print(
                        f"warning: temporary snapshot asset could not be removed: "
                        f"{temporary_path}: {error}",
                        file=sys.stderr,
                    )


if __name__ == "__main__":
    raise SystemExit(main())

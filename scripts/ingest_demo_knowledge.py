"""Validate and ingest the curated NanoLoop demo knowledge package through the public API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import httpx


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(package: Path) -> dict[str, Any]:
    path = package / "manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid manifest {path}: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("manifest must be an object with schema_version=1")
    documents = value.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("manifest.documents must be a non-empty list")
    return value


def safe_document_path(package: Path, relative_value: str) -> Path:
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"unsafe document path: {relative_value!r}")
    candidate = (package / Path(*relative.parts)).resolve()
    try:
        candidate.relative_to(package.resolve())
    except ValueError as error:
        raise ValueError(f"document path escapes package: {relative_value!r}") from error
    if not candidate.is_file():
        raise ValueError(f"knowledge document is missing: {candidate}")
    return candidate


def validate_document(package: Path, document: dict[str, Any]) -> tuple[Path, dict[str, Any], str]:
    asset_id = document.get("asset_id")
    if not isinstance(asset_id, str) or not asset_id.strip():
        raise ValueError("each document requires asset_id")
    relative_path = document.get("path")
    if not isinstance(relative_path, str):
        raise ValueError(f"{asset_id}: path must be a string")
    path = safe_document_path(package, relative_path)
    expected_sha = document.get("sha256")
    observed_sha = sha256_file(path)
    if expected_sha != observed_sha:
        raise ValueError(
            f"{asset_id}: SHA-256 mismatch; expected {expected_sha}, observed {observed_sha}"
        )
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{asset_id}: metadata must be an object")
    required = {
        "title",
        "source_type",
        "citation_text",
        "material_aliases",
        "license_note",
        "allowed_for_demo",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"{asset_id}: metadata missing {missing}")
    aliases = metadata.get("material_aliases")
    if not isinstance(aliases, list) or not aliases or not all(
        isinstance(item, str) and item.strip() for item in aliases
    ):
        raise ValueError(f"{asset_id}: material_aliases must be non-empty strings")
    if metadata.get("allowed_for_demo") is not True:
        raise ValueError(f"{asset_id}: curated demo document must be allowed_for_demo=true")
    return path, metadata, observed_sha


def request_headers() -> dict[str, str]:
    key = os.getenv("NANOLOOP_API_KEY", "").strip()
    return {"X-API-Key": key} if key else {}


def unwrap(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError as error:
        raise RuntimeError(
            f"API returned non-JSON response: {response.status_code} {response.text[:300]}"
        ) from error
    if not isinstance(body, dict):
        raise RuntimeError("API response must be a JSON object")
    if response.is_error or body.get("status") == "error":
        error_payload = body.get("error")
        raise RuntimeError(
            f"API request failed ({response.status_code}): "
            f"{json.dumps(error_payload or body, ensure_ascii=False)}"
        )
    data = body.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("API success response has no object data")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=Path,
        default=Path("demo_data/rag"),
        help="Curated package root containing manifest.json",
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--skip-reindex", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    package = args.package.expanduser().resolve(strict=True)
    manifest = load_manifest(package)
    base = args.api_base.rstrip("/")
    headers = request_headers()
    results: list[dict[str, Any]] = []
    try:
        with httpx.Client(base_url=base, headers=headers, timeout=args.timeout) as client:
            for raw in manifest["documents"]:
                if not isinstance(raw, dict):
                    raise ValueError("manifest.documents entries must be objects")
                path, metadata, observed_sha = validate_document(package, raw)
                with path.open("rb") as handle:
                    response = client.post(
                        "/api/v1/knowledge/documents",
                        files={"file": (path.name, handle, "text/markdown; charset=utf-8")},
                        data={"metadata_json": json.dumps(metadata, ensure_ascii=False)},
                    )
                data = unwrap(response)
                returned_sha = data.get("sha256")
                if returned_sha != observed_sha:
                    raise RuntimeError(
                        f"{raw['asset_id']}: API SHA mismatch; expected {observed_sha}, "
                        f"received {returned_sha}"
                    )
                results.append(
                    {
                        "asset_id": raw["asset_id"],
                        "doc_id": data.get("doc_id"),
                        "sha256": returned_sha,
                        "chunks_created": data.get("chunks_created"),
                        "warnings": data.get("warnings", []),
                    }
                )

            reindex_data: dict[str, Any] | None = None
            if not args.skip_reindex:
                reindex_data = unwrap(client.post("/api/v1/knowledge/reindex", json={"force": False}))

            catalogue = unwrap(client.get("/api/v1/knowledge/documents"))
            indexed = catalogue.get("documents", [])
            if not isinstance(indexed, list):
                raise RuntimeError("knowledge catalogue has invalid documents field")
            expected_shas = {item["sha256"] for item in results}
            present_shas = {
                item.get("sha256")
                for item in indexed
                if isinstance(item, dict) and item.get("status") == "ready"
            }
            missing = sorted(expected_shas - present_shas)
            if missing:
                raise RuntimeError(f"ingested demo documents missing from ready catalogue: {missing}")

        report = {
            "status": "ok",
            "package": manifest.get("package"),
            "documents": results,
            "reindex": reindex_data,
            "ready_document_count": len(present_shas),
        }
        rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0
    except Exception as error:
        print(
            json.dumps(
                {"status": "error", "error_type": type(error).__name__, "message": str(error)},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

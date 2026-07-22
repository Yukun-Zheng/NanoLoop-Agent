"""Run one fail-closed RAG acceptance pass against the public HTTP API.

The checked-in package is intentionally a draft. This command refuses network
access until corpus files, licences, independent review, question pages and the
local embedding snapshot all satisfy ``validate_rag_assets --require-runnable``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import urlsplit

import httpx

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_rag_assets import (
    AssetValidationError,
    ValidatedAcceptancePackage,
    validate_acceptance_package,
)

_RUN_LABEL = re.compile(r"^[a-z0-9][a-z0-9_-]{1,30}$")
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


class AcceptanceRunError(RuntimeError):
    """The remote run failed or returned evidence that violates the contract."""


class HttpClient(Protocol):
    def get(self, url: str, **kwargs: Any) -> httpx.Response: ...

    def post(self, url: str, **kwargs: Any) -> httpx.Response: ...


def normalize_api_base(value: str, *, allow_insecure_remote_http: bool = False) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise AcceptanceRunError("api base must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise AcceptanceRunError("api base must not contain credentials, query, or fragment")
    if (
        parsed.scheme == "http"
        and parsed.hostname not in _LOOPBACK_HOSTS
        and not allow_insecure_remote_http
    ):
        raise AcceptanceRunError("remote acceptance requires HTTPS")
    return value.strip().rstrip("/")


def api_headers(environment_name: str) -> dict[str, str]:
    api_key = os.environ.get(environment_name, "")
    return {"X-API-Key": api_key} if api_key else {}


def _response_payload(response: httpx.Response, *, operation: str) -> tuple[str, dict[str, Any]]:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        raise AcceptanceRunError(
            f"{operation} failed with HTTP {response.status_code}"
        ) from error
    try:
        payload = response.json()
    except ValueError as error:
        raise AcceptanceRunError(f"{operation} returned non-JSON data") from error
    if not isinstance(payload, dict):
        raise AcceptanceRunError(f"{operation} returned a non-object envelope")
    request_id = payload.get("request_id")
    status = payload.get("status")
    data = payload.get("data")
    error = payload.get("error")
    if not isinstance(request_id, str) or not request_id:
        raise AcceptanceRunError(f"{operation} response is missing request_id")
    expected_status = "accepted" if response.status_code == 202 else "success"
    if status != expected_status or error is not None:
        raise AcceptanceRunError(
            f"{operation} returned an invalid success envelope: "
            f"expected status={expected_status!r} and error=null"
        )
    if not isinstance(data, dict):
        raise AcceptanceRunError(f"{operation} response is missing object data")
    return request_id, cast(dict[str, Any], data)


def fetch_health(client: HttpClient, api_base: str) -> dict[str, Any]:
    response = client.get(f"{api_base}/api/v1/health", timeout=30.0)
    request_id, data = _response_payload(response, operation="health")
    return {"request_id": request_id, "data": data}


def _material_aliases(row: Mapping[str, str]) -> list[str]:
    aliases = {alias.strip() for alias in row["aliases"].split(";") if alias.strip()}
    return sorted(aliases, key=lambda value: (value.casefold(), value))


def _document_metadata(row: Mapping[str, str]) -> dict[str, Any]:
    return {
        "title": row["title"],
        "source_type": "paper",
        "year": int(row["year"]),
        "citation_text": row["citation_text"],
        "material_aliases": _material_aliases(row),
        "license_note": (
            f"{row['license']}; terms={row['license_url']}; "
            f"evidence={row['license_evidence_url']}; reviewed_by={row['reviewed_by']}; "
            f"reviewed_at={row['reviewed_at']}"
        ),
        "allowed_for_demo": True,
    }


def ingest_corpus(
    client: HttpClient,
    api_base: str,
    headers: Mapping[str, str],
    package: ValidatedAcceptancePackage,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    results: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}
    sources = package.root / "corpus" / "sources"
    for row in package.accepted_rows:
        asset_id = row["asset_id"]
        source = sources / row["file_path"]
        metadata = _document_metadata(row)
        with source.open("rb") as handle:
            response = client.post(
                f"{api_base}/api/v1/knowledge/documents",
                headers=dict(headers),
                files={"file": (source.name, handle, "application/octet-stream")},
                data={"metadata_json": json.dumps(metadata, ensure_ascii=False)},
                timeout=120.0,
            )
        if response.status_code == 409:
            raise AcceptanceRunError(
                f"ingest {asset_id} conflicts with an existing document; "
                "a 409 means the stored metadata differs and cannot be reused"
            )
        request_id, data = _response_payload(response, operation=f"ingest {asset_id}")
        doc_id = data.get("doc_id")
        observed_sha = data.get("sha256")
        if not isinstance(doc_id, str) or not doc_id:
            raise AcceptanceRunError(f"ingest {asset_id} did not return doc_id")
        if observed_sha != row["file_sha256"]:
            raise AcceptanceRunError(
                f"ingest {asset_id} SHA mismatch: expected {row['file_sha256']}, "
                f"observed {observed_sha}"
            )
        mapping[asset_id] = doc_id
        results.append(
            {
                "asset_id": asset_id,
                "doc_id": doc_id,
                "source_sha256": observed_sha,
                "ingest_status": "accepted",
                "request_id": request_id,
                "pages_total": data.get("pages_total"),
                "pages_extracted": data.get("pages_extracted"),
                "chunks_created": data.get("chunks_created"),
                "warnings": data.get("warnings", []),
                "index_version": data.get("index_version"),
            }
        )
    return results, mapping


def verify_runtime_mapping(
    client: HttpClient,
    api_base: str,
    headers: Mapping[str, str],
    package: ValidatedAcceptancePackage,
    asset_to_doc: Mapping[str, str],
) -> list[dict[str, str]]:
    """Bind a saved mapping back to authoritative runtime document facts."""

    response = client.get(
        f"{api_base}/api/v1/knowledge/documents",
        headers=dict(headers),
        timeout=30.0,
    )
    _, data = _response_payload(response, operation="list knowledge documents")
    documents = data.get("documents")
    if not isinstance(documents, list):
        raise AcceptanceRunError("knowledge document list is missing documents")
    by_id: dict[str, dict[str, Any]] = {}
    for document in documents:
        if not isinstance(document, dict):
            raise AcceptanceRunError("knowledge document list contains a non-object")
        doc_id = document.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            raise AcceptanceRunError("knowledge document list contains an invalid doc_id")
        if doc_id in by_id:
            raise AcceptanceRunError(f"knowledge document list repeats doc_id: {doc_id}")
        by_id[doc_id] = document

    expected_assets = {row["asset_id"] for row in package.accepted_rows}
    if set(asset_to_doc) != expected_assets:
        raise AcceptanceRunError("runtime mapping does not cover exactly the accepted assets")
    if len(set(asset_to_doc.values())) != len(asset_to_doc):
        raise AcceptanceRunError("runtime mapping assigns one doc_id to multiple assets")

    verified: list[dict[str, str]] = []
    for row in package.accepted_rows:
        asset_id = row["asset_id"]
        doc_id = asset_to_doc[asset_id]
        document = by_id.get(doc_id)
        if document is None:
            raise AcceptanceRunError(
                f"runtime mapping for {asset_id} references missing doc_id {doc_id}"
            )
        metadata = _document_metadata(row)
        expected = {
            "sha256": row["file_sha256"],
            "title": metadata["title"],
            "source_type": metadata["source_type"],
            "year": metadata["year"],
            "citation_text": metadata["citation_text"],
            "material_aliases": metadata["material_aliases"],
            "license_note": metadata["license_note"],
            "allowed_for_demo": True,
            "status": "ready",
        }
        mismatches = [
            field
            for field, expected_value in expected.items()
            if document.get(field) != expected_value
        ]
        if mismatches:
            raise AcceptanceRunError(
                f"runtime document {doc_id} for {asset_id} mismatches fields: {mismatches}"
            )
        verified.append(
            {
                "asset_id": asset_id,
                "doc_id": doc_id,
                "sha256": row["file_sha256"],
                "status": "ready",
            }
        )
    return verified


def load_mapping(path: Path, package: ValidatedAcceptancePackage) -> dict[str, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AcceptanceRunError(f"invalid mapping file {path}: {error}") from error
    if not isinstance(value, list):
        raise AcceptanceRunError("mapping file must contain the prior ingest result list")
    expected_shas = {
        row["asset_id"]: row["file_sha256"] for row in package.accepted_rows
    }
    mapping: dict[str, str] = {}
    for item in value:
        if not isinstance(item, dict):
            raise AcceptanceRunError("mapping file entries must be objects")
        asset_id = item.get("asset_id")
        doc_id = item.get("doc_id")
        source_sha256 = item.get("source_sha256")
        if not isinstance(asset_id, str) or not isinstance(doc_id, str) or not doc_id:
            raise AcceptanceRunError("mapping file entry is missing asset_id/doc_id")
        if not isinstance(source_sha256, str) or not source_sha256:
            raise AcceptanceRunError(
                f"mapping file entry for {asset_id} is missing source_sha256"
            )
        if asset_id not in expected_shas:
            raise AcceptanceRunError(
                f"mapping file contains non-accepted or unknown asset: {asset_id}"
            )
        if asset_id in mapping:
            raise AcceptanceRunError(f"mapping file repeats asset_id: {asset_id}")
        if source_sha256 != expected_shas[asset_id]:
            raise AcceptanceRunError(
                f"mapping file SHA mismatch for {asset_id}: "
                f"expected {expected_shas[asset_id]}, observed {source_sha256}"
            )
        mapping[asset_id] = doc_id
    if len(set(mapping.values())) != len(mapping):
        raise AcceptanceRunError("mapping file maps multiple assets to the same doc_id")
    missing = sorted(expected_shas.keys() - mapping.keys())
    if missing:
        raise AcceptanceRunError(f"mapping file is missing accepted assets: {missing}")
    return mapping


def _citation_records(
    citations: object,
    doc_to_asset: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    if not isinstance(citations, list):
        raise AcceptanceRunError("query response citations must be a list")
    records: list[dict[str, Any]] = []
    asset_ids: list[str] = []
    unmapped: list[str] = []
    for citation in citations:
        if not isinstance(citation, dict):
            raise AcceptanceRunError("query response citation must be an object")
        doc_id = citation.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            raise AcceptanceRunError("query response citation is missing doc_id")
        asset_id = doc_to_asset.get(doc_id)
        if asset_id is None:
            unmapped.append(doc_id)
        else:
            asset_ids.append(asset_id)
        records.append(
            {
                "citation_id": citation.get("citation_id"),
                "asset_id": asset_id,
                "doc_id": doc_id,
                "page": citation.get("page"),
                "chunk_id": citation.get("chunk_id"),
                "excerpt": citation.get("excerpt"),
                "retrieval_score": citation.get("retrieval_score"),
                "citation_text": citation.get("citation_text"),
            }
        )
    return records, asset_ids, unmapped


def run_queries(
    client: HttpClient,
    api_base: str,
    headers: Mapping[str, str],
    job_id: str,
    package: ValidatedAcceptancePackage,
    asset_to_doc: Mapping[str, str],
) -> list[dict[str, Any]]:
    doc_to_asset = {doc_id: asset_id for asset_id, doc_id in asset_to_doc.items()}
    if len(doc_to_asset) != len(asset_to_doc):
        raise AcceptanceRunError("multiple asset IDs map to the same runtime doc_id")
    results: list[dict[str, Any]] = []
    for question in package.questions:
        query_id = cast(str, question["query_id"])
        response = client.post(
            f"{api_base}/api/v1/analyses/{job_id}/query",
            headers=dict(headers),
            json={
                "question": question["question"],
                "query_type": question["query_type"],
                "material_context": question["material_context"],
            },
            timeout=120.0,
        )
        request_id, data = _response_payload(response, operation=f"query {query_id}")
        citations, observed_assets, unmapped = _citation_records(
            data.get("citations", []), doc_to_asset
        )
        observed = set(observed_assets)
        relevant = set(cast(Sequence[str], question["relevant_asset_ids"]))
        forbidden = set(cast(Sequence[str], question["must_not_return_asset_ids"]))
        expected_outcome = question["expected_outcome"]
        observed_outcome = data.get("outcome_code")
        checks = {
            "query_type_matches": data.get("query_type") == question["query_type"],
            "outcome_matches": observed_outcome == expected_outcome,
            "relevant_asset_retrieved": (
                bool(observed & relevant) if expected_outcome == "OK" else True
            ),
            "forbidden_asset_absent": not bool(observed & forbidden),
            "no_unmapped_citation": not unmapped,
            "insufficient_has_zero_citations": (
                not citations if expected_outcome == "INSUFFICIENT_EVIDENCE" else True
            ),
        }
        results.append(
            {
                "query_id": query_id,
                "request_id": request_id,
                "case_type": question["case_type"],
                "expected_outcome": expected_outcome,
                "observed_outcome": observed_outcome,
                "expected_relevant_asset_ids": sorted(relevant),
                "observed_asset_ids": observed_assets,
                "forbidden_asset_ids": sorted(forbidden),
                "unmapped_doc_ids": unmapped,
                "citations": citations,
                "answer": data.get("answer"),
                "limitations": data.get("limitations", []),
                "automated_checks": checks,
                "automated_checks_passed": all(checks.values()),
                "human_review": {
                    "status": "pending",
                    "citation_correct": None,
                    "material_leakage": None,
                    "unsupported_fact": None,
                    "reviewed_by": None,
                    "reviewed_at": None,
                },
                "final_passed": None,
            }
        )
    return results


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    text = "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values)
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None, *, client: HttpClient | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one real-asset RAG acceptance pass")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--package", default="rag-acceptance-v1")
    parser.add_argument("--job-id", required=True)
    parser.add_argument(
        "--run-label", required=True, help="for example keyword-only, hybrid, restart"
    )
    parser.add_argument("--api-key-env", default="NANOLOOP_API_KEY")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--mapping-file", type=Path)
    parser.add_argument(
        "--expect-rag-health", choices=("healthy", "degraded", "unavailable"), required=True
    )
    parser.add_argument("--allow-insecure-remote-http", action="store_true")
    args = parser.parse_args(argv)

    if not _RUN_LABEL.fullmatch(args.run_label):
        parser.error("--run-label must match [a-z0-9][a-z0-9_-]{1,30}")
    if args.skip_ingest != (args.mapping_file is not None):
        parser.error("--skip-ingest and --mapping-file must be supplied together")
    try:
        package = validate_acceptance_package(
            args.package,
            require_runnable=True,
            verify_files=True,
        )
        api_base = normalize_api_base(
            args.api_base,
            allow_insecure_remote_http=args.allow_insecure_remote_http,
        )
    except (AssetValidationError, AcceptanceRunError) as error:
        parser.error(str(error))

    owned_client = client is None
    runtime_client = cast(HttpClient, client or httpx.Client())
    try:
        headers = api_headers(args.api_key_env)
        health_before = fetch_health(runtime_client, api_base)
        rag_health = health_before["data"].get("rag_index")
        observed_health = rag_health.get("status") if isinstance(rag_health, dict) else None
        if observed_health != args.expect_rag_health:
            raise AcceptanceRunError(
                "RAG health mismatch: "
                f"expected {args.expect_rag_health}, observed {observed_health}"
            )
        evaluation_dir = package.root / "evaluation"
        if args.skip_ingest:
            ingest_results: list[dict[str, Any]] = []
            asset_to_doc = load_mapping(args.mapping_file, package)
        else:
            ingest_results, asset_to_doc = ingest_corpus(
                runtime_client, api_base, headers, package
            )
            _write_json(
                evaluation_dir / f"ingest-results.{args.run_label}.json", ingest_results
            )
        verified_runtime_documents = verify_runtime_mapping(
            runtime_client,
            api_base,
            headers,
            package,
            asset_to_doc,
        )
        query_results = run_queries(
            runtime_client,
            api_base,
            headers,
            args.job_id,
            package,
            asset_to_doc,
        )
        health_after = fetch_health(runtime_client, api_base)
        rag_health_after = health_after["data"].get("rag_index")
        observed_health_after = (
            rag_health_after.get("status") if isinstance(rag_health_after, dict) else None
        )
        if observed_health_after != args.expect_rag_health:
            raise AcceptanceRunError(
                "RAG health changed during the run: "
                f"expected {args.expect_rag_health}, observed {observed_health_after}"
            )
        result_path = evaluation_dir / f"query-results.{args.run_label}.jsonl"
        _write_jsonl(result_path, query_results)
        failures = [
            result["query_id"]
            for result in query_results
            if not result["automated_checks_passed"]
        ]
        summary = {
            "run_label": args.run_label,
            "health_before": health_before,
            "health_after": health_after,
            "accepted_assets": sorted(asset_to_doc),
            "verified_runtime_documents": verified_runtime_documents,
            "question_count": len(query_results),
            "automated_failure_query_ids": failures,
            "human_review_pending": len(query_results),
            "real_acceptance_completed": False,
            "note": (
                "Automated checks do not establish citation correctness. Independent "
                "human review and restart/mismatch evidence remain required."
            ),
        }
        _write_json(evaluation_dir / f"acceptance-summary.{args.run_label}.json", summary)
        if failures:
            raise AcceptanceRunError(f"automated acceptance failures: {failures}")
        print(
            json.dumps(
                {
                    "automated_checks_passed": True,
                    "human_review_pending": len(query_results),
                    "result_path": str(result_path),
                    "real_acceptance_completed": False,
                },
                ensure_ascii=False,
            )
        )
        return 0
    except (httpx.HTTPError, OSError, AcceptanceRunError) as error:
        print(
            json.dumps(
                {"error": str(error), "real_acceptance_completed": False},
                ensure_ascii=False,
            )
        )
        return 2
    finally:
        if owned_client and isinstance(runtime_client, httpx.Client):
            runtime_client.close()


if __name__ == "__main__":
    raise SystemExit(main())

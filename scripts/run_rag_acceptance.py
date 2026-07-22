"""RAG acceptance driver for D's rag-acceptance-v1 package (v4.0 FR-09 / FR-11).

Orchestrates the real-asset RAG acceptance:
  1. Ingest licensed corpus from corpus-manifest.csv via the running API.
  2. For each question in evaluation/questions.jsonl, run the unified query
     endpoint and record the answer + citations.
  3. Emit ingest / query result files for manual + F review.

Uses ONLY the public HTTP API (no private DB access) so it runs on any machine
that can reach the API, including the team asset machine.

PREREQUISITES (external assets provided by the team -- NOT in this repo):
  * 5-10 licensed corpus files under rag-acceptance-v1/corpus/sources/
  * A fixed local embedding snapshot mounted at EMBEDDING_MODEL
  * A running API (see docs/DEVELOPMENT.md) with the RAG extra installed

USAGE:
  python scripts/run_rag_acceptance.py \
      --api-base http://127.0.0.1:8000 \
      --package rag-acceptance-v1 \
      --job-id <any_valid_analysis_job_id> \
      --api-key <runtime-secret>

NOTE: material_knowledge / mixed queries route through
POST /api/v1/analyses/{job_id}/query, which requires a job scope. Any valid
analysis job_id works for knowledge-only questions (the knowledge path does not
need analysis data). To get keyword-only vs hybrid results, run this script in
two environments: one without the embedding asset (degradation) and one with it.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("requests is required: pip install requests")


def _headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": api_key} if api_key else {}


def ingest_corpus(api_base: str, api_key: str, package: Path) -> list[dict]:
    """Ingest every row of corpus-manifest.csv via POST /api/v1/knowledge/documents."""
    manifest = package / "corpus" / "corpus-manifest.csv"
    sources = package / "corpus" / "sources"
    results: list[dict] = []
    with manifest.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            file_path = sources / row["file_path"]
            if not file_path.exists():
                results.append({"asset_id": row["asset_id"], "status": "missing-source"})
                continue
            metadata = {
                "title": row["title"],
                "source_type": "paper",
                "year": int(row["year"] or 0),
                "citation_text": row["citation_text"],
                "material_aliases": [
                    a.strip() for a in row["aliases"].split(";") if a.strip()
                ],
                "license_note": row["license"],
                "allowed_for_demo": row["allowed_for_demo"].strip().lower() == "true",
            }
            with file_path.open("rb") as fh2:
                resp = requests.post(
                    f"{api_base}/api/v1/knowledge/documents",
                    headers=_headers(api_key),
                    files={"file": (file_path.name, fh2)},
                    data={"metadata_json": json.dumps(metadata)},
                    timeout=120,
                )
            resp.raise_for_status()
            results.append({"asset_id": row["asset_id"], "status": resp.status_code})
    return results


def run_queries(api_base: str, api_key: str, job_id: str, package: Path) -> list[dict]:
    questions_path = package / "evaluation" / "questions.jsonl"
    questions = [
        json.loads(line)
        for line in questions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out: list[dict] = []
    for q in questions:
        payload = {
            "question": q["question"],
            "query_type": q.get("query_type", "material_knowledge"),
            "material_context": q.get("material_context"),
        }
        resp = requests.post(
            f"{api_base}/api/v1/analyses/{job_id}/query",
            headers=_headers(api_key),
            json=payload,
            timeout=120,
        )
        data = resp.json().get("data", {})
        citations = data.get("citations", [])
        out.append(
            {
                "query_id": q["query_id"],
                "case_type": q.get("case_type"),
                "expected_outcome": q.get("expected_outcome"),
                "outcome_code": data.get("outcome_code"),
                "citation_ids": [c.get("citation_id") for c in citations],
                "doc_ids": [c.get("doc_id") for c in citations],
                "pages": [c.get("page") for c in citations],
                "chunk_ids": [c.get("chunk_id") for c in citations],
                "answer_chars": len(data.get("answer", "")),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG acceptance driver")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--package", default="rag-acceptance-v1")
    parser.add_argument(
        "--job-id",
        required=True,
        help="any valid analysis job_id (knowledge-only queries still need a job scope)",
    )
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()

    package = Path(args.package)
    ingest = ingest_corpus(args.api_base, args.api_key, package)
    queries = run_queries(args.api_base, args.api_key, args.job_id, package)

    (package / "evaluation" / "ingest-results.json").write_text(
        json.dumps(ingest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (package / "evaluation" / "query-results.json").write_text(
        json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Ingested {len(ingest)} docs; ran {len(queries)} queries. "
        f"Results in {package}/evaluation/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

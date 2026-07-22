from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest

from scripts import run_rag_acceptance as runner
from scripts.validate_rag_assets import (
    AssetValidationError,
    directory_tree_sha256,
    validate_acceptance_package,
)

_FIELDS = [
    "asset_id",
    "decision",
    "title",
    "authors",
    "year",
    "source_url",
    "doi",
    "license",
    "license_url",
    "license_evidence_url",
    "citation_text",
    "material_names",
    "formula",
    "aliases",
    "file_sha256",
    "allowed_for_demo",
    "page_text_verified",
    "reviewed_by",
    "reviewed_at",
    "file_path",
]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_runnable_package(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    package = tmp_path / "rag-package"
    sources = package / "corpus" / "sources"
    evaluation = package / "evaluation"
    embedding_dir = tmp_path / "private-embedding"
    sources.mkdir(parents=True)
    evaluation.mkdir()
    embedding_dir.mkdir()
    (embedding_dir / "model.bin").write_bytes(b"fixed embedding bytes")

    rows: list[dict[str, str]] = []
    expected_shas: dict[str, str] = {}
    for index in range(5):
        asset_id = f"asset_{index}"
        filename = f"source-{index}.txt"
        content = f"licensed source {index}".encode()
        (sources / filename).write_bytes(content)
        digest = _sha256(content)
        expected_shas[asset_id] = digest
        rows.append(
            {
                "asset_id": asset_id,
                "decision": "ACCEPT_FULLTEXT",
                "title": f"Title {asset_id}",
                "authors": "Author",
                "year": "2025",
                "source_url": f"https://example.test/{asset_id}",
                "doi": f"10.1000/{asset_id}",
                "license": "CC BY 4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "license_evidence_url": f"https://example.test/{asset_id}/license",
                "citation_text": f"Author. Title {asset_id}. 2025.",
                "material_names": "二氧化钛",
                "formula": "TiO2",
                "aliases": "二氧化钛;titanium dioxide",
                "file_sha256": digest,
                "allowed_for_demo": "true",
                "page_text_verified": "true",
                "reviewed_by": "Reviewer",
                "reviewed_at": "2026-07-23",
                "file_path": filename,
            }
        )
    with (package / "corpus" / "corpus-manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    questions = []
    for index in range(20):
        asset_id = f"asset_{index % 5}"
        questions.append(
            {
                "query_id": f"q{index + 1:03d}",
                "question": f"Question {index}",
                "language": "en",
                "query_type": "material_knowledge",
                "material_context": {
                    "formula": "TiO2",
                    "name": "titanium dioxide",
                    "aliases": ["TiO2"],
                    "source": "request",
                },
                "case_type": "direct",
                "relevant_asset_ids": [asset_id],
                "relevant_pages": [1],
                "expected_outcome": "OK",
                "must_not_return_asset_ids": [],
                "annotation_status": "final",
                "annotated_by": "Annotator",
                "reviewed_by": "Reviewer",
            }
        )
    (evaluation / "questions.jsonl").write_text(
        "".join(json.dumps(question) + "\n" for question in questions), encoding="utf-8"
    )
    (package / "embedding").mkdir()
    (package / "embedding" / "model-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "verified",
                "model": "example/model",
                "revision": "a" * 40,
                "dimension": 2,
                "normalize": True,
                "max_length": 512,
                "pooling": "mean",
                "license": "Apache-2.0",
                "license_url": "https://example.test/license",
                "local_dir": str(embedding_dir),
                "tree_sha256": directory_tree_sha256(embedding_dir),
                "size_bytes": (embedding_dir / "model.bin").stat().st_size,
                "resource": {"device": "cpu", "memory_mb": 10, "cold_start_seconds": 1},
                "verified_by": "Reviewer",
                "verified_at": "2026-07-23",
                "verified": True,
            }
        ),
        encoding="utf-8",
    )
    (package / "asset-ledger.json").write_text(
        json.dumps(
            {
                "corpus_summary": {
                    "total_candidates": 5,
                    "candidate_fulltext": 0,
                    "accepted_fulltext": 5,
                },
                "embedding_status": "verified",
                "real_acceptance_completed": False,
            }
        ),
        encoding="utf-8",
    )
    (evaluation / "judgment-schema.json").write_text(
        json.dumps(
            {
                "expected_source": "questions.jsonl",
                "observed_output_pattern": "query-results.<run_label>.jsonl",
            }
        ),
        encoding="utf-8",
    )
    return package, expected_shas


class _FakeClient:
    def __init__(self, expected_shas: dict[str, str]) -> None:
        self.expected_shas = expected_shas
        self.headers_seen: list[dict[str, str]] = []
        self.calls = 0

    @staticmethod
    def _response(method: str, url: str, status: int, data: dict[str, Any]) -> httpx.Response:
        return httpx.Response(
            status,
            request=httpx.Request(method, url),
            json={"request_id": f"req-{status}", "status": "success", "data": data, "error": None},
        )

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        del kwargs
        self.calls += 1
        return self._response(
            "GET",
            url,
            200,
            {
                "service": {"status": "healthy"},
                "database": {"status": "healthy"},
                "model_registry": {"status": "degraded"},
                "rag_index": {"status": "healthy"},
                "version": "test",
            },
        )

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        self.calls += 1
        self.headers_seen.append(dict(kwargs.get("headers", {})))
        if url.endswith("/knowledge/documents"):
            metadata = json.loads(kwargs["data"]["metadata_json"])
            asset_id = metadata["title"].removeprefix("Title ")
            return self._response(
                "POST",
                url,
                202,
                {
                    "doc_id": f"doc-{asset_id}",
                    "sha256": self.expected_shas[asset_id],
                    "pages_total": 1,
                    "pages_extracted": 1,
                    "chunks_created": 1,
                    "chunks_skipped": 0,
                    "warnings": [],
                    "index_version": "fts5-v1",
                },
            )
        question_index = int(kwargs["json"]["question"].split()[-1])
        asset_id = f"asset_{question_index % 5}"
        return self._response(
            "POST",
            url,
            200,
            {
                "query_type": "material_knowledge",
                "answer": "Supported answer [C1].",
                "data_evidence": [],
                "citations": [
                    {
                        "citation_id": "C1",
                        "doc_id": f"doc-{asset_id}",
                        "title": f"Title {asset_id}",
                        "page": 1,
                        "chunk_id": f"chunk-{asset_id}",
                        "excerpt": "Supported evidence.",
                        "retrieval_score": 1.0,
                        "citation_text": f"Citation {asset_id}",
                    }
                ],
                "tool_calls": [],
                "material_context": kwargs["json"]["material_context"],
                "confidence": "medium",
                "limitations": [],
                "needs_clarification": False,
                "outcome_code": "OK",
            },
        )


def test_checked_in_package_is_schema_valid_but_not_runnable() -> None:
    repository = Path(__file__).resolve().parents[3]
    package = validate_acceptance_package(repository / "rag-acceptance-v1")
    assert len(package.corpus_rows) == 17
    assert len(package.accepted_rows) == 0
    assert len(package.questions) == 32
    with pytest.raises(AssetValidationError, match="requires 5-10 ACCEPT_FULLTEXT"):
        validate_acceptance_package(
            repository / "rag-acceptance-v1", require_runnable=True, verify_files=True
        )


def test_runnable_package_checks_files_embedding_and_independent_review(tmp_path: Path) -> None:
    package_path, _ = _write_runnable_package(tmp_path)
    package = validate_acceptance_package(
        package_path, require_runnable=True, verify_files=True
    )
    assert len(package.accepted_rows) == 5
    assert package.embedding["verified"] is True


def test_csv_column_drift_is_rejected(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[3]
    package = tmp_path / "rag-acceptance-v1"
    shutil.copytree(repository / "rag-acceptance-v1", package)
    manifest = package / "corpus" / "corpus-manifest.csv"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    lines[1] += ",unexpected"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(AssetValidationError, match="header column count"):
        validate_acceptance_package(package)


def test_private_inputs_and_generated_results_are_gitignored() -> None:
    repository = Path(__file__).resolve().parents[3]
    targets = [
        "rag-acceptance-v1/corpus/sources/licensed.pdf",
        "rag-acceptance-v1/evaluation/query-results.hybrid.jsonl",
        "rag-acceptance-v1/evaluation/ingest-results.hybrid.json",
        "rag-acceptance-v1/evaluation/acceptance-summary.hybrid.json",
        "rag-acceptance-v1/index-evidence/private.index",
        "rag-acceptance-v1/run-record/restart-smoke.json",
    ]
    for target in targets:
        completed = subprocess.run(
            ["git", "check-ignore", "--quiet", target],
            cwd=repository,
            check=False,
        )
        assert completed.returncode == 0, target


def test_real_runner_maps_asset_ids_and_keeps_human_review_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    package_path, expected_shas = _write_runnable_package(tmp_path)
    fake = _FakeClient(expected_shas)
    monkeypatch.setenv("RAG_TEST_KEY", "do-not-print-this-secret")
    result = runner.main(
        [
            "--api-base",
            "http://127.0.0.1:8000",
            "--package",
            str(package_path),
            "--job-id",
            "job-test",
            "--run-label",
            "hybrid",
            "--api-key-env",
            "RAG_TEST_KEY",
            "--expect-rag-health",
            "healthy",
        ],
        client=fake,
    )
    assert result == 0
    assert fake.calls == 27
    assert all(
        headers.get("X-API-Key") == "do-not-print-this-secret"
        for headers in fake.headers_seen
    )
    output = capsys.readouterr().out
    assert "do-not-print-this-secret" not in output
    results = [
        json.loads(line)
        for line in (package_path / "evaluation" / "query-results.hybrid.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(results) == 20
    assert all(result["automated_checks_passed"] is True for result in results)
    assert all(result["human_review"]["status"] == "pending" for result in results)
    assert all(result["final_passed"] is None for result in results)
    summary = json.loads(
        (package_path / "evaluation" / "acceptance-summary.hybrid.json").read_text()
    )
    assert summary["real_acceptance_completed"] is False
    assert summary["human_review_pending"] == 20


def test_restart_mapping_must_match_current_manifest_sha(tmp_path: Path) -> None:
    package_path, expected_shas = _write_runnable_package(tmp_path)
    package = validate_acceptance_package(
        package_path, require_runnable=True, verify_files=True
    )
    mapping = [
        {
            "asset_id": asset_id,
            "doc_id": f"doc-{asset_id}",
            "source_sha256": digest,
        }
        for asset_id, digest in expected_shas.items()
    ]
    mapping[0]["source_sha256"] = "0" * 64
    mapping_path = tmp_path / "stale-ingest-results.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    with pytest.raises(runner.AcceptanceRunError, match="mapping file SHA mismatch"):
        runner.load_mapping(mapping_path, package)


def test_duplicate_ingest_must_report_matching_sha(tmp_path: Path) -> None:
    package_path, expected_shas = _write_runnable_package(tmp_path)

    class DuplicateMismatchClient(_FakeClient):
        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            self.calls += 1
            self.headers_seen.append(dict(kwargs.get("headers", {})))
            return httpx.Response(
                409,
                request=httpx.Request("POST", url),
                json={
                    "request_id": "req-409",
                    "status": "error",
                    "data": None,
                    "error": {
                        "code": "DUPLICATE_DOCUMENT",
                        "message": "already ingested",
                        "details": {
                            "existing_doc_id": "doc-existing",
                            "sha256": "0" * 64,
                        },
                    },
                },
            )

    fake = DuplicateMismatchClient(expected_shas)
    result = runner.main(
        [
            "--api-base",
            "http://127.0.0.1:8000",
            "--package",
            str(package_path),
            "--job-id",
            "job-test",
            "--run-label",
            "hybrid",
            "--expect-rag-health",
            "healthy",
        ],
        client=fake,
    )

    assert result == 2


def test_draft_refuses_network_access(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[3]
    fake = _FakeClient({})
    with pytest.raises(SystemExit) as raised:
        runner.main(
            [
                "--package",
                str(repository / "rag-acceptance-v1"),
                "--job-id",
                "job-test",
                "--run-label",
                "hybrid",
                "--expect-rag-health",
                "healthy",
            ],
            client=fake,
        )
    assert raised.value.code == 2
    assert fake.calls == 0


def test_remote_plain_http_is_rejected() -> None:
    with pytest.raises(runner.AcceptanceRunError, match="requires HTTPS"):
        runner.normalize_api_base("http://example.test:8000")

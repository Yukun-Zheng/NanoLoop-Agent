"""Integrity and offline-answer tests for the checked-in curated demo corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.contracts.common import HealthComponent
from app.contracts.knowledge import RetrievedChunk
from app.contracts.queries import MaterialContext
from app.rag.ingestion import IngestionPipeline
from app.rag.keyword_store import KeywordSearchHit
from app.rag.providers import ExtractiveAnswerProvider
from app.rag.retrieval import RetrievalService
from app.rag.service import KnowledgeService

_REPOSITORY = Path(__file__).resolve().parents[3]
_PACKAGE = _REPOSITORY / "demo_data" / "rag"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict[str, Any]:
    payload = json.loads((_PACKAGE / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _retrieved_chunks() -> list[RetrievedChunk]:
    output: list[RetrievedChunk] = []
    pipeline = IngestionPipeline()
    for document in _manifest()["documents"]:
        metadata = document["metadata"]
        prepared = pipeline.prepare(
            _PACKAGE / document["path"],
            doc_id=document["asset_id"],
            title=metadata["title"],
            material_tags=tuple(metadata["material_aliases"]),
        )
        output.extend(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                title=chunk.title,
                source_type=metadata["source_type"],
                citation_text=metadata["citation_text"],
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_title=chunk.section_title,
                text=chunk.text,
                material_tags=list(chunk.material_tags),
                retrieval_score=0.0,
            )
            for chunk in prepared.chunks
        )
    return output


class _KeywordStore:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def health(self) -> HealthComponent:
        return HealthComponent(status="healthy", detail="curated demo fixture")

    def search(self, query: str, *, limit: int) -> list[KeywordSearchHit]:
        tokens = [token for token in query.casefold().split() if token]
        scored = [
            (sum(token in chunk.text.casefold() for token in tokens), chunk)
            for chunk in self.chunks
        ]
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            KeywordSearchHit(chunk=chunk, rank=rank)
            for rank, (score, chunk) in enumerate(scored, start=1)
            if score > 0
        ][:limit]

    def get_many(self, chunk_ids: list[str]) -> dict[str, RetrievedChunk]:
        wanted = set(chunk_ids)
        return {chunk.chunk_id: chunk for chunk in self.chunks if chunk.chunk_id in wanted}


def test_curated_manifest_binds_all_checked_in_documents() -> None:
    manifest = _manifest()

    assert manifest["schema_version"] == 1
    assert manifest["package"] == "nanoloop-demo-knowledge-v1"
    assert len(manifest["documents"]) == 7
    for document in manifest["documents"]:
        path = (_PACKAGE / document["path"]).resolve(strict=True)
        path.relative_to(_PACKAGE.resolve(strict=True))
        assert _sha256(path) == document["sha256"]
        assert document["metadata"]["allowed_for_demo"] is True
        assert document["metadata"]["material_aliases"]


def test_curated_corpus_chunks_and_answers_with_citations() -> None:
    chunks = _retrieved_chunks()
    service = KnowledgeService(
        RetrievalService(_KeywordStore(chunks)),
        provider=ExtractiveAnswerProvider(),
    )

    answer = service.answer(
        "钙钛矿氧化物 析出 晶格",
        material_context=MaterialContext(
            name="钙钛矿氧化物",
            aliases=["perovskite oxide", "ABO3"],
        ),
    )

    assert len(chunks) >= 7
    assert answer.outcome_code == "OK"
    assert answer.citations
    assert all(citation.doc_id.startswith("demo_") for citation in answer.citations)
    assert "[C1]" in answer.answer


def test_curated_question_set_contains_knowledge_mixed_and_refusal_cases() -> None:
    questions = [
        json.loads(line)
        for line in (_PACKAGE / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(questions) == 30
    assert {item["query_type"] for item in questions} == {"material_knowledge", "mixed"}
    assert sum(item["expected_outcome"] == "INSUFFICIENT_EVIDENCE" for item in questions) == 2
    assert any("编造" in item["question"] for item in questions)
    contract = json.loads(
        (_PACKAGE / "evaluation_contract.json").read_text(encoding="utf-8")
    )
    assert contract["schema_version"] == 1
    assert set(contract["expected_asset_ids"]) == {
        item["query_id"] for item in questions
    }
    assert contract["scope_requirements"] == {
        "q025": "image",
        "q026": "image",
        "q027": "job",
        "q028": "image",
    }
    assert set(contract["evidence_requirements"]) == {
        "q025",
        "q026",
        "q027",
        "q028",
    }

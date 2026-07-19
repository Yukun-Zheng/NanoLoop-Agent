"""Tests for offline/remote answer providers and evidence-only service behavior."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from app.contracts.common import HealthComponent
from app.contracts.knowledge import RetrievalRequest, RetrievedChunk
from app.contracts.queries import MaterialContext
from app.rag.keyword_store import SQLiteFTS5KeywordStore
from app.rag.providers import (
    CitationContext,
    CitationValidationError,
    ExtractiveAnswerProvider,
    OpenAICompatibleProvider,
    ProviderAnswer,
    validate_provider_answer,
)
from app.rag.retrieval import RetrievalReport, RetrievalService
from app.rag.service import KnowledgeService


def _context() -> CitationContext:
    return CitationContext(
        citation_id="C1",
        chunk=RetrievedChunk(
            chunk_id="chunk_1",
            doc_id="doc_1",
            title="材料论文",
            source_type="paper",
            citation_text="材料论文规范引用，2026。",
            page_start=4,
            page_end=4,
            text="该材料被报道具有催化应用。",
            material_tags=["SrNi"],
            retrieval_score=0.83,
        ),
    )


def test_extractive_provider_returns_only_supplied_evidence() -> None:
    answer = ExtractiveAnswerProvider().generate(
        question="有什么用途？",
        contexts=[_context()],
        material_context=None,
    )

    assert answer.used_citation_ids == ("C1",)
    assert "[C1]" in answer.answer
    assert "催化应用" in answer.answer
    assert answer.confidence == "medium"


def test_extractive_provider_marks_every_evidence_sentence() -> None:
    base = _context()
    context = CitationContext(
        citation_id=base.citation_id,
        chunk=base.chunk.model_copy(
            update={"text": "该材料具有催化应用。第二项实验观察支持这一结论！"}
        ),
    )

    answer = ExtractiveAnswerProvider().generate(
        question="有什么用途？",
        contexts=[context],
        material_context=None,
    )

    evidence_lines = [line for line in answer.answer.splitlines() if line.startswith("- ")]
    assert len(evidence_lines) == 2
    assert all(line.startswith("- [C1] ") for line in evidence_lines)


@pytest.mark.parametrize(
    "answer",
    [
        ProviderAnswer("材料具有催化活性。", (), "medium"),
        ProviderAnswer("材料具有催化活性。[C2]", ("C2",), "medium"),
        ProviderAnswer("材料具有催化活性。[C1]", (), "medium"),
        ProviderAnswer(
            "材料具有催化活性。[C1] 但具体机理尚不明确。",
            ("C1",),
            "medium",
        ),
    ],
)
def test_strict_citation_validator_rejects_ungrounded_answers(answer: ProviderAnswer) -> None:
    with pytest.raises(CitationValidationError):
        validate_provider_answer(answer, {"C1"})


@dataclass
class FakeResponse:
    body: dict[str, Any]

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self.body


class FakeHttpClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        *,
        headers: Any,
        json: Any,
        timeout: float,
    ) -> FakeResponse:
        self.calls.append(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        return FakeResponse(
            {"choices": [{"message": {"content": self.content}}]}
        )


def test_openai_compatible_provider_parses_json_and_validates_citations() -> None:
    client = FakeHttpClient(
        '{"answer":"文献报道其具有催化用途。[C1]",'
        '"used_citation_ids":["C1"],"confidence":"medium","limitations":[]}'
    )
    provider = OpenAICompatibleProvider(
        base_url="http://llm.test/v1/",
        api_key="secret",
        model="fixture-model",
        client=client,
    )

    answer = provider.generate(
        question="用途？",
        contexts=[_context()],
        material_context=MaterialContext(formula="SrNi"),
    )

    assert answer.used_citation_ids == ("C1",)
    assert client.calls[0]["url"] == "http://llm.test/v1/chat/completions"
    assert client.calls[0]["json"]["temperature"] == 0


def test_openai_provider_is_unavailable_without_configuration() -> None:
    provider = OpenAICompatibleProvider(base_url=None, api_key=None, model=None)

    assert provider.health().status == "unavailable"
    assert "LLM_BASE_URL" in (provider.health().detail or "")


class FakeRetrieval:
    def __init__(self, report: RetrievalReport) -> None:
        self.report = report
        self.requests: list[object] = []

    def retrieve_with_report(self, request: object) -> RetrievalReport:
        self.requests.append(request)
        return self.report

    def health(self) -> HealthComponent:
        return self.report.health


def test_knowledge_service_returns_insufficient_without_corpus() -> None:
    retrieval = FakeRetrieval(
        RetrievalReport(
            chunks=(),
            health=HealthComponent(status="degraded", detail="knowledge index is empty"),
        )
    )
    service = KnowledgeService(cast(RetrievalService, retrieval))

    answer = service.answer("有什么用途？")

    assert answer.outcome_code == "INSUFFICIENT_EVIDENCE"
    assert not answer.citations
    assert "证据不足" in answer.answer


def test_knowledge_service_bounds_combined_identity_aliases() -> None:
    retrieval = FakeRetrieval(
        RetrievalReport(
            chunks=(),
            health=HealthComponent(status="degraded", detail="knowledge index is empty"),
        )
    )
    service = KnowledgeService(cast(RetrievalService, retrieval))

    service.answer(
        "有什么用途？",
        material_context=MaterialContext(
            formula="Formula",
            name="Material",
            aliases=[f"alias-{index}" for index in range(32)],
        ),
    )

    request = retrieval.requests[0]
    assert isinstance(request, RetrievalRequest)
    assert len(request.material_aliases) == 32
    assert request.material_aliases[:2] == ["Formula", "Material"]


def test_knowledge_health_is_degraded_for_empty_or_keyword_only_index() -> None:
    retrieval = FakeRetrieval(
        RetrievalReport(
            chunks=(),
            health=HealthComponent(status="degraded", detail="knowledge index is empty"),
        )
    )
    service = KnowledgeService(cast(RetrievalService, retrieval))

    assert service.health().status == "degraded"


def test_knowledge_service_builds_page_citations_and_offline_limitations() -> None:
    retrieval = FakeRetrieval(
        RetrievalReport(
            chunks=(_context().chunk,),
            health=HealthComponent(status="degraded", detail="keyword only"),
            warnings=("vector retrieval unavailable; keyword retrieval only",),
        )
    )
    service = KnowledgeService(cast(RetrievalService, retrieval))

    answer = service.answer(
        "用途？",
        material_context=MaterialContext(formula="SrNi", source="image_metadata"),
    )

    assert answer.outcome_code == "OK"
    assert answer.citations[0].page == 4
    assert answer.citations[0].chunk_id == "chunk_1"
    assert answer.citations[0].source_type == "paper"
    assert answer.citations[0].citation_text == "材料论文规范引用，2026。"
    assert len(answer.citations[0].excerpt) <= 160
    assert any("离线摘录" in limitation for limitation in answer.limitations)


def test_invalid_remote_citations_degrade_to_extractive_evidence() -> None:
    retrieval = FakeRetrieval(
        RetrievalReport(
            chunks=(_context().chunk,),
            health=HealthComponent(status="healthy", detail="fixture"),
        )
    )
    client = FakeHttpClient(
        '{"answer":"模型声称这是事实。","used_citation_ids":[],'
        '"confidence":"high","limitations":[]}'
    )
    remote = OpenAICompatibleProvider(
        base_url="http://llm.test/v1",
        api_key="secret",
        model="fixture",
        client=client,
    )
    service = KnowledgeService(cast(RetrievalService, retrieval), provider=remote)

    answer = service.answer("用途？")

    assert answer.outcome_code == "OK"
    assert "[C1]" in answer.answer
    assert answer.citations
    assert any("降级" in limitation for limitation in answer.limitations)


def test_empty_runtime_is_degraded_and_never_fabricates(tmp_path: Path) -> None:
    database = tmp_path / "knowledge.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE knowledge_documents (
            doc_id TEXT, title TEXT, source_type TEXT, citation_text TEXT, status TEXT
        );
        CREATE TABLE knowledge_chunks (
            chunk_id TEXT, doc_id TEXT, page_start INTEGER, page_end INTEGER,
            section_title TEXT, text TEXT, material_tags_json TEXT
        );
        CREATE VIRTUAL TABLE knowledge_chunks_fts USING fts5(
            chunk_id UNINDEXED, text, section_title, material_tags
        );
        """
    )
    connection.close()
    retrieval = RetrievalService(SQLiteFTS5KeywordStore(database))
    service = KnowledgeService(retrieval)

    assert service.health().status == "degraded"
    answer = service.answer("这个材料有什么用途？")
    assert answer.outcome_code == "INSUFFICIENT_EVIDENCE"
    assert answer.citations == ()


def test_material_mismatch_returns_insufficient_instead_of_other_material_citation(
    tmp_path: Path,
) -> None:
    database = tmp_path / "tagged-knowledge.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE knowledge_documents (
            doc_id TEXT, title TEXT, source_type TEXT, citation_text TEXT, status TEXT
        );
        CREATE TABLE knowledge_chunks (
            chunk_id TEXT, doc_id TEXT, page_start INTEGER, page_end INTEGER,
            section_title TEXT, text TEXT, material_tags_json TEXT
        );
        CREATE VIRTUAL TABLE knowledge_chunks_fts USING fts5(
            chunk_id UNINDEXED, text, section_title, material_tags
        );
        INSERT INTO knowledge_documents VALUES (
            'doc_ycu', 'YCu evidence', 'paper', 'YCu evidence citation.', 'ready'
        );
        INSERT INTO knowledge_chunks VALUES (
            'chunk_ycu', 'doc_ycu', 1, 1, 'Applications',
            'catalyst evidence belongs to YCu', '["YCu"]'
        );
        INSERT INTO knowledge_chunks_fts VALUES (
            'chunk_ycu', 'catalyst evidence belongs to YCu', 'Applications', 'YCu'
        );
        """
    )
    connection.close()
    service = KnowledgeService(RetrievalService(SQLiteFTS5KeywordStore(database)))

    answer = service.answer(
        "What catalyst evidence is available?",
        material_context=MaterialContext(formula="SrNi"),
    )

    assert answer.outcome_code == "INSUFFICIENT_EVIDENCE"
    assert answer.citations == ()
    assert any("other-material evidence" in limitation for limitation in answer.limitations)

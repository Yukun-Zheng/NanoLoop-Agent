"""Tests for offline/remote answer providers and evidence-only service behavior."""

from __future__ import annotations

import json
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
    is_pure_insufficient_answer,
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
            text="该材料被报道具有催化应用和催化用途。",
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
            update={"text": "该材料具有催化用途。第二项实验观察支持这一结论！"}
        ),
    )

    answer = ExtractiveAnswerProvider().generate(
        question="有什么用途？",
        contexts=[context],
        material_context=None,
    )

    evidence_lines = [line for line in answer.answer.splitlines() if line.startswith("- ")]
    assert evidence_lines
    assert all(line.startswith("- [C1] ") for line in evidence_lines)


def test_extractive_provider_selects_relevant_sentences_beyond_chunk_prefix() -> None:
    base = _context()
    context = CitationContext(
        citation_id=base.citation_id,
        chunk=base.chunk.model_copy(
            update={
                "text": (
                    "# SEM 掩码形貌指标\n"
                    "这是一段与问题无关的开场说明。\n"
                    "另一段背景也不包含目标指标。\n"
                    "- 颗粒数量与单位面积数密度；\n"
                    "- 等效圆直径与粒径分布；"
                )
            }
        ),
    )

    answer = ExtractiveAnswerProvider().generate(
        question="SEM 掩码应该计算哪些颗粒形貌指标？",
        contexts=[context],
        material_context=None,
    )

    assert "数密度" in answer.answer
    assert all(
        line.startswith("- [C1] ")
        for line in answer.answer.splitlines()
        if line.startswith("- ")
    )


def test_extractive_provider_prefers_identity_boundary_for_label_without_formula() -> None:
    base = _context()
    context = CitationContext(
        citation_id=base.citation_id,
        chunk=base.chunk.model_copy(
            update={
                "text": (
                    "该材料有多种潜在应用。"
                    "讨论性质前应先核对实验条件。"
                    "如果完整化学式缺失，应把回答限定为相关体系的一般规律。"
                    "其他测试仍然需要独立完成。"
                )
            }
        ),
    )

    answer = ExtractiveAnswerProvider().generate(
        question="这个材料有什么特性和应用？",
        contexts=[context],
        material_context=MaterialContext(name="LaNi"),
    )

    assert "完整化学式" in answer.answer
    assert "一般规律" in answer.answer


def test_extractive_provider_supports_a_grounded_negative_claim_clause() -> None:
    base = _context()
    context = CitationContext(
        citation_id=base.citation_id,
        chunk=base.chunk.model_copy(
            update={
                "text": (
                    "仅凭 SEM 形貌不能确认颗粒元素组成，"
                    "需要 EDS、XPS 或 XRD 等证据进一步确认。"
                )
            }
        ),
    )

    answer = ExtractiveAnswerProvider().generate(
        question="为什么不能只看 SEM 就确认颗粒是 Ni？",
        contexts=[context],
        material_context=MaterialContext(name="LaNi"),
    )

    assert answer.confidence == "medium"
    assert answer.used_citation_ids == ("C1",)
    assert "不能确认" in answer.answer


def test_extractive_provider_supports_grounded_english_predicate_terms() -> None:
    base = _context()
    context = CitationContext(
        citation_id=base.citation_id,
        chunk=base.chunk.model_copy(
            update={"text": "Ni 析出可用于电池电极和其他催化应用。"}
        ),
    )

    answer = ExtractiveAnswerProvider().generate(
        question="What are the catalytic applications of Ni exsolution?",
        contexts=[context],
        material_context=MaterialContext(name="LaNi"),
    )

    assert answer.confidence == "medium"
    assert answer.used_citation_ids == ("C1",)
    assert "催化应用" in answer.answer


@pytest.mark.parametrize(
    "question",
    [
        "How is Ni exsolution used in catalysis?",
        "What factors affect Ni particle morphology?",
        "What determines Ni particle stability?",
        "What is the Ni particle density?",
        "What is the Ni particle diameter?",
        "What Ni catalyst evidence is available?",
    ],
)
def test_extractive_provider_supports_common_bilingual_science_questions(
    question: str,
) -> None:
    base = _context()
    context = CitationContext(
        citation_id=base.citation_id,
        chunk=base.chunk.model_copy(
            update={
                "text": (
                    "Ni 析出用于电池电极等催化应用。温度和气氛是影响颗粒形貌与"
                    "稳定性的因素。已有实验催化证据。数密度为 61.0 μm^-2，"
                    "平均粒径为 18 nm。"
                )
            }
        ),
    )

    answer = ExtractiveAnswerProvider().generate(
        question=question,
        contexts=[context],
        material_context=MaterialContext(name="LaNi"),
    )

    assert answer.confidence == "medium"
    assert answer.used_citation_ids == ("C1",)


def test_extractive_provider_preserves_what_in_chinese_definition_question() -> None:
    base = _context()
    context = CitationContext(
        citation_id=base.citation_id,
        chunk=base.chunk.model_copy(
            update={
                "text": (
                    "所谓析出，是指晶格中的可还原金属离子迁移至表面并形成纳米颗粒。"
                )
            }
        ),
    )

    answer = ExtractiveAnswerProvider().generate(
        question="钙钛矿氧化物中的析出是什么？",
        contexts=[context],
        material_context=MaterialContext(name="钙钛矿氧化物"),
    )

    assert answer.confidence == "medium"
    assert "所谓析出" in answer.answer


@pytest.mark.parametrize(
    "question",
    [
        "材料会不会在午夜自动唱歌？",
        "这种材料是否能让时间倒流？",
        "LaNi 是否可以治愈癌症？",
        "Ni 外析颗粒能否让人隐身？",
        "析出颗粒能否产生无限能源？",
        "析出颗粒的催化应用能让人隐身吗？",
        "Ni 颗粒的电化学性能可以预测彩票号码吗？",
        "颗粒尺寸和密度能否证明外星人存在？",
        "析出颗粒的催化应用能操控天气吗？",
        "Ni 颗粒的电化学性能会控制人类思想吗？",
        "颗粒尺寸和密度足以判定外星文明吗？",
        "析出颗粒能让人隐身并用于催化吗？",
        "Ni 颗粒能通过操控天气提高电化学性能吗？",
        "Ni 颗粒具有预测彩票号码的电化学性能吗？",
        "请解释 Ni 颗粒预测彩票号码的催化机制。",
        "Can Ni particles cure cancer?",
        "Can Ni exsolution predict lottery numbers?",
        "Does Ni exsolution enable time travel?",
    ],
)
def test_extractive_provider_refuses_accidental_generic_overlap(question: str) -> None:
    base = _context()
    context = CitationContext(
        citation_id=base.citation_id,
        chunk=base.chunk.model_copy(
            update={
                "text": (
                    "LaNi 是 Ni 相关析出颗粒样品，具有催化应用和电化学性能。"
                    "颗粒尺寸和密度是常用形貌指标。"
                    "该材料采用自动分割，并记录处理时间和常规实验条件。"
                ),
                "retrieval_score": 1.0,
            }
        ),
    )

    answer = ExtractiveAnswerProvider().generate(
        question=question,
        contexts=[context],
        material_context=MaterialContext(name="LaNi"),
    )

    assert answer.confidence == "low"
    assert answer.used_citation_ids == ()
    assert "证据不足" in answer.answer


@pytest.mark.parametrize(
    "question",
    [
        "材料会不会在午夜自动唱歌？",
        "这种材料是否能让时间倒流？",
        "LaNi 是否可以治愈癌症？",
        "Ni 外析颗粒能否让人隐身？",
        "析出颗粒能否产生无限能源？",
        "析出颗粒的催化应用能让人隐身吗？",
        "Ni 颗粒的电化学性能可以预测彩票号码吗？",
        "颗粒尺寸和密度能否证明外星人存在？",
        "析出颗粒的催化应用能操控天气吗？",
        "Ni 颗粒的电化学性能会控制人类思想吗？",
        "颗粒尺寸和密度足以判定外星文明吗？",
        "析出颗粒能让人隐身并用于催化吗？",
        "Ni 颗粒能通过操控天气提高电化学性能吗？",
        "Ni 颗粒具有预测彩票号码的电化学性能吗？",
        "请解释 Ni 颗粒预测彩票号码的催化机制。",
        "Can Ni particles cure cancer?",
        "Can Ni exsolution predict lottery numbers?",
        "Does Ni exsolution enable time travel?",
    ],
)
def test_knowledge_service_maps_unsupported_predicates_to_insufficient(
    question: str,
) -> None:
    base = _context()
    chunk = base.chunk.model_copy(
        update={
            "text": (
                "LaNi 是 Ni 相关析出颗粒样品，具有催化应用和电化学性能。"
                "颗粒尺寸和密度是常用形貌指标。"
                "该材料采用自动分割，并记录处理时间和常规实验条件。"
            ),
            "retrieval_score": 1.0,
        }
    )
    retrieval = FakeRetrieval(
        RetrievalReport(
            chunks=(chunk,),
            health=HealthComponent(status="healthy", detail="fixture"),
        )
    )
    service = KnowledgeService(cast(RetrievalService, retrieval))

    answer = service.answer(
        question,
        material_context=MaterialContext(name="LaNi"),
    )

    assert answer.outcome_code == "INSUFFICIENT_EVIDENCE"
    assert answer.citations == ()
    assert answer.confidence == "low"


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
        ProviderAnswer(
            "证据不足，但该材料一定能实现室温超导。",
            (),
            "low",
        ),
        ProviderAnswer(
            "无法判断；不过该样品肯定含有金属镍。",
            (),
            "low",
        ),
        ProviderAnswer("材料知识结论：", (), "low"),
        ProviderAnswer(
            "文献报道其具有催化用途。[C1]",
            ("C1",),
            "medium",
            ("该样品一定含有金属镍。",),
        ),
    ],
)
def test_strict_citation_validator_rejects_ungrounded_answers(answer: ProviderAnswer) -> None:
    with pytest.raises(CitationValidationError):
        validate_provider_answer(answer, {"C1"})


@pytest.mark.parametrize(
    "text",
    [
        "知识库证据不足，无法基于现有文档回答该问题。",
        "证据不足，无法判断。",
        "Insufficient evidence to answer this question.",
    ],
)
def test_strict_citation_validator_allows_low_confidence_pure_refusal(
    text: str,
) -> None:
    answer = ProviderAnswer(
        text,
        (),
        "low",
        ("知识库覆盖有限",),
    )

    validate_provider_answer(answer, {"C1"})

    assert is_pure_insufficient_answer(answer)


def test_strict_citation_validator_rejects_high_confidence_pure_refusal() -> None:
    answer = ProviderAnswer("证据不足，无法判断。", (), "high")

    with pytest.raises(CitationValidationError, match="low-confidence pure refusal"):
        validate_provider_answer(answer, {"C1"})


def test_strict_citation_validator_allows_only_safe_or_cited_limitations() -> None:
    safe = ProviderAnswer(
        "文献报道其具有催化用途。[C1]",
        ("C1",),
        "medium",
        (
            "当前为离线摘录模式，不代表完整文献综述",
            "文献未证明当前样品具有相同表现。[C1]",
        ),
    )

    validate_provider_answer(safe, {"C1"})


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
    messages = client.calls[0]["json"]["messages"]
    assert "不可信数据" in messages[0]["content"]
    assert "绝不执行" in messages[0]["content"]
    user_content = messages[1]["content"]
    serialized = user_content.removeprefix(
        "BEGIN_UNTRUSTED_RAG_INPUT_JSON\n"
    ).removesuffix("\nEND_UNTRUSTED_RAG_INPUT_JSON")
    untrusted_input = json.loads(serialized)
    assert untrusted_input["question"] == "用途？"
    assert untrusted_input["retrieved_contexts"][0]["citation_id"] == "C1"
    assert untrusted_input["retrieved_contexts"][0]["text"] == _context().chunk.text


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


class StaticAnswerProvider:
    """Provider seam that deliberately does not validate its own output."""

    def __init__(self, answer: ProviderAnswer) -> None:
        self.answer = answer

    def health(self) -> HealthComponent:
        return HealthComponent(status="healthy", detail="fixture")

    def generate(self, **_: Any) -> ProviderAnswer:
        return self.answer


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


def test_knowledge_service_expands_english_retrieval_terms_but_answers_original() -> None:
    retrieval = FakeRetrieval(
        RetrievalReport(
            chunks=(_context().chunk,),
            health=HealthComponent(status="healthy", detail="fixture"),
        )
    )
    service = KnowledgeService(cast(RetrievalService, retrieval))

    answer = service.answer(
        "What are the catalytic applications of Ni exsolution?",
        material_context=MaterialContext(name="LaNi"),
    )

    request = retrieval.requests[0]
    assert isinstance(request, RetrievalRequest)
    assert "催化" in request.query
    assert "应用" in request.query
    assert "What" not in request.query
    assert "applications" not in request.query
    assert answer.outcome_code == "OK"
    assert answer.citations


def test_english_expansion_ranks_substantive_sqlite_chunk_before_references(
    tmp_path: Path,
) -> None:
    database = tmp_path / "english-expansion.db"
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
            'doc_1', 'Ni 析出与催化应用', 'paper', 'Curated Ni source.', 'ready'
        );
        """
    )
    rows = [
        (
            "chunk_app",
            "典型应用场景",
            "Ni 析出的典型应用包括固体氧化物电池电极和异相催化。",
        ),
        *[
            (
                f"chunk_ref_{index}",
                "参考来源",
                (
                    "What are the catalytic applications of Ni exsolution? "
                    f"Reference catalyst entry {index}."
                ),
            )
            for index in range(8)
        ],
    ]
    connection.executemany(
        """
        INSERT INTO knowledge_chunks VALUES (?, 'doc_1', 1, 1, ?, ?, '["Ni", "析出"]')
        """,
        rows,
    )
    connection.executemany(
        "INSERT INTO knowledge_chunks_fts VALUES (?, ?, ?, 'Ni 析出')",
        [(chunk_id, text, section) for chunk_id, section, text in rows],
    )
    connection.commit()
    connection.close()

    retrieval = RetrievalService(SQLiteFTS5KeywordStore(database))
    service = KnowledgeService(retrieval)

    answer = service.answer(
        "What are the catalytic applications of Ni exsolution?",
        material_context=MaterialContext(name="LaNi", aliases=["Ni"]),
    )

    assert answer.outcome_code == "OK"
    assert any(citation.chunk_id == "chunk_app" for citation in answer.citations)
    assert "固体氧化物电池" in answer.answer


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


def test_remote_pure_refusal_maps_to_insufficient_evidence_without_citations() -> None:
    retrieval = FakeRetrieval(
        RetrievalReport(
            chunks=(_context().chunk,),
            health=HealthComponent(status="healthy", detail="fixture"),
        )
    )
    client = FakeHttpClient(
        '{"answer":"知识库证据不足，无法基于现有文档回答该问题。",'
        '"used_citation_ids":[],"confidence":"low",'
        '"limitations":["知识库覆盖有限"]}'
    )
    remote = OpenAICompatibleProvider(
        base_url="http://llm.test/v1",
        api_key="secret",
        model="fixture",
        client=client,
    )
    service = KnowledgeService(cast(RetrievalService, retrieval), provider=remote)

    answer = service.answer("无法由现有文档回答的问题？")

    assert answer.outcome_code == "INSUFFICIENT_EVIDENCE"
    assert answer.confidence == "low"
    assert answer.citations == ()
    assert "证据不足" in answer.answer


def test_knowledge_service_never_returns_ok_without_a_current_context_citation() -> None:
    retrieval = FakeRetrieval(
        RetrievalReport(
            chunks=(_context().chunk,),
            health=HealthComponent(status="healthy", detail="fixture"),
        )
    )
    unvalidated = StaticAnswerProvider(
        ProviderAnswer("该材料一定具有室温超导性。", (), "high")
    )
    service = KnowledgeService(
        cast(RetrievalService, retrieval),
        provider=unvalidated,
        fallback_provider=unvalidated,
    )

    answer = service.answer("这个材料是否具有室温超导性？")

    assert answer.outcome_code == "INSUFFICIENT_EVIDENCE"
    assert answer.citations == ()
    assert "未能产生可验证引用" in answer.answer


def test_uncited_factual_remote_limitation_degrades_to_extractive_evidence() -> None:
    retrieval = FakeRetrieval(
        RetrievalReport(
            chunks=(_context().chunk,),
            health=HealthComponent(status="healthy", detail="fixture"),
        )
    )
    client = FakeHttpClient(
        '{"answer":"文献报道其具有催化用途。[C1]",'
        '"used_citation_ids":["C1"],"confidence":"medium",'
        '"limitations":["该样品一定含有金属镍。"]}'
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
    assert answer.citations
    assert "金属镍" not in answer.answer
    assert all("金属镍" not in limitation for limitation in answer.limitations)
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

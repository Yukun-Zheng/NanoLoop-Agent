"""Tests for evidence-preserving data/knowledge query composition."""

from __future__ import annotations

from typing import cast

import pytest

from app.agent.router import QueryRouter
from app.agent.unified_query import (
    DataQuery,
    DataQueryResult,
    DataToolService,
    UnifiedQueryService,
)
from app.contracts.enums import QueryType
from app.contracts.identity import LEGACY_TENANT_ID, AuthMode
from app.contracts.queries import (
    Citation,
    MaterialContext,
    ToolCallLog,
    ToolEvidence,
    UnifiedQueryRequest,
)
from app.core.errors import ServiceUnavailableError
from app.rag.service import KnowledgeAnswer, KnowledgeService


class FakeDataTools:
    def __init__(self) -> None:
        self.queries: list[DataQuery] = []

    def answer(self, query: DataQuery) -> DataQueryResult:
        self.queries.append(query)
        evidence = ToolEvidence(
            tool_name="rank_samples",
            validated_arguments={"metric": "coverage_ratio"},
            rows=[{"sample": "A", "coverage_ratio": 0.42}],
            units={"coverage_ratio": "ratio"},
            source_run_ids=["run_1"],
        )
        call = ToolCallLog(
            tool_name="rank_samples",
            arguments={"metric": "coverage_ratio"},
            outcome="success",
            source_run_ids=["run_1"],
        )
        return DataQueryResult(
            answer="样品 A 覆盖率最高，为 0.42。",
            evidence=(evidence,),
            tool_calls=(call,),
            confidence="high",
        )


class FakeKnowledgeService:
    def __init__(self, outcome: str = "OK") -> None:
        self.calls = 0
        self.outcome = outcome

    def answer(
        self,
        question: str,
        *,
        material_context: MaterialContext | None = None,
    ) -> KnowledgeAnswer:
        del question
        self.calls += 1
        if self.outcome == "INSUFFICIENT_EVIDENCE":
            return KnowledgeAnswer(
                answer="知识库证据不足。",
                citations=(),
                confidence="low",
                limitations=("empty corpus",),
                outcome_code="INSUFFICIENT_EVIDENCE",
                material_context=material_context,
            )
        citation = Citation(
            citation_id="C1",
            doc_id="doc_1",
            title="论文",
            page=2,
            chunk_id="chunk_1",
            excerpt="材料具有催化用途。",
            retrieval_score=0.9,
        )
        return KnowledgeAnswer(
            answer="文献报道该材料具有催化用途。[C1]",
            citations=(citation,),
            confidence="medium",
            limitations=(),
            outcome_code="OK",
            material_context=material_context,
        )


def _service(
    data: FakeDataTools,
    knowledge: FakeKnowledgeService,
) -> UnifiedQueryService:
    return UnifiedQueryService(
        router=QueryRouter(),
        knowledge_service=cast(KnowledgeService, knowledge),
        data_tools=cast(DataToolService, data),
    )


def _answer(
    service: UnifiedQueryService,
    request: UnifiedQueryRequest,
    *,
    auth_mode: AuthMode = AuthMode.DISABLED,
    tenant_id: str = LEGACY_TENANT_ID,
):
    return service.answer(
        "job_1",
        request,
        tenant_id=tenant_id,
        auth_mode=auth_mode,
    )


def test_unknown_auto_query_clarifies_without_calling_tools() -> None:
    data = FakeDataTools()
    knowledge = FakeKnowledgeService()

    response = _answer(
        _service(data, knowledge),
        UnifiedQueryRequest(question="帮我看看"),
    )

    assert response.needs_clarification
    assert response.query_type == QueryType.AUTO
    assert not data.queries
    assert knowledge.calls == 0


def test_data_query_delegates_all_numbers_to_injected_tool_service() -> None:
    data = FakeDataTools()
    knowledge = FakeKnowledgeService()

    response = _answer(
        _service(data, knowledge),
        UnifiedQueryRequest(
            question="哪组覆盖率最高？",
            query_type=QueryType.ANALYSIS_DATA,
            run_ids=["run_1"],
        ),
    )

    assert response.answer == "样品 A 覆盖率最高，为 0.42。"
    assert response.data_evidence[0].rows[0]["coverage_ratio"] == 0.42
    assert data.queries[0].job_id == "job_1"
    assert data.queries[0].tenant_id == LEGACY_TENANT_ID
    assert knowledge.calls == 0


def test_auto_mixed_query_keeps_data_and_knowledge_in_separate_sections() -> None:
    data = FakeDataTools()
    knowledge = FakeKnowledgeService()

    response = _answer(
        _service(data, knowledge),
        UnifiedQueryRequest(
            question="已有研究用途是什么，我们这批覆盖率最高吗？",
            material_context=MaterialContext(formula="SrNi"),
        ),
    )

    assert response.query_type == QueryType.MIXED
    assert response.answer.startswith("实验数据结论：")
    assert "\n\n材料知识结论：" in response.answer
    assert response.data_evidence
    assert response.citations
    assert response.confidence == "medium"
    assert data.queries and knowledge.calls == 1


def test_mixed_query_exposes_partial_insufficiency() -> None:
    response = _answer(
        _service(FakeDataTools(), FakeKnowledgeService("INSUFFICIENT_EVIDENCE")),
        UnifiedQueryRequest(
            question="已有研究用途是什么，我们这批覆盖率最高吗？",
            query_type=QueryType.MIXED,
        ),
    )

    assert response.outcome_code == "INSUFFICIENT_EVIDENCE"
    assert response.data_evidence
    assert not response.citations
    assert "empty corpus" in response.limitations


@pytest.mark.parametrize(
    "query_request",
    [
        UnifiedQueryRequest(
            question="该材料有什么用途？",
            query_type=QueryType.MATERIAL_KNOWLEDGE,
        ),
        UnifiedQueryRequest(
            question="我们这批覆盖率最高吗，已有研究用途是什么？",
            query_type=QueryType.MIXED,
        ),
        UnifiedQueryRequest(question="已有研究用途是什么？"),
    ],
)
def test_principal_knowledge_routes_fail_before_any_tool_or_provider(
    query_request: UnifiedQueryRequest,
) -> None:
    data = FakeDataTools()
    knowledge = FakeKnowledgeService()

    with pytest.raises(ServiceUnavailableError) as error:
        _answer(
            _service(data, knowledge),
            query_request,
            auth_mode=AuthMode.PRINCIPAL,
            tenant_id=f"tnt_{'a' * 32}",
        )

    assert error.value.details == {"component": "knowledge_tenant_scope"}
    assert not data.queries
    assert knowledge.calls == 0


@pytest.mark.parametrize("auth_mode", [AuthMode.DISABLED, AuthMode.SHARED_KEY])
def test_compatibility_modes_keep_global_knowledge_behavior(auth_mode: AuthMode) -> None:
    data = FakeDataTools()
    knowledge = FakeKnowledgeService()

    response = _answer(
        _service(data, knowledge),
        UnifiedQueryRequest(
            question="该材料有什么用途？",
            query_type=QueryType.MATERIAL_KNOWLEDGE,
        ),
        auth_mode=auth_mode,
    )

    assert response.query_type is QueryType.MATERIAL_KNOWLEDGE
    assert knowledge.calls == 1
    assert not data.queries

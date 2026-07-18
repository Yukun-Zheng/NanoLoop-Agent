"""Compose data-tool and knowledge answers without performing numeric analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from app.agent.router import QueryRouter
from app.contracts.enums import QueryType
from app.contracts.identity import AuthMode
from app.contracts.queries import (
    ToolCallLog,
    ToolEvidence,
    UnifiedQueryRequest,
    UnifiedQueryResponse,
)
from app.core.errors import ServiceUnavailableError
from app.rag.service import KnowledgeAnswer, KnowledgeService

Confidence = Literal["low", "medium", "high"]
OutcomeCode = Literal["OK", "INSUFFICIENT_EVIDENCE"]


@dataclass(frozen=True, slots=True)
class DataQuery:
    job_id: str
    tenant_id: str
    question: str
    image_id: str | None
    run_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DataQueryResult:
    """A precomputed, evidence-bearing response returned by the analysis owner."""

    answer: str
    evidence: tuple[ToolEvidence, ...] = ()
    tool_calls: tuple[ToolCallLog, ...] = ()
    confidence: Confidence = "low"
    limitations: tuple[str, ...] = ()
    needs_clarification: bool = False
    outcome_code: OutcomeCode = "OK"


class DataToolService(Protocol):
    """Injected boundary; implementations own SQL, statistics, and argument validation."""

    def answer(self, query: DataQuery) -> DataQueryResult: ...


class UnavailableDataToolService:
    def answer(self, query: DataQuery) -> DataQueryResult:
        del query
        return DataQueryResult(
            answer="当前实验数据工具尚未就绪，无法计算或比较统计结果。",
            confidence="low",
            limitations=("DataToolService unavailable",),
            outcome_code="INSUFFICIENT_EVIDENCE",
        )


class UnifiedQueryService:
    def __init__(
        self,
        *,
        router: QueryRouter,
        knowledge_service: KnowledgeService,
        data_tools: DataToolService | None = None,
    ) -> None:
        self.router = router
        self.knowledge_service = knowledge_service
        self.data_tools = data_tools or UnavailableDataToolService()

    def answer(
        self,
        job_id: str,
        request: UnifiedQueryRequest,
        *,
        tenant_id: str,
        auth_mode: AuthMode,
    ) -> UnifiedQueryResponse:
        query_type = request.query_type
        if query_type == QueryType.AUTO:
            decision = self.router.classify(
                request.question,
                material_context=request.material_context,
            )
            if decision.needs_clarification:
                return UnifiedQueryResponse(
                    query_type=QueryType.AUTO,
                    answer="请明确选择数据分析、材料知识或混合问答类型。",
                    material_context=request.material_context,
                    confidence="low",
                    limitations=("查询意图不明确，未调用任何工具",),
                    needs_clarification=True,
                )
            query_type = decision.query_type

        if auth_mode is AuthMode.PRINCIPAL and query_type in {
            QueryType.MATERIAL_KNOWLEDGE,
            QueryType.MIXED,
        }:
            # Knowledge documents are still global.  Principal-mode callers must
            # never reach retrieval or answer providers until that corpus has a
            # tenant boundary of its own.
            raise ServiceUnavailableError(details={"component": "knowledge_tenant_scope"})

        if query_type == QueryType.ANALYSIS_DATA:
            return self._data_response(job_id, request, tenant_id=tenant_id)
        if query_type == QueryType.MATERIAL_KNOWLEDGE:
            return self._knowledge_response(request)
        if query_type == QueryType.MIXED:
            return self._mixed_response(job_id, request, tenant_id=tenant_id)
        raise ValueError(f"unsupported query type: {query_type}")

    def _data_result(
        self,
        job_id: str,
        request: UnifiedQueryRequest,
        *,
        tenant_id: str,
    ) -> DataQueryResult:
        return self.data_tools.answer(
            DataQuery(
                job_id=job_id,
                tenant_id=tenant_id,
                question=request.question,
                image_id=request.image_id,
                run_ids=tuple(request.run_ids),
            )
        )

    def _data_response(
        self,
        job_id: str,
        request: UnifiedQueryRequest,
        *,
        tenant_id: str,
    ) -> UnifiedQueryResponse:
        result = self._data_result(job_id, request, tenant_id=tenant_id)
        return UnifiedQueryResponse(
            query_type=QueryType.ANALYSIS_DATA,
            answer=result.answer,
            data_evidence=list(result.evidence),
            tool_calls=list(result.tool_calls),
            material_context=request.material_context,
            confidence=result.confidence,
            limitations=list(result.limitations),
            needs_clarification=result.needs_clarification,
            outcome_code=result.outcome_code,
        )

    def _knowledge_result(self, request: UnifiedQueryRequest) -> KnowledgeAnswer:
        return self.knowledge_service.answer(
            request.question,
            material_context=request.material_context,
        )

    def _knowledge_response(self, request: UnifiedQueryRequest) -> UnifiedQueryResponse:
        result = self._knowledge_result(request)
        return UnifiedQueryResponse(
            query_type=QueryType.MATERIAL_KNOWLEDGE,
            answer=result.answer,
            citations=list(result.citations),
            material_context=result.material_context,
            confidence=result.confidence,
            limitations=list(result.limitations),
            outcome_code=result.outcome_code,
        )

    def _mixed_response(
        self,
        job_id: str,
        request: UnifiedQueryRequest,
        *,
        tenant_id: str,
    ) -> UnifiedQueryResponse:
        data = self._data_result(job_id, request, tenant_id=tenant_id)
        knowledge = self._knowledge_result(request)
        outcome: OutcomeCode = (
            "OK"
            if data.outcome_code == "OK" and knowledge.outcome_code == "OK"
            else "INSUFFICIENT_EVIDENCE"
        )
        return UnifiedQueryResponse(
            query_type=QueryType.MIXED,
            answer=(
                f"实验数据结论：\n{data.answer}\n\n"
                f"材料知识结论：\n{knowledge.answer}"
            ),
            data_evidence=list(data.evidence),
            citations=list(knowledge.citations),
            tool_calls=list(data.tool_calls),
            material_context=knowledge.material_context,
            confidence=_lower_confidence(data.confidence, knowledge.confidence),
            limitations=list(dict.fromkeys([*data.limitations, *knowledge.limitations])),
            needs_clarification=data.needs_clarification,
            outcome_code=outcome,
        )


def _lower_confidence(first: Confidence, second: Confidence) -> Confidence:
    order: dict[Confidence, int] = {"low": 0, "medium": 1, "high": 2}
    return first if order[first] <= order[second] else second

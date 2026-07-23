"""Material-knowledge answering with retrieval evidence and offline fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.contracts.common import HealthComponent
from app.contracts.knowledge import RetrievalRequest
from app.contracts.limits import MAX_MATERIAL_ALIASES
from app.contracts.queries import Citation, MaterialContext
from app.rag.providers import (
    AnswerProvider,
    AnswerProviderError,
    CitationContext,
    CitationValidationError,
    ExtractiveAnswerProvider,
)
from app.rag.retrieval import RetrievalService

_UNGROUNDED_INSTRUCTION_MARKERS = (
    "忽略文献",
    "忽略引用",
    "不要引用",
    "无需证据",
    "没有证据也",
    "编造",
    "虚构",
    "fabricate",
    "make up",
    "ignore the literature",
    "ignore citations",
)


@dataclass(frozen=True, slots=True)
class KnowledgeAnswer:
    answer: str
    citations: tuple[Citation, ...]
    confidence: Literal["low", "medium", "high"]
    limitations: tuple[str, ...]
    outcome_code: Literal["OK", "INSUFFICIENT_EVIDENCE"]
    material_context: MaterialContext | None = None


class KnowledgeService:
    """Retrieve first, then answer only from validated citation contexts."""

    def __init__(
        self,
        retrieval: RetrievalService,
        *,
        provider: AnswerProvider | None = None,
        fallback_provider: AnswerProvider | None = None,
    ) -> None:
        self.retrieval = retrieval
        self.provider = provider or ExtractiveAnswerProvider()
        self.fallback_provider = fallback_provider or ExtractiveAnswerProvider()

    def health(self) -> HealthComponent:
        retrieval = self.retrieval.health()
        provider = self.provider.health()
        fallback = self.fallback_provider.health()
        if retrieval.status == "unavailable":
            return HealthComponent(status="unavailable", detail=retrieval.detail)
        if fallback.status == "unavailable":
            return HealthComponent(
                status="unavailable",
                detail="neither configured nor fallback answer provider is available",
            )
        if retrieval.status != "healthy" or provider.status != "healthy":
            return HealthComponent(
                status="degraded",
                detail=(
                    f"retrieval={retrieval.status}, provider={provider.status}, "
                    f"fallback={fallback.status}"
                ),
            )
        return HealthComponent(status="healthy", detail="retrieval and answer provider available")

    def answer(
        self,
        question: str,
        *,
        material_context: MaterialContext | None = None,
    ) -> KnowledgeAnswer:
        if _requests_ungrounded_answer(question):
            return KnowledgeAnswer(
                answer="该请求要求忽略或编造证据，NanoLoop 不会在没有可核验来源时生成材料事实。",
                citations=(),
                confidence="low",
                limitations=("拒绝绕过知识库引用与证据约束",),
                outcome_code="INSUFFICIENT_EVIDENCE",
                material_context=material_context,
            )

        aliases = _material_aliases(material_context)
        report = self.retrieval.retrieve_with_report(
            RetrievalRequest(query=question, material_aliases=aliases)
        )
        base_limitations = list(report.warnings)
        base_limitations.append("知识库仅包含团队已导入文档，不代表完整文献综述")
        if not report.chunks:
            if report.health.detail:
                base_limitations.append(report.health.detail)
            return KnowledgeAnswer(
                answer="知识库证据不足，无法基于当前已导入文档回答该问题。",
                citations=(),
                confidence="low",
                limitations=tuple(dict.fromkeys(base_limitations)),
                outcome_code="INSUFFICIENT_EVIDENCE",
                material_context=material_context,
            )

        contexts = tuple(
            CitationContext(citation_id=f"C{index}", chunk=chunk)
            for index, chunk in enumerate(report.chunks, start=1)
        )
        provider = self.provider
        provider_fallback = False
        if provider.health().status != "healthy":
            provider = self.fallback_provider
            provider_fallback = True
        try:
            generated = provider.generate(
                question=question,
                contexts=contexts,
                material_context=material_context,
            )
        except (AnswerProviderError, CitationValidationError) as error:
            if provider is self.fallback_provider:
                return KnowledgeAnswer(
                    answer="知识库证据已检索，但回答提供器未能产生可验证引用。",
                    citations=(),
                    confidence="low",
                    limitations=tuple(
                        dict.fromkeys([*base_limitations, f"answer provider failure: {error}"])
                    ),
                    outcome_code="INSUFFICIENT_EVIDENCE",
                    material_context=material_context,
                )
            generated = self.fallback_provider.generate(
                question=question,
                contexts=contexts,
                material_context=material_context,
            )
            provider_fallback = True

        used_ids = set(generated.used_citation_ids)
        citations = tuple(
            _citation_from_context(context)
            for context in contexts
            if context.citation_id in used_ids
        )
        limitations = [*base_limitations, *generated.limitations]
        if provider_fallback:
            limitations.append("生成式回答不可用，已降级为离线引用摘录")
        return KnowledgeAnswer(
            answer=generated.answer,
            citations=citations,
            confidence=generated.confidence,
            limitations=tuple(dict.fromkeys(limitations)),
            outcome_code="OK",
            material_context=material_context,
        )


def _requests_ungrounded_answer(question: str) -> bool:
    normalized = " ".join(question.casefold().split())
    return any(marker in normalized for marker in _UNGROUNDED_INSTRUCTION_MARKERS)


def _material_aliases(context: MaterialContext | None) -> list[str]:
    if context is None:
        return []
    values = [context.formula, context.name, *context.aliases]
    aliases = list(
        dict.fromkeys(value.strip() for value in values if value and value.strip())
    )
    return aliases[:MAX_MATERIAL_ALIASES]


def _citation_from_context(context: CitationContext) -> Citation:
    compact = " ".join(context.chunk.text.split())
    excerpt = compact if len(compact) <= 160 else compact[:159].rstrip() + "…"
    return Citation(
        citation_id=context.citation_id,
        doc_id=context.chunk.doc_id,
        title=context.chunk.title,
        page=context.chunk.page_start,
        chunk_id=context.chunk.chunk_id,
        excerpt=excerpt,
        retrieval_score=context.chunk.retrieval_score,
        source_type=context.chunk.source_type,
        citation_text=context.chunk.citation_text,
    )

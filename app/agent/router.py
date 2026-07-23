"""Deterministic first-pass classifier for data, knowledge, and mixed questions."""

from __future__ import annotations

from dataclasses import dataclass

from app.contracts.enums import QueryType
from app.contracts.queries import MaterialContext

_DATA_SIGNALS = (
    "数量",
    "颗粒数",
    "粒径",
    "密度",
    "覆盖率",
    "周长",
    "哪组",
    "哪张",
    "最高",
    "最低",
    "排序",
    "分布",
    "当前结果",
    "当前数据",
    "任务概览",
    "结果概览",
    "数据概览",
    "结果汇总",
    "我们的结果",
    "我们这批",
    "复核",
    "模型结果",
    "run",
)
_KNOWLEDGE_SIGNALS = (
    "特性",
    "性质",
    "机理",
    "用途",
    "应用",
    "作用",
    "领域",
    "优势",
    "结构",
    "稳定性",
    "催化",
    "氧空位",
    "析出",
    "成核",
    "粗化",
    "团聚",
    "应变",
    "已有研究",
    "文献",
    "报道",
    "为什么",
    "研究背景",
    "known",
    "literature",
    "application",
    "property",
    "mechanism",
)
_CONTEXTUAL_MATERIAL_QUESTIONS = ("这个材料", "该材料", "这种材料")


@dataclass(frozen=True, slots=True)
class RouteDecision:
    query_type: QueryType
    confidence: float
    needs_clarification: bool
    matched_data_signals: tuple[str, ...] = ()
    matched_knowledge_signals: tuple[str, ...] = ()


class QueryRouter:
    @staticmethod
    def requires_material_context(question: str) -> bool:
        normalized = question.casefold().strip()
        return any(signal in normalized for signal in _CONTEXTUAL_MATERIAL_QUESTIONS)

    def classify(
        self,
        question: str,
        *,
        material_context: MaterialContext | None = None,
    ) -> RouteDecision:
        normalized = question.casefold().strip()
        data = tuple(signal for signal in _DATA_SIGNALS if signal in normalized)
        knowledge = tuple(signal for signal in _KNOWLEDGE_SIGNALS if signal in normalized)
        contextual = self.requires_material_context(question)
        has_material = material_context is not None and bool(
            material_context.formula or material_context.name or material_context.aliases
        )
        if data and knowledge:
            return RouteDecision(QueryType.MIXED, 0.95, False, data, knowledge)
        if data:
            return RouteDecision(QueryType.ANALYSIS_DATA, 0.90, False, data, knowledge)
        if knowledge or (contextual and has_material):
            return RouteDecision(QueryType.MATERIAL_KNOWLEDGE, 0.90, False, data, knowledge)
        return RouteDecision(
            QueryType.AUTO,
            0.0,
            True,
            data,
            knowledge,
        )

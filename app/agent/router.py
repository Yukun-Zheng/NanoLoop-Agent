"""Deterministic first-pass classifier for data, knowledge, and mixed questions.

Metric words such as ``密度`` or ``分布`` occur in both scientific-literature
questions and requests about the caller's measurements.  Treating those words as
standalone data intent turns ordinary knowledge questions into mixed queries.  A
data route therefore requires either an experimental-scope anchor (for example
``我们这张图``), an explicit quantitative operation, or an overview/review
command.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.contracts.enums import QueryType
from app.contracts.queries import MaterialContext

_DATA_METRIC_SIGNALS = (
    "颗粒数密度",
    "颗粒密度",
    "number density",
    "颗粒数",
    "particle count",
    "平均粒径",
    "mean diameter",
    "粒径",
    "覆盖率",
    "coverage",
    "周长密度",
    "边界密度",
    "perimeter density",
    "boundary density",
    "周长",
    "数量",
    "密度",
    "分布",
)
_DATA_SCOPE_SIGNALS = (
    "我们这张图",
    "这张图",
    "当前样品",
    "这个样品",
    "我们的",
    "我们这批",
    "我们",
    "当前",
    "本次",
    "哪组",
    "哪张",
    "哪个样品",
    "任务",
    "结果",
    "数据",
    "模型结果",
    "our ",
    "current ",
    "this image",
    "this sample",
    "which image",
    "which sample",
    "which group",
    "job ",
    "run",
)
_DATA_OPERATION_SIGNALS = (
    "多少",
    "是多少",
    "最高",
    "最低",
    "排序",
    "比较",
    "差异",
    "how many",
    "highest",
    "lowest",
    "rank",
    "compare",
)
_DATA_COMMAND_SIGNALS = (
    "当前结果",
    "当前数据",
    "任务概览",
    "结果概览",
    "数据概览",
    "结果汇总",
    "我们的结果",
    "我们这批",
    "复核",
    "质量门控",
    "模型结果",
    "quality gate",
    "review",
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
    "缺位",
    "高温",
    "处理时间",
    "已有研究",
    "文献",
    "知识库",
    "报道",
    "为什么",
    "研究背景",
    "材料因素",
    "完整化学式",
    "化学式",
    "样品标签",
    "标签",
    "谨慎结论",
    "形貌指标",
    "掩码",
    "图像统计",
    "替代",
    "电化学性能",
    "报告",
    "一致",
    "known",
    "literature",
    "reported",
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
        metrics = tuple(signal for signal in _DATA_METRIC_SIGNALS if signal in normalized)
        scope = tuple(signal for signal in _DATA_SCOPE_SIGNALS if signal in normalized)
        operations = tuple(
            signal for signal in _DATA_OPERATION_SIGNALS if signal in normalized
        )
        commands = tuple(signal for signal in _DATA_COMMAND_SIGNALS if signal in normalized)
        data = (
            tuple(dict.fromkeys((*metrics, *scope, *operations, *commands)))
            if commands or (metrics and (scope or operations))
            else ()
        )
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

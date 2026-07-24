"""Deterministic first-pass classifier for conversational and evidence questions.

Metric words such as ``密度`` or ``分布`` occur in both scientific-literature
questions and requests about the caller's measurements.  Treating those words as
standalone data intent turns ordinary questions into tool calls.  NanoLoop is a
general assistant first: a data route requires an experimental-scope anchor or
quantitative operation, while RAG requires an explicit request for sources.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.contracts.enums import QueryType
from app.contracts.queries import MaterialContext

_DATA_METRIC_SIGNALS = (
    "颗粒数密度",
    "颗粒密度",
    "number density",
    "颗粒数",
    "颗粒",
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
    "哪个模型",
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
    "概括当前任务",
    "总结当前任务",
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
_RAG_REQUEST_SIGNALS = (
    "文献",
    "知识库",
    "已有研究",
    "研究报道",
    "论文",
    "引用",
    "出处",
    "来源",
    "检索",
    "查资料",
    "有依据",
    "reference",
    "references",
    "literature",
    "knowledge base",
    "reported",
    "source",
    "citation",
)
_CONTEXTUAL_MATERIAL_QUESTIONS = ("这个材料", "该材料", "这种材料")
_GENERAL_CHAT_SIGNALS = (
    "你好",
    "您好",
    "hello",
    "hi",
    "你是谁",
    "能做什么",
    "怎么用",
    "如何使用",
    "这个系统",
    "当前页面",
    "下一步",
    "操作",
    "帮助",
    "概括当前任务",
    "总结当前任务",
)
_FOLLOW_UP_PREFIXES = ("那", "那么", "这个", "这种", "它", "为什么", "呢", "再说")
_EVIDENCE_FOLLOW_UP_SIGNALS = (
    "这个差异",
    "这种差异",
    "上述差异",
    "这个结果",
    "这种结果",
    "上述结果",
    "刚才的结果",
    "这个数",
    "这个数据",
    "这些数据",
)
_MATERIAL_TOKEN = re.compile(r"\b(?:La|Nd|Ba|Sr|Ca)[A-Z][A-Za-z0-9]*\b")


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
        previous_query_type: QueryType | None = None,
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
        rag_request = tuple(
            signal for signal in _RAG_REQUEST_SIGNALS if signal in normalized
        )
        if _MATERIAL_TOKEN.search(question):
            knowledge = tuple(dict.fromkeys((*knowledge, "material token")))
        contextual = self.requires_material_context(question)
        if data and knowledge:
            return RouteDecision(QueryType.MIXED, 0.95, False, data, knowledge)
        if (
            "为什么" in normalized
            and previous_query_type in {QueryType.ANALYSIS_DATA, QueryType.MIXED}
            and any(signal in normalized for signal in _EVIDENCE_FOLLOW_UP_SIGNALS)
        ):
            return RouteDecision(QueryType.MIXED, 0.92, False, data, knowledge)
        if data:
            return RouteDecision(QueryType.ANALYSIS_DATA, 0.90, False, data, knowledge)
        if rag_request:
            return RouteDecision(QueryType.MATERIAL_KNOWLEDGE, 0.90, False, data, knowledge)
        if (
            previous_query_type is QueryType.MATERIAL_KNOWLEDGE
            and normalized.startswith(_FOLLOW_UP_PREFIXES)
            and "material token" in knowledge
        ):
            return RouteDecision(
                QueryType.MATERIAL_KNOWLEDGE,
                0.82,
                False,
                data,
                knowledge,
            )
        if any(signal in normalized for signal in _GENERAL_CHAT_SIGNALS):
            return RouteDecision(QueryType.GENERAL_CHAT, 0.95, False, data, knowledge)
        # Contextual material questions, scientific background questions, and
        # ordinary open-ended requests remain conversational unless the user
        # explicitly asks for current measurements or sourced knowledge.
        confidence = 0.75 if contextual or knowledge else 0.55
        return RouteDecision(QueryType.GENERAL_CHAT, confidence, False, data, knowledge)

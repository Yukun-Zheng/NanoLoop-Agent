"""Tests for deterministic query classification and clarification."""

import pytest

from app.agent.router import QueryRouter
from app.contracts.enums import QueryType
from app.contracts.queries import MaterialContext


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("哪组颗粒数密度最高？", QueryType.ANALYSIS_DATA),
        ("SrNi 有哪些已知应用和文献？", QueryType.MATERIAL_KNOWLEDGE),
        ("已有研究怎么说，我们这批覆盖率最高吗？", QueryType.MIXED),
        ("这个材料可以在什么领域发挥作用？", QueryType.MATERIAL_KNOWLEDGE),
        ("A 位缺位为什么可能促进析出？", QueryType.MATERIAL_KNOWLEDGE),
        ("当前平均粒径较大，是否可能与高温粗化有关？", QueryType.MIXED),
    ],
)
def test_classifies_frozen_signal_groups(question: str, expected: QueryType) -> None:
    context = MaterialContext(name="LaNi") if "这个材料" in question else None
    decision = QueryRouter().classify(question, material_context=context)

    assert decision.query_type == expected
    assert not decision.needs_clarification


def test_contextual_material_question_requires_material_context() -> None:
    router = QueryRouter()

    missing = router.classify("这个材料怎么样？")
    resolved = router.classify(
        "这个材料怎么样？",
        material_context=MaterialContext(formula="SrNiO3-x"),
    )

    assert missing.needs_clarification
    assert resolved.query_type == QueryType.MATERIAL_KNOWLEDGE
    assert router.requires_material_context("这个材料怎么样？")
    assert not router.requires_material_context("SrNi 有哪些已知应用？")


def test_unknown_intent_requests_clarification_without_guessing() -> None:
    decision = QueryRouter().classify("帮我看看")

    assert decision.query_type == QueryType.AUTO
    assert decision.needs_clarification
    assert decision.confidence == 0

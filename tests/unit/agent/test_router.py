"""Tests for deterministic query classification and clarification."""

import json
from pathlib import Path

import pytest

from app.agent.router import QueryRouter
from app.contracts.enums import QueryType
from app.contracts.queries import MaterialContext


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("哪组颗粒数密度最高？", QueryType.ANALYSIS_DATA),
        ("当前样品的周长密度是多少？", QueryType.ANALYSIS_DATA),
        ("SrNi 有哪些已知应用和文献？", QueryType.MATERIAL_KNOWLEDGE),
        ("已有研究怎么说，我们这批覆盖率最高吗？", QueryType.MIXED),
        ("这个材料可以在什么领域发挥作用？", QueryType.GENERAL_CHAT),
        ("A 位缺位为什么可能促进析出？", QueryType.GENERAL_CHAT),
        ("当前平均粒径较大，是否可能与高温粗化有关？", QueryType.MIXED),
    ],
)
def test_classifies_frozen_signal_groups(question: str, expected: QueryType) -> None:
    context = MaterialContext(name="LaNi") if "这个材料" in question else None
    decision = QueryRouter().classify(question, material_context=context)

    assert decision.query_type == expected
    assert not decision.needs_clarification


def test_contextual_material_question_stays_conversational() -> None:
    router = QueryRouter()

    missing = router.classify("这个材料怎么样？")
    resolved = router.classify(
        "这个材料怎么样？",
        material_context=MaterialContext(formula="SrNiO3-x"),
    )

    assert missing.query_type is QueryType.GENERAL_CHAT
    assert resolved.query_type is QueryType.GENERAL_CHAT
    assert not missing.needs_clarification
    assert not resolved.needs_clarification
    assert router.requires_material_context("这个材料怎么样？")
    assert not router.requires_material_context("SrNi 有哪些已知应用？")


def test_unknown_intent_stays_conversational_without_guessing_science() -> None:
    decision = QueryRouter().classify("帮我看看")

    assert decision.query_type == QueryType.GENERAL_CHAT
    assert not decision.needs_clarification
    assert decision.confidence == 0.55


def test_workflow_why_follow_up_stays_in_general_chat() -> None:
    decision = QueryRouter().classify(
        "为什么局部区域可以跳过？",
        previous_query_type=QueryType.GENERAL_CHAT,
    )

    assert decision.query_type is QueryType.GENERAL_CHAT
    assert not decision.needs_clarification


def test_general_chat_and_history_aware_follow_up_are_deterministic() -> None:
    router = QueryRouter()

    greeting = router.classify("你好，你能帮我做什么？")
    follow_up = router.classify(
        "那 NdNi 呢？",
        previous_query_type=QueryType.MATERIAL_KNOWLEDGE,
    )
    mechanism_follow_up = router.classify(
        "为什么可能出现这种差异？",
        previous_query_type=QueryType.ANALYSIS_DATA,
    )

    assert greeting.query_type is QueryType.GENERAL_CHAT
    assert follow_up.query_type is QueryType.MATERIAL_KNOWLEDGE
    assert mechanism_follow_up.query_type is QueryType.MIXED


@pytest.mark.parametrize(
    "question",
    [
        "这个数据和实验是做什么的？",
        "那我现在是什么情况？",
        "你能做什么？",
        "帮我写一段 Python 代码",
        "为什么天空是蓝色的？",
    ],
)
def test_open_ended_questions_are_general_conversation_first(question: str) -> None:
    decision = QueryRouter().classify(
        question,
        previous_query_type=QueryType.ANALYSIS_DATA,
    )

    assert decision.query_type is QueryType.GENERAL_CHAT
    assert not decision.needs_clarification


def test_curated_rag_questions_only_auto_route_when_evidence_is_explicit() -> None:
    questions_path = (
        Path(__file__).resolve().parents[3] / "demo_data" / "rag" / "questions.jsonl"
    )
    records = [
        json.loads(line)
        for line in questions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {record["query_id"]: record for record in records}
    observed: dict[str, QueryType] = {}
    for record in records:
        raw_context = record.get("material_context")
        context = MaterialContext.model_validate(raw_context) if raw_context else None
        observed[record["query_id"]] = QueryRouter().classify(
            record["question"],
            material_context=context,
        ).query_type

    assert len(observed) == 30
    assert set(by_id) == set(observed)
    assert observed["q002"] is QueryType.GENERAL_CHAT
    assert observed["q023"] is QueryType.MATERIAL_KNOWLEDGE
    assert observed["q025"] is QueryType.MIXED
    assert observed["q026"] is QueryType.MIXED
    assert observed["q029"] is QueryType.GENERAL_CHAT
    assert observed["q030"] is QueryType.MATERIAL_KNOWLEDGE

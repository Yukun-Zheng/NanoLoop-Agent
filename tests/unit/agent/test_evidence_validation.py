from __future__ import annotations

import pytest

from app.agent.evidence_validation import validate_conversation_answer
from app.contracts.knowledge import RetrievedChunk
from app.contracts.queries import ToolEvidence
from app.rag.providers import CitationContext, CitationValidationError


def _evidence() -> ToolEvidence:
    return ToolEvidence(
        tool_name="compare_models",
        validated_arguments={"metric": "particle_count"},
        rows=[{"run_id": "run_1", "particle_count": 3}],
        aggregates={"maximum": 3},
        units={"particle_count": "count"},
        source_run_ids=["run_1"],
    )


def test_data_reference_accepts_exact_backend_value() -> None:
    validate_conversation_answer(
        answer="当前运行检测到 3 个颗粒 [D1]。",
        limitations=[],
        used_data_ids=["D1"],
        used_citation_ids=[],
        data_evidence=[_evidence()],
        citation_contexts=[],
        allow_uncited_general_chat=False,
    )


@pytest.mark.parametrize(
    ("answer", "used_ids"),
    [
        ("当前运行检测到 4 个颗粒 [D1]。", ["D1"]),
        ("当前运行检测到 3 个颗粒 [D99]。", ["D99"]),
        ("当前运行数据不足 [D#]。", []),
        ("当前运行检测到 3 nm 颗粒 [D1]。", ["D1"]),
    ],
)
def test_data_reference_rejects_changed_value_unknown_id_or_unit(
    answer: str,
    used_ids: list[str],
) -> None:
    with pytest.raises(CitationValidationError):
        validate_conversation_answer(
            answer=answer,
            limitations=[],
            used_data_ids=used_ids,
            used_citation_ids=[],
            data_evidence=[_evidence()],
            citation_contexts=[],
            allow_uncited_general_chat=False,
        )


def test_data_reference_rejects_substring_number_match() -> None:
    evidence = _evidence().model_copy(
        update={
            "rows": [{"run_id": "run_x", "particle_count": 13}],
            "aggregates": {"maximum": 13},
            "source_run_ids": ["run_x"],
        }
    )
    with pytest.raises(CitationValidationError):
        validate_conversation_answer(
            answer="当前运行检测到 3 个颗粒 [D1]。",
            limitations=[],
            used_data_ids=["D1"],
            used_citation_ids=[],
            data_evidence=[evidence],
            citation_contexts=[],
            allow_uncited_general_chat=False,
        )


def test_knowledge_reference_must_match_current_context() -> None:
    context = CitationContext(
        citation_id="C1",
        chunk=RetrievedChunk(
            chunk_id="chunk_1",
            doc_id="doc_1",
            title="fixture",
            source_type="paper",
            citation_text="fixture citation",
            page_start=1,
            page_end=1,
            text="LaNi 标签不能证明完整化学式。",
            material_tags=["LaNi"],
            retrieval_score=1.0,
        ),
    )

    validate_conversation_answer(
        answer="LaNi 标签不能证明完整化学式 [C1]。",
        limitations=[],
        used_data_ids=[],
        used_citation_ids=["C1"],
        data_evidence=[],
        citation_contexts=[context],
        allow_uncited_general_chat=False,
    )
    with pytest.raises(CitationValidationError):
        validate_conversation_answer(
            answer="LaNi 就是 LaNiO3 [C99]。",
            limitations=[],
            used_data_ids=[],
            used_citation_ids=["C99"],
            data_evidence=[],
            citation_contexts=[context],
            allow_uncited_general_chat=False,
        )

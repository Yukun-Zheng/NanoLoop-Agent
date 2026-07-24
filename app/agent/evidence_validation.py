"""Validate only final user-visible prose against current-turn evidence."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation

from app.contracts.queries import ToolEvidence
from app.rag.providers import CitationContext, CitationValidationError

_DATA_REFERENCE = re.compile(r"\[(D\d+)\]")
_CITATION_REFERENCE = re.compile(r"\[(C\d+)\]")
_ANY_EVIDENCE_REFERENCE = re.compile(r"\[([CD][^\]]*)\]")
_VALID_EVIDENCE_ID = re.compile(r"[CD]\d+")
_NUMBER = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d+)?|\.\d+)")
_UNIT = re.compile(r"(?<![A-Za-z])(?:nm|µm|μm|um|px|px²|px2|%)(?![A-Za-z])", re.I)
_SENTENCE = re.compile(r".*?[。！？.!?；;\n]+|.+$", re.S)


def validate_conversation_answer(
    *,
    answer: str,
    limitations: Sequence[str],
    used_data_ids: Sequence[str],
    used_citation_ids: Sequence[str],
    data_evidence: Sequence[ToolEvidence],
    citation_contexts: Sequence[CitationContext],
    allow_uncited_general_chat: bool,
) -> None:
    """Reject unknown references, altered data values, and citation-list drift."""

    if not answer.strip():
        raise CitationValidationError("conversation answer cannot be empty")
    visible = "\n".join((answer, *limitations))
    data_refs = set(_DATA_REFERENCE.findall(visible))
    citation_refs = set(_CITATION_REFERENCE.findall(visible))
    malformed_refs = {
        value
        for value in _ANY_EVIDENCE_REFERENCE.findall(visible)
        if _VALID_EVIDENCE_ID.fullmatch(value) is None
    }
    if malformed_refs:
        raise CitationValidationError("answer contains a malformed evidence reference")
    valid_data = {f"D{index}" for index in range(1, len(data_evidence) + 1)}
    valid_citations = {context.citation_id for context in citation_contexts}
    if data_refs != set(used_data_ids) or not data_refs <= valid_data:
        raise CitationValidationError("data evidence references do not match this turn")
    if citation_refs != set(used_citation_ids) or not citation_refs <= valid_citations:
        raise CitationValidationError("knowledge citations do not match this turn")
    if not data_refs and not citation_refs:
        if allow_uncited_general_chat:
            return
        if _NUMBER.search(_remove_reference_numbers(visible)):
            raise CitationValidationError("numeric claims require data evidence")
    if citation_contexts and not allow_uncited_general_chat:
        for sentence in _SENTENCE.findall(answer):
            if _DATA_REFERENCE.search(sentence) or _CITATION_REFERENCE.search(sentence):
                continue
            normalized = _remove_reference_numbers(sentence).strip(" \t\r\n-*#：:。！？.!?；;")
            if not normalized or _is_refusal(normalized):
                continue
            raise CitationValidationError(
                "material knowledge sentences require a [C#] reference"
            )

    evidence_text = {
        f"D{index}": json.dumps(item.model_dump(mode="json"), ensure_ascii=False)
        for index, item in enumerate(data_evidence, start=1)
    }
    evidence_numbers = {
        evidence_id: _numbers_in_text(text)
        for evidence_id, text in evidence_text.items()
    }
    for sentence in _SENTENCE.findall(visible):
        clean = _remove_reference_numbers(sentence)
        numbers = _NUMBER.findall(clean)
        units = _UNIT.findall(clean)
        sentence_refs = _DATA_REFERENCE.findall(sentence)
        if numbers and not sentence_refs:
            raise CitationValidationError("every numeric sentence requires a [D#] reference")
        if not sentence_refs:
            continue
        source = "\n".join(evidence_text[ref] for ref in sentence_refs)
        allowed_numbers = set().union(
            *(evidence_numbers[ref] for ref in sentence_refs)
        )
        if any(_normalize_number(number) not in allowed_numbers for number in numbers):
            raise CitationValidationError("answer changed a value from data evidence")
        normalized_source = source.casefold().replace("μ", "µ").replace("um", "µm")
        for unit in units:
            normalized_unit = unit.casefold().replace("μ", "µ").replace("um", "µm")
            if normalized_unit not in normalized_source:
                raise CitationValidationError("answer changed a unit from data evidence")


def _remove_reference_numbers(value: str) -> str:
    return _DATA_REFERENCE.sub("", _CITATION_REFERENCE.sub("", value))


def _normalize_number(value: str) -> Decimal | None:
    try:
        return Decimal(value).normalize()
    except InvalidOperation:
        return None


def _numbers_in_text(value: str) -> set[Decimal]:
    numbers: set[Decimal] = set()
    for raw in _NUMBER.findall(value):
        normalized_number = _normalize_number(raw)
        if normalized_number is not None:
            numbers.add(normalized_number)
    return numbers


def _is_refusal(value: str) -> bool:
    return any(
        marker in value
        for marker in (
            "证据不足",
            "无法确认",
            "无法回答",
            "不能确认",
            "insufficient evidence",
            "cannot confirm",
        )
    )

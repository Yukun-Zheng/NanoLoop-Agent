"""Grounded answer providers with strict citation validation."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import httpx

from app.contracts.common import HealthComponent
from app.contracts.knowledge import RetrievedChunk
from app.contracts.queries import MaterialContext

Confidence = Literal["low", "medium", "high"]
_CITATION_PATTERN = re.compile(r"\[(C\d+)\]")
_FACTUAL_UNIT_PATTERN = re.compile(r".*?[。！？.!?](?:\s*\[C\d+\])*|.+$")
_INSUFFICIENT_PHRASES = (
    "证据不足",
    "无法判断",
    "无法确认",
    "not enough evidence",
    "insufficient evidence",
)


class AnswerProviderError(RuntimeError):
    """Raised when a configured answer provider fails or violates its contract."""


class CitationValidationError(AnswerProviderError):
    """Raised when generated prose is not grounded in supplied citation IDs."""


@dataclass(frozen=True, slots=True)
class CitationContext:
    citation_id: str
    chunk: RetrievedChunk


@dataclass(frozen=True, slots=True)
class ProviderAnswer:
    answer: str
    used_citation_ids: tuple[str, ...]
    confidence: Confidence
    limitations: tuple[str, ...] = ()


class AnswerProvider(Protocol):
    def health(self) -> HealthComponent: ...

    def generate(
        self,
        *,
        question: str,
        contexts: Sequence[CitationContext],
        material_context: MaterialContext | None,
    ) -> ProviderAnswer: ...


class HttpResponse(Protocol):
    def raise_for_status(self) -> None: ...

    def json(self) -> Any: ...


class HttpClient(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any],
        timeout: float,
    ) -> HttpResponse: ...


class ExtractiveAnswerProvider:
    """Offline fallback that presents retrieved excerpts without invented synthesis."""

    def __init__(self, *, max_contexts: int = 6) -> None:
        if max_contexts < 1:
            raise ValueError("max_contexts must be positive")
        self.max_contexts = max_contexts

    def health(self) -> HealthComponent:
        return HealthComponent(status="healthy", detail="offline extractive answer provider")

    def generate(
        self,
        *,
        question: str,
        contexts: Sequence[CitationContext],
        material_context: MaterialContext | None,
    ) -> ProviderAnswer:
        del question, material_context
        selected = list(contexts[: self.max_contexts])
        if not selected:
            return ProviderAnswer(
                answer="知识库证据不足，无法基于现有文档回答该问题。",
                used_citation_ids=(),
                confidence="low",
                limitations=("没有达到检索阈值的知识片段",),
            )
        lines = ["知识库检索到以下相关证据摘录："]
        for context in selected:
            excerpt = _compact_excerpt(context.chunk.text, limit=160)
                        for sentence in _split_sentences(excerpt):
                lines.append(f"- [{context.citation_id}] {sentence}")

        answer = ProviderAnswer(
            answer="\n".join(lines),
            used_citation_ids=tuple(context.citation_id for context in selected),
            confidence="medium",
            limitations=("当前为离线摘录模式，不代表完整文献综述",),
        )
        validate_provider_answer(answer, {context.citation_id for context in selected})
        return answer


class OpenAICompatibleProvider:
    """Strict JSON answer generation through an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        *,
        base_url: str | None,
        api_key: str | None,
        model: str | None,
        client: HttpClient | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key
        self.model = model
        self.client = client or cast(HttpClient, httpx.Client())
        self.timeout_seconds = timeout_seconds

    def health(self) -> HealthComponent:
        missing = [
            name
            for name, value in (
                ("LLM_BASE_URL", self.base_url),
                ("LLM_API_KEY", self.api_key),
                ("LLM_MODEL", self.model),
            )
            if not value
        ]
        if missing:
            return HealthComponent(
                status="unavailable",
                detail=f"missing OpenAI-compatible configuration: {', '.join(missing)}",
            )
        return HealthComponent(status="healthy", detail="OpenAI-compatible provider configured")

    def generate(
        self,
        *,
        question: str,
        contexts: Sequence[CitationContext],
        material_context: MaterialContext | None,
    ) -> ProviderAnswer:
        if self.health().status != "healthy":
            raise AnswerProviderError(self.health().detail or "LLM provider unavailable")
        if not contexts:
            raise AnswerProviderError("generation requires at least one retrieved context")

        valid_ids = {context.citation_id for context in contexts}
        context_text = "\n\n".join(
            f"[{context.citation_id}] title={context.chunk.title}; "
            f"page={context.chunk.page_start}; text={context.chunk.text}"
            for context in contexts
        )
        material = material_context.model_dump(mode="json") if material_context else None
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是材料科研知识助手，只能依据 CONTEXT 回答。每个事实句必须引用 "
                        "[C#]；区分文献报道与当前样品；证据不足时明确拒答。只输出 JSON，字段为 "
                        "answer, used_citation_ids, confidence, limitations。"
                        "不得提供危险实验操作细节。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"MATERIAL_CONTEXT: {json.dumps(material, ensure_ascii=False)}\n"
                        f"QUESTION: {question}\nCONTEXT:\n{context_text}"
                    ),
                },
            ],
        }
        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(_strip_json_fence(content))
            confidence = parsed["confidence"]
            if confidence not in {"low", "medium", "high"}:
                raise ValueError("invalid confidence")
            answer = ProviderAnswer(
                answer=str(parsed["answer"]).strip(),
                used_citation_ids=tuple(str(item) for item in parsed["used_citation_ids"]),
                confidence=confidence,
                limitations=tuple(str(item) for item in parsed.get("limitations", [])),
            )
        except (
            httpx.HTTPError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise AnswerProviderError(
                "OpenAI-compatible provider returned an invalid response"
            ) from error
        except Exception as error:
            raise AnswerProviderError("OpenAI-compatible provider request failed") from error
        validate_provider_answer(answer, valid_ids)
        return answer


def validate_provider_answer(answer: ProviderAnswer, valid_citation_ids: set[str]) -> None:
    """Require known citations and a citation marker on each factual answer sentence."""

    if not answer.answer.strip():
        raise CitationValidationError("provider answer cannot be empty")
    referenced = set(_CITATION_PATTERN.findall(answer.answer))
    used = set(answer.used_citation_ids)
    if not referenced <= valid_citation_ids or not used <= valid_citation_ids:
        raise CitationValidationError("provider used a citation outside retrieved contexts")
    if referenced != used:
        raise CitationValidationError("used_citation_ids must match citations present in answer")

    for line in answer.answer.splitlines():
        for sentence in _FACTUAL_UNIT_PATTERN.findall(line):
            cleaned = sentence.strip().lstrip("-• ")
            if not cleaned or cleaned.endswith((":", "：")):
                continue
            lower = cleaned.casefold()
            if any(phrase in lower for phrase in _INSUFFICIENT_PHRASES):
                continue
            if not _CITATION_PATTERN.search(cleaned):
                raise CitationValidationError(
                    "every factual sentence must contain a citation marker"
                )


def _compact_excerpt(text: str, *, limit: int) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"

_SENTENCE_SPLIT = re.compile(r"[^。！？.!?]*[。！？.!?]|[^。！？.!?]+")


def _split_sentences(text: str) -> list[str]:
    """Split an excerpt into sentences so each can carry its own citation marker.

    The strict citation validator requires every factual sentence to hold a
    ``[C#]`` marker. Retrieved knowledge excerpts are often multi-sentence, so we
    ground each sentence individually instead of only the first one.
    """

    return [piece.strip() for piece in _SENTENCE_SPLIT.findall(text) if piece.strip()]


def _strip_json_fence(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("message content must be a string")
    stripped = value.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()
    return stripped

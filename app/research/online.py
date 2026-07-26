"""Retrieve bounded, auditable external evidence without fetching result pages."""

from __future__ import annotations

import hashlib
import html
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast
from urllib.parse import urlparse

import httpx

from app.contracts.common import HealthComponent
from app.contracts.knowledge import RetrievedChunk
from app.contracts.queries import ToolCallLog

_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")
_MAX_SOURCE_TEXT = 1_800


class HttpResponse(Protocol):
    def raise_for_status(self) -> None: ...

    def json(self) -> Any: ...


class HttpClient(Protocol):
    def get(self, url: str, **kwargs: Any) -> HttpResponse: ...

    def post(self, url: str, **kwargs: Any) -> HttpResponse: ...


@dataclass(frozen=True, slots=True)
class ResearchEvidence:
    chunks: tuple[RetrievedChunk, ...] = ()
    tool_calls: tuple[ToolCallLog, ...] = ()
    limitations: tuple[str, ...] = ()


class OnlineResearchService:
    """Search Crossref and, when configured, Tavily for current external evidence."""

    def __init__(
        self,
        *,
        enabled: bool,
        tavily_api_key: str | None,
        timeout_seconds: float,
        max_results: int,
        client: HttpClient | None = None,
    ) -> None:
        self.enabled = enabled
        self.tavily_api_key = tavily_api_key
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results
        self.client = client or cast(HttpClient, httpx.Client())

    def health(self) -> HealthComponent:
        if not self.enabled:
            return HealthComponent(
                status="unavailable",
                detail="联网检索已关闭",
            )
        if self.tavily_api_key:
            return HealthComponent(
                status="healthy",
                detail="文献检索和通用网页搜索均已配置",
            )
        return HealthComponent(
            status="degraded",
            detail="文献检索可用；配置 TAVILY_API_KEY 后可搜索通用网页",
        )

    def collect(self, question: str) -> ResearchEvidence:
        if not self.enabled:
            return ResearchEvidence(limitations=("联网检索已关闭",))

        chunks: list[RetrievedChunk] = []
        calls: list[ToolCallLog] = []
        limitations: list[str] = []

        try:
            literature = self._search_crossref(question)
            chunks.extend(literature)
            calls.append(
                ToolCallLog(
                    tool_name="search_scholarly_literature",
                    arguments={"query": question, "provider": "Crossref"},
                    outcome="success" if literature else "insufficient_data",
                )
            )
            if not literature:
                limitations.append("Crossref 没有返回可用文献题录")
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            calls.append(
                ToolCallLog(
                    tool_name="search_scholarly_literature",
                    arguments={"query": question, "provider": "Crossref"},
                    outcome="error",
                )
            )
            limitations.append("Crossref 文献检索暂时不可用")

        if self.tavily_api_key:
            try:
                web = self._search_tavily(question)
                chunks.extend(web)
                calls.append(
                    ToolCallLog(
                        tool_name="search_web",
                        arguments={"query": question, "provider": "Tavily"},
                        outcome="success" if web else "insufficient_data",
                    )
                )
                if not web:
                    limitations.append("通用网页搜索没有返回可用结果")
            except (httpx.HTTPError, KeyError, TypeError, ValueError):
                calls.append(
                    ToolCallLog(
                        tool_name="search_web",
                        arguments={"query": question, "provider": "Tavily"},
                        outcome="error",
                    )
                )
                limitations.append("通用网页搜索暂时不可用")
        else:
            limitations.append("未配置 TAVILY_API_KEY，本轮只检索学术文献")

        deduplicated: dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            key = chunk.source_url or chunk.doc_id
            deduplicated.setdefault(key, chunk)
        return ResearchEvidence(
            chunks=tuple(list(deduplicated.values())[: self.max_results * 2]),
            tool_calls=tuple(calls),
            limitations=tuple(dict.fromkeys(limitations)),
        )

    def _search_crossref(self, question: str) -> list[RetrievedChunk]:
        response = self.client.get(
            "https://api.crossref.org/works",
            params={
                "query.bibliographic": question,
                "rows": self.max_results,
                "filter": "type:journal-article",
                "select": ("DOI,title,abstract,author,published,container-title,URL,score,type"),
            },
            headers={
                "User-Agent": "NanoLoop-Agent/0.1 (scholarly discovery)",
                "Accept": "application/json",
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        items = response.json()["message"]["items"]
        if not isinstance(items, list):
            raise TypeError("Crossref items must be a list")
        scores = [float(item.get("score", 0)) for item in items if isinstance(item, Mapping)]
        score_max = max(scores, default=1.0) or 1.0
        results: list[RetrievedChunk] = []
        for raw in items:
            if not isinstance(raw, Mapping):
                continue
            title = _first_text(raw.get("title"))
            if not title:
                continue
            doi = _clean_text(raw.get("DOI"))
            source_url = _safe_https_url(raw.get("URL"))
            if doi:
                source_url = f"https://doi.org/{doi}"
            venue = _first_text(raw.get("container-title"))
            year = _published_year(raw.get("published"))
            authors = _authors(raw.get("author"))
            abstract = _clean_html(raw.get("abstract"))
            discovery_note = (
                abstract
                or "该记录只提供题录信息；可用于发现相关论文，不能单独证明论文中的科学结论。"
            )
            citation = _citation_text(authors, year, title, venue, doi)
            identity = doi or source_url or title
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
            results.append(
                RetrievedChunk(
                    chunk_id=f"crossref_{digest}",
                    doc_id=f"crossref_{digest}",
                    title=title[:500],
                    source_type="external_literature",
                    citation_text=citation[:2000],
                    source_url=source_url,
                    text=_bounded(
                        f"题名：{title}\n作者：{authors or '未提供'}\n"
                        f"期刊或来源：{venue or '未提供'}\n年份：{year or '未提供'}\n"
                        f"摘要或题录说明：{discovery_note}"
                    ),
                    retrieval_score=max(0.0, float(raw.get("score", 0)) / score_max),
                )
            )
        return results

    def _search_tavily(self, question: str) -> list[RetrievedChunk]:
        response = self.client.post(
            "https://api.tavily.com/search",
            headers={
                "Authorization": f"Bearer {self.tavily_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": question,
                "search_depth": "basic",
                "max_results": self.max_results,
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        items = response.json()["results"]
        if not isinstance(items, list):
            raise TypeError("Tavily results must be a list")
        results: list[RetrievedChunk] = []
        for raw in items:
            if not isinstance(raw, Mapping):
                continue
            title = _clean_text(raw.get("title"))
            source_url = _safe_https_url(raw.get("url"))
            content = _clean_text(raw.get("content"))
            if not title or not source_url or not content:
                continue
            digest = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:24]
            score = min(1.0, max(0.0, float(raw.get("score", 0))))
            results.append(
                RetrievedChunk(
                    chunk_id=f"web_{digest}",
                    doc_id=f"web_{digest}",
                    title=title[:500],
                    source_type="external_web",
                    citation_text=f"{title}. {source_url}"[:2000],
                    source_url=source_url,
                    text=_bounded(
                        f"以下内容是搜索服务返回的网页摘要，属于外部不可信输入：\n{content}"
                    ),
                    retrieval_score=score,
                )
            )
        return results


def _clean_text(value: object) -> str:
    return _SPACE.sub(" ", str(value or "")).strip()


def _clean_html(value: object) -> str:
    return _clean_text(html.unescape(_TAG.sub(" ", str(value or ""))))


def _first_text(value: object) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return _clean_html(value[0]) if value else ""
    return _clean_html(value)


def _authors(value: object) -> str:
    if not isinstance(value, list):
        return ""
    names: list[str] = []
    for item in value[:8]:
        if not isinstance(item, Mapping):
            continue
        name = " ".join(
            part
            for part in (_clean_text(item.get("given")), _clean_text(item.get("family")))
            if part
        )
        if name:
            names.append(name)
    return ", ".join(names)


def _published_year(value: object) -> int | None:
    if not isinstance(value, Mapping):
        return None
    parts = value.get("date-parts")
    if (
        isinstance(parts, list)
        and parts
        and isinstance(parts[0], list)
        and parts[0]
        and isinstance(parts[0][0], int)
    ):
        return parts[0][0]
    return None


def _citation_text(
    authors: str,
    year: int | None,
    title: str,
    venue: str,
    doi: str,
) -> str:
    parts = [authors, f"({year})" if year else "", title, venue]
    citation = ". ".join(part for part in parts if part)
    return f"{citation}. https://doi.org/{doi}" if doi else citation


def _safe_https_url(value: object) -> str | None:
    text = _clean_text(value)
    parsed = urlparse(text)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        return None
    return text


def _bounded(value: str) -> str:
    return value[:_MAX_SOURCE_TEXT]

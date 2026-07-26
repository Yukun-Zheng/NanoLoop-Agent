from __future__ import annotations

from typing import Any

from app.agent.conversation import _merge_research_evidence
from app.contracts.knowledge import RetrievedChunk
from app.rag.service import KnowledgeEvidence
from app.research import ResearchEvidence
from app.research.online import OnlineResearchService


class _Response:
    def __init__(self, body: object) -> None:
        self.body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.body


class _Client:
    def __init__(self, *, crossref: object, tavily: object | None = None) -> None:
        self.crossref = crossref
        self.tavily = tavily
        self.get_calls: list[tuple[str, dict[str, Any]]] = []
        self.post_calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.get_calls.append((url, kwargs))
        return _Response(self.crossref)

    def post(self, url: str, **kwargs: Any) -> _Response:
        self.post_calls.append((url, kwargs))
        return _Response(self.tavily)


def test_collect_combines_literature_and_web_with_auditable_links() -> None:
    client = _Client(
        crossref={
            "message": {
                "items": [
                    {
                        "DOI": "10.1000/example",
                        "title": ["Exsolution in perovskites"],
                        "abstract": "<jats:p>Supported abstract claim.</jats:p>",
                        "author": [{"given": "A.", "family": "Researcher"}],
                        "published": {"date-parts": [[2025, 1, 2]]},
                        "container-title": ["Journal of Examples"],
                        "URL": "https://doi.org/10.1000/example",
                        "score": 20.0,
                    }
                ]
            }
        },
        tavily={
            "results": [
                {
                    "title": "University research overview",
                    "url": "https://example.edu/research",
                    "content": "A bounded external search snippet.",
                    "score": 0.8,
                }
            ]
        },
    )
    service = OnlineResearchService(
        enabled=True,
        tavily_api_key="secret",
        timeout_seconds=5,
        max_results=3,
        client=client,
    )

    evidence = service.collect("perovskite exsolution literature")

    assert [chunk.source_type for chunk in evidence.chunks] == [
        "external_literature",
        "external_web",
    ]
    assert evidence.chunks[0].source_url == "https://doi.org/10.1000/example"
    assert evidence.chunks[1].source_url == "https://example.edu/research"
    assert [call.tool_name for call in evidence.tool_calls] == [
        "search_scholarly_literature",
        "search_web",
    ]
    assert client.get_calls[0][0] == "https://api.crossref.org/works"
    assert client.get_calls[0][1]["params"]["filter"] == "type:journal-article"
    assert client.post_calls[0][0] == "https://api.tavily.com/search"
    assert client.post_calls[0][1]["headers"]["Authorization"] == "Bearer secret"


def test_crossref_remains_available_without_general_web_key() -> None:
    client = _Client(
        crossref={
            "message": {
                "items": [
                    {
                        "DOI": "10.1000/metadata-only",
                        "title": ["Metadata-only discovery"],
                        "score": 1.0,
                    }
                ]
            }
        }
    )
    service = OnlineResearchService(
        enabled=True,
        tavily_api_key=None,
        timeout_seconds=5,
        max_results=2,
        client=client,
    )

    evidence = service.collect("find papers")

    assert len(evidence.chunks) == 1
    assert "不能单独证明" in evidence.chunks[0].text
    assert evidence.tool_calls[0].outcome == "success"
    assert any("未配置 TAVILY_API_KEY" in item for item in evidence.limitations)
    assert client.post_calls == []
    assert service.health().status == "degraded"


def test_non_https_web_results_are_discarded() -> None:
    client = _Client(
        crossref={"message": {"items": []}},
        tavily={
            "results": [
                {
                    "title": "Unsafe result",
                    "url": "http://example.com/result",
                    "content": "Not accepted.",
                    "score": 0.9,
                }
            ]
        },
    )
    service = OnlineResearchService(
        enabled=True,
        tavily_api_key="secret",
        timeout_seconds=5,
        max_results=2,
        client=client,
    )

    evidence = service.collect("find sources")

    assert evidence.chunks == ()
    assert all(call.outcome == "insufficient_data" for call in evidence.tool_calls)


def test_external_chunks_receive_normalized_citation_ids_and_urls() -> None:
    chunk = RetrievedChunk(
        chunk_id="crossref_example",
        doc_id="crossref_example",
        title="Example paper",
        source_type="external_literature",
        citation_text="Example citation",
        source_url="https://doi.org/10.1000/example",
        text="An abstract-backed statement.",
        retrieval_score=0.9,
    )

    merged = _merge_research_evidence(
        KnowledgeEvidence((), (), (), "INSUFFICIENT_EVIDENCE"),
        ResearchEvidence(chunks=(chunk,)),
    )

    assert merged.outcome_code == "OK"
    assert merged.contexts[0].citation_id == "C1"
    assert merged.citations[0].url == "https://doi.org/10.1000/example"

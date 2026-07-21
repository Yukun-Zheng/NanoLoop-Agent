"""Offline extractive knowledge pipeline regression tests (no external assets).

These tests cover developer D's retrieval + extractive-answer path without any
embedding weights, licensed corpus, or API key, so they run in plain CI. They
use a deliberately small in-memory keyword seam: as long as a document has been
chunked, it must retrieve relevant evidence and the extractive provider must
answer only from citations, with honest limitations and an explicit
insufficient-evidence outcome otherwise.

They do not replace production SQLite FTS5, FAISS/embedding, persistence/restart,
Chinese retrieval, authorization, or licensed-corpus acceptance tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.contracts.common import HealthComponent
from app.contracts.knowledge import RetrievedChunk
from app.contracts.queries import MaterialContext
from app.rag.ingestion import IngestionPipeline
from app.rag.keyword_store import KeywordSearchHit
from app.rag.providers import ExtractiveAnswerProvider
from app.rag.retrieval import RetrievalService
from app.rag.service import KnowledgeService


def _chunk_to_retrieved(chunk) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        title=chunk.title,
        source_type="manual_note",
        citation_text="self-authored demo note (placeholder; replace with licensed corpus)",
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section_title=chunk.section_title,
        text=chunk.text,
        material_tags=list(chunk.material_tags),
        retrieval_score=0.0,
    )


class _InMemoryKeywordStore:
    """Tiny substring-match keyword store used only for offline tests."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def health(self) -> HealthComponent:
        return HealthComponent(status="healthy", detail="in-memory offline keyword store")

    def search(self, query: str, *, limit: int) -> list[KeywordSearchHit]:
        tokens = [token for token in query.casefold().split() if token]
        ranked = sorted(
            self._chunks,
            key=lambda chunk: -sum(token in chunk.text.casefold() for token in tokens),
        )
        hits = [
            KeywordSearchHit(chunk=chunk, rank=index + 1)
            for index, chunk in enumerate(ranked)
            if any(token in chunk.text.casefold() for token in tokens)
        ]
        return hits[:limit]

    def get_many(self, chunk_ids: list[str]) -> dict[str, RetrievedChunk]:
        wanted = set(chunk_ids)
        return {chunk.chunk_id: chunk for chunk in self._chunks if chunk.chunk_id in wanted}


def _prepare_chunks(path: Path, *, doc_id: str, title: str, material_tags: tuple[str, ...]):
    prepared = IngestionPipeline().prepare(
        path, doc_id=doc_id, title=title, material_tags=material_tags
    )
    return [_chunk_to_retrieved(chunk) for chunk in prepared.chunks]


def _service_for(chunks: list[RetrievedChunk]) -> KnowledgeService:
    store = _InMemoryKeywordStore(chunks)
    retrieval = RetrievalService(keyword_store=store)
    return KnowledgeService(retrieval=retrieval, provider=ExtractiveAnswerProvider())


_SILICA_TEXT = (
    "# 二氧化硅纳米颗粒表征\n\n"
    "二氧化硅（SiO2）纳米颗粒常用 Stöber 法合成，通过控制氨水与正硅酸乙酯比例调节粒径。\n\n"
    "透射电子显微镜（TEM）用于测量单颗粒直径，扫描电镜（SEM）则适合统计大批量颗粒分布。\n\n"
    "粒径分布常用 D10、D50、D90 描述，其中 D50 表示累计体积占比 50% 对应的粒径。\n"
)

_TITANIA_TEXT = (
    "# 二氧化钛纳米颗粒表征\n\n"
    "二氧化钛（TiO2）纳米颗粒常用于光催化，可通过水热法控制其晶型与粒径。\n\n"
    "扫描电镜（SEM）适合统计大批量 TiO2 颗粒的粒径分布。\n"
)


@pytest.fixture
def silica_service(tmp_path: Path) -> KnowledgeService:
    doc = tmp_path / "silica.md"
    doc.write_text(_SILICA_TEXT, encoding="utf-8")
    chunks = _prepare_chunks(
        doc, doc_id="doc-silica", title="SiO2 表征笔记", material_tags=("SiO2",)
    )
    return _service_for(chunks)


def test_relevant_query_returns_cited_extractive_answer(silica_service: KnowledgeService) -> None:
    answer = silica_service.answer("二氧化硅 粒径 分布 D50 测量")

    assert answer.outcome_code == "OK"
    assert answer.confidence == "medium"
    assert answer.citations, "answer must carry at least one citation"
    assert "[C1]" in answer.answer
    assert any("不代表完整文献综述" in limit for limit in answer.limitations)
    assert answer.citations[0].doc_id == "doc-silica"


def test_unrelated_query_yields_insufficient_evidence(silica_service: KnowledgeService) -> None:
    answer = silica_service.answer("如何合成聚苯乙烯微球？")

    assert answer.outcome_code == "INSUFFICIENT_EVIDENCE"
    assert answer.confidence == "low"
    assert answer.citations == ()


def test_material_context_excludes_other_materials(tmp_path: Path) -> None:
    silica = tmp_path / "silica.md"
    silica.write_text(_SILICA_TEXT, encoding="utf-8")
    titania = tmp_path / "titania.md"
    titania.write_text(_TITANIA_TEXT, encoding="utf-8")
    chunks = [
        *_prepare_chunks(
            silica, doc_id="doc-silica", title="SiO2 表征笔记", material_tags=("SiO2",)
        ),
        *_prepare_chunks(
            titania, doc_id="doc-titania", title="TiO2 表征笔记", material_tags=("TiO2",)
        ),
    ]
    service = _service_for(chunks)

    material = MaterialContext(formula="TiO2")
    answer = service.answer("粒径 分布 统计 SEM", material_context=material)

    assert answer.outcome_code == "OK"
    returned_docs = {citation.doc_id for citation in answer.citations}
    assert returned_docs == {"doc-titania"}


def test_no_material_context_keeps_all_retrieved_chunks(tmp_path: Path) -> None:
    silica = tmp_path / "silica.md"
    silica.write_text(_SILICA_TEXT, encoding="utf-8")
    titania = tmp_path / "titania.md"
    titania.write_text(_TITANIA_TEXT, encoding="utf-8")
    chunks = [
        *_prepare_chunks(
            silica, doc_id="doc-silica", title="SiO2 表征笔记", material_tags=("SiO2",)
        ),
        *_prepare_chunks(
            titania, doc_id="doc-titania", title="TiO2 表征笔记", material_tags=("TiO2",)
        ),
    ]
    service = _service_for(chunks)

    answer = service.answer("粒径 分布 统计 SEM")

    assert answer.outcome_code == "OK"
    returned_docs = {citation.doc_id for citation in answer.citations}
    assert returned_docs == {"doc-silica", "doc-titania"}


def test_ingestion_produces_expected_chunks(tmp_path: Path) -> None:
    doc = tmp_path / "silica.md"
    doc.write_text(_SILICA_TEXT, encoding="utf-8")
    chunks = _prepare_chunks(
        doc, doc_id="doc-silica", title="SiO2 表征笔记", material_tags=("SiO2",)
    )

    assert chunks, "document must be chunked"
    assert all("SiO2" in chunk.material_tags for chunk in chunks)
    assert all(chunk.text.strip() for chunk in chunks)

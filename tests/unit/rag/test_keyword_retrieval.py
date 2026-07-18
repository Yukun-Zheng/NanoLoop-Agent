"""Tests for SQLite FTS and normalized hybrid retrieval."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.contracts.common import HealthComponent
from app.contracts.knowledge import RetrievalRequest, RetrievedChunk
from app.rag.embeddings import CallableEmbeddingProvider
from app.rag.keyword_store import KeywordSearchHit, SQLiteFTS5KeywordStore
from app.rag.retrieval import RetrievalService
from app.rag.vector_store import InMemoryVectorStore


def _knowledge_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE knowledge_documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            source_type TEXT NOT NULL,
            citation_text TEXT NOT NULL,
            status TEXT NOT NULL
        );
        CREATE TABLE knowledge_chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            page_start INTEGER,
            page_end INTEGER,
            section_title TEXT,
            text TEXT NOT NULL,
            material_tags_json TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE knowledge_chunks_fts USING fts5(
            chunk_id UNINDEXED,
            text,
            section_title,
            material_tags
        );
        """
    )
    return connection


def _insert_chunk(
    connection: sqlite3.Connection,
    *,
    chunk_id: str,
    doc_id: str,
    title: str,
    text: str,
    tags: list[str],
    status: str = "ready",
) -> None:
    connection.execute(
        """INSERT OR IGNORE INTO knowledge_documents(
            doc_id, title, source_type, citation_text, status
        ) VALUES (?, ?, 'paper', 'Fixture citation, 2026.', ?)""",
        (doc_id, title, status),
    )
    tags_json = json.dumps(tags, ensure_ascii=False)
    connection.execute(
        """
        INSERT INTO knowledge_chunks(
            chunk_id, doc_id, page_start, page_end, section_title, text, material_tags_json
        ) VALUES (?, ?, 2, 2, '用途', ?, ?)
        """,
        (chunk_id, doc_id, text, tags_json),
    )
    connection.execute(
        "INSERT INTO knowledge_chunks_fts VALUES (?, ?, '用途', ?)",
        (chunk_id, text, " ".join(tags)),
    )
    connection.commit()


def test_sqlite_fts_health_search_and_chunk_lookup(tmp_path: Path) -> None:
    database = tmp_path / "knowledge.db"
    connection = _knowledge_database(database)
    _insert_chunk(
        connection,
        chunk_id="chunk_1",
        doc_id="doc_1",
        title="催化研究",
        text="SrNi 催化 应用 性质",
        tags=["SrNi"],
    )
    connection.close()
    store = SQLiteFTS5KeywordStore(database)

    assert store.health().status == "healthy"
    hits = store.search("催化？", limit=10)
    assert [hit.chunk.chunk_id for hit in hits] == ["chunk_1"]
    assert hits[0].chunk.page_start == 2
    assert hits[0].chunk.material_tags == ["SrNi"]
    assert hits[0].chunk.source_type == "paper"
    assert hits[0].chunk.citation_text == "Fixture citation, 2026."
    assert store.get_many(["chunk_1"])["chunk_1"].title == "催化研究"


def test_sqlite_fts_reports_empty_and_missing_indexes(tmp_path: Path) -> None:
    empty_database = tmp_path / "empty.db"
    connection = _knowledge_database(empty_database)
    connection.close()

    assert SQLiteFTS5KeywordStore(empty_database).health().status == "degraded"
    assert SQLiteFTS5KeywordStore(tmp_path / "missing.db").health().status == "unavailable"


def test_sqlite_fts_excludes_disabled_documents_from_all_retrieval_paths(
    tmp_path: Path,
) -> None:
    database = tmp_path / "disabled.db"
    connection = _knowledge_database(database)
    _insert_chunk(
        connection,
        chunk_id="chunk_disabled",
        doc_id="doc_disabled",
        title="Disabled evidence",
        text="catalyst evidence that must remain hidden",
        tags=["TiO2"],
        status="disabled",
    )
    connection.close()
    store = SQLiteFTS5KeywordStore(database)

    assert store.health().status == "degraded"
    assert store.search("catalyst", limit=10) == []
    assert store.get_many(["chunk_disabled"]) == {}


class FakeKeywordStore:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def health(self) -> HealthComponent:
        return HealthComponent(status="healthy", detail="fixture")

    def search(self, query: str, *, limit: int) -> list[KeywordSearchHit]:
        del query
        return [
            KeywordSearchHit(chunk=chunk, rank=index)
            for index, chunk in enumerate(self.chunks[:limit], start=1)
        ]

    def get_many(self, chunk_ids: list[str]) -> dict[str, RetrievedChunk]:
        return {chunk.chunk_id: chunk for chunk in self.chunks if chunk.chunk_id in chunk_ids}


def _chunk(chunk_id: str, *, tags: list[str]) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        title=chunk_id,
        page_start=1,
        page_end=1,
        text=f"evidence for {chunk_id}",
        material_tags=tags,
        retrieval_score=0,
    )


def test_keyword_only_rrf_is_normalized_before_documented_threshold() -> None:
    store = FakeKeywordStore([_chunk("c1", tags=[]), _chunk("c2", tags=[])])
    retrieval = RetrievalService(store)

    report = retrieval.retrieve_with_report(RetrievalRequest(query="evidence", min_score=0.2))

    assert report.health.status == "degraded"
    assert [chunk.chunk_id for chunk in report.chunks] == ["c1", "c2"]
    assert report.chunks[0].retrieval_score == 1.0
    assert all(0.2 <= chunk.retrieval_score <= 1 for chunk in report.chunks)
    assert any("keyword retrieval only" in warning for warning in report.warnings)


def test_hybrid_rrf_fuses_ranks_and_drops_below_threshold() -> None:
    chunks = [_chunk("c1", tags=[]), _chunk("c2", tags=[])]
    store = FakeKeywordStore(chunks)
    embedding = CallableEmbeddingProvider(lambda texts: [[1.0, 0.0] for _ in texts])
    vectors = InMemoryVectorStore({"c1": [0.0, 1.0], "c2": [1.0, 0.0]})
    retrieval = RetrievalService(
        store,
        embedding_provider=embedding,
        vector_store=vectors,
    )

    chunks = retrieval.retrieve(RetrievalRequest(query="evidence", min_score=0.6))

    assert [chunk.chunk_id for chunk in chunks] == ["c2"]
    assert chunks[0].retrieval_score > 0.9


def test_material_filter_never_falls_back_without_a_matching_tag() -> None:
    store = FakeKeywordStore(
        [
            _chunk("match", tags=["Sr-Ni"]),
            _chunk("generic", tags=[]),
            _chunk("other", tags=["YCu"]),
        ]
    )
    retrieval = RetrievalService(store)

    matching = retrieval.retrieve_with_report(
        RetrievalRequest(query="evidence", material_aliases=["Sr_Ni"])
    )
    assert [chunk.chunk_id for chunk in matching.chunks] == ["match", "generic"]
    assert not matching.material_filter_fallback

    generic = retrieval.retrieve_with_report(
        RetrievalRequest(query="evidence", material_aliases=["unknown"])
    )
    assert not generic.chunks
    assert not generic.material_filter_fallback
    assert any("other-material evidence" in warning for warning in generic.warnings)

    tagged_only = RetrievalService(
        FakeKeywordStore([_chunk("other", tags=["YCu"])])
    ).retrieve_with_report(
        RetrievalRequest(query="evidence", material_aliases=["unknown"])
    )
    assert not tagged_only.chunks
    assert not tagged_only.material_filter_fallback

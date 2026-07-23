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


def test_sqlite_keyword_search_falls_back_for_unsegmented_chinese(
    tmp_path: Path,
) -> None:
    database = tmp_path / "knowledge.db"
    connection = _knowledge_database(database)
    _insert_chunk(
        connection,
        chunk_id="chunk_relevant",
        doc_id="doc_relevant",
        title="析出稳定性",
        text="析出颗粒嵌入晶格形成锚定界面，因此可能具有更好的抗团聚稳定性。",
        tags=["钙钛矿氧化物"],
    )
    _insert_chunk(
        connection,
        chunk_id="chunk_other",
        doc_id="doc_other",
        title="其他主题",
        text="氧化物样品需要记录烧结温度与保温时间。",
        tags=["钙钛矿氧化物"],
    )
    connection.close()

    hits = SQLiteFTS5KeywordStore(database).search(
        "析出颗粒与外部沉积颗粒相比有什么潜在优势？",
        limit=5,
    )

    assert hits
    assert hits[0].chunk.chunk_id == "chunk_relevant"


def test_sqlite_keyword_search_keeps_cjk_ranking_when_ascii_fts_already_matches(
    tmp_path: Path,
) -> None:
    database = tmp_path / "knowledge.db"
    connection = _knowledge_database(database)
    _insert_chunk(
        connection,
        chunk_id="chunk_broad_ascii_match",
        doc_id="doc_broad",
        title="SEM 通用说明",
        text="SEM 图像需要保存原始文件。",
        tags=["钙钛矿氧化物"],
    )
    _insert_chunk(
        connection,
        chunk_id="chunk_relevant",
        doc_id="doc_relevant",
        title="应重点统计的图像指标",
        text="颗粒形貌指标包括单位面积数密度、平均粒径和粒径分布。",
        tags=["钙钛矿氧化物"],
    )
    connection.close()

    hits = SQLiteFTS5KeywordStore(database).search(
        "应该从 SEM 掩码计算哪些颗粒形貌指标？",
        limit=5,
    )

    assert [hit.chunk.chunk_id for hit in hits[:2]] == [
        "chunk_relevant",
        "chunk_broad_ascii_match",
    ]


def test_sqlite_cjk_scoring_happens_before_limit_not_chunk_id_order(
    tmp_path: Path,
) -> None:
    database = tmp_path / "knowledge.db"
    connection = _knowledge_database(database)
    for index in range(8):
        _insert_chunk(
            connection,
            chunk_id=f"a_weak_{index}",
            doc_id=f"doc_weak_{index}",
            title="一般颗粒说明",
            text="颗粒需要结合实验上下文解释。",
            tags=["钙钛矿氧化物"],
        )
    _insert_chunk(
        connection,
        chunk_id="z_exact",
        doc_id="doc_exact",
        title="颗粒形貌指标",
        text="颗粒形貌指标包括单位面积数密度、平均粒径和粒径分布。",
        tags=["钙钛矿氧化物"],
    )
    connection.close()

    hits = SQLiteFTS5KeywordStore(database).search(
        "颗粒形貌指标",
        limit=1,
    )

    assert [hit.chunk.chunk_id for hit in hits] == ["z_exact"]


def test_sqlite_mixed_query_preserves_specific_ascii_fts_match(
    tmp_path: Path,
) -> None:
    database = tmp_path / "knowledge.db"
    connection = _knowledge_database(database)
    _insert_chunk(
        connection,
        chunk_id="chunk_generic_chinese",
        doc_id="doc_generic",
        title="材料性质",
        text="材料性质需要结合实验条件解释。",
        tags=["氧化物"],
    )
    _insert_chunk(
        connection,
        chunk_id="chunk_lani",
        doc_id="doc_lani",
        title="LaNi",
        text="LaNi 是项目样品标签。",
        tags=["LaNi"],
    )
    connection.close()

    hits = SQLiteFTS5KeywordStore(database).search(
        "LaNi 材料性质",
        limit=1,
    )

    assert [hit.chunk.chunk_id for hit in hits] == ["chunk_lani"]


def test_sqlite_fts_health_detects_ready_chunk_parity_mismatch(tmp_path: Path) -> None:
    database = tmp_path / "knowledge.db"
    connection = _knowledge_database(database)
    connection.execute(
        """
        INSERT INTO knowledge_documents(
            doc_id, title, source_type, citation_text, status
        ) VALUES ('doc_1', 'title', 'paper', 'citation', 'ready')
        """
    )
    connection.execute(
        """
        INSERT INTO knowledge_chunks(
            chunk_id, doc_id, page_start, page_end, section_title, text,
            material_tags_json
        ) VALUES ('chunk_1', 'doc_1', 1, 1, 'section', '未索引文本', '[]')
        """
    )
    connection.commit()
    connection.close()

    health = SQLiteFTS5KeywordStore(database).health()

    assert health.status == "degraded"
    assert "parity mismatch" in (health.detail or "")


def test_sqlite_fts_reports_empty_and_missing_indexes(tmp_path: Path) -> None:
    empty_database = tmp_path / "empty.db"
    connection = _knowledge_database(empty_database)
    connection.close()

    assert SQLiteFTS5KeywordStore(empty_database).health().status == "degraded"
    missing = tmp_path / "missing.db"
    assert SQLiteFTS5KeywordStore(missing).health().status == "unavailable"
    assert not missing.exists()


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


def test_vector_only_retrieval_applies_raw_cosine_threshold_before_rrf() -> None:
    chunk = _chunk("weak", tags=[])
    store = FakeKeywordStore([])
    embedding = CallableEmbeddingProvider(
        lambda texts: [[1.0, 0.0] for _ in texts],
        model_fingerprint="a" * 64,
    )
    vectors = InMemoryVectorStore({"weak": [0.001, 1.0]})
    retrieval = RetrievalService(
        store,
        embedding_provider=embedding,
        vector_store=vectors,
        chunk_source=FakeKeywordStore([chunk]),
    )

    report = retrieval.retrieve_with_report(
        RetrievalRequest(query="unrelated", min_score=0.99)
    )

    assert report.chunks == ()
    assert any("minimum cosine similarity" in warning for warning in report.warnings)


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

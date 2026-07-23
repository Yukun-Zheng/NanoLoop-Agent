"""SQLite FTS5 keyword retrieval over the frozen knowledge tables."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.contracts.common import HealthComponent
from app.contracts.knowledge import RetrievedChunk

_QUERY_TOKEN = re.compile(r"[^\W_]+(?:[-.][^\W_]+)*", flags=re.UNICODE)
_HAN_SEQUENCE = re.compile(r"[\u3400-\u9fff]+")
_MAX_CJK_TERMS = 96
_CJK_QUERY_STOP_TERMS = frozenset(
    {
        "什么",
        "为什么",
        "怎么",
        "怎样",
        "哪些",
        "是否",
        "能否",
        "这个",
        "这种",
        "当前",
        "我们",
        "已有",
        "根据",
        "可以",
        "可能",
        "一定",
        "直接",
        "较高",
        "有关",
    }
)


class KeywordStoreUnavailableError(RuntimeError):
    """Raised when the migrated FTS index cannot be queried."""


@dataclass(frozen=True, slots=True)
class KeywordSearchHit:
    chunk: RetrievedChunk
    rank: int


class ChunkSource(Protocol):
    def get_many(self, chunk_ids: list[str]) -> dict[str, RetrievedChunk]: ...


class KeywordStore(ChunkSource, Protocol):
    def health(self) -> HealthComponent: ...

    def search(self, query: str, *, limit: int) -> list[KeywordSearchHit]: ...


class UnavailableKeywordStore:
    def __init__(self, detail: str = "SQLite keyword index is not configured") -> None:
        self.detail = detail

    def health(self) -> HealthComponent:
        return HealthComponent(status="unavailable", detail=self.detail)

    def search(self, query: str, *, limit: int) -> list[KeywordSearchHit]:
        del query, limit
        raise KeywordStoreUnavailableError(self.detail)

    def get_many(self, chunk_ids: list[str]) -> dict[str, RetrievedChunk]:
        del chunk_ids
        return {}


class SQLiteFTS5KeywordStore:
    """Read keyword candidates from SQLite's migrated FTS5 index."""

    def __init__(self, database: str | Path | sqlite3.Connection) -> None:
        self._database = database

    def health(self) -> HealthComponent:
        try:
            with self._connect() as connection:
                required = {
                    "knowledge_documents",
                    "knowledge_chunks",
                    "knowledge_chunks_fts",
                }
                placeholders = ",".join("?" for _ in required)
                rows = connection.execute(
                    f"SELECT name FROM sqlite_master WHERE name IN ({placeholders})",
                    tuple(sorted(required)),
                ).fetchall()
                present = {str(row[0]) for row in rows}
                if present != required:
                    missing = ", ".join(sorted(required - present))
                    return HealthComponent(
                        status="unavailable",
                        detail=f"missing SQLite knowledge tables: {missing}",
                    )
                indexed_count = int(
                    connection.execute(
                        """
                        SELECT count(*)
                        FROM knowledge_chunks_fts
                        JOIN knowledge_chunks AS c
                          ON c.chunk_id = knowledge_chunks_fts.chunk_id
                        JOIN knowledge_documents AS d ON d.doc_id = c.doc_id
                        WHERE d.status = 'ready'
                        """
                    ).fetchone()[0]
                )
                ready_count = int(
                    connection.execute(
                        """
                        SELECT count(*)
                        FROM knowledge_chunks AS c
                        JOIN knowledge_documents AS d ON d.doc_id = c.doc_id
                        WHERE d.status = 'ready'
                        """
                    ).fetchone()[0]
                )
        except sqlite3.Error as error:
            return HealthComponent(status="unavailable", detail=f"SQLite FTS error: {error}")
        if ready_count == 0:
            return HealthComponent(status="degraded", detail="knowledge index is empty")
        if indexed_count != ready_count:
            return HealthComponent(
                status="degraded",
                detail=(
                    "SQLite FTS parity mismatch: "
                    f"{indexed_count} indexed, {ready_count} ready chunks"
                ),
            )
        return HealthComponent(status="healthy", detail=f"{indexed_count} indexed chunks")

    def search(self, query: str, *, limit: int) -> list[KeywordSearchHit]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        match_query = self._safe_match_query(query)
        if not match_query:
            return []
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT
                        c.chunk_id,
                        c.doc_id,
                        d.title,
                        d.source_type,
                        d.citation_text,
                        c.page_start,
                        c.page_end,
                        c.section_title,
                        c.text,
                        c.material_tags_json,
                        bm25(knowledge_chunks_fts) AS lexical_rank
                    FROM knowledge_chunks_fts
                    JOIN knowledge_chunks AS c
                      ON c.chunk_id = knowledge_chunks_fts.chunk_id
                    JOIN knowledge_documents AS d ON d.doc_id = c.doc_id
                    WHERE knowledge_chunks_fts MATCH ? AND d.status = 'ready'
                    ORDER BY lexical_rank ASC, c.chunk_id ASC
                    LIMIT ?
                    """,
                    (match_query, limit),
                ).fetchall()
                if _HAN_SEQUENCE.search(query):
                    # A mixed Chinese/ASCII question can produce a broad FTS hit
                    # (for example, every chunk containing ``SEM``) even though
                    # unicode61 cannot segment the meaningful Han phrase. Keep
                    # the bounded CJK scorer active whenever Han text is present,
                    # then rank the union by actual query-term coverage.
                    cjk_rows = self._search_cjk_rows(connection, query, limit=limit)
                    rows = self._merge_ranked_rows(
                        query,
                        cjk_rows,
                        rows,
                        limit=limit,
                    )
        except sqlite3.Error as error:
            raise KeywordStoreUnavailableError(f"SQLite FTS search failed: {error}") from error
        return [
            KeywordSearchHit(chunk=self._row_to_chunk(row), rank=index)
            for index, row in enumerate(rows, start=1)
        ]

    @classmethod
    def _merge_ranked_rows(
        cls,
        query: str,
        cjk_rows: list[sqlite3.Row],
        fts_rows: list[sqlite3.Row],
        *,
        limit: int,
    ) -> list[sqlite3.Row]:
        cjk_ranks = {str(row[0]): rank for rank, row in enumerate(cjk_rows, start=1)}
        fts_ranks = {str(row[0]): rank for rank, row in enumerate(fts_rows, start=1)}
        rows_by_id = {str(row[0]): row for row in (*cjk_rows, *fts_rows)}
        cjk_terms = cls._cjk_terms(query)
        ascii_terms = [
            token
            for token in _QUERY_TOKEN.findall(query.casefold())
            if not _HAN_SEQUENCE.search(token)
        ]

        def ranking_key(row: sqlite3.Row) -> tuple[int, int, int, int, str]:
            chunk_id = str(row[0])
            haystack = " ".join(
                str(value or "") for value in (row[7], row[8], row[9])
            ).casefold()
            coverage = sum(
                len(term) ** 2 for term in cjk_terms if term in haystack
            ) + sum(
                4 * len(term) ** 2 for term in ascii_terms if term in haystack
            )
            channel_count = int(chunk_id in cjk_ranks) + int(chunk_id in fts_ranks)
            return (
                -coverage,
                -channel_count,
                cjk_ranks.get(chunk_id, len(cjk_rows) + 1),
                fts_ranks.get(chunk_id, len(fts_rows) + 1),
                chunk_id,
            )

        return sorted(rows_by_id.values(), key=ranking_key)[:limit]

    def get_many(self, chunk_ids: list[str]) -> dict[str, RetrievedChunk]:
        unique_ids = list(dict.fromkeys(chunk_ids))
        if not unique_ids:
            return {}
        placeholders = ",".join("?" for _ in unique_ids)
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    f"""
                    SELECT
                        c.chunk_id,
                        c.doc_id,
                        d.title,
                        d.source_type,
                        d.citation_text,
                        c.page_start,
                        c.page_end,
                        c.section_title,
                        c.text,
                        c.material_tags_json,
                        0.0 AS lexical_rank
                    FROM knowledge_chunks AS c
                    JOIN knowledge_documents AS d ON d.doc_id = c.doc_id
                    WHERE c.chunk_id IN ({placeholders}) AND d.status = 'ready'
                    """,
                    tuple(unique_ids),
                ).fetchall()
        except sqlite3.Error as error:
            raise KeywordStoreUnavailableError(f"SQLite chunk lookup failed: {error}") from error
        return {str(row[0]): self._row_to_chunk(row) for row in rows}

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if isinstance(self._database, sqlite3.Connection):
            previous_factory = self._database.row_factory
            self._database.row_factory = sqlite3.Row
            try:
                yield self._database
            finally:
                self._database.row_factory = previous_factory
            return

        path = Path(self._database).expanduser().resolve()
        if not path.is_file():
            raise sqlite3.OperationalError(f"knowledge database does not exist: {path}")
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _safe_match_query(query: str) -> str:
        tokens = _QUERY_TOKEN.findall(query.casefold())
        unique = list(dict.fromkeys(token for token in tokens if token))
        return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in unique)

    @classmethod
    def _search_cjk_rows(
        cls,
        connection: sqlite3.Connection,
        query: str,
        *,
        limit: int,
    ) -> list[sqlite3.Row]:
        """Bounded substring fallback for CJK text indexed by ``unicode61``.

        SQLite's default FTS tokenizer commonly treats a continuous Han sentence
        as one token, so a different unsegmented question has no lexical match.
        The fallback remains read-only and parameterized: it selects a bounded
        candidate set by 2-6 character n-gram overlap and ranks those candidates
        deterministically in Python.
        """

        terms = cls._cjk_terms(query)
        if not terms:
            return []
        score_clauses = " + ".join(
            "(CASE WHEN instr(c.text, ?) > 0 "
            "OR instr(coalesce(c.section_title, ''), ?) > 0 "
            "OR instr(c.material_tags_json, ?) > 0 THEN ? ELSE 0 END)"
            for _ in terms
        )
        score_parameters = tuple(
            value
            for term in terms
            for value in (term, term, term, len(term) ** 2)
        )
        predicates = " OR ".join(
            "(instr(c.text, ?) > 0 OR instr(coalesce(c.section_title, ''), ?) > 0 "
            "OR instr(c.material_tags_json, ?) > 0)"
            for _ in terms
        )
        parameters = tuple(term for term in terms for _ in range(3))
        return list(
            connection.execute(
            f"""
            SELECT
                c.chunk_id,
                c.doc_id,
                d.title,
                d.source_type,
                d.citation_text,
                c.page_start,
                c.page_end,
                c.section_title,
                c.text,
                c.material_tags_json,
                ({score_clauses}) AS lexical_rank
            FROM knowledge_chunks AS c
            JOIN knowledge_documents AS d ON d.doc_id = c.doc_id
            WHERE d.status = 'ready' AND ({predicates})
            ORDER BY lexical_rank DESC, c.chunk_id ASC
            LIMIT ?
            """,
                (*score_parameters, *parameters, limit),
            ).fetchall()
        )

    @staticmethod
    def _cjk_terms(query: str) -> list[str]:
        terms: list[str] = []
        for sequence in _HAN_SEQUENCE.findall(query.casefold()):
            for width in (2, 3, 6, 5, 4):
                if len(sequence) < width:
                    continue
                for start in range(len(sequence) - width + 1):
                    term = sequence[start : start + width]
                    if term in _CJK_QUERY_STOP_TERMS or term in terms:
                        continue
                    terms.append(term)
                    if len(terms) >= _MAX_CJK_TERMS:
                        return terms
        return terms

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> RetrievedChunk:
        raw_tags = row[9]
        try:
            parsed_tags = json.loads(raw_tags) if isinstance(raw_tags, str) else raw_tags
        except json.JSONDecodeError:
            parsed_tags = []
        tags = [str(tag) for tag in parsed_tags] if isinstance(parsed_tags, list) else []
        return RetrievedChunk(
            chunk_id=str(row[0]),
            doc_id=str(row[1]),
            title=str(row[2]),
            source_type=str(row[3]),
            citation_text=str(row[4]),
            page_start=int(row[5]) if row[5] is not None else None,
            page_end=int(row[6]) if row[6] is not None else None,
            section_title=str(row[7]) if row[7] is not None else None,
            text=str(row[8]),
            material_tags=tags,
            retrieval_score=0.0,
        )

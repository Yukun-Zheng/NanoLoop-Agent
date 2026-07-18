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
                count = int(
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
        except sqlite3.Error as error:
            return HealthComponent(status="unavailable", detail=f"SQLite FTS error: {error}")
        if count == 0:
            return HealthComponent(status="degraded", detail="knowledge index is empty")
        return HealthComponent(status="healthy", detail=f"{count} indexed chunks")

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
        except sqlite3.Error as error:
            raise KeywordStoreUnavailableError(f"SQLite FTS search failed: {error}") from error
        return [
            KeywordSearchHit(chunk=self._row_to_chunk(row), rank=index)
            for index, row in enumerate(rows, start=1)
        ]

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

        connection = sqlite3.connect(str(self._database))
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

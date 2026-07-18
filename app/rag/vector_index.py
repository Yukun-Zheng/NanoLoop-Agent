"""Full-corpus vector-index rebuilding from the authoritative SQL database."""

from __future__ import annotations

from threading import RLock
from typing import Protocol

from sqlalchemy import func, select

from app.contracts.enums import KnowledgeDocumentStatus
from app.db.models import KnowledgeChunk, KnowledgeDocument
from app.db.session import Database
from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import (
    PersistentFaissVectorStore,
    VectorIndexRecord,
    VectorPublishResult,
)

DEFAULT_EMBEDDING_INDEX_BATCH_SIZE = 128
DEFAULT_MAX_VECTOR_INDEX_CHUNKS = 100_000


class VectorIndexCapacityError(RuntimeError):
    """Raised before a rebuild can allocate an unbounded corpus projection."""

    def __init__(self, *, limit: int, observed: int) -> None:
        super().__init__(
            f"ready knowledge corpus has {observed} chunks; vector index limit is {limit}"
        )
        self.limit = limit
        self.observed = observed


class VectorIndexPublisher(Protocol):
    def rebuild(self) -> VectorPublishResult: ...


class DatabaseVectorIndexPublisher:
    """Embed every ready chunk and publish one internally consistent generation."""

    def __init__(
        self,
        database: Database,
        embedding_provider: EmbeddingProvider,
        vector_store: PersistentFaissVectorStore,
        *,
        batch_size: int = DEFAULT_EMBEDDING_INDEX_BATCH_SIZE,
        max_chunks: int = DEFAULT_MAX_VECTOR_INDEX_CHUNKS,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("vector embedding batch_size must be positive")
        if max_chunks <= 0:
            raise ValueError("vector index max_chunks must be positive")
        self.database = database
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.batch_size = batch_size
        self.max_chunks = max_chunks
        self._lock = RLock()

    def rebuild(self) -> VectorPublishResult:
        with self._lock:
            return self._rebuild_locked()

    def _rebuild_locked(self) -> VectorPublishResult:
        with self.database.session_factory() as session:
            ready_filter = KnowledgeDocument.status == KnowledgeDocumentStatus.READY.value
            observed = int(
                session.scalar(
                    select(func.count(KnowledgeChunk.chunk_id))
                    .join(
                        KnowledgeDocument,
                        KnowledgeDocument.doc_id == KnowledgeChunk.doc_id,
                    )
                    .where(ready_filter)
                )
                or 0
            )
            if observed > self.max_chunks:
                raise VectorIndexCapacityError(limit=self.max_chunks, observed=observed)
            rows = session.execute(
                select(KnowledgeChunk.chunk_id, KnowledgeChunk.text)
                .join(KnowledgeDocument, KnowledgeDocument.doc_id == KnowledgeChunk.doc_id)
                .where(ready_filter)
                .order_by(KnowledgeChunk.chunk_id)
                .limit(self.max_chunks + 1)
            ).all()
        if not rows:
            return self.vector_store.publish_empty()
        if len(rows) > self.max_chunks:
            raise VectorIndexCapacityError(limit=self.max_chunks, observed=len(rows))
        records: list[VectorIndexRecord] = []
        for start in range(0, len(rows), self.batch_size):
            batch = rows[start : start + self.batch_size]
            texts = [str(text_value) for _, text_value in batch]
            vectors = self.embedding_provider.embed_documents(texts)
            if len(vectors) != len(batch):
                raise ValueError(
                    "embedding provider returned an unexpected vector-index batch size"
                )
            records.extend(
                VectorIndexRecord(
                    chunk_id=str(chunk_id),
                    vector=vector,
                    content_sha256=self.vector_store.content_sha256(str(text_value)),
                )
                for (chunk_id, text_value), vector in zip(batch, vectors, strict=True)
            )
        return self.vector_store.publish(records)


__all__ = [
    "DatabaseVectorIndexPublisher",
    "VectorIndexCapacityError",
    "VectorIndexPublisher",
]

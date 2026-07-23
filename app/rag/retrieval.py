"""Normalized reciprocal-rank fusion with honest lexical-only degradation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.contracts.common import HealthComponent
from app.contracts.knowledge import RetrievalRequest, RetrievedChunk
from app.rag.embeddings import EmbeddingProvider, UnavailableEmbeddingProvider
from app.rag.keyword_store import (
    ChunkSource,
    KeywordStore,
    KeywordStoreUnavailableError,
)
from app.rag.vector_store import UnavailableVectorStore, VectorStore

_RRF_K = 60
_ALIAS_SEPARATORS = re.compile(r"[\s_\-–—/·]+")


@dataclass(frozen=True, slots=True)
class RetrievalReport:
    chunks: tuple[RetrievedChunk, ...]
    health: HealthComponent
    warnings: tuple[str, ...] = ()
    material_filter_fallback: bool = False


class RetrievalService:
    """Fuse independent keyword/vector ranks without comparing raw score scales."""

    def __init__(
        self,
        keyword_store: KeywordStore,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        chunk_source: ChunkSource | None = None,
    ) -> None:
        self.keyword_store = keyword_store
        self.embedding_provider = embedding_provider or UnavailableEmbeddingProvider()
        self.vector_store = vector_store or UnavailableVectorStore()
        self.chunk_source = chunk_source or keyword_store

    def health(self) -> HealthComponent:
        keyword = self.keyword_store.health()
        embedding = self.embedding_provider.health()
        vector = self.vector_store.health()
        vector_ready = embedding.status == "healthy" and vector.status == "healthy"
        if keyword.status == "unavailable" and not vector_ready:
            return HealthComponent(
                status="unavailable",
                detail="neither keyword nor vector retrieval is available",
            )
        if keyword.status != "healthy" or not vector_ready:
            details = [
                f"keyword={keyword.status}",
                f"embedding={embedding.status}",
                f"vector={vector.status}",
            ]
            return HealthComponent(status="degraded", detail=", ".join(details))
        return HealthComponent(status="healthy", detail="keyword and vector retrieval available")

    def retrieve(self, request: RetrievalRequest) -> list[RetrievedChunk]:
        return list(self.retrieve_with_report(request).chunks)

    def retrieve_with_report(self, request: RetrievalRequest) -> RetrievalReport:
        warnings: list[str] = []
        channel_ranks: list[dict[str, int]] = []
        chunks_by_id: dict[str, RetrievedChunk] = {}

        try:
            keyword_hits = self.keyword_store.search(
                request.query,
                limit=request.candidate_k,
            )
        except KeywordStoreUnavailableError as error:
            keyword_hits = []
            warnings.append(str(error))
        if keyword_hits:
            keyword_ranks = {hit.chunk.chunk_id: hit.rank for hit in keyword_hits}
            channel_ranks.append(keyword_ranks)
            chunks_by_id.update({hit.chunk.chunk_id: hit.chunk for hit in keyword_hits})

        embedding_health = self.embedding_provider.health()
        vector_health = self.vector_store.health()
        if embedding_health.status != "unavailable" and vector_health.status == "healthy":
            try:
                query_vector = self.embedding_provider.embed_query(request.query)
                vector_hits = self.vector_store.search(query_vector, limit=request.candidate_k)
                if vector_hits:
                    vector_ranks = {
                        hit.chunk_id: rank for rank, hit in enumerate(vector_hits, start=1)
                    }
                    channel_ranks.append(vector_ranks)
                    missing_ids = [
                        chunk_id for chunk_id in vector_ranks if chunk_id not in chunks_by_id
                    ]
                    chunks_by_id.update(self.chunk_source.get_many(missing_ids))
            except Exception as error:  # provider boundary: degrade instead of fabricating
                warnings.append(f"vector retrieval unavailable for this query: {error}")
        else:
            warnings.append("vector retrieval unavailable; keyword retrieval only")

        active_channels = [ranks for ranks in channel_ranks if ranks]
        if not active_channels:
            return RetrievalReport(
                chunks=(),
                health=self.health(),
                warnings=tuple(dict.fromkeys(warnings)),
            )

        raw_scores: dict[str, float] = {}
        for ranks in active_channels:
            for chunk_id, rank in ranks.items():
                if chunk_id in chunks_by_id:
                    raw_scores[chunk_id] = raw_scores.get(chunk_id, 0.0) + 1.0 / (_RRF_K + rank)
        theoretical_max = len(active_channels) / (_RRF_K + 1)
        scored = [
            chunks_by_id[chunk_id].model_copy(
                update={"retrieval_score": min(raw_score / theoretical_max, 1.0)}
            )
            for chunk_id, raw_score in raw_scores.items()
            if raw_score / theoretical_max >= request.min_score
        ]
        scored.sort(key=lambda chunk: (-chunk.retrieval_score, chunk.chunk_id))

        filtered, fallback, material_warning = self._filter_material(
            scored,
            request.material_aliases,
        )
        if material_warning:
            warnings.append(material_warning)
        return RetrievalReport(
            chunks=tuple(filtered[: request.top_k]),
            health=self.health(),
            warnings=tuple(dict.fromkeys(warnings)),
            material_filter_fallback=fallback,
        )

    @classmethod
    def _filter_material(
        cls,
        chunks: list[RetrievedChunk],
        aliases: list[str],
    ) -> tuple[list[RetrievedChunk], bool, str | None]:
        normalized_aliases = {cls._normalize_alias(alias) for alias in aliases if alias.strip()}
        if not normalized_aliases or not chunks:
            return chunks, False, None
        matching: list[RetrievedChunk] = []
        generic: list[RetrievedChunk] = []
        for chunk in chunks:
            tags = {cls._normalize_alias(tag) for tag in chunk.material_tags if tag.strip()}
            if not tags:
                generic.append(chunk)
            elif tags & normalized_aliases:
                matching.append(chunk)
        if matching:
            return [*matching, *generic], False, None
        return (
            [],
            False,
            "no material-tag match; excluded untagged and other-material evidence",
        )

    @staticmethod
    def _normalize_alias(value: str) -> str:
        return _ALIAS_SEPARATORS.sub("", value.casefold())

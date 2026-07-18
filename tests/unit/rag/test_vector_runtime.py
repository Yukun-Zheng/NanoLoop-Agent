"""Offline embedding and atomic persistent-vector runtime tests."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.contracts.enums import KnowledgeDocumentStatus
from app.core.config import Settings
from app.db.base import Base
from app.db.models import KnowledgeChunk, KnowledgeDocument
from app.db.session import Database
from app.rag.application import KnowledgeApplicationService
from app.rag.embeddings import (
    CallableEmbeddingProvider,
    EmbeddingUnavailableError,
    InvalidEmbeddingError,
    SentenceTransformerEmbeddingProvider,
)
from app.rag.vector_index import DatabaseVectorIndexPublisher, VectorIndexCapacityError
from app.rag.vector_store import (
    PersistentFaissVectorStore,
    VectorIndexRecord,
    VectorStoreUnavailableError,
)


class _FakeSentenceModel:
    def __init__(self, outputs: list[object], *, reported_dimension: int | None = None) -> None:
        self.outputs = outputs
        self.reported_dimension = reported_dimension
        self.encode_calls: list[dict[str, object]] = []

    def get_sentence_embedding_dimension(self) -> int | None:
        return self.reported_dimension

    def encode(self, _texts: list[str], **kwargs: object) -> object:
        self.encode_calls.append(kwargs)
        return self.outputs.pop(0)


def test_sentence_transformer_provider_is_lazy_local_only_and_normalizes() -> None:
    model = _FakeSentenceModel([[[3.0, 4.0], [0.0, 2.0]]], reported_dimension=2)
    factory_calls: list[tuple[str, dict[str, object]]] = []

    def factory(model_id: str, **kwargs: object) -> object:
        factory_calls.append((model_id, kwargs))
        return model

    provider = SentenceTransformerEmbeddingProvider(
        "local/model",
        device="cpu",
        model_fingerprint="a" * 64,
        model_factory=factory,
    )

    assert factory_calls == []
    vectors = provider.embed_documents(["first", "second"])

    assert factory_calls == [
        ("local/model", {"local_files_only": True, "device": "cpu"})
    ]
    assert provider.dimension == 2
    assert np.allclose(vectors, [[0.6, 0.8], [0.0, 1.0]])
    assert model.encode_calls == [
        {
            "batch_size": 32,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }
    ]


@pytest.mark.parametrize(
    "output, match",
    [
        ([[float("nan"), 1.0]], "non-finite"),
        ([[0.0, 0.0]], "empty vector"),
        ([[]], "invalid shape"),
    ],
)
def test_sentence_transformer_provider_rejects_invalid_vectors(
    output: object,
    match: str,
) -> None:
    model = _FakeSentenceModel([output])
    provider = SentenceTransformerEmbeddingProvider(
        "local/model",
        model_fingerprint="a" * 64,
        model_factory=lambda *_args, **_kwargs: model,
    )

    with pytest.raises(InvalidEmbeddingError, match=match):
        provider.embed_query("query")


def test_sentence_transformer_provider_rejects_dimension_changes() -> None:
    model = _FakeSentenceModel([[[1.0, 0.0]], [[1.0, 0.0, 0.0]]])
    provider = SentenceTransformerEmbeddingProvider(
        "local/model",
        model_fingerprint="a" * 64,
        model_factory=lambda *_args, **_kwargs: model,
    )

    provider.embed_query("first")
    with pytest.raises(InvalidEmbeddingError, match="dimension changed"):
        provider.embed_query("second")


def test_sentence_transformer_load_failure_is_cached_as_unavailable() -> None:
    calls = 0

    def unavailable(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise ImportError("not installed")

    provider = SentenceTransformerEmbeddingProvider(
        "missing/model",
        model_fingerprint="a" * 64,
        model_factory=unavailable,
    )

    assert provider.health().status == "degraded"
    assert calls == 0
    with pytest.raises(EmbeddingUnavailableError, match="not installed"):
        provider.embed_query("query")
    assert provider.health().status == "unavailable"
    assert provider.health().status == "unavailable"
    assert calls == 1


def test_embedding_identity_requires_revision_or_hashes_local_snapshot(
    tmp_path: Path,
) -> None:
    unresolved = SentenceTransformerEmbeddingProvider("org/mutable-model")
    with pytest.raises(EmbeddingUnavailableError, match="immutable revision"):
        _ = unresolved.fingerprint

    model_dir = tmp_path / "embedding-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"dimension": 2}', encoding="utf-8")
    first = SentenceTransformerEmbeddingProvider(str(model_dir)).fingerprint
    (model_dir / "config.json").write_text('{"dimension": 3}', encoding="utf-8")
    second = SentenceTransformerEmbeddingProvider(str(model_dir)).fingerprint

    assert len(first) == len(second) == 64
    assert first != second


class _FakeFlatIndex:
    def __init__(self, dimension: int) -> None:
        self.d = dimension


class _FakeIdMapIndex:
    def __init__(self, owner: _FakeFaiss, flat: _FakeFlatIndex) -> None:
        self.owner = owner
        self.d = flat.d
        self.vectors = np.empty((0, flat.d), dtype=np.float32)
        self.id_map = np.empty((0,), dtype=np.int64)

    @property
    def ntotal(self) -> int:
        return int(self.id_map.size)

    def add_with_ids(
        self,
        vectors: np.ndarray[Any, np.dtype[np.float32]],
        vector_ids: np.ndarray[Any, np.dtype[np.int64]],
    ) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32).copy()
        self.id_map = np.asarray(vector_ids, dtype=np.int64).copy()

    def search(
        self,
        query: np.ndarray[Any, np.dtype[np.float32]],
        limit: int,
    ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, np.dtype[np.int64]]]:
        with self.owner.monitor_lock:
            self.owner.active_searches += 1
            self.owner.max_active_searches = max(
                self.owner.max_active_searches,
                self.owner.active_searches,
            )
        try:
            if self.owner.search_delay:
                time.sleep(self.owner.search_delay)
            raw_scores = np.asarray(query @ self.vectors.T, dtype=np.float32)
            order = np.argsort(-raw_scores, axis=1)[:, :limit]
            scores = np.take_along_axis(raw_scores, order, axis=1)
            ids = self.id_map[order]
            return scores, ids
        finally:
            with self.owner.monitor_lock:
                self.owner.active_searches -= 1


class _FakeFaiss:
    def __init__(self) -> None:
        self.fail_write = False
        self.search_delay = 0.0
        self.active_searches = 0
        self.max_active_searches = 0
        self.monitor_lock = threading.Lock()

    def IndexFlatIP(self, dimension: int) -> _FakeFlatIndex:
        return _FakeFlatIndex(dimension)

    def IndexIDMap2(self, flat: _FakeFlatIndex) -> _FakeIdMapIndex:
        return _FakeIdMapIndex(self, flat)

    def write_index(self, index: _FakeIdMapIndex, path: str) -> None:
        if self.fail_write:
            raise OSError("simulated write failure")
        payload = {
            "dimension": index.d,
            "vectors": index.vectors.tolist(),
            "ids": index.id_map.tolist(),
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    def read_index(self, path: str) -> _FakeIdMapIndex:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        index = _FakeIdMapIndex(self, _FakeFlatIndex(int(payload["dimension"])))
        index.add_with_ids(
            np.asarray(payload["vectors"], dtype=np.float32),
            np.asarray(payload["ids"], dtype=np.int64),
        )
        return index

    @staticmethod
    def vector_to_array(value: object) -> np.ndarray[Any, np.dtype[np.int64]]:
        return np.asarray(value, dtype=np.int64)


def _create_vector_database(path: Path, *, text_value: str = "alpha evidence") -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE knowledge_documents (
            doc_id TEXT PRIMARY KEY,
            status TEXT NOT NULL
        );
        CREATE TABLE knowledge_chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            text TEXT NOT NULL,
            vector_id INTEGER
        );
        INSERT INTO knowledge_documents(doc_id, status) VALUES ('doc_1', 'ready');
        """
    )
    connection.execute(
        "INSERT INTO knowledge_chunks(chunk_id, doc_id, text) VALUES ('chunk_1', 'doc_1', ?)",
        (text_value,),
    )
    connection.commit()
    connection.close()


def _record(
    chunk_id: str,
    vector: list[float],
    text_value: str,
) -> VectorIndexRecord:
    return VectorIndexRecord(
        chunk_id=chunk_id,
        vector=vector,
        content_sha256=PersistentFaissVectorStore.content_sha256(text_value),
    )


def test_faiss_store_publishes_atomic_manifest_and_validates_database(
    tmp_path: Path,
) -> None:
    database = tmp_path / "knowledge.db"
    _create_vector_database(database)
    backend = _FakeFaiss()
    store = PersistentFaissVectorStore(
        tmp_path / "faiss.index",
        model_id="local/model",
        model_fingerprint="a" * 64,
        database_path=database,
        faiss_loader=lambda: backend,
    )

    result = store.publish([_record("chunk_1", [1.0, 0.0], "alpha evidence")])

    assert result.vector_count == 1
    assert result.dimension == 2
    assert store.manifest_path.is_file()
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_id"] == "local/model"
    assert manifest["model_fingerprint"] == "a" * 64
    assert manifest["entries"][0]["vector_id"] == store.stable_vector_id("chunk_1")
    assert (tmp_path / manifest["index_file"]).is_file()
    assert store.health().status == "healthy"
    assert [hit.chunk_id for hit in store.search([1.0, 0.0], limit=4)] == ["chunk_1"]

    wrong_model = PersistentFaissVectorStore(
        tmp_path / "faiss.index",
        model_id="different/model",
        model_fingerprint="a" * 64,
        database_path=database,
        faiss_loader=lambda: backend,
    )
    assert wrong_model.health().status == "unavailable"
    assert "model mismatch" in (wrong_model.health().detail or "")

    wrong_fingerprint = PersistentFaissVectorStore(
        tmp_path / "faiss.index",
        model_id="local/model",
        model_fingerprint="b" * 64,
        database_path=database,
        faiss_loader=lambda: backend,
    )
    assert wrong_fingerprint.health().status == "unavailable"
    assert "fingerprint mismatch" in (wrong_fingerprint.health().detail or "")

    connection = sqlite3.connect(database)
    connection.execute(
        "UPDATE knowledge_chunks SET text = 'changed evidence' WHERE chunk_id = 'chunk_1'"
    )
    connection.commit()
    connection.close()
    assert store.health().status == "unavailable"
    assert "content mismatch" in (store.health().detail or "")


def test_failed_faiss_publish_keeps_previous_generation(tmp_path: Path) -> None:
    backend = _FakeFaiss()
    store = PersistentFaissVectorStore(
        tmp_path / "faiss.index",
        model_id="local/model",
        model_fingerprint="a" * 64,
        faiss_loader=lambda: backend,
    )
    store.publish([_record("chunk_1", [1.0, 0.0], "alpha")])
    previous_manifest = store.manifest_path.read_bytes()
    previous_hits = store.search([1.0, 0.0], limit=2)

    backend.fail_write = True
    with pytest.raises(VectorStoreUnavailableError, match="FAISS index write failed"):
        store.publish([_record("chunk_2", [0.0, 1.0], "beta")])

    assert store.manifest_path.read_bytes() == previous_manifest
    assert store.search([1.0, 0.0], limit=2) == previous_hits


def test_faiss_store_serializes_search_and_empty_publish_needs_no_dependency(
    tmp_path: Path,
) -> None:
    backend = _FakeFaiss()
    backend.search_delay = 0.05
    store = PersistentFaissVectorStore(
        tmp_path / "faiss.index",
        model_id="local/model",
        model_fingerprint="a" * 64,
        faiss_loader=lambda: backend,
    )
    store.publish([_record("chunk_1", [1.0, 0.0], "alpha")])

    threads = [
        threading.Thread(target=store.search, args=([1.0, 0.0],), kwargs={"limit": 1})
        for _ in range(3)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)
    assert backend.max_active_searches == 1

    empty = PersistentFaissVectorStore(
        tmp_path / "empty.index",
        model_id="local/model",
        model_fingerprint="a" * 64,
        faiss_loader=lambda: (_ for _ in ()).throw(ImportError("missing")),
    )
    empty.publish_empty()
    assert empty.health().status == "degraded"
    assert empty.search([1.0], limit=1) == []


def test_database_vector_publisher_builds_only_ready_chunks(tmp_path: Path) -> None:
    database_path = tmp_path / "publisher.db"
    database = Database(
        Settings(app_env="test", database_url=f"sqlite:///{database_path}")
    )
    Base.metadata.create_all(database.engine)
    with database.session() as session:
        ready = KnowledgeDocument(
            doc_id="doc_ready",
            title="Ready",
            source_type="paper",
            storage_path="ready.md",
            sha256="a" * 64,
            citation_text="Ready citation",
            status=KnowledgeDocumentStatus.READY.value,
            metadata_json={},
        )
        ready.chunks = [
            KnowledgeChunk(
                chunk_id="chunk_ready",
                doc_id=ready.doc_id,
                text="ready evidence",
                material_tags_json=[],
            )
        ]
        disabled = KnowledgeDocument(
            doc_id="doc_disabled",
            title="Disabled",
            source_type="paper",
            storage_path="disabled.md",
            sha256="b" * 64,
            citation_text="Disabled citation",
            status=KnowledgeDocumentStatus.DISABLED.value,
            metadata_json={},
        )
        disabled.chunks = [
            KnowledgeChunk(
                chunk_id="chunk_disabled",
                doc_id=disabled.doc_id,
                text="disabled evidence",
                material_tags_json=[],
            )
        ]
        session.add_all([ready, disabled])

    backend = _FakeFaiss()
    store = PersistentFaissVectorStore(
        tmp_path / "publisher.index",
        model_id="local/model",
        model_fingerprint="a" * 64,
        database_path=database_path,
        faiss_loader=lambda: backend,
    )
    embedded_batches: list[list[str]] = []

    def embed(texts: Sequence[str]) -> list[Sequence[float]]:
        embedded_batches.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    publisher = DatabaseVectorIndexPublisher(
        database,
        CallableEmbeddingProvider(embed),
        store,
    )

    result = publisher.rebuild()

    assert result.vector_count == 1
    assert embedded_batches == [["ready evidence"]]
    assert [hit.chunk_id for hit in store.search([1.0, 0.0], limit=2)] == [
        "chunk_ready"
    ]
    with database.engine.begin() as connection:
        connection.exec_driver_sql(
            """
            CREATE VIRTUAL TABLE knowledge_chunks_fts USING fts5(
                chunk_id UNINDEXED,
                text,
                section_title,
                material_tags
            )
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TRIGGER knowledge_chunks_fts_insert
            AFTER INSERT ON knowledge_chunks BEGIN
                INSERT INTO knowledge_chunks_fts(
                    chunk_id, text, section_title, material_tags
                ) VALUES (
                    new.chunk_id, new.text, new.section_title, new.material_tags_json
                );
            END
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TRIGGER knowledge_chunks_fts_delete
            AFTER DELETE ON knowledge_chunks BEGIN
                DELETE FROM knowledge_chunks_fts WHERE chunk_id = old.chunk_id;
            END
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TRIGGER knowledge_chunks_fts_update
            AFTER UPDATE ON knowledge_chunks BEGIN
                DELETE FROM knowledge_chunks_fts WHERE chunk_id = old.chunk_id;
                INSERT INTO knowledge_chunks_fts(
                    chunk_id, text, section_title, material_tags
                ) VALUES (
                    new.chunk_id, new.text, new.section_title, new.material_tags_json
                );
            END
            """
        )
        connection.exec_driver_sql(
            """
            INSERT INTO knowledge_chunks_fts(chunk_id, text, section_title, material_tags)
            VALUES ('chunk_ready', 'ready evidence', NULL, '[]')
            """
        )
    source_root = tmp_path / "sources"
    service = KnowledgeApplicationService(
        database,
        source_root,
        vector_index_publisher=publisher,
    )

    service.set_document_enabled("doc_ready", enabled=False)

    assert store.health().status == "degraded"
    assert store.search([1.0, 0.0], limit=2) == []

    service.set_document_enabled("doc_ready", enabled=True)

    assert store.health().status == "healthy"
    assert [hit.chunk_id for hit in store.search([1.0, 0.0], limit=2)] == [
        "chunk_ready"
    ]
    database.dispose()


def test_database_vector_publisher_batches_and_rejects_oversized_corpus(
    tmp_path: Path,
) -> None:
    database = Database(
        Settings(app_env="test", database_url=f"sqlite:///{tmp_path / 'bounded.db'}")
    )
    Base.metadata.create_all(database.engine)
    with database.session() as session:
        document = KnowledgeDocument(
            doc_id="doc_ready",
            title="Ready",
            source_type="paper",
            storage_path="ready.md",
            sha256="c" * 64,
            citation_text="Ready citation",
            status=KnowledgeDocumentStatus.READY.value,
            metadata_json={},
        )
        document.chunks = [
            KnowledgeChunk(
                chunk_id=f"chunk_{index}",
                doc_id=document.doc_id,
                text=f"evidence {index}",
                material_tags_json=[],
            )
            for index in range(5)
        ]
        session.add(document)

    backend = _FakeFaiss()
    store = PersistentFaissVectorStore(
        tmp_path / "bounded.index",
        model_id="local/model",
        model_fingerprint="a" * 64,
        faiss_loader=lambda: backend,
    )
    batches: list[list[str]] = []

    def embed(texts: Sequence[str]) -> list[Sequence[float]]:
        batches.append(list(texts))
        return [[1.0, float(index + 1)] for index, _ in enumerate(texts)]

    publisher = DatabaseVectorIndexPublisher(
        database,
        CallableEmbeddingProvider(embed),
        store,
        batch_size=2,
        max_chunks=5,
    )
    assert publisher.rebuild().vector_count == 5
    assert [len(batch) for batch in batches] == [2, 2, 1]

    oversized = DatabaseVectorIndexPublisher(
        database,
        CallableEmbeddingProvider(embed),
        store,
        batch_size=2,
        max_chunks=4,
    )
    batches.clear()
    with pytest.raises(VectorIndexCapacityError, match="5 chunks"):
        oversized.rebuild()
    assert batches == []
    database.dispose()

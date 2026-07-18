"""Tests for knowledge persistence, idempotency, and real FTS5 trigger maintenance."""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from sqlalchemy import delete, func, select, text

from app.contracts.enums import (
    KnowledgeDocumentStatus,
    KnowledgeSourceType,
)
from app.contracts.knowledge import IngestDocumentMetadata, ReindexRequest
from app.core.config import Settings
from app.db.base import Base
from app.db.models import KnowledgeChunk, KnowledgeDocument
from app.db.session import Database
from app.rag.application import (
    DuplicateKnowledgeDocumentError,
    KnowledgeApplicationService,
    KnowledgeDocumentNotFoundError,
    KnowledgeDocumentStateError,
    KnowledgeIndexUnavailableError,
    KnowledgeSourcePathError,
)
from app.rag.chunking import ParagraphChunker
from app.rag.ingestion import (
    DocumentExtractionUnavailableError,
    IngestionPipeline,
    PreparedDocument,
)
from app.rag.vector_index import VectorIndexPublisher
from app.rag.vector_store import VectorPublishResult


@dataclass(frozen=True, slots=True)
class KnowledgeHarness:
    database: Database
    source_root: Path

    def service(
        self,
        *,
        pipeline: IngestionPipeline | None = None,
        vector_index_publisher: VectorIndexPublisher | None = None,
    ) -> KnowledgeApplicationService:
        return KnowledgeApplicationService(
            self.database,
            self.source_root,
            pipeline=pipeline,
            vector_index_publisher=vector_index_publisher,
        )


@pytest.fixture
def knowledge_harness(tmp_path: Path) -> Iterator[KnowledgeHarness]:
    database = Database(
        Settings(
            app_env="test",
            database_url=f"sqlite:///{tmp_path / 'knowledge.db'}",
            knowledge_source_dir=tmp_path / "sources",
        )
    )
    Base.metadata.create_all(database.engine)
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
    source_root = tmp_path / "sources"
    source_root.mkdir()
    yield KnowledgeHarness(database=database, source_root=source_root)
    database.dispose()


def _metadata(
    *,
    title: str = "Sr-Ni perovskite note",
    aliases: list[str] | None = None,
) -> IngestDocumentMetadata:
    return IngestDocumentMetadata(
        title=title,
        source_type=KnowledgeSourceType.MATERIAL_NOTE,
        year=2026,
        citation_text="NanoLoop internal material note, 2026.",
        material_aliases=aliases or ["SrNiO3-x", "Sr-Ni perovskite"],
        license_note="团队合法获取，仅用于比赛演示和内部检索",
        allowed_for_demo=True,
    )


def _write_source(root: Path, name: str, text_value: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text_value, encoding="utf-8")
    return path


def _count(database: Database, model: type[KnowledgeDocument] | type[KnowledgeChunk]) -> int:
    with database.session_factory() as session:
        return int(session.scalar(select(func.count()).select_from(model)) or 0)


def test_ingest_persists_ready_document_metadata_chunks_and_fts(
    knowledge_harness: KnowledgeHarness,
) -> None:
    source = _write_source(
        knowledge_harness.source_root,
        "nested/material.md",
        "# Catalysis\n\nSrNi catalyst supports oxygen evolution applications.",
    )

    report = knowledge_harness.service().ingest_document(source, _metadata())

    assert report.pages_total == 1
    assert report.pages_extracted == 1
    assert report.chunks_created == 1
    assert report.chunks_skipped == 0
    assert report.index_version == "fts5-v1"
    with knowledge_harness.database.session_factory() as session:
        document = session.get(KnowledgeDocument, report.doc_id)
        assert document is not None
        assert document.status == KnowledgeDocumentStatus.READY.value
        assert document.storage_path == "nested/material.md"
        assert document.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
        assert document.metadata_json["license_note"] == _metadata().license_note
        assert document.metadata_json["allowed_for_demo"] is True
        assert document.metadata_json["material_aliases"] == [
            "Sr-Ni perovskite",
            "SrNiO3-x",
        ]
        fts_rows = session.execute(
            text(
                "SELECT chunk_id, text, section_title, material_tags "
                "FROM knowledge_chunks_fts WHERE knowledge_chunks_fts MATCH 'catalyst'"
            )
        ).all()
    assert len(fts_rows) == 1
    assert fts_rows[0].chunk_id.startswith(report.doc_id)
    assert fts_rows[0].section_title == "Catalysis"
    assert "SrNiO3-x" in fts_rows[0].material_tags


def test_duplicate_sha_is_idempotent_but_conflicting_metadata_is_rejected(
    knowledge_harness: KnowledgeHarness,
) -> None:
    source = _write_source(
        knowledge_harness.source_root,
        "duplicate.txt",
        "stable evidence text",
    )
    service = knowledge_harness.service()
    first = service.ingest(source, _metadata())

    duplicate = service.ingest(source, _metadata(aliases=["Sr-Ni perovskite", "SrNiO3-x"]))

    assert duplicate.doc_id == first.doc_id
    assert duplicate.sha256 == first.sha256
    assert duplicate.chunks_created == 0
    assert duplicate.chunks_skipped == first.chunks_created
    assert duplicate.warnings == ["duplicate_sha256: existing document reused"]
    assert _count(knowledge_harness.database, KnowledgeDocument) == 1
    assert _count(knowledge_harness.database, KnowledgeChunk) == first.chunks_created

    with pytest.raises(DuplicateKnowledgeDocumentError) as error:
        service.ingest(source, _metadata(title="Conflicting title"))
    assert error.value.existing_doc_id == first.doc_id
    assert error.value.sha256 == first.sha256
    assert _count(knowledge_harness.database, KnowledgeDocument) == 1


def test_list_documents_reconstructs_public_metadata_dto(
    knowledge_harness: KnowledgeHarness,
) -> None:
    source = _write_source(
        knowledge_harness.source_root,
        "catalogue.md",
        "# Properties\n\nA traceable property statement.",
    )
    metadata = _metadata()
    report = knowledge_harness.service().ingest(source, metadata)

    listed = knowledge_harness.service().list_documents()

    assert len(listed.documents) == 1
    document = listed.documents[0]
    assert document.doc_id == report.doc_id
    assert document.title == metadata.title
    assert document.source_type == metadata.source_type
    assert document.status == KnowledgeDocumentStatus.READY
    assert document.material_aliases == ["Sr-Ni perovskite", "SrNiO3-x"]
    assert document.license_note == metadata.license_note
    assert document.allowed_for_demo is True


def test_document_can_be_disabled_and_reenabled_idempotently(
    knowledge_harness: KnowledgeHarness,
) -> None:
    source = _write_source(
        knowledge_harness.source_root,
        "toggle.md",
        "# Evidence\n\nA searchable catalyst marker.",
    )
    service = knowledge_harness.service()
    report = service.ingest(source, _metadata())

    disabled = service.set_document_enabled(report.doc_id, enabled=False)
    disabled_again = service.set_document_enabled(report.doc_id, enabled=False)

    assert disabled.status == KnowledgeDocumentStatus.DISABLED
    assert disabled_again.status == KnowledgeDocumentStatus.DISABLED
    assert service.list_documents().documents[0].status == KnowledgeDocumentStatus.DISABLED

    enabled = service.set_document_enabled(report.doc_id, enabled=True)
    enabled_again = service.set_document_enabled(report.doc_id, enabled=True)

    assert enabled.status == KnowledgeDocumentStatus.READY
    assert enabled_again.status == KnowledgeDocumentStatus.READY
    assert _count(knowledge_harness.database, KnowledgeChunk) == report.chunks_created


def test_document_toggle_rejects_missing_and_maintenance_states(
    knowledge_harness: KnowledgeHarness,
) -> None:
    service = knowledge_harness.service()
    with pytest.raises(KnowledgeDocumentNotFoundError) as missing:
        service.set_document_enabled("doc_missing", enabled=False)
    assert missing.value.doc_id == "doc_missing"

    source = _write_source(
        knowledge_harness.source_root,
        "maintenance.md",
        "maintenance state evidence",
    )
    report = service.ingest(source, _metadata())
    with knowledge_harness.database.session() as session:
        document = session.get(KnowledgeDocument, report.doc_id)
        assert document is not None
        document.status = KnowledgeDocumentStatus.UNAVAILABLE.value

    with pytest.raises(KnowledgeDocumentStateError) as conflict:
        service.set_document_enabled(report.doc_id, enabled=False)
    assert conflict.value.current_status == KnowledgeDocumentStatus.UNAVAILABLE.value
    assert conflict.value.requested_status == KnowledgeDocumentStatus.DISABLED


def test_disabled_document_without_chunks_cannot_be_enabled(
    knowledge_harness: KnowledgeHarness,
) -> None:
    source = _write_source(
        knowledge_harness.source_root,
        "empty-index.md",
        "evidence that will lose its index fixture",
    )
    service = knowledge_harness.service()
    report = service.ingest(source, _metadata())
    service.set_document_enabled(report.doc_id, enabled=False)
    with knowledge_harness.database.session() as session:
        session.execute(
            delete(KnowledgeChunk).where(KnowledgeChunk.doc_id == report.doc_id)
        )

    with pytest.raises(KnowledgeDocumentStateError, match="without indexed chunks"):
        service.set_document_enabled(report.doc_id, enabled=True)


def test_reindex_reextracts_replaces_chunks_and_removes_stale_fts_rows(
    knowledge_harness: KnowledgeHarness,
) -> None:
    source = _write_source(
        knowledge_harness.source_root,
        "reindex.md",
        "\n\n".join(
            [
                "First catalyst paragraph contains stable evidence and several useful words.",
                "Second paragraph discusses electrochemistry and material applications.",
                "Third paragraph records limitations and experimental context.",
            ]
        ),
    )
    small_chunks = IngestionPipeline(
        chunker=ParagraphChunker(target_chars=80, overlap_chars=10)
    )
    first = knowledge_harness.service(pipeline=small_chunks).ingest(source, _metadata())
    assert first.chunks_created > 1
    with knowledge_harness.database.session_factory() as session:
        old_fts_ids = set(
            session.scalars(text("SELECT chunk_id FROM knowledge_chunks_fts")).all()
        )

    large_chunks = IngestionPipeline(
        chunker=ParagraphChunker(target_chars=1000, overlap_chars=80)
    )
    report = knowledge_harness.service(pipeline=large_chunks).reindex(
        ReindexRequest(force=False)
    )

    assert report.documents_indexed == 1
    assert report.chunks_indexed == 1
    assert report.chunks_skipped == 0
    with knowledge_harness.database.session_factory() as session:
        db_ids = set(
            session.scalars(
                select(KnowledgeChunk.chunk_id).where(KnowledgeChunk.doc_id == first.doc_id)
            ).all()
        )
        fts_ids = set(session.scalars(text("SELECT chunk_id FROM knowledge_chunks_fts")).all())
    assert len(db_ids) == 1
    assert fts_ids == db_ids
    assert old_fts_ids - fts_ids


def test_reindex_failure_is_isolated_and_not_counted_as_success(
    knowledge_harness: KnowledgeHarness,
) -> None:
    service = knowledge_harness.service()
    present = _write_source(
        knowledge_harness.source_root,
        "present.txt",
        "present source evidence",
    )
    missing = _write_source(
        knowledge_harness.source_root,
        "missing.txt",
        "source that will disappear",
    )
    present_report = service.ingest(present, _metadata(title="Present"))
    missing_report = service.ingest(missing, _metadata(title="Missing"))
    missing.unlink()

    report = service.reindex()

    assert report.documents_indexed == 1
    assert report.chunks_indexed == present_report.chunks_created
    assert report.chunks_skipped == missing_report.chunks_created
    assert any(
        warning.startswith(f"{missing_report.doc_id}: source_unavailable:")
        for warning in report.warnings
    )
    with knowledge_harness.database.session_factory() as session:
        retained = session.scalar(
            select(func.count(KnowledgeChunk.chunk_id)).where(
                KnowledgeChunk.doc_id == missing_report.doc_id
            )
        )
    assert retained == missing_report.chunks_created


class _FailingVectorPublisher:
    def __init__(self) -> None:
        self.calls = 0

    def rebuild(self) -> VectorPublishResult:
        self.calls += 1
        raise RuntimeError("offline embedding asset missing")


class _RecoveringVectorPublisher:
    def __init__(self) -> None:
        self.calls = 0

    def rebuild(self) -> VectorPublishResult:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary embedding outage")
        return VectorPublishResult(generation="recovered", vector_count=1, dimension=3)


class _CountingVectorPublisher:
    def __init__(self) -> None:
        self.calls = 0

    def rebuild(self) -> VectorPublishResult:
        self.calls += 1
        return VectorPublishResult(
            generation=f"generation-{self.calls}",
            vector_count=1,
            dimension=3,
        )


def test_duplicate_ingest_retries_degraded_vector_projection(
    knowledge_harness: KnowledgeHarness,
) -> None:
    publisher = _RecoveringVectorPublisher()
    source = _write_source(
        knowledge_harness.source_root,
        "retry-vector.txt",
        "stable evidence for a retryable projection",
    )
    service = knowledge_harness.service(vector_index_publisher=publisher)

    first = service.ingest(source, _metadata())
    duplicate = service.ingest(source, _metadata())

    assert publisher.calls == 2
    assert any("vector_index_degraded" in warning for warning in first.warnings)
    assert duplicate.doc_id == first.doc_id
    assert duplicate.warnings == ["duplicate_sha256: existing document reused"]


def test_idempotent_document_toggles_republish_vector_projection(
    knowledge_harness: KnowledgeHarness,
) -> None:
    publisher = _CountingVectorPublisher()
    source = _write_source(
        knowledge_harness.source_root,
        "toggle-vector.txt",
        "stable evidence for repeated toggle publication",
    )
    service = knowledge_harness.service(vector_index_publisher=publisher)
    report = service.ingest(source, _metadata())

    service.set_document_enabled(report.doc_id, enabled=False)
    service.set_document_enabled(report.doc_id, enabled=False)
    service.set_document_enabled(report.doc_id, enabled=True)
    service.set_document_enabled(report.doc_id, enabled=True)

    assert publisher.calls == 5


def test_vector_publish_failure_degrades_without_rolling_back_fts(
    knowledge_harness: KnowledgeHarness,
) -> None:
    publisher = _FailingVectorPublisher()
    source = _write_source(
        knowledge_harness.source_root,
        "vector-degraded.txt",
        "keyword evidence remains available",
    )
    service = knowledge_harness.service(vector_index_publisher=publisher)

    ingested = service.ingest(source, _metadata())

    assert publisher.calls == 1
    assert any("vector_index_degraded" in warning for warning in ingested.warnings)
    assert _count(knowledge_harness.database, KnowledgeDocument) == 1
    assert _count(knowledge_harness.database, KnowledgeChunk) == ingested.chunks_created
    with knowledge_harness.database.session_factory() as session:
        assert session.scalar(
            text("SELECT COUNT(*) FROM knowledge_chunks_fts")
        ) == ingested.chunks_created

    reindexed = service.reindex()

    assert publisher.calls == 2
    assert reindexed.documents_indexed == 1
    assert any("vector_index_degraded" in warning for warning in reindexed.warnings)


def test_reindex_requires_force_for_changed_source_and_records_hash_history(
    knowledge_harness: KnowledgeHarness,
) -> None:
    source = _write_source(
        knowledge_harness.source_root,
        "changed.txt",
        "original marker evidence",
    )
    service = knowledge_harness.service()
    ingested = service.ingest(source, _metadata())
    source.write_text("replacement marker evidence", encoding="utf-8")

    refused = service.reindex()

    assert refused.documents_indexed == 0
    assert refused.chunks_skipped == ingested.chunks_created
    assert any("source sha256 changed" in warning for warning in refused.warnings)
    with knowledge_harness.database.session_factory() as session:
        unchanged = session.get(KnowledgeDocument, ingested.doc_id)
        assert unchanged is not None
        assert unchanged.sha256 == ingested.sha256

    accepted = service.reindex(force=True)

    expected_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    assert accepted.documents_indexed == 1
    assert any("sha256 updated by force" in warning for warning in accepted.warnings)
    with knowledge_harness.database.session_factory() as session:
        updated = session.get(KnowledgeDocument, ingested.doc_id)
        assert updated is not None
        assert updated.sha256 == expected_sha256
        assert updated.metadata_json["sha256_history"] == [ingested.sha256]
        fts_text = session.scalar(
            text(
                "SELECT text FROM knowledge_chunks_fts "
                "WHERE chunk_id IN ("
                "SELECT chunk_id FROM knowledge_chunks WHERE doc_id = :doc_id"
                ")"
            ),
            {"doc_id": ingested.doc_id},
        )
    assert fts_text == "replacement marker evidence"


class _DisableDuringPrepare(IngestionPipeline):
    def __init__(self, database: Database, doc_id: str) -> None:
        super().__init__()
        self.database = database
        self.doc_id = doc_id

    def prepare(
        self,
        path: str | Path,
        *,
        doc_id: str,
        title: str,
        material_tags: list[str] | tuple[str, ...] = (),
    ) -> PreparedDocument:
        with self.database.session() as session:
            document = session.get(KnowledgeDocument, self.doc_id)
            assert document is not None
            document.status = KnowledgeDocumentStatus.DISABLED.value
        return super().prepare(
            path,
            doc_id=doc_id,
            title=title,
            material_tags=material_tags,
        )


def test_reindex_does_not_overwrite_a_concurrent_disable(
    knowledge_harness: KnowledgeHarness,
) -> None:
    source = _write_source(
        knowledge_harness.source_root,
        "disable-race.txt",
        "evidence remains indexed while retrieval is disabled",
    )
    original = knowledge_harness.service().ingest(source, _metadata())
    service = knowledge_harness.service(
        pipeline=_DisableDuringPrepare(knowledge_harness.database, original.doc_id)
    )

    report = service.reindex()

    assert report.documents_indexed == 1
    with knowledge_harness.database.session_factory() as session:
        document = session.get(KnowledgeDocument, original.doc_id)
        assert document is not None
        assert document.status == KnowledgeDocumentStatus.DISABLED.value


def test_source_paths_cannot_escape_or_traverse_symlinks(
    knowledge_harness: KnowledgeHarness,
    tmp_path: Path,
) -> None:
    outside = _write_source(tmp_path, "outside.txt", "outside evidence")
    service = knowledge_harness.service()

    with pytest.raises(KnowledgeSourcePathError, match="inside"):
        service.ingest(outside, _metadata())

    link = knowledge_harness.source_root / "linked.txt"
    link.symlink_to(outside)
    with pytest.raises(KnowledgeSourcePathError, match="symlinks"):
        service.ingest(link, _metadata())


def test_missing_pdf_dependency_is_explicit_and_does_not_persist(
    knowledge_harness: KnowledgeHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = knowledge_harness.source_root / "paper.pdf"
    source.write_bytes(b"%PDF-1.4\nfixture")
    real_import_module = importlib.import_module

    def import_without_fitz(name: str, package: str | None = None) -> object:
        if name == "fitz":
            raise ImportError("fixture without PyMuPDF")
        return real_import_module(name, package)

    monkeypatch.setattr("app.rag.ingestion.importlib.import_module", import_without_fitz)

    with pytest.raises(
        DocumentExtractionUnavailableError,
        match="optional PyMuPDF dependency",
    ):
        knowledge_harness.service().ingest(source, _metadata())
    assert _count(knowledge_harness.database, KnowledgeDocument) == 0


def test_broken_fts_trigger_rolls_back_document_instead_of_reporting_ready(
    knowledge_harness: KnowledgeHarness,
) -> None:
    with knowledge_harness.database.engine.begin() as connection:
        connection.exec_driver_sql("DROP TRIGGER knowledge_chunks_fts_insert")
        connection.exec_driver_sql(
            """
            CREATE TRIGGER knowledge_chunks_fts_insert
            AFTER INSERT ON knowledge_chunks BEGIN
                SELECT 1;
            END
            """
        )
    source = _write_source(
        knowledge_harness.source_root,
        "broken-index.txt",
        "evidence that must be indexed",
    )

    with pytest.raises(KnowledgeIndexUnavailableError, match="projection mismatch"):
        knowledge_harness.service().ingest(source, _metadata())

    assert _count(knowledge_harness.database, KnowledgeDocument) == 0
    with knowledge_harness.database.session_factory() as session:
        fts_count = session.scalar(text("SELECT COUNT(*) FROM knowledge_chunks_fts"))
    assert fts_count == 0


def test_missing_fts_migration_is_reported_before_extraction(
    knowledge_harness: KnowledgeHarness,
) -> None:
    with knowledge_harness.database.engine.begin() as connection:
        connection.exec_driver_sql("DROP TRIGGER knowledge_chunks_fts_update")
    source = _write_source(
        knowledge_harness.source_root,
        "not-extracted.txt",
        "the missing migration is the primary failure",
    )

    with pytest.raises(KnowledgeIndexUnavailableError, match="apply database migrations"):
        knowledge_harness.service().ingest(source, _metadata())

    assert _count(knowledge_harness.database, KnowledgeDocument) == 0

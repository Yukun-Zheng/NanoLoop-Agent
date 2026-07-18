"""Application service for auditable knowledge ingestion and FTS5 maintenance."""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.contracts.enums import (
    KnowledgeDocumentStatus,
    KnowledgeSourceType,
)
from app.contracts.knowledge import (
    IngestDocumentMetadata,
    IngestReport,
    KnowledgeDocumentDTO,
    KnowledgeDocumentListData,
    ReindexReport,
    ReindexRequest,
)
from app.db.models import KnowledgeChunk, KnowledgeDocument
from app.db.session import Database
from app.rag.ingestion import (
    DocumentExtractionError,
    DocumentExtractionUnavailableError,
    IngestionPipeline,
    PreparedDocument,
)
from app.rag.vector_index import VectorIndexPublisher

logger = logging.getLogger(__name__)

_SUPPORTED_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".pdf"})
_FTS_OBJECTS = {
    "knowledge_chunks_fts": "table",
    "knowledge_chunks_fts_insert": "trigger",
    "knowledge_chunks_fts_delete": "trigger",
    "knowledge_chunks_fts_update": "trigger",
}


class KnowledgeApplicationError(RuntimeError):
    """Base error for knowledge application-service failures."""


class KnowledgeSourcePathError(KnowledgeApplicationError):
    """Raised when a persisted source is missing or escapes the configured root."""


class KnowledgeIndexUnavailableError(KnowledgeApplicationError):
    """Raised when the migrated SQLite FTS5 index cannot be used safely."""


class DuplicateKnowledgeDocumentError(KnowledgeApplicationError):
    """Raised when the same bytes are submitted with conflicting metadata."""

    def __init__(self, *, sha256: str, existing_doc_id: str) -> None:
        super().__init__(
            "a document with this sha256 already exists with different metadata"
        )
        self.sha256 = sha256
        self.existing_doc_id = existing_doc_id


class KnowledgeDocumentNotFoundError(KnowledgeApplicationError):
    """Raised when a requested knowledge document does not exist."""

    def __init__(self, doc_id: str) -> None:
        super().__init__(f"knowledge document not found: {doc_id}")
        self.doc_id = doc_id


class KnowledgeDocumentStateError(KnowledgeApplicationError):
    """Raised when a document cannot safely enter the requested public state."""

    def __init__(
        self,
        *,
        doc_id: str,
        current_status: str,
        requested_status: KnowledgeDocumentStatus,
        reason: str,
    ) -> None:
        super().__init__(reason)
        self.doc_id = doc_id
        self.current_status = current_status
        self.requested_status = requested_status
        self.reason = reason


@dataclass(frozen=True, slots=True)
class _DocumentSnapshot:
    doc_id: str
    title: str
    storage_path: str
    sha256: str
    status: str
    material_aliases: tuple[str, ...]
    metadata_json: dict[str, Any]
    chunk_count: int


class KnowledgeApplicationService:
    """Persist documents and maintain FTS5 plus an optional vector projection.

    File extraction and hashing happen outside write transactions. Each document is
    inserted or reindexed in its own short transaction, and FTS trigger output is
    verified before that transaction commits. Vector publication happens afterward;
    failure is reported as degradation and never rolls back authoritative SQL/FTS data.
    """

    def __init__(
        self,
        database: Database,
        source_root: str | Path,
        *,
        pipeline: IngestionPipeline | None = None,
        index_version: str = "fts5-v1",
        vector_index_publisher: VectorIndexPublisher | None = None,
    ) -> None:
        root = Path(source_root).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        self.database = database
        self.source_root = root.resolve(strict=True)
        if not self.source_root.is_dir():
            raise KnowledgeSourcePathError("knowledge source root must be a directory")
        self.pipeline = pipeline or IngestionPipeline()
        self.index_version = index_version
        self.vector_index_publisher = vector_index_publisher

    def ingest(
        self,
        source_path: str | Path,
        metadata: IngestDocumentMetadata,
    ) -> IngestReport:
        """Index one already-persisted source, with SHA-based idempotency."""

        return self.ingest_document(source_path, metadata)

    def ingest_document(
        self,
        source_path: str | Path,
        metadata: IngestDocumentMetadata,
    ) -> IngestReport:
        """Index one already-persisted source, with SHA-based idempotency."""

        if not isinstance(metadata, IngestDocumentMetadata):
            raise TypeError("metadata must be IngestDocumentMetadata")
        self._assert_fts5_ready()
        source = self._resolve_source(source_path)

        initial_sha256 = self._sha256(source)
        duplicate = self._find_by_sha256(initial_sha256)
        if duplicate is not None:
            return self._duplicate_result_with_vector_refresh(duplicate, metadata)

        prepared = self.pipeline.prepare(
            source,
            doc_id=f"doc_{uuid4().hex}",
            title=metadata.title,
            material_tags=self._canonical_aliases(metadata.material_aliases),
        )
        if prepared.sha256 != initial_sha256:
            raise DocumentExtractionError("knowledge source changed while it was being ingested")
        if not prepared.chunks:
            raise DocumentExtractionError("knowledge document contains no extractable text")

        try:
            self._persist_prepared(source, prepared, metadata)
        except IntegrityError:
            # A concurrent ingestion can win the unique sha256 race. It is still
            # idempotent only when all user-controlled metadata agrees.
            duplicate = self._find_by_sha256(prepared.sha256)
            if duplicate is None:
                raise
            return self._duplicate_result_with_vector_refresh(duplicate, metadata)

        warnings = list(prepared.warnings)
        vector_warning = self._refresh_vector_index()
        if vector_warning is not None:
            warnings.append(vector_warning)
        return IngestReport(
            doc_id=prepared.doc_id,
            sha256=prepared.sha256,
            pages_total=prepared.pages_total,
            pages_extracted=prepared.pages_extracted,
            chunks_created=len(prepared.chunks),
            chunks_skipped=max(prepared.pages_total - prepared.pages_extracted, 0),
            warnings=warnings,
            index_version=self.index_version,
        )

    def list_documents(self) -> KnowledgeDocumentListData:
        """Return the persisted knowledge catalogue as public DTOs."""

        with self.database.session_factory() as session:
            records = session.scalars(
                select(KnowledgeDocument).order_by(
                    KnowledgeDocument.created_at.desc(),
                    KnowledgeDocument.doc_id,
                )
            ).all()
            documents = [self._to_dto(record) for record in records]
        return KnowledgeDocumentListData(documents=documents)

    def set_document_enabled(
        self,
        doc_id: str,
        *,
        enabled: bool,
    ) -> KnowledgeDocumentDTO:
        """Idempotently toggle retrieval eligibility for an indexed document.

        Public toggles deliberately operate only between ``ready`` and ``disabled``.
        ``indexing`` and ``unavailable`` are maintenance states and must be repaired by
        ingestion/reindexing rather than promoted by this endpoint.
        """

        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError("doc_id must be a non-empty string")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")

        target = (
            KnowledgeDocumentStatus.READY
            if enabled
            else KnowledgeDocumentStatus.DISABLED
        )
        if target == KnowledgeDocumentStatus.READY:
            self._assert_fts5_ready()

        with self.database.session_factory() as session, session.begin():
            document = session.get(KnowledgeDocument, doc_id)
            if document is None:
                raise KnowledgeDocumentNotFoundError(doc_id)

            current_status = document.status
            if current_status not in {
                KnowledgeDocumentStatus.READY.value,
                KnowledgeDocumentStatus.DISABLED.value,
            }:
                raise KnowledgeDocumentStateError(
                    doc_id=doc_id,
                    current_status=current_status,
                    requested_status=target,
                    reason="only ready and disabled documents can be toggled",
                )

            if target == KnowledgeDocumentStatus.READY:
                chunk_count = int(
                    session.scalar(
                        select(func.count(KnowledgeChunk.chunk_id)).where(
                            KnowledgeChunk.doc_id == doc_id
                        )
                    )
                    or 0
                )
                if chunk_count == 0:
                    raise KnowledgeDocumentStateError(
                        doc_id=doc_id,
                        current_status=current_status,
                        requested_status=target,
                        reason="a document without indexed chunks cannot be enabled",
                    )
                self._verify_fts_rows(session, doc_id, chunk_count)

            if current_status != target.value:
                document.status = target.value
            session.flush()
            result = self._to_dto(document)
        vector_warning = self._refresh_vector_index()
        if vector_warning is not None:
            logger.warning(
                "knowledge_vector_index_degraded",
                extra={"event": "component_degraded", "detail": vector_warning},
            )
        return result

    def reindex(
        self,
        request: ReindexRequest | None = None,
        *,
        force: bool = False,
    ) -> ReindexReport:
        """Re-extract and atomically replace chunks for every non-disabled document.

        A changed source hash is rejected by default because it is no longer the
        originally ingested artifact. ``force=True`` explicitly accepts that content
        change, records the previous hash in metadata, and updates the document hash.
        """

        if request is not None and not isinstance(request, ReindexRequest):
            raise TypeError("request must be ReindexRequest")
        effective_force = force or bool(request and request.force)
        self._assert_fts5_ready()

        documents_indexed = 0
        chunks_indexed = 0
        chunks_skipped = 0
        warnings: list[str] = []

        for snapshot in self._document_snapshots():
            if snapshot.status == KnowledgeDocumentStatus.DISABLED.value:
                chunks_skipped += snapshot.chunk_count
                warnings.append(f"{snapshot.doc_id}: disabled document skipped")
                continue
            try:
                source = self._resolve_source(snapshot.storage_path)
                prepared = self.pipeline.prepare(
                    source,
                    doc_id=snapshot.doc_id,
                    title=snapshot.title,
                    material_tags=snapshot.material_aliases,
                )
                if not prepared.chunks:
                    raise DocumentExtractionError(
                        "knowledge document contains no extractable text"
                    )
                content_changed = prepared.sha256 != snapshot.sha256
                if content_changed and not effective_force:
                    raise KnowledgeApplicationError(
                        "source sha256 changed; pass force=True to accept new content"
                    )
                self._replace_chunks(
                    snapshot,
                    prepared,
                    accept_changed_content=content_changed,
                )
            except Exception as error:
                chunks_skipped += snapshot.chunk_count
                warnings.append(self._reindex_warning(snapshot.doc_id, error))
                continue

            documents_indexed += 1
            chunks_indexed += len(prepared.chunks)
            chunks_skipped += max(prepared.pages_total - prepared.pages_extracted, 0)
            warnings.extend(f"{snapshot.doc_id}: {warning}" for warning in prepared.warnings)
            if content_changed:
                warnings.append(
                    f"{snapshot.doc_id}: source content changed; sha256 updated by force"
                )

        vector_warning = self._refresh_vector_index()
        if vector_warning is not None:
            warnings.append(vector_warning)

        return ReindexReport(
            documents_indexed=documents_indexed,
            chunks_indexed=chunks_indexed,
            chunks_skipped=chunks_skipped,
            warnings=warnings,
            index_version=self.index_version,
        )

    def _persist_prepared(
        self,
        source: Path,
        prepared: PreparedDocument,
        metadata: IngestDocumentMetadata,
    ) -> None:
        aliases = self._canonical_aliases(metadata.material_aliases)
        document = KnowledgeDocument(
            doc_id=prepared.doc_id,
            title=metadata.title,
            source_type=metadata.source_type.value,
            storage_path=source.relative_to(self.source_root).as_posix(),
            sha256=prepared.sha256,
            year=metadata.year,
            citation_text=metadata.citation_text,
            status=KnowledgeDocumentStatus.READY.value,
            metadata_json=self._metadata_json(
                metadata,
                aliases=aliases,
                prepared=prepared,
            ),
        )
        document.chunks = [
            KnowledgeChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_title=chunk.section_title,
                text=chunk.text,
                material_tags_json=list(chunk.material_tags),
            )
            for chunk in prepared.chunks
        ]
        with self.database.session_factory() as session, session.begin():
            session.add(document)
            session.flush()
            self._verify_fts_rows(session, prepared.doc_id, len(prepared.chunks))

    def _refresh_vector_index(self) -> str | None:
        """Publish a full vector generation without weakening committed FTS data."""

        if self.vector_index_publisher is None:
            return None
        try:
            self.vector_index_publisher.rebuild()
        except Exception as error:
            message = str(error).strip() or type(error).__name__
            return f"vector_index_degraded: {type(error).__name__}: {message}"
        return None

    def _replace_chunks(
        self,
        snapshot: _DocumentSnapshot,
        prepared: PreparedDocument,
        *,
        accept_changed_content: bool,
    ) -> None:
        with self.database.session_factory() as session, session.begin():
            document = session.get(KnowledgeDocument, snapshot.doc_id)
            if document is None:
                raise KnowledgeApplicationError("document disappeared during reindex")

            session.execute(
                delete(KnowledgeChunk).where(KnowledgeChunk.doc_id == snapshot.doc_id)
            )
            session.flush()
            session.add_all(
                KnowledgeChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_title=chunk.section_title,
                    text=chunk.text,
                    material_tags_json=list(chunk.material_tags),
                )
                for chunk in prepared.chunks
            )
            metadata_json = dict(document.metadata_json or {})
            metadata_json.update(
                {
                    "pages_total": prepared.pages_total,
                    "pages_extracted": prepared.pages_extracted,
                    "ingestion_warnings": list(prepared.warnings),
                    "index_version": self.index_version,
                }
            )
            if accept_changed_content:
                history = self._string_list(metadata_json.get("sha256_history"))
                if document.sha256 not in history:
                    history.append(document.sha256)
                metadata_json["sha256_history"] = history
                document.sha256 = prepared.sha256
            document.metadata_json = metadata_json
            if document.status != KnowledgeDocumentStatus.DISABLED.value:
                document.status = KnowledgeDocumentStatus.READY.value
            session.flush()
            self._verify_fts_rows(session, snapshot.doc_id, len(prepared.chunks))

    def _find_by_sha256(self, sha256: str) -> KnowledgeDocument | None:
        with self.database.session_factory() as session:
            return session.scalar(
                select(KnowledgeDocument).where(KnowledgeDocument.sha256 == sha256)
            )

    def _duplicate_result(
        self,
        document: KnowledgeDocument,
        metadata: IngestDocumentMetadata,
    ) -> IngestReport:
        if not self._metadata_matches(document, metadata):
            raise DuplicateKnowledgeDocumentError(
                sha256=document.sha256,
                existing_doc_id=document.doc_id,
            )
        with self.database.session_factory() as session:
            chunk_count = int(
                session.scalar(
                    select(func.count(KnowledgeChunk.chunk_id)).where(
                        KnowledgeChunk.doc_id == document.doc_id
                    )
                )
                or 0
            )
        stored = dict(document.metadata_json or {})
        return IngestReport(
            doc_id=document.doc_id,
            sha256=document.sha256,
            pages_total=self._nonnegative_int(stored.get("pages_total")),
            pages_extracted=self._nonnegative_int(stored.get("pages_extracted")),
            chunks_created=0,
            chunks_skipped=chunk_count,
            warnings=["duplicate_sha256: existing document reused"],
            index_version=str(stored.get("index_version") or self.index_version),
        )

    def _duplicate_result_with_vector_refresh(
        self,
        document: KnowledgeDocument,
        metadata: IngestDocumentMetadata,
    ) -> IngestReport:
        """Reuse authoritative SQL while retrying an optional failed projection.

        A duplicate request is still useful after a transient embedding or vector-store
        outage.  Treating it as a pure early return would leave a committed document
        permanently absent from the vector generation until an unrelated mutation.
        """

        report = self._duplicate_result(document, metadata)
        vector_warning = self._refresh_vector_index()
        if vector_warning is None:
            return report
        return report.model_copy(
            update={"warnings": [*report.warnings, vector_warning]}
        )

    def _metadata_matches(
        self,
        document: KnowledgeDocument,
        metadata: IngestDocumentMetadata,
    ) -> bool:
        stored = dict(document.metadata_json or {})
        return (
            document.title == metadata.title
            and document.source_type == metadata.source_type.value
            and document.year == metadata.year
            and document.citation_text == metadata.citation_text
            and self._canonical_aliases(self._string_list(stored.get("material_aliases")))
            == self._canonical_aliases(metadata.material_aliases)
            and stored.get("license_note") == metadata.license_note
            and stored.get("allowed_for_demo") is metadata.allowed_for_demo
        )

    def _document_snapshots(self) -> list[_DocumentSnapshot]:
        with self.database.session_factory() as session:
            rows = session.execute(
                select(
                    KnowledgeDocument,
                    func.count(KnowledgeChunk.chunk_id),
                )
                .outerjoin(
                    KnowledgeChunk,
                    KnowledgeChunk.doc_id == KnowledgeDocument.doc_id,
                )
                .group_by(KnowledgeDocument.doc_id)
                .order_by(KnowledgeDocument.created_at, KnowledgeDocument.doc_id)
            ).all()
            return [
                _DocumentSnapshot(
                    doc_id=document.doc_id,
                    title=document.title,
                    storage_path=document.storage_path,
                    sha256=document.sha256,
                    status=document.status,
                    material_aliases=self._canonical_aliases(
                        self._string_list(
                            dict(document.metadata_json or {}).get("material_aliases")
                        )
                    ),
                    metadata_json=dict(document.metadata_json or {}),
                    chunk_count=int(chunk_count),
                )
                for document, chunk_count in rows
            ]

    def _to_dto(self, document: KnowledgeDocument) -> KnowledgeDocumentDTO:
        metadata = dict(document.metadata_json or {})
        try:
            source_type = KnowledgeSourceType(document.source_type)
            status = KnowledgeDocumentStatus(document.status)
        except ValueError as error:
            raise KnowledgeApplicationError(
                f"document {document.doc_id} contains an invalid enum value"
            ) from error

        license_note = metadata.get("license_note")
        if not isinstance(license_note, str) or not license_note:
            license_note = "许可说明缺失（需人工补录）"
        allowed_for_demo = metadata.get("allowed_for_demo")
        return KnowledgeDocumentDTO(
            doc_id=document.doc_id,
            title=document.title,
            source_type=source_type,
            sha256=document.sha256,
            year=document.year,
            citation_text=document.citation_text,
            status=status,
            material_aliases=self._string_list(metadata.get("material_aliases")),
            license_note=license_note,
            allowed_for_demo=(
                allowed_for_demo if isinstance(allowed_for_demo, bool) else False
            ),
            created_at=document.created_at,
        )

    def _resolve_source(self, source_path: str | Path) -> Path:
        supplied = Path(source_path).expanduser()
        candidate = supplied if supplied.is_absolute() else self.source_root / supplied
        lexical = Path(os.path.abspath(candidate))
        try:
            relative = lexical.relative_to(self.source_root)
        except ValueError as error:
            raise KnowledgeSourcePathError(
                "knowledge source must stay inside the configured source root"
            ) from error

        current = self.source_root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise KnowledgeSourcePathError("knowledge source cannot traverse symlinks")
        try:
            resolved = lexical.resolve(strict=True)
        except (FileNotFoundError, NotADirectoryError) as error:
            raise KnowledgeSourcePathError("knowledge source does not exist") from error
        try:
            resolved.relative_to(self.source_root)
        except ValueError as error:
            raise KnowledgeSourcePathError(
                "knowledge source resolved outside the configured source root"
            ) from error
        if not resolved.is_file():
            raise KnowledgeSourcePathError("knowledge source must be a regular file")
        suffix = resolved.suffix.casefold()
        if suffix not in _SUPPORTED_SUFFIXES:
            raise DocumentExtractionError(
                f"unsupported knowledge document type: {suffix or '<none>'}"
            )
        return resolved

    def _assert_fts5_ready(self) -> None:
        if self.database.engine.dialect.name != "sqlite":
            raise KnowledgeIndexUnavailableError(
                "knowledge indexing currently requires migrated SQLite FTS5"
            )
        with self.database.session_factory() as session:
            rows = session.execute(
                text(
                    "SELECT name, type FROM sqlite_master "
                    "WHERE name IN ("
                    "'knowledge_chunks_fts', "
                    "'knowledge_chunks_fts_insert', "
                    "'knowledge_chunks_fts_delete', "
                    "'knowledge_chunks_fts_update'"
                    ")"
                )
            ).all()
        found = {str(name): str(object_type) for name, object_type in rows}
        missing = [
            name for name, object_type in _FTS_OBJECTS.items() if found.get(name) != object_type
        ]
        if missing:
            raise KnowledgeIndexUnavailableError(
                "knowledge FTS5 schema is unavailable; apply database migrations: "
                + ", ".join(sorted(missing))
            )

    @staticmethod
    def _verify_fts_rows(session: Session, doc_id: str, expected: int) -> None:
        actual = int(
            session.scalar(
                text(
                    "SELECT COUNT(*) "
                    "FROM knowledge_chunks_fts AS f "
                    "JOIN knowledge_chunks AS c ON c.chunk_id = f.chunk_id "
                    "WHERE c.doc_id = :doc_id"
                ),
                {"doc_id": doc_id},
            )
            or 0
        )
        if actual != expected:
            raise KnowledgeIndexUnavailableError(
                f"FTS5 trigger projection mismatch for {doc_id}: "
                f"expected {expected}, found {actual}"
            )

    def _metadata_json(
        self,
        metadata: IngestDocumentMetadata,
        *,
        aliases: tuple[str, ...],
        prepared: PreparedDocument,
    ) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "material_aliases": list(aliases),
            "license_note": metadata.license_note,
            "allowed_for_demo": metadata.allowed_for_demo,
            "pages_total": prepared.pages_total,
            "pages_extracted": prepared.pages_extracted,
            "ingestion_warnings": list(prepared.warnings),
            "index_version": self.index_version,
        }

    @staticmethod
    def _canonical_aliases(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
        unique = {value.strip() for value in values if value.strip()}
        return tuple(sorted(unique, key=lambda value: (value.casefold(), value)))

    @staticmethod
    def _string_list(value: object) -> list[str]:
        if not isinstance(value, (list, tuple)):
            return []
        return [item for item in value if isinstance(item, str) and item]

    @staticmethod
    def _nonnegative_int(value: object) -> int:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        return 0

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _reindex_warning(doc_id: str, error: Exception) -> str:
        if isinstance(error, DocumentExtractionUnavailableError):
            category = "extractor_unavailable"
        elif isinstance(error, DocumentExtractionError):
            category = "extraction_failed"
        elif isinstance(error, KnowledgeSourcePathError):
            category = "source_unavailable"
        elif isinstance(error, IntegrityError):
            category = "sha256_conflict"
        elif isinstance(error, KnowledgeIndexUnavailableError):
            category = "index_unavailable"
        else:
            category = "reindex_failed"
        message = str(error).strip() or type(error).__name__
        return f"{doc_id}: {category}: {message}"


__all__ = [
    "DuplicateKnowledgeDocumentError",
    "KnowledgeApplicationError",
    "KnowledgeApplicationService",
    "KnowledgeDocumentNotFoundError",
    "KnowledgeDocumentStateError",
    "KnowledgeIndexUnavailableError",
    "KnowledgeSourcePathError",
]

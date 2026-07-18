"""Knowledge ingestion, indexing, retrieval, and answer-provider DTOs."""

from datetime import datetime

from pydantic import Field, StrictBool

from app.contracts.common import ContractModel
from app.contracts.enums import KnowledgeDocumentStatus, KnowledgeSourceType
from app.contracts.limits import MAX_MATERIAL_ALIASES, MaterialAlias


class IngestDocumentMetadata(ContractModel):
    title: str = Field(min_length=1, max_length=500)
    source_type: KnowledgeSourceType
    year: int | None = Field(default=None, ge=1000, le=3000)
    citation_text: str = Field(min_length=1, max_length=2000)
    material_aliases: list[MaterialAlias] = Field(
        default_factory=list,
        max_length=MAX_MATERIAL_ALIASES,
    )
    license_note: str = Field(min_length=1, max_length=2000)
    allowed_for_demo: bool = False


class KnowledgeDocumentDTO(ContractModel):
    doc_id: str
    title: str
    source_type: KnowledgeSourceType
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    year: int | None = None
    citation_text: str
    status: KnowledgeDocumentStatus
    material_aliases: list[str] = Field(default_factory=list)
    license_note: str
    allowed_for_demo: bool
    created_at: datetime


class IngestReport(ContractModel):
    doc_id: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pages_total: int = Field(ge=0)
    pages_extracted: int = Field(ge=0)
    chunks_created: int = Field(ge=0)
    chunks_skipped: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    index_version: str


class ReindexRequest(ContractModel):
    force: bool = False


class ReindexReport(ContractModel):
    documents_indexed: int = Field(ge=0)
    chunks_indexed: int = Field(ge=0)
    chunks_skipped: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    index_version: str


class KnowledgeDocumentListData(ContractModel):
    documents: list[KnowledgeDocumentDTO] = Field(default_factory=list)


class UpdateKnowledgeDocumentRequest(ContractModel):
    """Enable or disable one already-indexed knowledge document."""

    enabled: StrictBool


class RetrievalRequest(ContractModel):
    query: str = Field(min_length=1, max_length=2000)
    material_aliases: list[MaterialAlias] = Field(
        default_factory=list,
        max_length=MAX_MATERIAL_ALIASES,
    )
    top_k: int = Field(default=6, ge=1, le=50)
    candidate_k: int = Field(default=20, ge=1, le=200)
    min_score: float = Field(default=0.20, ge=0)


class RetrievedChunk(ContractModel):
    chunk_id: str
    doc_id: str
    title: str
    source_type: str | None = None
    citation_text: str | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    section_title: str | None = None
    text: str
    material_tags: list[str] = Field(default_factory=list)
    retrieval_score: float = Field(ge=0)

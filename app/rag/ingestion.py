"""Lazy document extraction and persistence-neutral ingestion preparation."""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path

from app.rag.chunking import (
    ChunkLimitExceededError,
    ExtractedPage,
    KnowledgeChunkRecord,
    ParagraphChunker,
)

DEFAULT_MAX_PDF_PAGES = 2_000
DEFAULT_MAX_EXTRACTED_CHARS = 10_000_000
DEFAULT_MAX_CHUNKS_PER_DOCUMENT = 20_000


class DocumentExtractionError(ValueError):
    """Raised for unsupported or unreadable knowledge documents."""


class DocumentExtractionUnavailableError(DocumentExtractionError):
    """Raised when an optional extractor dependency is not installed."""


class KnowledgeDocumentLimitError(DocumentExtractionError):
    """Raised when valid document content exceeds an operational safety bound."""

    def __init__(self, *, resource: str, limit: int, observed: int | None = None) -> None:
        message = f"knowledge document exceeds the {resource} limit of {limit}"
        if observed is not None:
            message += f" (observed {observed})"
        super().__init__(message)
        self.resource = resource
        self.limit = limit
        self.observed = observed


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    pages: tuple[ExtractedPage, ...]
    pages_total: int
    pages_extracted: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PreparedDocument:
    doc_id: str
    title: str
    source_path: Path
    sha256: str
    pages_total: int
    pages_extracted: int
    chunks: tuple[KnowledgeChunkRecord, ...]
    warnings: tuple[str, ...]


class DocumentExtractor:
    """Extract TXT, Markdown, and optional PDF without importing PyMuPDF at startup."""

    text_suffixes = frozenset({".txt", ".md", ".markdown"})

    def __init__(
        self,
        *,
        max_pdf_pages: int = DEFAULT_MAX_PDF_PAGES,
        max_extracted_chars: int = DEFAULT_MAX_EXTRACTED_CHARS,
    ) -> None:
        if max_pdf_pages <= 0:
            raise ValueError("max_pdf_pages must be positive")
        if max_extracted_chars <= 0:
            raise ValueError("max_extracted_chars must be positive")
        self.max_pdf_pages = max_pdf_pages
        self.max_extracted_chars = max_extracted_chars

    def extract(self, path: str | Path) -> ExtractionResult:
        source = Path(path)
        if not source.is_file():
            raise DocumentExtractionError(f"knowledge document does not exist: {source.name}")
        suffix = source.suffix.casefold()
        if suffix in self.text_suffixes:
            try:
                # Stop after one character beyond the accepted bound. Reading the
                # complete file first would defeat the operational safety limit.
                with source.open("r", encoding="utf-8-sig") as stream:
                    text = stream.read(self.max_extracted_chars + 1)
            except UnicodeDecodeError as error:
                raise DocumentExtractionError("text documents must use UTF-8 encoding") from error
            self._assert_character_limit(len(text))
            stripped = text.strip()
            warnings = () if stripped else ("page_1_empty",)
            return ExtractionResult(
                pages=(ExtractedPage(page_number=1, text=text),),
                pages_total=1,
                pages_extracted=1 if stripped else 0,
                warnings=warnings,
            )
        if suffix == ".pdf":
            return self._extract_pdf(source)
        raise DocumentExtractionError(f"unsupported knowledge document type: {suffix or '<none>'}")

    def _extract_pdf(self, path: Path) -> ExtractionResult:
        try:
            fitz = importlib.import_module("fitz")
        except ImportError as error:
            raise DocumentExtractionUnavailableError(
                "PDF extraction requires the optional PyMuPDF dependency"
            ) from error

        pages: list[ExtractedPage] = []
        warnings: list[str] = []
        try:
            with fitz.open(path) as document:
                page_count = len(document)
                if page_count > self.max_pdf_pages:
                    raise KnowledgeDocumentLimitError(
                        resource="PDF page count",
                        limit=self.max_pdf_pages,
                        observed=page_count,
                    )
                extracted_chars = 0
                for index, page in enumerate(document, start=1):
                    text = page.get_text("text") or ""
                    extracted_chars += len(text)
                    self._assert_character_limit(extracted_chars)
                    pages.append(ExtractedPage(page_number=index, text=text))
                    if not text.strip():
                        warnings.append(f"page_{index}_empty")
        except KnowledgeDocumentLimitError:
            raise
        except Exception as error:
            raise DocumentExtractionError("PDF text extraction failed") from error
        return ExtractionResult(
            pages=tuple(pages),
            pages_total=len(pages),
            pages_extracted=sum(bool(page.text.strip()) for page in pages),
            warnings=tuple(warnings),
        )

    def _assert_character_limit(self, observed: int) -> None:
        if observed > self.max_extracted_chars:
            raise KnowledgeDocumentLimitError(
                resource="extracted character count",
                limit=self.max_extracted_chars,
                observed=observed,
            )


class IngestionPipeline:
    """Extract, hash, and chunk a source without claiming it has been indexed."""

    def __init__(
        self,
        *,
        extractor: DocumentExtractor | None = None,
        chunker: ParagraphChunker | None = None,
        max_chunks_per_document: int = DEFAULT_MAX_CHUNKS_PER_DOCUMENT,
    ) -> None:
        if max_chunks_per_document <= 0:
            raise ValueError("max_chunks_per_document must be positive")
        self.extractor = extractor or DocumentExtractor()
        self.chunker = chunker or ParagraphChunker()
        self.max_chunks_per_document = max_chunks_per_document

    def prepare(
        self,
        path: str | Path,
        *,
        doc_id: str,
        title: str,
        material_tags: list[str] | tuple[str, ...] = (),
    ) -> PreparedDocument:
        source = Path(path).resolve(strict=True)
        extraction = self.extractor.extract(source)
        try:
            chunks = self.chunker.chunk_pages(
                list(extraction.pages),
                doc_id=doc_id,
                title=title,
                material_tags=material_tags,
                max_chunks=self.max_chunks_per_document,
            )
        except ChunkLimitExceededError as error:
            raise KnowledgeDocumentLimitError(
                resource="chunk count",
                limit=error.limit,
                observed=error.limit + 1,
            ) from error
        return PreparedDocument(
            doc_id=doc_id,
            title=title,
            source_path=source,
            sha256=self._sha256(source),
            pages_total=extraction.pages_total,
            pages_extracted=extraction.pages_extracted,
            chunks=tuple(chunks),
            warnings=extraction.warnings,
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

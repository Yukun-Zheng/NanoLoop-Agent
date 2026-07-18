"""Tests for page-preserving extraction and paragraph-first chunking."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from app.rag.chunking import ExtractedPage, ParagraphChunker
from app.rag.ingestion import (
    DocumentExtractionError,
    DocumentExtractionUnavailableError,
    DocumentExtractor,
    IngestionPipeline,
    KnowledgeDocumentLimitError,
)


def test_extracts_utf8_text_and_markdown_as_page_one(tmp_path: Path) -> None:
    source = tmp_path / "knowledge.md"
    source.write_text("# 性质\n\nSrNi 催化 性质", encoding="utf-8")

    result = DocumentExtractor().extract(source)

    assert result.pages_total == 1
    assert result.pages_extracted == 1
    assert result.pages[0].page_number == 1
    assert result.pages[0].text.startswith("# 性质")
    assert result.warnings == ()


def test_empty_text_page_is_retained_and_warned(tmp_path: Path) -> None:
    source = tmp_path / "empty.txt"
    source.write_text(" \n", encoding="utf-8")

    result = DocumentExtractor().extract(source)

    assert len(result.pages) == 1
    assert result.pages_extracted == 0
    assert result.warnings == ("page_1_empty",)


def test_pdf_import_is_lazy_and_missing_dependency_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"not opened because dependency is absent")

    def missing_module(name: str) -> Any:
        assert name == "fitz"
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("app.rag.ingestion.importlib.import_module", missing_module)

    with pytest.raises(DocumentExtractionUnavailableError, match="PyMuPDF"):
        DocumentExtractor().extract(source)


def test_pdf_extraction_keeps_original_one_based_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"fixture")

    class FakePage:
        def __init__(self, text: str) -> None:
            self.text = text

        def get_text(self, mode: str) -> str:
            assert mode == "text"
            return self.text

    class FakeDocument(list[FakePage]):
        def __enter__(self) -> FakeDocument:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class FakeFitz:
        @staticmethod
        def open(path: Path) -> FakeDocument:
            assert path == source
            return FakeDocument([FakePage("第一页"), FakePage(""), FakePage("第三页")])

    monkeypatch.setattr(
        "app.rag.ingestion.importlib.import_module",
        lambda name: FakeFitz,
    )

    result = DocumentExtractor().extract(source)

    assert [page.page_number for page in result.pages] == [1, 2, 3]
    assert result.pages_total == 3
    assert result.pages_extracted == 2
    assert result.warnings == ("page_2_empty",)


def test_pdf_page_limit_is_checked_before_page_text_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "many-pages.pdf"
    source.write_bytes(b"fixture")
    page_reads = 0

    class FakePage:
        def get_text(self, mode: str) -> str:
            nonlocal page_reads
            page_reads += 1
            return mode

    class FakeDocument(list[FakePage]):
        def __enter__(self) -> FakeDocument:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class FakeFitz:
        @staticmethod
        def open(path: Path) -> FakeDocument:
            assert path == source
            return FakeDocument([FakePage(), FakePage(), FakePage()])

    monkeypatch.setattr("app.rag.ingestion.importlib.import_module", lambda _: FakeFitz)

    with pytest.raises(KnowledgeDocumentLimitError, match="PDF page count limit of 2"):
        DocumentExtractor(max_pdf_pages=2).extract(source)

    assert page_reads == 0


def test_pdf_extracted_character_limit_stops_incremental_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "verbose.pdf"
    source.write_bytes(b"fixture")
    page_reads = 0

    class FakePage:
        def get_text(self, mode: str) -> str:
            nonlocal page_reads
            assert mode == "text"
            page_reads += 1
            return "1234"

    class FakeDocument(list[FakePage]):
        def __enter__(self) -> FakeDocument:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class FakeFitz:
        @staticmethod
        def open(path: Path) -> FakeDocument:
            assert path == source
            return FakeDocument([FakePage(), FakePage(), FakePage()])

    monkeypatch.setattr("app.rag.ingestion.importlib.import_module", lambda _: FakeFitz)

    with pytest.raises(KnowledgeDocumentLimitError, match="character count limit of 6"):
        DocumentExtractor(max_extracted_chars=6).extract(source)

    assert page_reads == 2


def test_extracted_character_and_chunk_limits_raise_domain_errors(
    tmp_path: Path,
) -> None:
    too_long = tmp_path / "too-long.txt"
    too_long.write_text("123456", encoding="utf-8")

    with pytest.raises(KnowledgeDocumentLimitError, match="character count limit of 5"):
        DocumentExtractor(max_extracted_chars=5).extract(too_long)

    many_chunks = tmp_path / "many-chunks.txt"
    many_chunks.write_text("甲" * 1_400, encoding="utf-8")
    pipeline = IngestionPipeline(max_chunks_per_document=2)
    with pytest.raises(KnowledgeDocumentLimitError, match="chunk count limit of 2"):
        pipeline.prepare(many_chunks, doc_id="doc_bounded", title="Bounded")


def test_rejects_unsupported_or_non_utf8_documents(tmp_path: Path) -> None:
    unsupported = tmp_path / "source.docx"
    unsupported.write_bytes(b"docx")
    invalid_text = tmp_path / "invalid.txt"
    invalid_text.write_bytes(b"\xff\xfe")

    with pytest.raises(DocumentExtractionError, match="unsupported"):
        DocumentExtractor().extract(unsupported)
    with pytest.raises(DocumentExtractionError, match="UTF-8"):
        DocumentExtractor().extract(invalid_text)


def test_chunker_prefers_paragraphs_and_carries_configured_overlap() -> None:
    first = "甲" * 300
    second = "乙" * 300
    third = "丙" * 300
    chunker = ParagraphChunker(target_chars=600, overlap_chars=80)

    chunks = chunker.chunk_pages(
        [ExtractedPage(page_number=4, text=f"# 材料性质\n\n{first}\n\n{second}\n\n{third}")],
        doc_id="doc_001",
        title="材料文档",
        material_tags=["SrNi", "SrNi"],
    )

    assert all(len(chunk.text) <= 600 for chunk in chunks)
    assert all(chunk.page_start == 4 and chunk.page_end == 4 for chunk in chunks)
    assert all(chunk.section_title == "材料性质" for chunk in chunks)
    assert chunks[0].chunk_id == "doc_001_0004_001"
    assert chunks[0].material_tags == ("SrNi",)
    assert chunks[1].text.startswith(chunks[0].text[-80:])


def test_long_paragraph_is_split_without_losing_page_metadata() -> None:
    text = "纳" * 1400
    chunks = ParagraphChunker().chunk_pages(
        [ExtractedPage(page_number=2, text=text)],
        doc_id="doc_long",
        title="长段落",
    )

    assert len(chunks) == 3
    assert all(len(chunk.text) <= 600 for chunk in chunks)
    assert chunks[1].text.startswith(chunks[0].text[-80:])
    assert chunks[2].text.startswith(chunks[1].text[-80:])


def test_ingestion_prepares_hash_chunks_and_warnings_without_claiming_indexing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.txt"
    content = "材料 性质\n\n催化 应用".encode()
    source.write_bytes(content)

    prepared = IngestionPipeline().prepare(
        source,
        doc_id="doc_001",
        title="来源",
        material_tags=["SrNi"],
    )

    assert prepared.sha256 == hashlib.sha256(content).hexdigest()
    assert prepared.pages_total == 1
    assert prepared.pages_extracted == 1
    assert prepared.chunks
    assert prepared.chunks[0].doc_id == "doc_001"

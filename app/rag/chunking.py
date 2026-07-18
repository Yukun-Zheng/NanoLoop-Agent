"""Page-aware, paragraph-first chunking for material knowledge documents."""

from __future__ import annotations

import re
from dataclasses import dataclass

_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*$")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？.!?；;])\s*")


class ChunkLimitExceededError(ValueError):
    """Raised before a chunker can materialize an unbounded document."""

    def __init__(self, *, limit: int) -> None:
        super().__init__(f"knowledge document exceeds the {limit} chunk limit")
        self.limit = limit


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """Text extracted from one original, one-based document page."""

    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class KnowledgeChunkRecord:
    """Persistence-neutral chunk produced before DB and vector indexing."""

    chunk_id: str
    doc_id: str
    title: str
    page_start: int
    page_end: int
    section_title: str | None
    text: str
    material_tags: tuple[str, ...]


class ParagraphChunker:
    """Create deterministic chunks while preferring paragraph/sentence boundaries."""

    def __init__(self, *, target_chars: int = 600, overlap_chars: int = 80) -> None:
        if target_chars <= 0:
            raise ValueError("target_chars must be positive")
        if overlap_chars < 0 or overlap_chars >= target_chars:
            raise ValueError("overlap_chars must be non-negative and smaller than target_chars")
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars

    def chunk_pages(
        self,
        pages: list[ExtractedPage],
        *,
        doc_id: str,
        title: str,
        material_tags: list[str] | tuple[str, ...] = (),
        max_chunks: int | None = None,
    ) -> list[KnowledgeChunkRecord]:
        if max_chunks is not None and max_chunks <= 0:
            raise ValueError("max_chunks must be positive when configured")
        chunks: list[KnowledgeChunkRecord] = []
        tags = tuple(dict.fromkeys(tag.strip() for tag in material_tags if tag.strip()))
        for page in pages:
            if page.page_number < 1:
                raise ValueError("page numbers are one-based")
            remaining = None if max_chunks is None else max_chunks - len(chunks)
            try:
                page_chunks = self._chunk_page(page.text, max_chunks=remaining)
            except ChunkLimitExceededError as error:
                raise ChunkLimitExceededError(
                    limit=max_chunks if max_chunks is not None else error.limit
                ) from error
            for index, (section_title, text) in enumerate(page_chunks, start=1):
                if max_chunks is not None and len(chunks) >= max_chunks:
                    raise ChunkLimitExceededError(limit=max_chunks)
                chunks.append(
                    KnowledgeChunkRecord(
                        chunk_id=f"{doc_id}_{page.page_number:04d}_{index:03d}",
                        doc_id=doc_id,
                        title=title,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        section_title=section_title,
                        text=text,
                        material_tags=tags,
                    )
                )
        return chunks

    def _chunk_page(
        self,
        text: str,
        *,
        max_chunks: int | None = None,
    ) -> list[tuple[str | None, str]]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []

        output: list[tuple[str | None, str]] = []
        buffer = ""
        buffer_section: str | None = None
        current_section: str | None = None

        def append_output(section: str | None, content: str) -> None:
            if max_chunks is not None and len(output) >= max_chunks:
                raise ChunkLimitExceededError(limit=max_chunks)
            output.append((section, content))

        def emit() -> str:
            nonlocal buffer, buffer_section
            emitted = buffer.strip()
            if emitted:
                append_output(buffer_section, emitted)
            buffer = ""
            buffer_section = None
            return emitted

        for paragraph in _PARAGRAPH_BREAK.split(normalized):
            cleaned = "\n".join(line.rstrip() for line in paragraph.strip().splitlines())
            if not cleaned:
                continue
            heading = _MARKDOWN_HEADING.match(cleaned)
            if heading:
                if buffer:
                    emit()
                current_section = heading.group("title").strip()

            if len(cleaned) > self.target_chars and not _SENTENCE_BOUNDARY.search(cleaned):
                if buffer:
                    emit()
                for window in self._sliding_windows(cleaned):
                    append_output(current_section, window)
                continue

            for unit in self._split_unit(cleaned):
                if not buffer:
                    buffer = unit
                    buffer_section = current_section
                    continue
                separator = "\n\n"
                if len(buffer) + len(separator) + len(unit) <= self.target_chars:
                    buffer += separator + unit
                    continue

                previous = emit()
                available_overlap = max(self.target_chars - len(unit) - len(separator), 0)
                overlap = self._overlap_tail(previous, min(self.overlap_chars, available_overlap))
                buffer = f"{overlap}{separator}{unit}" if overlap else unit
                buffer_section = current_section

        if buffer:
            emit()
        return output

    def _split_unit(self, text: str) -> list[str]:
        if len(text) <= self.target_chars:
            return [text]

        sentences = [item.strip() for item in _SENTENCE_BOUNDARY.split(text) if item.strip()]
        if len(sentences) > 1:
            units: list[str] = []
            current = ""
            for sentence in sentences:
                if len(sentence) > self.target_chars:
                    if current:
                        units.append(current)
                        current = ""
                    units.extend(self._sliding_windows(sentence))
                elif not current:
                    current = sentence
                elif len(current) + len(sentence) <= self.target_chars:
                    current += " " + sentence
                else:
                    units.append(current)
                    current = sentence
            if current:
                units.append(current)
            return units
        return self._sliding_windows(text)

    def _sliding_windows(self, text: str) -> list[str]:
        step = self.target_chars - self.overlap_chars
        return [
            text[start : start + self.target_chars]
            for start in range(0, len(text), step)
            if text[start : start + self.target_chars]
        ]

    @staticmethod
    def _overlap_tail(text: str, length: int) -> str:
        if length <= 0:
            return ""
        tail = text[-length:]
        first_space = tail.find(" ")
        if 0 <= first_space < len(tail) // 3:
            tail = tail[first_space + 1 :]
        return tail.strip()

"""Build the NanoLoop Agent v4 developer handoff DOCX."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from docx.document import Document as DocumentObject
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt
from docx.text.paragraph import Paragraph
from docx.text.run import Run

import scripts.build_v3_handoff_doc as base

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "docs" / "NanoLoop_Agent_协同开发规格与接口总文档_v4.0.md"
OUTPUT_PATH = REPO_ROOT / "docs" / "NanoLoop_Agent_协同开发规格与接口总文档_v4.0.docx"
CJK_FONT = "Hiragino Sans GB"

_base_header_paragraph = base._header_paragraph
_base_set_metadata = base._set_metadata
_base_set_fonts = base._set_fonts
_base_style_paragraphs = base._style_paragraphs


def _set_fonts(
    run: Run, *, latin: str = base.LATIN_FONT, cjk: str = CJK_FONT
) -> None:
    _base_set_fonts(run, latin=latin, cjk=cjk)


def _header_paragraph(paragraph: Paragraph) -> None:
    _base_header_paragraph(paragraph)
    paragraph.runs[-1].text = "\tv4.0 · 2026-07-22"
    paragraph.runs[-1].font.size = Pt(8.5)
    paragraph.runs[-1].font.color.rgb = base._rgb(base.MUTED)
    base._set_fonts(paragraph.runs[-1])


def _set_metadata(doc: DocumentObject, timestamp: datetime) -> None:
    _base_set_metadata(doc, timestamp)
    props = doc.core_properties
    props.title = "NanoLoop Agent 协同开发规格与接口总文档 v4.0"
    props.subject = "真实资产接入、可演示 MVP 与 AI 协作执行手册"
    props.keywords = (
        "NanoLoop Agent, SEM, MVP, developer handoff, RAG, inference, acceptance"
    )


def _style_paragraphs(doc: DocumentObject) -> None:
    _base_style_paragraphs(doc)
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text == "协同开发规格与接口总文档 v4.0":
            paragraph.style = doc.styles["NanoLoop Cover Subtitle"]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
        elif text == "真实资产接入、可演示 MVP 与 AI 协作执行手册":
            paragraph.style = doc.styles["NanoLoop Cover Kicker"]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--pandoc", default=shutil.which("pandoc"))
    args = parser.parse_args()
    if not args.pandoc:
        parser.error("pandoc is required to build the handoff document")
    return args


def main() -> None:
    args = parse_args()
    base.CJK_FONT = CJK_FONT
    base._set_fonts = _set_fonts
    base._header_paragraph = _header_paragraph
    base._set_metadata = _set_metadata
    base._style_paragraphs = _style_paragraphs
    base.build(args.source.resolve(), args.output.resolve(), args.pandoc)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

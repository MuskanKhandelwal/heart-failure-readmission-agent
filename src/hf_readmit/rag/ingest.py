from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pdfplumber


@dataclass
class GuidelineChunk:
    """A single chunk of guideline text for retrieval."""

    chunk_id: str
    source_path: Path
    source_name: str
    text: str
    heading: Optional[str] = None
    page: Optional[int] = None

    def to_metadata(self) -> dict[str, str | int | None]:
        return {
            "chunk_id": self.chunk_id,
            "source_name": self.source_name,
            "source_path": str(self.source_path),
            "heading": self.heading or "",
            "page": self.page,
        }


def find_guideline_pdfs(source_dir: Path) -> list[Path]:
    """Find guideline PDF files under a source directory."""
    return sorted([p for p in source_dir.glob("**/*.pdf") if p.is_file()])


PAGE_BREAK_MARKER = "\n\n---PAGE_BREAK---\n\n"


def extract_pdf_text(source_path: Path) -> str:
    """Extract text from a PDF guideline using pdfplumber."""
    pages: list[str] = []
    with pdfplumber.open(source_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(layout=True) or ""
            if text.strip():
                pages.append(text)
    return PAGE_BREAK_MARKER.join(pages)


def _split_long_page(text: str, max_chars: int = 3000) -> list[str]:
    if len(text) <= max_chars:
        return [text.strip()]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        paragraph_len = len(paragraph)
        if paragraph_len > max_chars:
            if current:
                prefix = "\n\n".join(current).strip()
                available = max_chars - len(prefix) - (2 if prefix else 0)
                if available > 0:
                    first_part = paragraph[:available].strip()
                    chunks.append(f"{prefix}\n\n{first_part}" if prefix else first_part)
                    start = available
                else:
                    chunks.append(prefix)
                    start = 0
                current = []
                current_len = 0
            else:
                start = 0

            while start < paragraph_len:
                end = min(start + max_chars, paragraph_len)
                chunks.append(paragraph[start:end].strip())
                start = end
            continue

        if current_len + paragraph_len + (2 if current else 0) <= max_chars:
            current.append(paragraph)
            current_len += paragraph_len + (2 if current else 0)
            continue

        chunks.append("\n\n".join(current).strip())
        current = [paragraph]
        current_len = paragraph_len

    if current:
        chunks.append("\n\n".join(current).strip())

    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < 200:
            merged[-1] = f"{merged[-1]}\n\n{chunk}"
        else:
            merged.append(chunk)

    return [chunk for chunk in merged if chunk]


def chunk_guideline_text(text: str) -> list[str]:
    """Split extracted PDF text into page-based chunks."""
    if not text.strip():
        return []

    pages = [page.strip() for page in text.split("\n\n---PAGE_BREAK---\n\n") if page.strip()]
    chunks: list[str] = []

    for page in pages:
        if len(page) <= 3000:
            chunks.append(page)
            continue
        chunks.extend(_split_long_page(page, max_chars=3000))

    return [chunk for chunk in chunks if len(chunk) >= 200]


def build_guideline_corpus(source_dir: Path) -> list[GuidelineChunk]:
    """Build a corpus of guideline chunks from a directory of PDF files."""
    pdf_paths = find_guideline_pdfs(source_dir)
    chunks: list[GuidelineChunk] = []

    for source_path in pdf_paths:
        pdf_text = extract_pdf_text(source_path)
        pages = [page.strip() for page in pdf_text.split(PAGE_BREAK_MARKER) if page.strip()]

        for page_num, page_text in enumerate(pages, start=1):
            page_chunks = chunk_guideline_text(page_text)
            for chunk_index, chunk_text in enumerate(page_chunks, start=1):
                chunks.append(
                    GuidelineChunk(
                        chunk_id=f"{source_path.stem}_p{page_num}_{chunk_index}",
                        source_path=source_path,
                        source_name=source_path.name,
                        text=chunk_text,
                        heading=None,
                        page=page_num,
                    )
                )

    return chunks

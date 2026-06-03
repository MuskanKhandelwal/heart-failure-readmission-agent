from __future__ import annotations

from pathlib import Path

import pytest

from hf_readmit.rag.bm25_index import BM25Index
from hf_readmit.rag.ingest import GuidelineChunk, chunk_guideline_text, extract_pdf_text


def test_chunk_guideline_text_splits_long_text() -> None:
    page_text = "First paragraph.\n\n" + "word " * 3100 + "\n\nSecond paragraph."
    chunks = chunk_guideline_text(page_text)

    assert len(chunks) >= 2
    assert all(isinstance(chunk, str) and len(chunk) >= 200 for chunk in chunks)
    assert any("First paragraph" in chunk for chunk in chunks)


def test_extract_pdf_text_returns_text_from_pdf(tmp_path: Path) -> None:
    from pypdf import PdfWriter

    pdf_path = tmp_path / "sample.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with pdf_path.open("wb") as f:
        writer.write(f)

    result = extract_pdf_text(pdf_path)
    assert isinstance(result, str)
    assert result == ""


def test_bm25_index_returns_relevant_chunk() -> None:
    chunks = [
        GuidelineChunk(chunk_id="a-0", source_path=Path("guidelines/a.pdf"), source_name="a.pdf", text="Heart failure guideline overview."),
        GuidelineChunk(chunk_id="a-1", source_path=Path("guidelines/a.pdf"), source_name="a.pdf", text="Medication management and discharge planning."),
        GuidelineChunk(chunk_id="b-0", source_path=Path("guidelines/b.pdf"), source_name="b.pdf", text="Monitoring fluid status in heart failure."),
    ]
    index = BM25Index(chunks)
    hits = index.query("discharge planning for heart failure", top_k=2)

    assert hits
    assert hits[0].chunk_id == "a-1"
    assert any("heart failure" in hit.document.lower() for hit in hits)


def test_build_guideline_corpus_returns_chunks_for_each_page(tmp_path: Path) -> None:
    from pypdf import PdfWriter

    pdf_path = tmp_path / "sample.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with pdf_path.open("wb") as f:
        writer.write(f)

    # A blank PDF is valid but contains no text, so corpus should be empty.
    from hf_readmit.rag.ingest import build_guideline_corpus

    corpus = build_guideline_corpus(tmp_path)
    assert corpus == []


def test_bm25_index_can_save_and_load(tmp_path: Path) -> None:
    chunks = [
        GuidelineChunk(chunk_id="c-0", source_path=Path("guidelines/c.pdf"), source_name="c.pdf", text="Sample guideline text."),
    ]
    index = BM25Index(chunks)
    path = tmp_path / "bm25_index.pkl"
    index.save(path)

    loaded = BM25Index.load(path)
    assert loaded.ids == index.ids
    assert loaded.documents == index.documents
    assert loaded.metadatas == index.metadatas

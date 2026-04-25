import io
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from citverif.memory.chunker import chunk_pdf, Chunk, _CHUNK_CHARS, _OVERLAP_CHARS


def _make_fake_pdf(pages: list[str]):
    """Return a mock that makes fitz.open() yield given page texts."""
    mock_pages = []
    for text in pages:
        p = MagicMock()
        p.get_text.return_value = text
        mock_pages.append(p)

    mock_doc = MagicMock()
    mock_doc.__iter__ = lambda self: iter(mock_pages)
    mock_doc.__len__ = lambda self: len(mock_pages)
    return mock_doc


def _patch_fitz(pages: list[str]):
    mock_doc = _make_fake_pdf(pages)
    return patch("citverif.memory.chunker.fitz.open", return_value=mock_doc)


def test_single_short_page_single_chunk():
    with _patch_fitz(["This is a short paper abstract.\n\nIt has two paragraphs."]):
        chunks = chunk_pdf(Path("fake.pdf"), "paper-1")
    assert len(chunks) == 1
    assert chunks[0].paper_id == "paper-1"
    assert chunks[0].page == 1
    assert "abstract" in chunks[0].text.lower()


def test_chunk_indices_are_sequential():
    long_para = "word " * 600  # ~600 words, well over one chunk
    with _patch_fitz([long_para]):
        chunks = chunk_pdf(Path("fake.pdf"), "paper-x")
    assert len(chunks) >= 2
    for i, c in enumerate(chunks):
        assert c.chunk_idx == i


def test_overlap_between_adjacent_chunks():
    long_para = "alpha " * 700
    with _patch_fitz([long_para]):
        chunks = chunk_pdf(Path("fake.pdf"), "paper-y")
    if len(chunks) >= 2:
        tail = chunks[0].text[-_OVERLAP_CHARS:]
        head = chunks[1].text[:_OVERLAP_CHARS]
        # Some overlap content must appear in the next chunk
        assert tail.strip()[:20] in chunks[1].text


def test_section_label_detected():
    page = "Introduction\n\nWe propose a new method.\n\nMethod\n\nOur approach uses X."
    with _patch_fitz([page]):
        chunks = chunk_pdf(Path("fake.pdf"), "paper-z")
    sections = {c.section for c in chunks if c.section}
    assert any("Introduction" in s or "Method" in s for s in sections)


def test_empty_pdf_returns_no_chunks():
    with _patch_fitz([""]):
        chunks = chunk_pdf(Path("fake.pdf"), "paper-empty")
    assert chunks == []


def test_multipage_page_numbers():
    pages = ["Page one content.\n\nMore text.", "Page two content.\n\nMore text."]
    with _patch_fitz(pages):
        chunks = chunk_pdf(Path("fake.pdf"), "paper-mp")
    page_nums = {c.page for c in chunks}
    assert 1 in page_nums

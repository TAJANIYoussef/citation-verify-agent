import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # pymupdf


@dataclass
class Chunk:
    text: str
    page: int          # 1-indexed
    section: str | None
    chunk_idx: int     # position in the document chunk sequence
    paper_id: str


_SECTION_RE = re.compile(
    r"^\s*(?:\d+\.?\s+)?(?:Abstract|Introduction|Background|Related Work|"
    r"Method(?:ology)?|Approach|Experiment[s]?|Result[s]?|Discussion|"
    r"Conclusion[s]?|Reference[s]?|Appendix)\b",
    re.IGNORECASE | re.MULTILINE,
)

# Rough token estimator: 1 token ≈ 4 chars for English/Latin academic text
_CHARS_PER_TOKEN = 4
_CHUNK_TOKENS = 512
_OVERLAP_TOKENS = 64
_CHUNK_CHARS = _CHUNK_TOKENS * _CHARS_PER_TOKEN    # 2048
_OVERLAP_CHARS = _OVERLAP_TOKENS * _CHARS_PER_TOKEN  # 256


def _extract_text_by_page(pdf_path: Path) -> list[tuple[int, str]]:
    """Return list of (page_number_1indexed, page_text)."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append((i + 1, text))
    doc.close()
    return pages


def _detect_section(text: str, current: str | None) -> str | None:
    """Return updated section label if a section header is found in text."""
    m = _SECTION_RE.search(text)
    if m:
        return m.group(0).strip()
    return current


def chunk_pdf(pdf_path: Path, paper_id: str) -> list[Chunk]:
    """
    Extract text from a PDF and split into overlapping chunks.
    Chunks do not cross section boundaries when a section header is detected.
    Returns chunks with page number and section label.
    """
    pages = _extract_text_by_page(pdf_path)
    if not pages:
        return []

    chunks: list[Chunk] = []
    current_section: str | None = None
    chunk_idx = 0

    # Build a flat list of (page, paragraph) segments first
    segments: list[tuple[int, str, str | None]] = []
    for page_num, page_text in pages:
        # Split page into paragraphs on double newlines
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", page_text) if p.strip()]
        for para in paragraphs:
            current_section = _detect_section(para, current_section)
            segments.append((page_num, para, current_section))

    # Slide a window over segments, respecting section breaks
    buffer = ""
    buffer_page = 1
    buffer_section: str | None = None
    prev_section: str | None = None

    def flush(buf: str, pg: int, sec: str | None) -> None:
        nonlocal chunk_idx
        if buf.strip():
            chunks.append(Chunk(
                text=buf.strip(),
                page=pg,
                section=sec,
                chunk_idx=chunk_idx,
                paper_id=paper_id,
            ))
            chunk_idx += 1

    for page_num, para, section in segments:
        # Section break → flush immediately (don't mix sections)
        if section != prev_section and prev_section is not None and buffer:
            flush(buffer, buffer_page, prev_section)
            # Keep overlap from end of previous buffer
            buffer = buffer[-_OVERLAP_CHARS:] if len(buffer) > _OVERLAP_CHARS else buffer
            buffer_page = page_num
            buffer_section = section

        if not buffer:
            buffer_page = page_num
            buffer_section = section

        buffer += ("\n\n" if buffer else "") + para
        prev_section = section

        # Flush when chunk is full
        while len(buffer) >= _CHUNK_CHARS:
            flush(buffer[:_CHUNK_CHARS], buffer_page, buffer_section)
            buffer = buffer[_CHUNK_CHARS - _OVERLAP_CHARS:]
            buffer_page = page_num
            buffer_section = section

    flush(buffer, buffer_page, buffer_section)
    return chunks

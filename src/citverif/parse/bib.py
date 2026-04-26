import re
from dataclasses import dataclass
from pathlib import Path

import bibtexparser
from bibtexparser.middlewares import MonthIntMiddleware

try:
    from bibtexparser.middlewares import ResolveStringReferencesMiddleware as _ResolveStrings
except ImportError:
    from bibtexparser.middlewares import ResolveStringReferences as _ResolveStrings  # type: ignore[no-redef]


@dataclass
class BibEntry:
    cite_key: str
    title: str
    authors: list[str]
    year: str | None
    doi: str | None
    arxiv_id: str | None
    url: str | None
    raw: dict  # raw field values as strings


_ARXIV_FROM_URL = re.compile(
    r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", re.IGNORECASE
)
_ARXIV_BARE = re.compile(r"\b([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)\b")
_DOI_CLEAN = re.compile(r"^https?://(?:dx\.)?doi\.org/", re.IGNORECASE)


def _extract_arxiv(raw: dict) -> str | None:
    for field in ("eprint", "arxivid", "arxiv"):
        val = str(raw.get(field, "")).strip()
        if val:
            m = _ARXIV_BARE.search(val)
            if m:
                return m.group(1)
    for field in ("url", "howpublished", "note"):
        val = str(raw.get(field, "")).strip()
        m = _ARXIV_FROM_URL.search(val)
        if m:
            return m.group(1)
    return None


def _extract_doi(raw: dict) -> str | None:
    doi = str(raw.get("doi", "")).strip()
    if not doi:
        return None
    return _DOI_CLEAN.sub("", doi).strip() or None


def _extract_authors(raw: dict) -> list[str]:
    val = str(raw.get("author", "")).strip()
    if not val:
        return []
    return [a.strip() for a in re.split(r"\s+and\s+", val, flags=re.IGNORECASE)]


def _raw_fields(entry) -> dict:
    """Flatten bibtexparser v2 beta entry fields to {key: str_value}."""
    try:
        # v2 beta exposes fields_dict
        return {k: str(v) for k, v in entry.fields_dict.items()}
    except AttributeError:
        # fallback for slight API variation across betas
        return {f.key: str(f.value) for f in entry.fields}


def parse_bib(path: Path) -> list[BibEntry]:
    library = bibtexparser.parse_file(
        str(path),
        append_middleware=[_ResolveStrings(), MonthIntMiddleware()],
    )
    entries: list[BibEntry] = []
    for entry in library.entries:
        raw = _raw_fields(entry)
        entries.append(BibEntry(
            cite_key=entry.key,
            title=raw.get("title", "").strip("{}"),
            authors=_extract_authors(raw),
            year=raw.get("year"),
            doi=_extract_doi(raw),
            arxiv_id=_extract_arxiv(raw),
            url=raw.get("url", "").strip() or None,
            raw=raw,
        ))
    return entries

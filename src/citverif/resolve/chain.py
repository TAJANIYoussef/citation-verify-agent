import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import httpx

from citverif.memory.paper_cache import PaperCache
from citverif.parse.bib import BibEntry
from citverif.resolve.arxiv import resolve_arxiv
from citverif.resolve.openalex import resolve_openalex
from citverif.resolve.semantic_scholar import resolve_s2_by_doi, resolve_s2_by_title
from citverif.resolve.unpaywall import resolve_unpaywall

log = logging.getLogger(__name__)


@dataclass
class ResolvedRef:
    cite_key: str
    pdf_path: Path | None
    abstract: str | None
    # "arxiv" | "openalex" | "s2" | "unpaywall" | "web" | None
    source: str | None
    # True when we only have abstract — caps agent confidence at 0.7
    abstract_only: bool = False
    error: str | None = None


async def _download_pdf(url: str, cache: PaperCache, paper_id: str) -> Path | None:
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            r = await client.get(url)
            if r.status_code == 200 and b"%PDF" in r.content[:8]:
                return cache.save_pdf_bytes(paper_id, r.content)
    except Exception as exc:
        log.debug("PDF download failed for %s: %s", url, exc)
    return None


async def resolve_one(entry: BibEntry, cache: PaperCache) -> ResolvedRef:
    key = entry.cite_key

    # ── Step 1: arXiv ID ────────────────────────────────────────────────────
    if entry.arxiv_id:
        pdf = await resolve_arxiv(entry.arxiv_id, cache)
        if pdf:
            log.info("[%s] resolved via arXiv (%s)", key, entry.arxiv_id)
            return ResolvedRef(cite_key=key, pdf_path=pdf, abstract=None, source="arxiv")

    # ── Step 2: DOI → OpenAlex ──────────────────────────────────────────────
    if entry.doi:
        oa = await resolve_openalex(entry.doi)
        if oa:
            pdf_path = None
            if oa.oa_pdf_url:
                pdf_path = await _download_pdf(oa.oa_pdf_url, cache, entry.doi)
            if pdf_path:
                log.info("[%s] resolved via OpenAlex PDF", key)
                return ResolvedRef(cite_key=key, pdf_path=pdf_path, abstract=oa.abstract, source="openalex")
            if oa.abstract:
                log.info("[%s] OpenAlex: abstract only", key)
                # Continue to try to get a PDF, but keep the abstract
                abstract_fallback = oa.abstract
            else:
                abstract_fallback = None

            # ── Step 4: Unpaywall (we have DOI, no PDF yet) ─────────────────
            up_url = await resolve_unpaywall(entry.doi)
            if up_url:
                pdf_path = await _download_pdf(up_url, cache, entry.doi)
                if pdf_path:
                    log.info("[%s] resolved via Unpaywall", key)
                    return ResolvedRef(cite_key=key, pdf_path=pdf_path, abstract=abstract_fallback, source="unpaywall")

            # ── Step 2b: DOI → S2 (may have OA PDF or better abstract) ──────
            s2 = await resolve_s2_by_doi(entry.doi)
            if s2:
                if s2.open_access_pdf:
                    pdf_path = await _download_pdf(s2.open_access_pdf, cache, entry.doi)
                    if pdf_path:
                        log.info("[%s] resolved via S2 PDF", key)
                        return ResolvedRef(cite_key=key, pdf_path=pdf_path, abstract=s2.abstract or abstract_fallback, source="s2")
                abstract_fallback = abstract_fallback or s2.abstract

            # If arXiv ID found via S2, try that
            if s2 and s2.arxiv_id:
                pdf = await resolve_arxiv(s2.arxiv_id, cache)
                if pdf:
                    log.info("[%s] resolved via S2→arXiv", key)
                    return ResolvedRef(cite_key=key, pdf_path=pdf, abstract=abstract_fallback, source="arxiv")

            if abstract_fallback:
                cache.save_abstract(entry.doi, abstract_fallback)
                log.warning("[%s] abstract-only (no PDF found)", key)
                return ResolvedRef(cite_key=key, pdf_path=None, abstract=abstract_fallback, source="openalex", abstract_only=True)

    # ── Step 3: title → S2 fuzzy match ──────────────────────────────────────
    if entry.title:
        s2 = await resolve_s2_by_title(entry.title)
        if s2:
            pdf_path = None
            if s2.open_access_pdf:
                pid = s2.arxiv_id or s2.doi or entry.cite_key
                pdf_path = await _download_pdf(s2.open_access_pdf, cache, pid)
            if pdf_path:
                log.info("[%s] resolved via S2 title search", key)
                return ResolvedRef(cite_key=key, pdf_path=pdf_path, abstract=s2.abstract, source="s2")
            if s2.arxiv_id:
                pdf = await resolve_arxiv(s2.arxiv_id, cache)
                if pdf:
                    log.info("[%s] resolved via S2 title→arXiv", key)
                    return ResolvedRef(cite_key=key, pdf_path=pdf, abstract=s2.abstract, source="arxiv")
            if s2.abstract:
                pid = s2.doi or entry.cite_key
                cache.save_abstract(pid, s2.abstract)
                log.warning("[%s] abstract-only via S2 title search", key)
                return ResolvedRef(cite_key=key, pdf_path=None, abstract=s2.abstract, source="s2", abstract_only=True)

    log.error("[%s] unresolvable — no PDF or abstract found", key)
    return ResolvedRef(cite_key=key, pdf_path=None, abstract=None, source=None,
                       error="no PDF or abstract found")


async def resolve_all(
    entries: list[BibEntry],
    cache: PaperCache,
    concurrency: int = 4,
) -> list[ResolvedRef]:
    sem = asyncio.Semaphore(concurrency)

    async def bounded(entry: BibEntry) -> ResolvedRef:
        async with sem:
            return await resolve_one(entry, cache)

    return await asyncio.gather(*[bounded(e) for e in entries])

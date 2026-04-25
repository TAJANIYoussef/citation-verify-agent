import os
from dataclasses import dataclass

import httpx


@dataclass
class S2Result:
    title: str | None
    abstract: str | None
    arxiv_id: str | None
    doi: str | None
    open_access_pdf: str | None


_BASE = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "title,abstract,externalIds,openAccessPdf"


def _headers() -> dict:
    key = os.getenv("S2_API_KEY", "")
    return {"x-api-key": key} if key else {}


async def resolve_s2_by_title(title: str, threshold: float = 0.85) -> S2Result | None:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                f"{_BASE}/paper/search",
                params={"query": title, "fields": _FIELDS, "limit": 1},
                headers=_headers(),
            )
            if r.status_code != 200:
                return None
            data = r.json()
            papers = data.get("data", [])
            if not papers:
                return None
            paper = papers[0]

        # Fuzzy title match guard
        from difflib import SequenceMatcher
        candidate_title = (paper.get("title") or "").lower()
        if SequenceMatcher(None, title.lower(), candidate_title).ratio() < threshold:
            return None

        ext = paper.get("externalIds", {})
        oa = paper.get("openAccessPdf") or {}
        return S2Result(
            title=paper.get("title"),
            abstract=paper.get("abstract"),
            arxiv_id=ext.get("ArXiv"),
            doi=ext.get("DOI"),
            open_access_pdf=oa.get("url"),
        )
    except Exception:
        return None


async def resolve_s2_by_doi(doi: str) -> S2Result | None:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                f"{_BASE}/paper/DOI:{doi}",
                params={"fields": _FIELDS},
                headers=_headers(),
            )
            if r.status_code != 200:
                return None
            paper = r.json()

        ext = paper.get("externalIds", {})
        oa = paper.get("openAccessPdf") or {}
        return S2Result(
            title=paper.get("title"),
            abstract=paper.get("abstract"),
            arxiv_id=ext.get("ArXiv"),
            doi=doi,
            open_access_pdf=oa.get("url"),
        )
    except Exception:
        return None

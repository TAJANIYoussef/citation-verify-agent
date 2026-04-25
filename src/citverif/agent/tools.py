from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

import httpx
from pydantic_ai import RunContext

from citverif.memory.vector_store import VectorStore

log = logging.getLogger(__name__)

_DDGO_URL = "https://html.duckduckgo.com/html/"
_SNIPPET_RE = re.compile(r'class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class VerifierDeps:
    store: VectorStore
    paper_id: str
    abstract: str | None = None   # set when PDF unavailable


# ── Tool functions ────────────────────────────────────────────────────────────
# Each takes RunContext[VerifierDeps] as first arg — pydantic-ai injects deps.

async def semantic_search(ctx: RunContext[VerifierDeps], query: str, k: int = 5) -> str:
    """
    Search the cited paper's text for passages most relevant to the query.
    Returns the top-k passages with their section and page number.
    If only an abstract is available, returns that instead.
    """
    deps = ctx.deps

    # Abstract-only fallback
    if deps.abstract and not deps.store.has_paper(deps.paper_id):
        log.debug("[tool:semantic_search] abstract-only for %s", deps.paper_id)
        return (
            f"[Abstract only — no full text available]\n\n{deps.abstract}\n\n"
            "Note: confidence must be capped at 0.7 for abstract-only verification."
        )

    results = deps.store.search(query, deps.paper_id, k=k)
    if not results:
        return "No relevant passages found in the cited paper."

    lines = []
    for i, r in enumerate(results, 1):
        loc = f"p.{r.page}" + (f" §{r.section}" if r.section else "")
        lines.append(f"[{i}] ({loc})\n{r.text}")
    return "\n\n---\n\n".join(lines)


async def fetch_section(ctx: RunContext[VerifierDeps], section_name: str) -> str:
    """
    Retrieve all text from a specific named section of the cited paper
    (e.g. 'Results', 'Experiments', 'Conclusion').
    Useful when you know which section should contain the claim.
    """
    deps = ctx.deps
    results = deps.store.search(
        query=section_name,
        paper_id=deps.paper_id,
        k=20,
    )
    # Filter to chunks whose section matches (case-insensitive prefix)
    section_lower = section_name.lower().strip()
    matching = [
        r for r in results
        if r.section and section_lower in r.section.lower()
    ]
    if not matching:
        # Fall back to top semantic result if no section label match
        matching = results[:3]

    if not matching:
        return f"Section '{section_name}' not found in the cited paper."

    lines = [f"[p.{r.page}] {r.text}" for r in matching]
    return f"=== {section_name} ===\n\n" + "\n\n---\n\n".join(lines)


async def web_search(ctx: RunContext[VerifierDeps], query: str) -> str:
    """
    Search the web for information about the claim or cited paper.
    Use this when the paper is unavailable or to find corroborating sources.
    Results are snippets only — do not treat them as ground truth.
    """
    try:
        async with httpx.AsyncClient(
            timeout=10,
            headers={"User-Agent": "citverif/0.1 citation-verification-research"},
            follow_redirects=True,
        ) as client:
            r = await client.post(_DDGO_URL, data={"q": query, "b": ""})

        snippets = _SNIPPET_RE.findall(r.text)
        cleaned = [_TAG_RE.sub("", s).strip() for s in snippets[:5] if s.strip()]

        if not cleaned:
            return "No web results found."

        return "Web search results (snippets only — verify against primary sources):\n\n" + \
               "\n\n".join(f"• {s}" for s in cleaned)

    except Exception as exc:
        log.warning("[tool:web_search] failed: %s", exc)
        return f"Web search failed: {exc}"

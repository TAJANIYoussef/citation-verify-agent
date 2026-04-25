"""
Integration tests for the resolution chain.
These hit real external APIs — run with: pytest tests/integration/ -v
"""
import asyncio
from pathlib import Path
import pytest

from citverif.parse.bib import parse_bib
from citverif.resolve.chain import resolve_all
from citverif.memory.paper_cache import PaperCache

TEST_PAPERS = Path(__file__).parent.parent.parent / ".context" / "test-papers"
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "papers"


@pytest.fixture
def cache(tmp_path):
    return PaperCache(tmp_path)


@pytest.mark.asyncio
async def test_paper1_resolution_rate(cache):
    entries = parse_bib(TEST_PAPERS / "paper1.bib")
    results = await resolve_all(entries, cache)

    resolved = [r for r in results if r.pdf_path or r.abstract]
    unresolvable = [r for r in results if r.cite_key == "broken2099"]

    # vaswani (arXiv) and devlin (DOI) should resolve; broken2099 should not
    rate = len(resolved) / len(results)
    assert rate >= 0.6, f"Resolution rate too low: {rate:.0%}"

    assert any(r.cite_key == "broken2099" and r.source is None for r in results), \
        "broken2099 should be unresolvable"


@pytest.mark.asyncio
async def test_arxiv_resolves(cache):
    entries = parse_bib(TEST_PAPERS / "paper1.bib")
    vaswani = next(e for e in entries if e.cite_key == "vaswani2017attention")
    from citverif.resolve.chain import resolve_one
    result = await resolve_one(vaswani, cache)
    assert result.pdf_path is not None or result.abstract is not None
    assert result.source in ("arxiv", "openalex", "s2")


@pytest.mark.asyncio
async def test_paper2_resolution_rate(cache):
    entries = parse_bib(TEST_PAPERS / "paper2.bib")
    results = await resolve_all(entries, cache)
    resolved = [r for r in results if r.pdf_path or r.abstract]
    rate = len(resolved) / len(results)
    assert rate >= 0.5, f"Resolution rate too low: {rate:.0%}"

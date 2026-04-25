from pathlib import Path

import arxiv

from citverif.memory.paper_cache import PaperCache


async def resolve_arxiv(arxiv_id: str, cache: PaperCache) -> Path | None:
    """Download PDF for an arXiv ID, return local path (or None on failure)."""
    cached = cache.get_pdf(arxiv_id)
    if cached:
        return cached

    try:
        client = arxiv.Client()
        results = list(client.results(arxiv.Search(id_list=[arxiv_id])))
        if not results:
            return None
        paper = results[0]
        dest = cache.pdf_path(arxiv_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        paper.download_pdf(dirpath=str(dest.parent), filename=dest.name)
        return dest if dest.exists() else None
    except Exception:
        return None

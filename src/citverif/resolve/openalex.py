from dataclasses import dataclass

import httpx


@dataclass
class OpenAlexResult:
    title: str | None
    abstract: str | None
    oa_pdf_url: str | None
    doi: str | None


async def resolve_openalex(doi: str) -> OpenAlexResult | None:
    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url, params={"mailto": "citverif"})
            if r.status_code != 200:
                return None
            data = r.json()

        abstract = _reconstruct_abstract(data.get("abstract_inverted_index"))
        oa_url = None
        oa = data.get("open_access", {})
        if oa.get("is_oa"):
            oa_url = oa.get("oa_url")

        return OpenAlexResult(
            title=data.get("title"),
            abstract=abstract,
            oa_pdf_url=oa_url,
            doi=doi,
        )
    except Exception:
        return None


def _reconstruct_abstract(inverted: dict | None) -> str | None:
    if not inverted:
        return None
    positions: list[tuple[int, str]] = []
    for word, indices in inverted.items():
        for i in indices:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)

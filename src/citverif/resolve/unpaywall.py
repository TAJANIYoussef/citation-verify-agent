import os

import httpx


async def resolve_unpaywall(doi: str) -> str | None:
    """Return OA PDF URL from Unpaywall, or None."""
    email = os.getenv("UNPAYWALL_EMAIL", "")
    if not email:
        return None
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                f"https://api.unpaywall.org/v2/{doi}",
                params={"email": email},
            )
            if r.status_code != 200:
                return None
            data = r.json()
        best = data.get("best_oa_location") or {}
        return best.get("url_for_pdf") or best.get("url")
    except Exception:
        return None

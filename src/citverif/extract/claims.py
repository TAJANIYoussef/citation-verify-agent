from __future__ import annotations

import logging
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.ollama import OllamaProvider

from citverif.parse.tex import CitationContext

log = logging.getLogger(__name__)

_OLLAMA_BASE = "http://localhost:11434/v1"
_MODEL = "llama3.1:8b"

_SYSTEM_PROMPT = """\
You are a scientific claim extractor. Given a passage from a research paper that contains \
a citation marker, your job is to extract the specific factual claim that the citation is \
meant to support.

Rules:
- Output a single, standalone, atomic claim — one sentence.
- Write it in third-person neutral ("The method achieves...", "X outperforms...").
- Remove hedging from the source paper's voice ("we show", "in this work").
- If the passage is not making a verifiable factual claim (e.g. it's a definition, \
a method description, or "as discussed in [X]"), set is_verifiable to false.
- Never fabricate details not present in the passage.
- Keep the claim concise: 10–40 words.
"""


class RefinedClaim(BaseModel):
    claim: str
    is_verifiable: bool


def _make_agent() -> Agent[None, RefinedClaim]:
    model = OpenAIModel(
        _MODEL,
        provider=OllamaProvider(base_url=_OLLAMA_BASE),
    )
    return Agent(
        model,
        output_type=RefinedClaim,
        system_prompt=_SYSTEM_PROMPT,
    )


# Module-level singleton — one agent per process
_agent: Agent[None, RefinedClaim] | None = None


def _get_agent() -> Agent[None, RefinedClaim]:
    global _agent
    if _agent is None:
        _agent = _make_agent()
    return _agent


async def refine_claim(ctx: CitationContext) -> RefinedClaim:
    """
    Use llama3.1:8b to turn raw citation context into a clean atomic claim.
    Falls back to a passthrough RefinedClaim on any model error.
    """
    user_prompt = (
        f"Passage: {ctx.claim_context}\n\n"
        f"The citation key is: {ctx.cite_key}\n\n"
        "Extract the factual claim this citation is meant to support."
    )
    try:
        result = await _get_agent().run(user_prompt)
        return result.data
    except Exception as exc:
        log.warning("Claim refinement failed for %s: %s — using raw context", ctx.cite_key, exc)
        return RefinedClaim(claim=ctx.claim_context[:300], is_verifiable=True)


async def refine_all(contexts: list[CitationContext]) -> list[tuple[CitationContext, RefinedClaim]]:
    """Refine claims sequentially (llama3.1:8b on CPU is single-threaded anyway)."""
    out = []
    for ctx in contexts:
        refined = await refine_claim(ctx)
        if refined.is_verifiable:
            log.info("[%s] claim: %s", ctx.cite_key, refined.claim[:80])
        else:
            log.debug("[%s] non-verifiable, skipping agent", ctx.cite_key)
        out.append((ctx, refined))
    return out

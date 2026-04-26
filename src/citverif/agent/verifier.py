from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.usage import UsageLimits

from citverif.agent.prompts import MAIN_VERIFIER_PROMPT
from citverif.agent.tools import VerifierDeps, fetch_section, semantic_search, web_search
from citverif.extract.claims import RefinedClaim
from citverif.memory.vector_store import VectorStore
from citverif.parse.tex import CitationContext
from citverif.resolve.chain import ResolvedRef
from citverif.schema import AgentResult, CitationVerdict, EvidenceSpan, Verdict

log = logging.getLogger(__name__)

_OLLAMA_BASE = "http://localhost:11434/v1"
_MODEL = "qwen2.5:14b"
_MAIN_LIMITS = UsageLimits(request_limit=5)  # 4 tool calls + 1 final response

# Alias so existing test imports of _AgentResult still resolve
_AgentResult = AgentResult


# ── Agent factory ─────────────────────────────────────────────────────────────

def _make_verifier() -> Agent[VerifierDeps, _AgentResult]:
    model = OpenAIModel(
        _MODEL,
        provider=OllamaProvider(base_url=_OLLAMA_BASE),
    )
    return Agent(
        model,
        deps_type=VerifierDeps,
        output_type=_AgentResult,
        system_prompt=MAIN_VERIFIER_PROMPT,
        tools=[semantic_search, fetch_section, web_search],
    )


_verifier: Agent[VerifierDeps, _AgentResult] | None = None


def _get_verifier() -> Agent[VerifierDeps, _AgentResult]:
    global _verifier
    if _verifier is None:
        _verifier = _make_verifier()
    return _verifier


# ── Main entry point ──────────────────────────────────────────────────────────

async def verify_citation(
    ctx: CitationContext,
    refined: RefinedClaim,
    ref: ResolvedRef,
    store: VectorStore,
) -> CitationVerdict:
    """
    Run the full verification pipeline for one citation:
      1. Main ReAct pass (max 4 tool calls).
      2. Reflexion pass when triggered (max 3 tool calls).
    """
    # Import here to avoid circular import at module load time
    from citverif.agent.reflexion import needs_reflexion, run_reflexion

    if not refined.is_verifiable:
        return CitationVerdict(
            cite_key=ctx.cite_key,
            claim=refined.claim,
            claim_context=ctx.claim_context,
            verdict="unverifiable",
            confidence=0.0,
            evidence=[],
            rationale="Claim was classified as non-verifiable (definition or method reference).",
            resolution_source=ref.source,
        )

    deps = VerifierDeps(
        store=store,
        paper_id=ctx.cite_key,
        abstract=ref.abstract,
    )

    user_prompt = (
        f"Claim: {refined.claim}\n\n"
        f"Context from source paper: {ctx.claim_context}\n\n"
        f"Cited paper ID: {ctx.cite_key}"
        + ("\n\nNote: Only the abstract is available — cap confidence at 0.7."
           if ref.abstract_only else "")
    )

    agent = _get_verifier()

    # ── Main pass ─────────────────────────────────────────────────────────
    try:
        main_run = await agent.run(
            user_prompt,
            deps=deps,
            usage_limits=_MAIN_LIMITS,
        )
        ar: _AgentResult = main_run.data
        tool_calls = max(0, main_run.usage().requests - 1)
    except Exception as exc:
        log.error("[%s] verifier agent failed: %s", ctx.cite_key, exc)
        return CitationVerdict(
            cite_key=ctx.cite_key,
            claim=refined.claim,
            claim_context=ctx.claim_context,
            verdict="unverifiable",
            confidence=0.0,
            evidence=[],
            rationale=f"Agent error: {exc}",
            resolution_source=ref.source,
        )

    # ── Reflexion pass ────────────────────────────────────────────────────
    if needs_reflexion(ar, tool_calls):
        log.info(
            "[%s] reflexion triggered (confidence=%.2f, tool_calls=%d, verdict=%s)",
            ctx.cite_key, ar.confidence, tool_calls, ar.verdict,
        )
        ar = await run_reflexion(
            refined.claim, ar, deps, abstract_only=ref.abstract_only
        )

    # Abstract-only confidence cap (applied after reflexion too)
    if ref.abstract_only:
        ar = _AgentResult(
            verdict=ar.verdict,
            confidence=min(ar.confidence, 0.7),
            evidence=ar.evidence,
            rationale=ar.rationale,
        )

    return CitationVerdict(
        cite_key=ctx.cite_key,
        claim=refined.claim,
        claim_context=ctx.claim_context,
        verdict=ar.verdict,
        confidence=ar.confidence,
        evidence=ar.evidence,
        rationale=ar.rationale,
        resolution_source=ref.source,
    )


async def verify_all(
    pairs: list[tuple[CitationContext, RefinedClaim]],
    refs: dict[str, ResolvedRef],
    store: VectorStore,
) -> list[CitationVerdict]:
    """Verify all citations sequentially (qwen2.5:14b is memory-intensive)."""
    verdicts: list[CitationVerdict] = []
    for ctx, refined in pairs:
        ref = refs.get(ctx.cite_key)
        if ref is None:
            verdicts.append(CitationVerdict(
                cite_key=ctx.cite_key,
                claim=refined.claim,
                claim_context=ctx.claim_context,
                verdict="unverifiable",
                confidence=0.0,
                evidence=[],
                rationale="Reference not found in .bib file.",
                resolution_source=None,
            ))
            continue
        verdict = await verify_citation(ctx, refined, ref, store)
        log.info("[%s] %s (confidence=%.2f)", ctx.cite_key, verdict.verdict, verdict.confidence)
        verdicts.append(verdict)
    return verdicts

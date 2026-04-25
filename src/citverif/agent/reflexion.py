from __future__ import annotations

import logging

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.usage import UsageLimits

from citverif.agent.prompts import REFLEXION_PROMPT
from citverif.agent.tools import VerifierDeps, fetch_section, semantic_search, web_search
from citverif.schema import AgentResult

log = logging.getLogger(__name__)

_OLLAMA_BASE = "http://localhost:11434/v1"
_MODEL = "qwen2.5:14b"
_REFLEX_LIMITS = UsageLimits(request_limit=4)  # 3 tool calls + 1 final response


def needs_reflexion(result: AgentResult, tool_calls_made: int) -> bool:
    """
    Reflexion triggers when:
    - confidence < 0.6  (uncertain — worth challenging)
    - verdict is 'supported' AND fewer than 2 tool calls were made
      (over-confident paraphrase: model agreed without finding real evidence)
    """
    if result.confidence < 0.6:
        return True
    if result.verdict == "supported" and tool_calls_made < 2:
        return True
    return False


def _make_reflexion_agent() -> Agent[VerifierDeps, AgentResult]:
    model = OpenAIModel(
        _MODEL,
        base_url=_OLLAMA_BASE,
        api_key="ollama",
    )
    return Agent(
        model,
        deps_type=VerifierDeps,
        result_type=AgentResult,
        system_prompt=REFLEXION_PROMPT,
        tools=[semantic_search, fetch_section, web_search],
    )


_reflexion_agent: Agent[VerifierDeps, AgentResult] | None = None


def _get_reflexion_agent() -> Agent[VerifierDeps, AgentResult]:
    global _reflexion_agent
    if _reflexion_agent is None:
        _reflexion_agent = _make_reflexion_agent()
    return _reflexion_agent


async def run_reflexion(
    claim: str,
    prev: AgentResult,
    deps: VerifierDeps,
    abstract_only: bool = False,
) -> AgentResult:
    """
    Run the counter-evidence pass against an initial verdict.

    The agent searches for evidence contradicting the previous conclusion.
    - No counter-evidence found → confidence increases.
    - Counter-evidence found → verdict revised.

    Never raises — falls back to the original verdict on any error.
    """
    agent = _get_reflexion_agent()

    user_prompt = (
        f"Previous verdict: {prev.verdict}\n"
        f"Previous confidence: {prev.confidence:.2f}\n"
        f"Previous rationale: {prev.rationale}\n\n"
        f"Claim to re-examine: {claim}\n\n"
        "Now perform the counter-evidence pass as instructed."
        + ("\n\nNote: Only the abstract is available — confidence must stay ≤ 0.7."
           if abstract_only else "")
    )

    try:
        result = await agent.run(user_prompt, deps=deps, usage_limits=_REFLEX_LIMITS)
        revised = result.data

        if abstract_only:
            revised = AgentResult(
                verdict=revised.verdict,
                confidence=min(revised.confidence, 0.7),
                evidence=revised.evidence,
                rationale=revised.rationale,
            )

        log.info(
            "Reflexion: %s→%s  confidence %.2f→%.2f",
            prev.verdict, revised.verdict,
            prev.confidence, revised.confidence,
        )
        return revised

    except Exception as exc:
        log.warning("Reflexion pass failed: %s — keeping original verdict", exc)
        return prev

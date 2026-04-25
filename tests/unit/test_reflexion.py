"""
Unit tests for the reflexion module.
No Ollama required — reflexion agent is mocked throughout.

Key regression requirement: reflexion must NOT flip a high-confidence
correct verdict to wrong when no counter-evidence exists.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from citverif.agent.reflexion import needs_reflexion, run_reflexion
from citverif.agent.tools import VerifierDeps
from citverif.schema import AgentResult, EvidenceSpan


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ar(verdict, confidence, rationale="Test.") -> AgentResult:
    return AgentResult(verdict=verdict, confidence=confidence, evidence=[], rationale=rationale)


def _ar_with_evidence(verdict, confidence) -> AgentResult:
    return AgentResult(
        verdict=verdict,
        confidence=confidence,
        evidence=[EvidenceSpan(passage="Direct quote.", section="Results", page=3, paper_id="p1")],
        rationale="Evidence found.",
    )


def _mock_deps() -> VerifierDeps:
    store = MagicMock()
    store.has_paper.return_value = True
    return VerifierDeps(store=store, paper_id="test-paper", abstract=None)


def _patch_reflexion_agent(result: AgentResult):
    run_result = MagicMock()
    run_result.data = result
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=run_result)
    return patch("citverif.agent.reflexion._get_reflexion_agent", return_value=mock_agent)


# ── needs_reflexion ───────────────────────────────────────────────────────────

def test_triggers_on_low_confidence():
    assert needs_reflexion(_ar("supported", 0.55), tool_calls_made=3) is True


def test_triggers_on_supported_with_one_tool_call():
    assert needs_reflexion(_ar("supported", 0.9), tool_calls_made=1) is True


def test_triggers_on_supported_with_zero_tool_calls():
    assert needs_reflexion(_ar("supported", 0.95), tool_calls_made=0) is True


def test_does_not_trigger_on_confident_unsupported():
    assert needs_reflexion(_ar("unsupported", 0.88), tool_calls_made=2) is False


def test_does_not_trigger_on_confident_misleading():
    assert needs_reflexion(_ar("misleading", 0.80), tool_calls_made=3) is False


def test_does_not_trigger_on_supported_with_enough_tool_calls():
    assert needs_reflexion(_ar("supported", 0.9), tool_calls_made=2) is False


def test_does_not_trigger_on_unverifiable():
    assert needs_reflexion(_ar("unverifiable", 0.0), tool_calls_made=4) is False


def test_triggers_exactly_at_threshold():
    # confidence = 0.6 is NOT below 0.6 — should not trigger on confidence alone
    assert needs_reflexion(_ar("unsupported", 0.6), tool_calls_made=3) is False
    # confidence = 0.599 should trigger
    assert needs_reflexion(_ar("unsupported", 0.599), tool_calls_made=3) is True


# ── run_reflexion — regression: must not flip correct high-confidence verdicts ─

@pytest.mark.asyncio
async def test_no_counter_evidence_keeps_verdict():
    """When reflexion finds no counter-evidence, verdict stays the same."""
    prev = _ar_with_evidence("supported", 0.91)
    revised = _ar_with_evidence("supported", 0.95)  # confidence went up, verdict unchanged

    with _patch_reflexion_agent(revised):
        result = await run_reflexion("The Transformer achieves 28.4 BLEU.", prev, _mock_deps())

    assert result.verdict == "supported"
    assert result.confidence >= prev.confidence


@pytest.mark.asyncio
async def test_counter_evidence_revises_verdict():
    """When reflexion finds counter-evidence, it can revise the verdict."""
    prev = _ar("supported", 0.55)
    revised = _ar("partially_supported", 0.72)

    with _patch_reflexion_agent(revised):
        result = await run_reflexion("GPT-3 achieves 99% on MMLU.", prev, _mock_deps())

    assert result.verdict == "partially_supported"


@pytest.mark.asyncio
async def test_reflexion_does_not_flip_high_confidence_correct_verdict():
    """
    Regression: if the main pass was correct and high-confidence, reflexion
    must not change the verdict (the mock simulates finding no counter-evidence).
    """
    prev = _ar_with_evidence("unsupported", 0.88)
    # Reflexion searched and confirmed no support — verdict unchanged, confidence up
    confirmed = _ar_with_evidence("unsupported", 0.92)

    with _patch_reflexion_agent(confirmed):
        result = await run_reflexion("Paper claims X but paper says Y.", prev, _mock_deps())

    assert result.verdict == "unsupported"
    assert result.confidence >= 0.88


@pytest.mark.asyncio
async def test_reflexion_error_returns_original():
    """If the reflexion agent errors, we silently return the original result."""
    prev = _ar("misleading", 0.75)
    deps = _mock_deps()

    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(side_effect=RuntimeError("Ollama crashed"))

    with patch("citverif.agent.reflexion._get_reflexion_agent", return_value=mock_agent):
        result = await run_reflexion("Some claim.", prev, deps)

    assert result.verdict == "misleading"
    assert result.confidence == 0.75


@pytest.mark.asyncio
async def test_abstract_only_caps_reflexion_confidence():
    """Even if reflexion would push confidence to 0.9, abstract-only caps at 0.7."""
    prev = _ar("supported", 0.55)
    reflexion_result = _ar("supported", 0.90)  # agent is very confident after reflexion

    with _patch_reflexion_agent(reflexion_result):
        result = await run_reflexion("Some claim.", prev, _mock_deps(), abstract_only=True)

    assert result.confidence <= 0.7


@pytest.mark.asyncio
async def test_reflexion_preserves_evidence_spans():
    """Evidence spans from the reflexion pass are returned in the result."""
    prev = _ar("unsupported", 0.50)
    revised = AgentResult(
        verdict="partially_supported",
        confidence=0.65,
        evidence=[EvidenceSpan(passage="Found something.", section="Abstract", page=1, paper_id="p")],
        rationale="Found partial support.",
    )

    with _patch_reflexion_agent(revised):
        result = await run_reflexion("Some claim.", prev, _mock_deps())

    assert len(result.evidence) == 1
    assert result.evidence[0].passage == "Found something."

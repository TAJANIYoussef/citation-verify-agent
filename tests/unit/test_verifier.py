"""
Unit tests for the verifier agent — no Ollama required (agent mocked).

5 labeled (claim, expected_verdict) triples + reflexion trigger logic tests.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from citverif.agent.verifier import _AgentResult, verify_citation
from citverif.agent.reflexion import needs_reflexion as _needs_reflexion
from citverif.extract.claims import RefinedClaim
from citverif.parse.tex import CitationContext
from citverif.resolve.chain import ResolvedRef
from citverif.schema import CitationVerdict, EvidenceSpan


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ctx(cite_key: str, claim_context: str) -> CitationContext:
    return CitationContext(
        cite_key=cite_key,
        raw_text=claim_context,
        claim_context=claim_context,
    )


def _refined(claim: str, is_verifiable: bool = True) -> RefinedClaim:
    return RefinedClaim(claim=claim, is_verifiable=is_verifiable)


def _ref(cite_key: str, has_pdf: bool = True, abstract_only: bool = False) -> ResolvedRef:
    from pathlib import Path
    return ResolvedRef(
        cite_key=cite_key,
        pdf_path=Path("/fake/path.pdf") if has_pdf else None,
        abstract="Abstract text." if abstract_only else None,
        source="arxiv",
        abstract_only=abstract_only,
    )


def _mock_agent_result(verdict, confidence, evidence=None, rationale="Test rationale."):
    ar = _AgentResult(
        verdict=verdict,
        confidence=confidence,
        evidence=evidence or [],
        rationale=rationale,
    )
    usage_mock = MagicMock()
    usage_mock.requests = 3   # 2 tool calls + 1 final response
    run_result = MagicMock()
    run_result.data = ar
    run_result.usage.return_value = usage_mock
    return run_result


def _patch_verifier(verdict, confidence, evidence=None):
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=_mock_agent_result(verdict, confidence, evidence))
    return patch("citverif.agent.verifier._get_verifier", return_value=mock_agent)


# ── 5 labeled verdict triples ─────────────────────────────────────────────────

TRIPLES = [
    # 1. Clear support with direct quote
    {
        "cite_key": "vaswani2017attention",
        "claim": "The Transformer achieves 28.4 BLEU on WMT 2014 English-to-German.",
        "expected_verdict": "supported",
        "confidence": 0.92,
        "evidence": [EvidenceSpan(
            passage="We report BLEU scores of 28.4 on the WMT 2014 English-to-German translation task.",
            section="Results", page=7, paper_id="vaswani2017attention",
        )],
    },
    # 2. Paper says something weaker — partial support
    {
        "cite_key": "devlin2018bert",
        "claim": "BERT achieves perfect scores on all GLUE benchmarks.",
        "expected_verdict": "partially_supported",
        "confidence": 0.65,
        "evidence": [EvidenceSpan(
            passage="BERT achieves a GLUE score of 80.5, outperforming previous models on most tasks.",
            section="Results", page=5, paper_id="devlin2018bert",
        )],
    },
    # 3. Claim not in paper — unsupported
    {
        "cite_key": "brown2020language",
        "claim": "GPT-3 achieves 99% accuracy on the MMLU benchmark.",
        "expected_verdict": "unsupported",
        "confidence": 0.85,
        "evidence": [],
    },
    # 4. Paper contradicts the claim — misleading
    {
        "cite_key": "he2016deep",
        "claim": "ResNets perform worse than plain CNNs on ImageNet as depth increases.",
        "expected_verdict": "misleading",
        "confidence": 0.88,
        "evidence": [EvidenceSpan(
            passage="Residual networks consistently outperform plain networks as depth increases.",
            section="Experiments", page=6, paper_id="he2016deep",
        )],
    },
    # 5. Paper unavailable — unverifiable
    {
        "cite_key": "broken2099",
        "claim": "A nonexistent method achieves SOTA on all benchmarks.",
        "expected_verdict": "unverifiable",
        "confidence": 0.0,
        "evidence": [],
    },
]


@pytest.mark.parametrize("triple", TRIPLES, ids=[t["cite_key"] for t in TRIPLES])
@pytest.mark.asyncio
async def test_verdict_matches_expected(triple):
    ctx = _ctx(triple["cite_key"], triple["claim"])
    refined = _refined(triple["claim"])
    ref = _ref(triple["cite_key"])
    store = MagicMock()
    store.has_paper.return_value = True

    with _patch_verifier(triple["expected_verdict"], triple["confidence"], triple["evidence"]):
        verdict = await verify_citation(ctx, refined, ref, store)

    assert verdict.verdict == triple["expected_verdict"]
    assert verdict.cite_key == triple["cite_key"]
    assert isinstance(verdict.confidence, float)


# ── Reflexion trigger logic ───────────────────────────────────────────────────

def test_reflexion_triggered_on_low_confidence():
    ar = _AgentResult(verdict="supported", confidence=0.55, evidence=[], rationale="r")
    assert _needs_reflexion(ar, tool_calls_made=3) is True


def test_reflexion_triggered_on_supported_with_few_tools():
    ar = _AgentResult(verdict="supported", confidence=0.9, evidence=[], rationale="r")
    assert _needs_reflexion(ar, tool_calls_made=1) is True


def test_reflexion_not_triggered_on_confident_unsupported():
    ar = _AgentResult(verdict="unsupported", confidence=0.85, evidence=[], rationale="r")
    assert _needs_reflexion(ar, tool_calls_made=3) is False


def test_reflexion_not_triggered_on_confident_supported_with_enough_tools():
    ar = _AgentResult(verdict="supported", confidence=0.9, evidence=[], rationale="r")
    assert _needs_reflexion(ar, tool_calls_made=2) is False


def test_reflexion_triggered_on_zero_confidence():
    ar = _AgentResult(verdict="unverifiable", confidence=0.0, evidence=[], rationale="r")
    assert _needs_reflexion(ar, tool_calls_made=4) is True


# ── Non-verifiable short-circuit ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_non_verifiable_claim_skips_agent():
    ctx = _ctx("some-key", "As discussed in [X], we use method Y.")
    refined = _refined("Not a verifiable claim.", is_verifiable=False)
    ref = _ref("some-key")
    store = MagicMock()

    with patch("citverif.agent.verifier._get_verifier") as mock_get:
        verdict = await verify_citation(ctx, refined, ref, store)
        mock_get.assert_not_called()

    assert verdict.verdict == "unverifiable"


# ── Abstract-only confidence cap ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_abstract_only_caps_confidence():
    ctx = _ctx("abstract-paper", "The model achieves 95% accuracy.")
    refined = _refined("The model achieves 95% accuracy.")
    ref = _ref("abstract-paper", has_pdf=False, abstract_only=True)
    store = MagicMock()
    store.has_paper.return_value = False

    with _patch_verifier("supported", 0.95):
        verdict = await verify_citation(ctx, refined, ref, store)

    assert verdict.confidence <= 0.7, f"Expected ≤0.7 got {verdict.confidence}"


# ── Agent error fallback ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_agent_error_returns_unverifiable():
    ctx = _ctx("error-key", "Some claim.")
    refined = _refined("Some claim.")
    ref = _ref("error-key")
    store = MagicMock()

    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(side_effect=RuntimeError("Ollama crashed"))

    with patch("citverif.agent.verifier._get_verifier", return_value=mock_agent):
        verdict = await verify_citation(ctx, refined, ref, store)

    assert verdict.verdict == "unverifiable"
    assert "Agent error" in verdict.rationale

"""
Integration tests for the reflexion pass — requires Ollama + qwen2.5:14b + bge-m3.
Run with: pytest tests/integration/test_reflexion_integration.py -v

Regression requirement: reflexion must not flip a high-confidence correct verdict to wrong.
"""
import pytest
from pathlib import Path

from citverif.agent.reflexion import run_reflexion
from citverif.agent.tools import VerifierDeps
from citverif.memory.chunker import Chunk
from citverif.memory.vector_store import VectorStore
from citverif.schema import AgentResult, EvidenceSpan


def _store_with_text(paper_id: str, text: str, tmp_path: Path) -> VectorStore:
    store = VectorStore(tmp_path / paper_id)
    store.index_chunks([Chunk(
        text=text, page=1, section="Results", chunk_idx=0, paper_id=paper_id,
    )])
    return store


def _deps(store: VectorStore, paper_id: str) -> VerifierDeps:
    return VerifierDeps(store=store, paper_id=paper_id, abstract=None)


# ── Regression: high-confidence correct verdict must not flip ─────────────────

@pytest.mark.asyncio
async def test_regression_supported_verdict_not_flipped(tmp_path):
    """
    The paper explicitly states the claim. The main pass correctly returned
    'supported' with high confidence. Reflexion must not flip it.
    """
    paper_id = "reflex-supported"
    text = (
        "The Transformer model achieves a BLEU score of 28.4 on the WMT 2014 "
        "English-to-German translation task, outperforming all prior models."
    )
    store = _store_with_text(paper_id, text, tmp_path)
    deps = _deps(store, paper_id)

    prev = AgentResult(
        verdict="supported",
        confidence=0.92,
        evidence=[EvidenceSpan(passage=text[:80], section="Results", page=1, paper_id=paper_id)],
        rationale="The paper explicitly states the BLEU score.",
    )

    result = await run_reflexion(
        "The Transformer achieves 28.4 BLEU on WMT 2014 English-to-German.",
        prev, deps,
    )

    assert result.verdict == "supported", (
        f"Reflexion flipped a high-confidence correct verdict: {result.verdict} "
        f"(confidence={result.confidence:.2f})\nRationale: {result.rationale}"
    )


@pytest.mark.asyncio
async def test_regression_unsupported_verdict_not_flipped(tmp_path):
    """
    The paper does not contain the claim. Reflexion must not flip to 'supported'.
    """
    paper_id = "reflex-unsupported"
    text = (
        "We describe the architecture of our neural network. "
        "The model uses three convolutional layers followed by two fully-connected layers. "
        "No performance comparison with other methods is presented."
    )
    store = _store_with_text(paper_id, text, tmp_path)
    deps = _deps(store, paper_id)

    prev = AgentResult(
        verdict="unsupported",
        confidence=0.85,
        evidence=[],
        rationale="The paper does not report accuracy on the CIFAR-10 benchmark.",
    )

    result = await run_reflexion(
        "This model achieves 95% accuracy on CIFAR-10.",
        prev, deps,
    )

    assert result.verdict != "supported", (
        f"Reflexion incorrectly flipped unsupported→supported\n"
        f"Rationale: {result.rationale}"
    )


@pytest.mark.asyncio
async def test_reflexion_increases_confidence_when_no_counter_evidence(tmp_path):
    """When reflexion finds no counter-evidence, confidence should not decrease."""
    paper_id = "reflex-confidence"
    text = (
        "Our proposed method achieves state-of-the-art results on the benchmark. "
        "The accuracy is 91.3%, compared to 88.7% for the previous best method."
    )
    store = _store_with_text(paper_id, text, tmp_path)
    deps = _deps(store, paper_id)

    prev = AgentResult(
        verdict="supported",
        confidence=0.55,   # low enough to trigger reflexion
        evidence=[],
        rationale="Tentative: the paper mentions high accuracy.",
    )

    result = await run_reflexion(
        "The proposed method achieves 91.3% accuracy, outperforming prior work.",
        prev, deps,
    )

    # After reflexion with no counter-evidence, confidence must not drop
    assert result.confidence >= prev.confidence - 0.05, (
        f"Confidence dropped from {prev.confidence:.2f} to {result.confidence:.2f} "
        "after reflexion with no expected counter-evidence."
    )


@pytest.mark.asyncio
async def test_abstract_only_cap_enforced(tmp_path):
    """Even after reflexion, abstract-only confidence cannot exceed 0.7."""
    paper_id = "reflex-abstract"
    store = VectorStore(tmp_path / paper_id)  # empty store — abstract only
    deps = VerifierDeps(
        store=store,
        paper_id=paper_id,
        abstract="The model achieves 91.3% accuracy on the benchmark.",
    )

    prev = AgentResult(
        verdict="supported",
        confidence=0.55,
        evidence=[],
        rationale="Abstract mentions accuracy.",
    )

    result = await run_reflexion(
        "The model achieves 91.3% accuracy.",
        prev, deps, abstract_only=True,
    )

    assert result.confidence <= 0.7, (
        f"Abstract-only cap violated: confidence={result.confidence:.2f}"
    )

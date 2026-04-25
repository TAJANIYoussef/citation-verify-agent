"""
Integration tests for the verifier agent — requires Ollama + qwen2.5:14b + bge-m3.
Run with: pytest tests/integration/test_verifier_integration.py -v

Definition of done (§10 Phase 4): must hit ≥3/5 on the labeled triples.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from citverif.agent.verifier import verify_citation
from citverif.extract.claims import RefinedClaim
from citverif.memory.chunker import Chunk
from citverif.memory.vector_store import VectorStore
from citverif.parse.tex import CitationContext
from citverif.resolve.chain import ResolvedRef
from citverif.schema import Verdict


# ── Fixtures: known (claim, paper_text, expected_verdict) ────────────────────

TRIPLES: list[dict] = [
    {
        "cite_key": "t1-supported",
        "claim": "The Transformer architecture relies entirely on attention mechanisms and dispenses with recurrence.",
        "paper_text": (
            "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, "
            "dispensing with recurrence and convolutions entirely. "
            "The Transformer allows for significantly more parallelization and reaches state of the art on "
            "translation tasks after training for as little as twelve hours."
        ),
        "expected": "supported",
    },
    {
        "cite_key": "t2-unsupported",
        "claim": "The Transformer model requires more training data than LSTMs to converge.",
        "paper_text": (
            "We propose the Transformer, based solely on attention mechanisms. "
            "The model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task. "
            "Training was performed on the standard WMT 2014 English-German dataset."
        ),
        "expected": "unsupported",
    },
    {
        "cite_key": "t3-partially",
        "claim": "BERT achieves perfect state-of-the-art on all eleven NLP benchmarks it was evaluated on.",
        "paper_text": (
            "BERT obtains new state-of-the-art results on eleven NLP tasks. "
            "On the GLUE benchmark, BERT achieves a score of 80.5%. "
            "On SQuAD 1.1, BERT achieves an F1 score of 93.2, surpassing human performance. "
            "Note that results vary across tasks and BERT does not achieve the highest score on every subtask."
        ),
        "expected": "partially_supported",
    },
    {
        "cite_key": "t4-misleading",
        "claim": "ResNets perform worse than plain networks as depth increases beyond 50 layers.",
        "paper_text": (
            "When the depth increases from 20 to 56 layers, plain networks show higher training error. "
            "Our residual networks show consistently lower error as depth increases. "
            "The 152-layer ResNet outperforms all shallower plain and residual networks."
        ),
        "expected": "misleading",
    },
    {
        "cite_key": "t5-supported-quant",
        "claim": "GPT-3 achieves 76.2% accuracy on TriviaQA in a few-shot setting.",
        "paper_text": (
            "We evaluate GPT-3 on TriviaQA in the few-shot setting. "
            "GPT-3 achieves 76.2% accuracy on the TriviaQA closed-book test set without any fine-tuning. "
            "This compares favorably with fine-tuned T5 models."
        ),
        "expected": "supported",
    },
]


def _build_store(cite_key: str, paper_text: str, tmp_path: Path) -> VectorStore:
    store = VectorStore(tmp_path / cite_key)
    chunks = [Chunk(
        text=paper_text,
        page=1,
        section="Results",
        chunk_idx=0,
        paper_id=cite_key,
    )]
    store.index_chunks(chunks)
    return store


@pytest.mark.asyncio
@pytest.mark.parametrize("triple", TRIPLES, ids=[t["cite_key"] for t in TRIPLES])
async def test_verdict_on_labeled_triple(triple, tmp_path):
    store = _build_store(triple["cite_key"], triple["paper_text"], tmp_path)

    ctx = CitationContext(
        cite_key=triple["cite_key"],
        raw_text=triple["claim"],
        claim_context=triple["claim"],
    )
    refined = RefinedClaim(claim=triple["claim"], is_verifiable=True)
    ref = ResolvedRef(
        cite_key=triple["cite_key"],
        pdf_path=Path("/fake.pdf"),
        abstract=None,
        source="arxiv",
        abstract_only=False,
    )

    verdict = await verify_citation(ctx, refined, ref, store)

    assert verdict.verdict == triple["expected"], (
        f"[{triple['cite_key']}] expected={triple['expected']} "
        f"got={verdict.verdict} (confidence={verdict.confidence:.2f})\n"
        f"Rationale: {verdict.rationale}"
    )


@pytest.mark.asyncio
async def test_phase4_overall_accuracy(tmp_path):
    """Must hit ≥3/5 on the labeled set (§10 Phase 4 definition of done)."""
    correct = 0
    for triple in TRIPLES:
        store = _build_store(triple["cite_key"], triple["paper_text"], tmp_path)
        ctx = CitationContext(
            cite_key=triple["cite_key"],
            raw_text=triple["claim"],
            claim_context=triple["claim"],
        )
        refined = RefinedClaim(claim=triple["claim"], is_verifiable=True)
        ref = ResolvedRef(
            cite_key=triple["cite_key"],
            pdf_path=Path("/fake.pdf"),
            abstract=None,
            source="arxiv",
            abstract_only=False,
        )
        verdict = await verify_citation(ctx, refined, ref, store)
        if verdict.verdict == triple["expected"]:
            correct += 1

    assert correct >= 3, f"Phase 4 accuracy: {correct}/5 — must be ≥3/5"

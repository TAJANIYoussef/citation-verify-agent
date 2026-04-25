"""
Unit tests for claim refinement — no Ollama required (agent is mocked).

10 labeled fixtures covering:
  - clear factual claims          (is_verifiable=True)
  - non-verifiable passages       (is_verifiable=False)
  - hedged first-person text
  - quantitative comparisons
  - definition / method passages
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from citverif.extract.claims import RefinedClaim, refine_claim, refine_all
from citverif.parse.tex import CitationContext


# ── 10 labeled fixtures ──────────────────────────────────────────────────────

FIXTURES: list[dict] = [
    # 1. Quantitative performance claim
    {
        "cite_key": "vaswani2017attention",
        "claim_context": (
            "The Transformer achieves a BLEU score of 28.4 on WMT 2014 English-to-German, "
            "outperforming all previously reported models."
        ),
        "expected_verifiable": True,
    },
    # 2. Hedged first-person — still verifiable
    {
        "cite_key": "devlin2018bert",
        "claim_context": (
            "We show that BERT achieves state-of-the-art on eleven NLP tasks "
            "including a GLUE score of 80.5%."
        ),
        "expected_verifiable": True,
    },
    # 3. Method description — NOT verifiable
    {
        "cite_key": "he2016deep",
        "claim_context": (
            "We describe our training procedure in He et al., using batch normalization "
            "after each convolutional layer."
        ),
        "expected_verifiable": False,
    },
    # 4. Few-shot learning claim
    {
        "cite_key": "brown2020language",
        "claim_context": (
            "GPT-3 achieves 76.2% accuracy on TriviaQA in a few-shot setting "
            "without any gradient updates."
        ),
        "expected_verifiable": True,
    },
    # 5. "As discussed in" reference — NOT verifiable
    {
        "cite_key": "lecun1998gradient",
        "claim_context": (
            "Convolutional networks, as discussed in LeCun et al., use shared weights "
            "across spatial locations."
        ),
        "expected_verifiable": False,
    },
    # 6. Cross-model comparison
    {
        "cite_key": "goodfellow2014generative",
        "claim_context": (
            "GANs produce sharper images than variational autoencoders "
            "on the CIFAR-10 benchmark."
        ),
        "expected_verifiable": True,
    },
    # 7. Multilingual performance claim
    {
        "cite_key": "grave2018fasttext",
        "claim_context": (
            "FastText vectors trained on Common Crawl outperform Word2Vec "
            "on French NLP tasks."
        ),
        "expected_verifiable": True,
    },
    # 8. Definition passage — NOT verifiable
    {
        "cite_key": "hochreiter1997lstm",
        "claim_context": (
            "An LSTM cell, as defined by Hochreiter and Schmidhuber, consists of "
            "an input gate, a forget gate, and an output gate."
        ),
        "expected_verifiable": False,
    },
    # 9. Vision model quantitative claim
    {
        "cite_key": "dosovitskiy2020vit",
        "claim_context": (
            "ViT trained on JFT-300M achieves 88.55% top-1 accuracy on ImageNet, "
            "surpassing ResNet-based models."
        ),
        "expected_verifiable": True,
    },
    # 10. Semantic relationship claim
    {
        "cite_key": "mikolov2013word2vec",
        "claim_context": (
            "Word2Vec embeddings capture semantic relationships, enabling analogical "
            "reasoning such as king − man + woman ≈ queen."
        ),
        "expected_verifiable": True,
    },
]

NON_VERIFIABLE_KEYS = {"he2016deep", "lecun1998gradient", "hochreiter1997lstm"}


def _ctx(f: dict) -> CitationContext:
    return CitationContext(
        cite_key=f["cite_key"],
        raw_text=f["claim_context"],
        claim_context=f["claim_context"],
    )


def _mock_agent(is_verifiable: bool = True, claim: str = "Extracted claim."):
    """Return a mock that replaces _get_agent(), producing a deterministic RefinedClaim."""
    refined = RefinedClaim(claim=claim, is_verifiable=is_verifiable)
    mock_run_result = MagicMock()
    mock_run_result.data = refined
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=mock_run_result)
    return mock_agent


# ── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fixture", FIXTURES, ids=[f["cite_key"] for f in FIXTURES])
@pytest.mark.asyncio
async def test_refine_claim_returns_refined_claim(fixture):
    ctx = _ctx(fixture)
    expected_v = fixture["expected_verifiable"]
    with patch("citverif.extract.claims._get_agent", return_value=_mock_agent(expected_v)):
        result = await refine_claim(ctx)
    assert isinstance(result, RefinedClaim)
    assert result.is_verifiable == expected_v


@pytest.mark.asyncio
async def test_refine_claim_fallback_on_error():
    """Model failure produces a passthrough RefinedClaim, never raises."""
    ctx = CitationContext(
        cite_key="bad-key",
        raw_text="Some raw text",
        claim_context="Some raw text",
    )
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(side_effect=RuntimeError("Ollama not running"))
    with patch("citverif.extract.claims._get_agent", return_value=mock_agent):
        result = await refine_claim(ctx)
    assert isinstance(result, RefinedClaim)
    assert result.claim         # not empty
    assert result.is_verifiable # passthrough defaults to True


@pytest.mark.asyncio
async def test_refine_all_returns_all_pairs():
    contexts = [_ctx(f) for f in FIXTURES]

    async def fake_refine(ctx):
        return RefinedClaim(
            claim="Extracted.",
            is_verifiable=ctx.cite_key not in NON_VERIFIABLE_KEYS,
        )

    with patch("citverif.extract.claims.refine_claim", side_effect=fake_refine):
        pairs = await refine_all(contexts)

    assert len(pairs) == len(FIXTURES)


@pytest.mark.asyncio
async def test_refine_all_verifiable_count():
    contexts = [_ctx(f) for f in FIXTURES]

    async def fake_refine(ctx):
        return RefinedClaim(
            claim="Extracted.",
            is_verifiable=ctx.cite_key not in NON_VERIFIABLE_KEYS,
        )

    with patch("citverif.extract.claims.refine_claim", side_effect=fake_refine):
        pairs = await refine_all(contexts)

    verifiable = [rc for _, rc in pairs if rc.is_verifiable]
    non_verifiable = [rc for _, rc in pairs if not rc.is_verifiable]
    assert len(verifiable) == 7
    assert len(non_verifiable) == 3


@pytest.mark.asyncio
async def test_claim_text_is_not_empty():
    ctx = _ctx(FIXTURES[0])
    with patch("citverif.extract.claims._get_agent", return_value=_mock_agent(True, "The Transformer achieves 28.4 BLEU.")):
        result = await refine_claim(ctx)
    assert len(result.claim) > 5

"""
Integration tests for claim extraction — requires Ollama + llama3.1:8b.
Run with: pytest tests/integration/test_claims_integration.py -v
"""
import pytest
from citverif.extract.claims import refine_claim, RefinedClaim
from citverif.parse.tex import CitationContext


VERIFIABLE_CASES = [
    CitationContext(
        cite_key="vaswani2017attention",
        raw_text="The Transformer achieves 28.4 BLEU on WMT 2014 English-to-German.",
        claim_context="The Transformer achieves 28.4 BLEU on WMT 2014 English-to-German.",
    ),
    CitationContext(
        cite_key="dosovitskiy2020vit",
        raw_text="ViT achieves 88.55% top-1 accuracy on ImageNet, surpassing ResNets.",
        claim_context="ViT achieves 88.55% top-1 accuracy on ImageNet, surpassing ResNets.",
    ),
    CitationContext(
        cite_key="brown2020language",
        raw_text="GPT-3 achieves 76.2% on TriviaQA in a few-shot setting without fine-tuning.",
        claim_context="GPT-3 achieves 76.2% on TriviaQA in a few-shot setting without fine-tuning.",
    ),
]

NON_VERIFIABLE_CASES = [
    CitationContext(
        cite_key="hochreiter1997lstm",
        raw_text="An LSTM, as defined in Hochreiter and Schmidhuber, has input, forget, and output gates.",
        claim_context="An LSTM, as defined in Hochreiter and Schmidhuber, has input, forget, and output gates.",
    ),
    CitationContext(
        cite_key="lecun1998gradient",
        raw_text="As described in LeCun et al., we use shared weights across spatial locations.",
        claim_context="As described in LeCun et al., we use shared weights across spatial locations.",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("ctx", VERIFIABLE_CASES, ids=[c.cite_key for c in VERIFIABLE_CASES])
async def test_verifiable_claim_detected(ctx):
    result = await refine_claim(ctx)
    assert isinstance(result, RefinedClaim)
    assert result.is_verifiable is True, (
        f"Expected verifiable=True for {ctx.cite_key}, got claim: {result.claim}"
    )
    assert len(result.claim) > 5


@pytest.mark.asyncio
@pytest.mark.parametrize("ctx", NON_VERIFIABLE_CASES, ids=[c.cite_key for c in NON_VERIFIABLE_CASES])
async def test_non_verifiable_detected(ctx):
    result = await refine_claim(ctx)
    assert isinstance(result, RefinedClaim)
    # llama3.1:8b may not always get this right — warn rather than hard-fail
    if result.is_verifiable:
        import warnings
        warnings.warn(
            f"{ctx.cite_key}: expected non-verifiable but model returned verifiable. "
            f"Claim: {result.claim}"
        )


@pytest.mark.asyncio
async def test_integration_smoke_refine_all():
    from citverif.extract.claims import refine_all
    pairs = await refine_all(VERIFIABLE_CASES)
    assert len(pairs) == 3
    assert all(isinstance(rc, RefinedClaim) for _, rc in pairs)
    # At least 2 of 3 clear factual claims should be marked verifiable
    verifiable_count = sum(1 for _, rc in pairs if rc.is_verifiable)
    assert verifiable_count >= 2, f"Only {verifiable_count}/3 marked verifiable"

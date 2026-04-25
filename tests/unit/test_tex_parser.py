from pathlib import Path
import pytest
from citverif.parse.tex import parse_tex, CitationContext, UncitedCandidate

TEST_PAPERS = Path(__file__).parent.parent.parent / ".context" / "test-papers"


def test_paper1_citations():
    citations, _ = parse_tex(TEST_PAPERS / "paper1.tex")
    keys = [c.cite_key for c in citations]
    assert "vaswani2017attention" in keys
    assert "devlin2018bert" in keys
    assert "broken2099" in keys


def test_paper1_uncited():
    _, uncited = parse_tex(TEST_PAPERS / "paper1.tex")
    assert len(uncited) >= 1
    texts = " ".join(u.sentence for u in uncited).lower()
    assert "neural network" in texts or "image classification" in texts


def test_paper2_citations():
    citations, _ = parse_tex(TEST_PAPERS / "paper2.tex")
    keys = [c.cite_key for c in citations]
    assert "he2016deep" in keys
    assert "brown2020language" in keys


def test_paper2_uncited():
    _, uncited = parse_tex(TEST_PAPERS / "paper2.tex")
    assert len(uncited) >= 1


def test_claim_context_is_cleaned():
    citations, _ = parse_tex(TEST_PAPERS / "paper1.tex")
    for c in citations:
        assert "\\" not in c.claim_context or c.claim_context.count("\\") == 0


def test_multi_cite_key():
    from tempfile import NamedTemporaryFile
    content = r"""
\documentclass{article}
\begin{document}
Several works \cite{key1, key2, key3} show that this is effective.
\end{document}
"""
    with NamedTemporaryFile(suffix=".tex", mode="w", delete=False) as f:
        f.write(content)
        tmp = Path(f.name)
    citations, _ = parse_tex(tmp)
    keys = [c.cite_key for c in citations]
    assert "key1" in keys
    assert "key2" in keys
    assert "key3" in keys
    tmp.unlink()

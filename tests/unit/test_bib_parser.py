from pathlib import Path
import pytest
from citverif.parse.bib import parse_bib

TEST_PAPERS = Path(__file__).parent.parent.parent / ".context" / "test-papers"


def test_paper1_entries():
    entries = parse_bib(TEST_PAPERS / "paper1.bib")
    keys = {e.cite_key for e in entries}
    assert "vaswani2017attention" in keys
    assert "devlin2018bert" in keys
    assert "broken2099" in keys


def test_arxiv_extracted():
    entries = parse_bib(TEST_PAPERS / "paper1.bib")
    vaswani = next(e for e in entries if e.cite_key == "vaswani2017attention")
    assert vaswani.arxiv_id == "1706.03762"


def test_doi_extracted():
    entries = parse_bib(TEST_PAPERS / "paper1.bib")
    devlin = next(e for e in entries if e.cite_key == "devlin2018bert")
    assert devlin.doi is not None
    assert "arXiv" in devlin.doi or "1810" in devlin.doi


def test_paper2_entries():
    entries = parse_bib(TEST_PAPERS / "paper2.bib")
    keys = {e.cite_key for e in entries}
    assert "he2016deep" in keys
    assert "brown2020language" in keys


def test_paper2_arxiv():
    entries = parse_bib(TEST_PAPERS / "paper2.bib")
    gpt3 = next(e for e in entries if e.cite_key == "brown2020language")
    assert gpt3.arxiv_id == "2005.14165"


def test_broken_entry_has_no_ids():
    entries = parse_bib(TEST_PAPERS / "paper1.bib")
    broken = next(e for e in entries if e.cite_key == "broken2099")
    assert broken.arxiv_id is None
    assert broken.doi is None

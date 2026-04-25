"""Unit tests for report/markdown.py — pure rendering, no Ollama."""
from pathlib import Path
import pytest

from citverif.parse.tex import UncitedCandidate
from citverif.report.markdown import render_report, _confidence_bar
from citverif.schema import CitationVerdict, EvidenceSpan


def _verdict(
    cite_key: str,
    verdict,
    confidence: float = 0.8,
    claim: str = "Test claim.",
    evidence: list | None = None,
    rationale: str = "Test rationale.",
    source: str | None = "arxiv",
) -> CitationVerdict:
    return CitationVerdict(
        cite_key=cite_key,
        claim=claim,
        claim_context=f"Context for {cite_key}.",
        verdict=verdict,
        confidence=confidence,
        evidence=evidence or [],
        rationale=rationale,
        resolution_source=source,
    )


def _render(verdicts, uncited=None):
    return render_report(
        verdicts=verdicts,
        uncited=uncited or [],
        tex_path=Path("paper.tex"),
        resolution_rate=0.9,
        abstract_only_count=0,
        failed_count=0,
    )


# ── Ordering ──────────────────────────────────────────────────────────────────

def test_unsupported_appears_before_supported():
    report = _render([
        _verdict("a", "supported"),
        _verdict("b", "unsupported"),
    ])
    assert report.index("unsupported".upper()) < report.index("supported".upper())


def test_misleading_appears_before_partially_supported():
    report = _render([
        _verdict("a", "partially_supported"),
        _verdict("b", "misleading"),
    ])
    assert report.index("MISLEADING") < report.index("PARTIALLY SUPPORTED")


def test_verdict_order_full():
    verdicts = [
        _verdict("e", "unverifiable"),
        _verdict("d", "supported"),
        _verdict("c", "partially_supported"),
        _verdict("b", "misleading"),
        _verdict("a", "unsupported"),
    ]
    report = _render(verdicts)
    positions = {v: report.index(v.upper().replace("_", " ")) for v in
                 ["unsupported", "misleading", "partially_supported", "supported", "unverifiable"]}
    assert positions["unsupported"] < positions["misleading"]
    assert positions["misleading"] < positions["partially_supported"]
    assert positions["partially_supported"] < positions["supported"]
    assert positions["supported"] < positions["unverifiable"]


# ── Content ───────────────────────────────────────────────────────────────────

def test_cite_key_present():
    report = _render([_verdict("vaswani2017attention", "supported")])
    assert "vaswani2017attention" in report


def test_claim_present():
    report = _render([_verdict("k", "unsupported", claim="Unique claim text XYZ.")])
    assert "Unique claim text XYZ." in report


def test_rationale_present():
    report = _render([_verdict("k", "misleading", rationale="This is the rationale ABC.")])
    assert "This is the rationale ABC." in report


def test_evidence_quote_present():
    ev = EvidenceSpan(passage="Direct quote here.", section="Results", page=5, paper_id="p")
    report = _render([_verdict("k", "supported", evidence=[ev])])
    assert "Direct quote here." in report
    assert "Results" in report
    assert "p.5" in report


def test_no_evidence_fallback_text():
    report = _render([_verdict("k", "unsupported", evidence=[])])
    assert "No evidence spans" in report


def test_critical_warning_shown_when_issues_exist():
    report = _render([
        _verdict("a", "unsupported"),
        _verdict("b", "misleading"),
    ])
    assert "require immediate attention" in report


def test_no_critical_warning_when_all_supported():
    report = _render([_verdict("a", "supported"), _verdict("b", "supported")])
    assert "require immediate attention" not in report


def test_resolution_rate_in_header():
    report = _render([_verdict("a", "supported")])
    assert "90%" in report   # 0.9 resolution rate


def test_empty_verdicts_renders_without_error():
    report = _render([])
    assert "Citation Verification Report" in report


# ── Uncited candidates ────────────────────────────────────────────────────────

def test_uncited_section_present_when_candidates_exist():
    uncited = [UncitedCandidate("Neural networks outperform SVMs on image tasks.")]
    report = _render([_verdict("a", "supported")], uncited=uncited)
    assert "Uncited Claim Candidates" in report
    assert "Neural networks outperform" in report


def test_uncited_section_absent_when_no_candidates():
    report = _render([_verdict("a", "supported")], uncited=[])
    assert "Uncited Claim Candidates" not in report


def test_multiple_uncited_candidates_numbered():
    uncited = [
        UncitedCandidate("Claim one."),
        UncitedCandidate("Claim two."),
        UncitedCandidate("Claim three."),
    ]
    report = _render([], uncited=uncited)
    assert "1." in report
    assert "2." in report
    assert "3." in report


# ── Confidence bar ────────────────────────────────────────────────────────────

def test_confidence_bar_full():
    bar = _confidence_bar(1.0)
    assert "█" * 10 in bar
    assert "100%" in bar


def test_confidence_bar_zero():
    bar = _confidence_bar(0.0)
    assert "░" * 10 in bar
    assert "0%" in bar


def test_confidence_bar_half():
    bar = _confidence_bar(0.5)
    assert "█" * 5 in bar

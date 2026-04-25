from __future__ import annotations

from datetime import date
from pathlib import Path

from citverif.parse.tex import UncitedCandidate
from citverif.schema import CitationVerdict, Verdict

# Verdict display order — highest priority issues first
_ORDER: list[Verdict] = [
    "unsupported",
    "misleading",
    "partially_supported",
    "supported",
    "unverifiable",
]

_EMOJI: dict[str, str] = {
    "unsupported": "🚫",
    "misleading": "⚠️",
    "partially_supported": "🔶",
    "supported": "✅",
    "unverifiable": "❓",
}

_SECTION_TITLE: dict[str, str] = {
    "unsupported": "Unsupported",
    "misleading": "Misleading",
    "partially_supported": "Partially Supported",
    "supported": "Supported",
    "unverifiable": "Unverifiable",
}


def _confidence_bar(c: float) -> str:
    filled = round(c * 10)
    return "█" * filled + "░" * (10 - filled) + f" {c:.0%}"


def _verdict_block(v: CitationVerdict) -> str:
    emoji = _EMOJI[v.verdict]
    lines: list[str] = [
        f"### {emoji} `{v.cite_key}` — {v.verdict.upper().replace('_', ' ')}",
        f"**Confidence:** {_confidence_bar(v.confidence)}  ",
        f"**Claim:** {v.claim}  ",
        f"**Context:** _{v.claim_context}_  ",
    ]

    if v.resolution_source:
        lines.append(f"**Source resolved via:** {v.resolution_source}  ")

    if v.evidence:
        lines.append("\n**Evidence:**")
        for span in v.evidence:
            loc_parts = []
            if span.section:
                loc_parts.append(f"§{span.section}")
            if span.page:
                loc_parts.append(f"p.{span.page}")
            loc = ", ".join(loc_parts) or "unknown location"
            lines.append(f"> "{span.passage}"  ")
            lines.append(f"> — {loc} ({span.paper_id})  ")
    else:
        lines.append("\n_No evidence spans retrieved._  ")

    lines.append(f"\n**Rationale:** {v.rationale}")
    return "\n".join(lines)


def render_report(
    verdicts: list[CitationVerdict],
    uncited: list[UncitedCandidate],
    tex_path: Path,
    resolution_rate: float,
    abstract_only_count: int,
    failed_count: int,
) -> str:
    grouped: dict[Verdict, list[CitationVerdict]] = {v: [] for v in _ORDER}
    for verdict in verdicts:
        grouped[verdict.verdict].append(verdict)

    total = len(verdicts)
    lines: list[str] = [
        "# Citation Verification Report",
        "",
        f"**Paper:** `{tex_path}`  ",
        f"**Generated:** {date.today()}  ",
        f"**Total citations verified:** {total}  ",
        f"**Resolution rate:** {len(verdicts) - failed_count}/{total} "
        f"({resolution_rate:.0%})"
        + (f" — {abstract_only_count} abstract-only" if abstract_only_count else "") + "  ",
        "",
    ]

    # Summary table
    lines += [
        "## Summary",
        "",
        "| Verdict | Count |",
        "|---|---|",
    ]
    for verdict in _ORDER:
        n = len(grouped[verdict])
        if n:
            emoji = _EMOJI[verdict]
            label = _SECTION_TITLE[verdict]
            lines.append(f"| {emoji} {label} | {n} |")
    lines.append("")

    # Priority flag
    critical = len(grouped["unsupported"]) + len(grouped["misleading"])
    if critical:
        lines += [
            f"> **{critical} citation(s) require immediate attention** "
            f"(`unsupported` or `misleading`).",
            "",
        ]

    lines.append("---")
    lines.append("")

    # Verdict sections
    for verdict in _ORDER:
        items = grouped[verdict]
        if not items:
            continue
        emoji = _EMOJI[verdict]
        title = _SECTION_TITLE[verdict]
        lines += [
            f"## {emoji} {title} ({len(items)})",
            "",
        ]
        for v in items:
            lines.append(_verdict_block(v))
            lines.append("")
            lines.append("---")
            lines.append("")

    # Uncited claim candidates (separate section, never a CitationVerdict)
    if uncited:
        lines += [
            "## 📌 Uncited Claim Candidates",
            "",
            "The following sentences appear to make factual claims "
            "without any `\\cite{}` marker. Consider adding citations.",
            "",
        ]
        for i, u in enumerate(uncited, 1):
            lines.append(f"{i}. _{u.sentence}_")
        lines.append("")

    return "\n".join(lines)

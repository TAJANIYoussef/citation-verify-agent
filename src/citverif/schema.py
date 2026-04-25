from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

Verdict = Literal["supported", "partially_supported", "unsupported", "misleading", "unverifiable"]


class EvidenceSpan(BaseModel):
    passage: str
    section: str | None = None
    page: int | None = None
    paper_id: str


class CitationVerdict(BaseModel):
    cite_key: str
    claim: str
    claim_context: str
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceSpan] = []
    rationale: str
    resolution_source: str | None = None


class AgentResult(BaseModel):
    """Internal result returned by both the main verifier and reflexion agents."""
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceSpan] = []
    rationale: str


class MissingCitationCandidate(BaseModel):
    sentence: str
    reason: str   # short explanation of why this looks like it needs a citation

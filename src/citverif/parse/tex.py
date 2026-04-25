import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CitationContext:
    cite_key: str
    raw_text: str        # sentence(s) containing the \cite{}
    claim_context: str   # cleaned version without LaTeX commands


@dataclass
class UncitedCandidate:
    sentence: str        # sentence that looks like a factual claim but has no \cite{}


_CITE_RE = re.compile(r"\\cite[tp]?\*?\{([^}]+)\}")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# Patterns that suggest a factual/empirical claim
_FACTUAL_SIGNALS = re.compile(
    r"\b(show[s]?|demonstrate[s]?|prove[s]?|report[s]?|found|achieve[s]?|outperform[s]?|"
    r"improve[s]?|reduce[s]?|increase[s]?|indicate[s]?|suggest[s]?|state[s]?|"
    r"according to|it is (known|shown|proven|established))\b",
    re.IGNORECASE,
)

# Commands to strip for clean claim text
_TEX_CMD = re.compile(r"\\[a-zA-Z]+\{([^}]*)\}|\\[a-zA-Z]+\s*")
_MULTI_SPACE = re.compile(r"\s{2,}")


def _clean(text: str) -> str:
    text = _TEX_CMD.sub(r"\1", text)
    text = text.replace("~", " ").replace("--", "–")
    return _MULTI_SPACE.sub(" ", text).strip()


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def _surrounding_context(sentences: list[str], idx: int, window: int = 1) -> str:
    start = max(0, idx - window)
    end = min(len(sentences), idx + window + 1)
    return " ".join(sentences[start:end])


def parse_tex(path: Path) -> tuple[list[CitationContext], list[UncitedCandidate]]:
    """Extract citation contexts and uncited factual claims from a .tex file."""
    src = path.read_text(encoding="utf-8", errors="replace")

    # Strip comments
    src = re.sub(r"%.*", "", src)

    sentences = _split_sentences(src)

    citations: list[CitationContext] = []
    uncited: list[UncitedCandidate] = []

    for i, sentence in enumerate(sentences):
        cite_matches = _CITE_RE.findall(sentence)
        if cite_matches:
            context = _surrounding_context(sentences, i)
            for raw_keys in cite_matches:
                for key in [k.strip() for k in raw_keys.split(",")]:
                    citations.append(CitationContext(
                        cite_key=key,
                        raw_text=context,
                        claim_context=_clean(context),
                    ))
        else:
            if _FACTUAL_SIGNALS.search(sentence) and len(sentence.split()) > 8:
                uncited.append(UncitedCandidate(sentence=_clean(sentence)))

    return citations, uncited

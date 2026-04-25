# Citation Verifier Agent — Claude Code Build Prompt

You are building a citation verification agent that takes a `.tex` paper + `.bib` references and verifies, for each claim with a citation, whether the cited source actually supports the claim. This document is your single source of truth for the build.

---

## 0. READ THIS FIRST — non-negotiable Claude Code setup

Before writing any code, do these three things in order:

### 0.1 Create `CLAUDE.md` at the repo root

This is the file Claude Code (you, in future sessions) will auto-load. It must contain:

- A 3-sentence project summary (what, why, who).
- The full tech stack with exact versions (see §3).
- The architecture summary (see §4) — copy the ASCII diagram in.
- All commands needed to run, test, and lint the project (see §9).
- The "do not do this" list (see §11).
- A pointer to this file: `See CITATION_VERIFIER_PROMPT.md for the full spec.`

Keep it under 200 lines. This file is loaded into context every session — every wasted token costs us.

### 0.2 Create the `.context/` directory

Use it to store:

- `.context/decisions.md` — running log of architectural decisions with rationale. Append to it whenever you make a non-trivial choice.
- `.context/known-issues.md` — anything broken, deferred, or worth revisiting.
- `.context/test-papers/` — the 2–3 `.tex` + `.bib` pairs we use for dev testing (see §10).

This directory is project memory. If a future session needs to know "why did we pick X", the answer must be in `.context/decisions.md`.

### 0.3 Create `.claude/commands/` for reusable slash commands

At minimum create:

- `.claude/commands/run-eval.md` — runs the benchmark on the labeled set and prints metrics.
- `.claude/commands/verify-paper.md` — runs the agent end-to-end on a single `.tex` + `.bib`.
- `.claude/commands/test-resolver.md` — runs only the reference resolution step on a `.bib`, useful for debugging fetching failures.

Each command file is a markdown prompt Claude Code can invoke with `/run-eval` etc.

---

## 1. Project goal

Build a CLI tool: `verify-citations paper.tex refs.bib --out report.md`

For every `\cite{key}` in the paper, the tool produces a verdict:

- `supported` — the cited paper clearly states what the claim says
- `partially_supported` — the cited paper supports part but not all of the claim
- `unsupported` — the cited paper does not contain the claim
- `misleading` — the cited paper says something contradictory or context-twisting
- `unverifiable` — couldn't fetch the paper or no usable evidence

Each verdict carries: `confidence ∈ [0,1]`, evidence spans (passage + page/section), and a short rationale.

The output is a markdown report grouped by verdict, with the highest-priority issues (`unsupported`, `misleading`, low-confidence `supported`) at the top.

---

## 2. What this is, architecturally

**Multi-model pipeline with one agentic component.** Not multi-agent. The distinction matters — don't add agent-coordination frameworks (CrewAI, AutoGen, multi-agent LangGraph). The verifier is the only LLM that loops over tools. Everything else is deterministic code or single-shot LLM calls.

---

## 3. Tech stack (exact)

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | matches Youssef's existing setup |
| LLM runtime | Ollama (local) | 24 GB RAM ceiling on dev machine |
| Extractor model | `llama3.1:8b` | fast, runs N times per paper |
| Verifier model | `qwen2.5:14b` | best 14B tool-calling on Ollama |
| Embedder model | `bge-m3` | multilingual (handles FR refs), long-context |
| Agent framework | `pydantic-ai` | Youssef knows it; clean tool API; works with Ollama via OpenAI-compatible endpoint |
| Vector store | `chromadb` (persistent, on-disk) | one collection per cited paper |
| TeX parsing | `pylatexenc` + custom regex for `\cite{}` extraction | |
| BibTeX parsing | `bibtexparser` v2 | |
| PDF text extraction | `pymupdf` (fitz) | fastest, handles most academic PDFs |
| arXiv fetching | `arxiv` package | |
| OpenAlex / S2 / Unpaywall | `httpx` directly — small clients, no SDK needed | |
| HTTP client | `httpx` (async) | |
| CLI | `typer` | |
| Logging | `rich` | |
| Tests | `pytest` + `pytest-asyncio` | |
| Lint / format | `ruff` (lint + format both) | one tool, no black needed |

**Do not add:** LangChain, LlamaIndex, CrewAI, AutoGen, Letta, mem0. We don't need them and they bring dependency bloat.

---

## 4. Architecture

```
                  paper.tex                refs.bib
                      │                        │
                      ▼                        ▼
              ┌──────────────┐         ┌──────────────┐
              │ TeX parser   │         │ BibTeX parser│        [deterministic]
              │ claims+ctx   │         │ DOI/arXiv/ttl│
              └──────┬───────┘         └──────┬───────┘
                     │                        │
                     ▼                        ▼
              ┌──────────────┐         ┌──────────────┐
              │ llama3.1:8b  │         │ Ref resolver │
              │ claim refine │         │ arXiv→OpenAlx│
              └──────┬───────┘         │ →S2→Unpywall │
                     │                 └──────┬───────┘
                     │                        │
                     │           ┌────────────┴──┐
                     │           ▼               ▼
                     │   ┌────────────┐   ┌────────────┐
                     │   │ paper cache│   │   bge-m3   │   [persistent memory]
                     │   │ (disk PDFs)│   │  embedder  │
                     │   └─────┬──────┘   └──────┬─────┘
                     │         │                 │
                     │         └────────┬────────┘
                     │                  ▼
                     │           ┌────────────┐
                     │           │  vector    │
                     │           │  store     │
                     │           │ (per paper)│
                     │           └──────┬─────┘
                     │                  │
                     └─────────┬────────┘
                               ▼
              ┌─────────────────────────────────────┐
              │   qwen2.5:14b — verifier agent      │
              │   ReAct loop: reason→tool→observe   │   [agentic]
              │                                     │
              │   tools:                            │
              │   • semantic_search(q, paper, k)    │
              │   • fetch_section(paper, sec)       │
              │   • web_search(q)                   │
              └──────────────┬──────────────────────┘
                             │
                             ▼
                  ┌──────────────────┐
                  │ confidence < τ ? │──no──┐
                  └────────┬─────────┘      │
                           │ yes            │
                           ▼                │
                  ┌──────────────────┐      │
                  │ Reflexion pass   │      │
                  │ counter-evidence │      │
                  └────────┬─────────┘      │
                           │                │
                           └───────┬────────┘
                                   ▼
                          ┌─────────────────┐
                          │ verdict + report│
                          └─────────────────┘
```

---

## 5. Repo layout

```
citation-verifier/
├── CLAUDE.md                     # Claude Code auto-context (≤200 lines)
├── CITATION_VERIFIER_PROMPT.md   # this file
├── README.md                     # user-facing
├── pyproject.toml
├── .context/
│   ├── decisions.md
│   ├── known-issues.md
│   └── test-papers/
├── .claude/
│   └── commands/
│       ├── run-eval.md
│       ├── verify-paper.md
│       └── test-resolver.md
├── src/
│   └── citverif/
│       ├── __init__.py
│       ├── cli.py                # typer entry point
│       ├── parse/
│       │   ├── tex.py            # \cite{} extraction + surrounding context
│       │   └── bib.py            # bibtexparser wrapper
│       ├── extract/
│       │   └── claims.py         # llama3.1:8b — claim refinement
│       ├── resolve/
│       │   ├── arxiv.py
│       │   ├── openalex.py
│       │   ├── semantic_scholar.py
│       │   ├── unpaywall.py
│       │   └── chain.py          # the fallback chain
│       ├── memory/
│       │   ├── paper_cache.py    # disk cache by DOI/arXiv ID
│       │   ├── chunker.py
│       │   └── vector_store.py   # chromadb wrapper, one collection per paper
│       ├── agent/
│       │   ├── verifier.py       # qwen2.5:14b — pydantic-ai agent
│       │   ├── tools.py          # semantic_search, fetch_section, web_search
│       │   ├── prompts.py        # system prompts (one per pass)
│       │   └── reflexion.py      # second-pass counter-evidence prompt
│       ├── report/
│       │   └── markdown.py       # final report rendering
│       └── eval/
│           ├── dataset.py        # load labeled (claim, cite_key, verdict) triples
│           ├── metrics.py        # macro-F1, evidence precision, hallucination rate
│           └── runner.py         # benchmark matrix
├── tests/
│   ├── unit/
│   └── integration/
└── data/
    ├── papers/                   # downloaded PDFs (gitignored)
    ├── chroma/                   # vector store (gitignored)
    └── eval/
        └── ground_truth.jsonl    # 50–100 hand-labeled triples
```

Anything in `data/` is gitignored. The `.context/test-papers/` are committed (small `.tex` + `.bib`, no PDFs).

---

## 6. Verdict schema (Pydantic)

```python
from pydantic import BaseModel, Field
from typing import Literal

Verdict = Literal["supported", "partially_supported", "unsupported", "misleading", "unverifiable"]

class EvidenceSpan(BaseModel):
    passage: str
    section: str | None = None
    page: int | None = None
    paper_id: str   # arXiv ID or DOI

class CitationVerdict(BaseModel):
    cite_key: str
    claim: str
    claim_context: str          # surrounding sentence(s) from the source paper
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceSpan] = []
    rationale: str              # 1–3 sentences, why this verdict
    resolution_source: str | None = None   # "arxiv" | "openalex" | "s2" | "unpaywall" | "web" | None
```

The verifier agent must return this exact schema. Use `pydantic-ai`'s `result_type=CitationVerdict`.

---

## 7. The agent — implementation contract

`src/citverif/agent/verifier.py`:

- One `Agent` per process, instantiated with `qwen2.5:14b` via Ollama's OpenAI-compat endpoint (`http://localhost:11434/v1`).
- Three tools registered: `semantic_search`, `fetch_section`, `web_search`.
- System prompt lives in `prompts.py`. Two prompts: `MAIN_VERIFIER_PROMPT` and `REFLEXION_PROMPT`.
- The main prompt instructs: "Search for evidence first. Make at most 4 tool calls. Then commit to a verdict. Default to `unverifiable` over guessing."
- The reflexion prompt instructs: "You previously concluded X with confidence C. Find evidence that *contradicts* this conclusion. If you find none, increase confidence; if you find some, revise the verdict."
- Reflexion runs only when `confidence < 0.6` OR verdict is `supported` AND the original ReAct loop made fewer than 2 tool calls (the over-confident-paraphrase failure mode).

**Tool call budget per claim: 4 in main pass, 3 in reflexion. Hard cap.**

---

## 8. Reference resolution — the part that will surprise us

This is the hardest layer. Implement and test it *first*, before the agent. Variance here will dominate end-to-end accuracy.

Resolution chain (`src/citverif/resolve/chain.py`):

1. If `arxiv_id` field in bib → arXiv API → PDF.
2. If `doi` → OpenAlex (free, has abstracts and OA PDF links).
3. If only title → Semantic Scholar Graph API search by title (top-1 with fuzzy match ≥ 0.85).
4. Try Unpaywall for OA PDF if we have a DOI but no PDF yet.
5. Fall back to abstract-only verification if no full text is available — record this and downgrade max possible confidence to 0.7.
6. If everything fails → `unverifiable`, `resolution_source=None`.

**Track resolution rate as a separate metric.** A 90% verdict accuracy on 60% resolved papers is a worse system than 80% verdict accuracy on 90% resolved.

---

## 9. Commands (also document these in `CLAUDE.md`)

```bash
# install
uv pip install -e ".[dev]"     # use uv, not pip

# pull models (one-time)
ollama pull llama3.1:8b
ollama pull qwen2.5:14b
ollama pull bge-m3

# run on a paper
verify-citations paper.tex refs.bib --out report.md

# run benchmark
python -m citverif.eval.runner --models qwen2.5:14b --loops react,react+reflexion

# tests
pytest tests/ -v
pytest tests/unit/ -v          # fast, no LLM calls
pytest tests/integration/ -v   # slow, hits Ollama

# lint + format
ruff check src/ tests/
ruff format src/ tests/
```

---

## 10. Build order (do not skip steps)

Build in phases. Open a PR / commit per phase. Do not start phase N+1 until phase N has tests passing.

**Phase 1 — parsing + resolution (no LLM yet)**
- `parse/tex.py`, `parse/bib.py`, `resolve/*`, `memory/paper_cache.py`.
- Goal: given a `.tex` + `.bib`, print `(cite_key, claim, resolved_pdf_path | None, source)` for every citation.
- Tests: 2 hand-built `.tex`+`.bib` in `.context/test-papers/` covering arXiv refs, DOI refs, and broken/unfindable refs.
- Definition of done: ≥80% resolution rate on a real ML paper's bib.

**Phase 2 — chunking + embedding + retrieval**
- `memory/chunker.py`, `memory/vector_store.py`.
- Chunk size: 512 tokens with 64 overlap. Section-aware (don't split across `\section`).
- Goal: given a paper and a query string, return top-5 passages with section + page metadata.
- Tests: golden-passage retrieval — given a known claim about a known paper, the gold passage must be in top-5.

**Phase 3 — claim extraction**
- `extract/claims.py`.
- llama3.1:8b refines the raw `(text-around-cite, cite_key)` into a clean atomic claim.
- Tests: 10 hand-labeled examples.

**Phase 4 — verifier agent (single pass, no reflexion)**
- `agent/verifier.py`, `agent/tools.py`, `agent/prompts.py`.
- Goal: end-to-end verdict for one citation.
- Tests: 5 hand-labeled `(claim, paper, expected_verdict)` triples — must hit ≥3/5.

**Phase 5 — reflexion pass**
- `agent/reflexion.py`.
- Triggered on the conditions in §7.
- Tests: regression — reflexion must not flip a high-confidence correct verdict to wrong.

**Phase 6 — eval harness**
- `eval/*`, ground_truth.jsonl with 50–100 triples.
- The triples come from Youssef's own papers (AutoSMOTE-NC, MTSOT). Ask him to label them — don't fabricate.
- Macro-F1 across the 5 verdict classes, evidence precision (manual spot-check on 20 random verdicts), hallucination rate (false-positive `supported`), latency, resolution rate.

**Phase 7 — report rendering + CLI polish**
- `report/markdown.py`, `cli.py`.
- Group by verdict; `unsupported` and `misleading` at top; include evidence quotes with paper IDs.

---

## 11. Do NOT do this

- **Do not** add multi-agent orchestration. One agent. One loop.
- **Do not** add LangChain, LlamaIndex, CrewAI, AutoGen.
- **Do not** expose the parsers, resolver, or embedder as agent tools. They run before the agent.
- **Do not** let the agent make unbounded tool calls. Hard cap at 4 (main) + 3 (reflexion).
- **Do not** use `langchain-ollama`. Use Ollama's OpenAI-compatible endpoint via `pydantic-ai` directly.
- **Do not** hardcode API keys. OpenAlex needs none; S2 takes an optional key from env; Unpaywall needs an email in env (`UNPAYWALL_EMAIL`).
- **Do not** invent ground-truth labels. The eval set must be hand-labeled by Youssef.
- **Do not** write code that assumes papers are in English only. `bge-m3` is multilingual for a reason.
- **Do not** silently swallow resolution failures. Log them, count them, surface them in the report.
- **Do not** put paper PDFs in git. `data/` is gitignored.
- **Do not** make the verifier prompt suggest a "default" verdict like `supported`. The default in genuine uncertainty is `unverifiable`.

---

## 12. Open questions to ask Youssef before starting

1. Should the tool also check **uncited claims** (claims that should have a citation but don't)? Out of scope for v1 unless you say yes.
2. Do you want the report in French, English, or both? Default: English, but the verifier reads multilingual sources.
3. For the eval ground truth: will you label 50 or 100 triples? More is better for the F1 stability but costs your time.
4. Do you want a `--fast` mode that skips reflexion entirely for quick checks?

Pause and ask these before phase 1. Don't assume.

---

## 13. First action when you start

1. Create `CLAUDE.md` per §0.1.
2. Create `.context/decisions.md` and write the first entry: "Adopted multi-model pipeline architecture per CITATION_VERIFIER_PROMPT.md §2."
3. Create the empty repo layout from §5 (`mkdir`s + empty `__init__.py`s).
4. Set up `pyproject.toml` with the deps from §3.
5. Ask the open questions in §12.
6. Begin phase 1.

Stop after step 5. Wait for answers.

# Citation Verifier

A local, fully offline CLI tool that checks every `\cite{key}` in a LaTeX paper and tells you whether the cited source actually supports the claim being made.

```
verify-citations paper.tex refs.bib --out report.md
```

No cloud APIs. No data leaves your machine. Runs entirely on [Ollama](https://ollama.com) with open-weight models.

---

## Why

Manual citation checking is slow and error-prone. A paper with 40 references can hide unsupported claims, misquoted statistics, or subtly misleading citations. This tool automates the grunt work: it fetches every cited paper, embeds it into a local vector store, and runs a reasoning agent to return a structured verdict with evidence quotes and confidence scores.

---

## What it produces

For every `\cite{key}` in your paper, the tool produces one of five verdicts:

| Verdict | Meaning |
|---|---|
| `supported` | The cited paper clearly states what the claim says |
| `partially_supported` | The cited paper supports part but not all of the claim |
| `unsupported` | The cited paper does not contain the claim |
| `misleading` | The cited paper says something that contradicts or twists the claim |
| `unverifiable` | Could not fetch the paper or find usable evidence |

Each verdict includes a **confidence score**, **evidence spans** (direct quotes with page and section), and a **rationale**.

The report also flags sentences in your paper that look like factual claims but have no `\cite{}` attached.

---

## Architecture

```
          paper.tex                refs.bib
              │                        │
              ▼                        ▼
      ┌──────────────┐         ┌──────────────┐
      │  TeX parser  │         │ BibTeX parser│   [deterministic]
      │  claims+ctx  │         │ DOI/arXiv/ttl│
      └──────┬───────┘         └──────┬───────┘
             │                        │
             ▼                        ▼
      ┌──────────────┐         ┌──────────────────────┐
      │ llama3.1:8b  │         │    Ref resolver       │
      │ claim refine │         │ arXiv → OpenAlex      │
      └──────┬───────┘         │ → S2 → Unpaywall      │
             │                 └──────────┬─────────────┘
             │                            │
             │              ┌─────────────┴──┐
             │              ▼                ▼
             │      ┌────────────┐   ┌────────────┐
             │      │ paper cache│   │   bge-m3   │
             │      │ (disk PDFs)│   │  embedder  │
             │      └─────┬──────┘   └──────┬─────┘
             │            └────────┬─────────┘
             │                     ▼
             │              ┌────────────┐
             │              │  chromadb  │
             │              │ (per paper)│
             │              └──────┬─────┘
             └─────────┬───────────┘
                       ▼
      ┌─────────────────────────────────────┐
      │   qwen2.5:14b  —  verifier agent    │
      │   ReAct loop  (max 4 tool calls)    │
      │   tools: semantic_search            │
      │           fetch_section             │
      │           web_search                │
      └──────────────┬──────────────────────┘
                     │
                     ▼
         confidence < 0.6 or fast verdict?
                     │
                     ▼
          Reflexion pass (max 3 calls)
                     │
                     ▼
              verdict + report.md
```

**One agentic component** (the verifier). Everything else is deterministic code or single-shot LLM calls.

---

## Models

| Role | Model | Why |
|---|---|---|
| Claim extraction | `llama3.1:8b` | Fast, runs once per citation |
| Verification agent | `qwen2.5:14b` | Best 14B tool-calling model on Ollama |
| Embedder | `bge-m3` | Multilingual, long-context |

All models run locally via Ollama. No API keys required for the models themselves.

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- ~10 GB disk for the three models
- 16 GB RAM minimum (24 GB recommended for qwen2.5:14b)

---

## Installation

```bash
# Clone the repo
git clone https://github.com/TAJANIYoussef/citation-verifier.git
cd citation-verifier

# Install (use uv for speed, or regular pip)
uv pip install -e ".[dev]"
# or: pip install -e ".[dev]"

# Pull the three models (one-time, ~10 GB total)
ollama pull llama3.1:8b
ollama pull qwen2.5:14b
ollama pull bge-m3
```

---

## Usage

### Verify a paper

```bash
verify-citations paper.tex refs.bib --out report.md
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--out` | `report.md` | Output report path |
| `--cache-dir` | `data/papers` | Where to cache downloaded PDFs |
| `--store-dir` | `data/chroma` | Where to store vector embeddings |
| `--verbose` | off | Show debug logs |

### Read the report

The report is a Markdown file grouped by verdict severity:

```
# Citation Verification Report

## Summary
| Verdict            | Count |
|--------------------|-------|
| 🚫 Unsupported     | 2     |
| ⚠️ Misleading      | 1     |
| 🔶 Partially Supported | 3  |
| ✅ Supported       | 18    |
| ❓ Unverifiable    | 1     |

> 3 citation(s) require immediate attention.

---

## 🚫 Unsupported (2)

### 🚫 `smith2020method` — UNSUPPORTED
**Confidence:** ███████░░░ 70%
**Claim:** The method achieves 99% accuracy on CIFAR-10.

**Evidence:**
> "Our method achieves 84.3% accuracy on CIFAR-10, comparable to prior work."
> — §Results, p.7 (smith2020method)

**Rationale:** The cited paper reports 84.3%, not 99%. The claim overstates the result.

...

## 📌 Uncited Claim Candidates

1. _Transformer models have been shown to outperform RNNs on all sequence tasks._
```

---

## Reference resolution

The tool tries four sources in order for each `.bib` entry:

1. **arXiv** — if `eprint` or `archivePrefix` is set
2. **OpenAlex** — if `doi` is set (free, no key needed)
3. **Semantic Scholar** — title fuzzy-match (≥ 0.85 similarity)
4. **Unpaywall** — OA PDF fallback (requires `UNPAYWALL_EMAIL` env var)

If only an abstract is available (no full PDF), the agent still runs but confidence is capped at **0.7**.

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `UNPAYWALL_EMAIL` | Optional | Your email for the Unpaywall API |
| `S2_API_KEY` | Optional | Semantic Scholar API key (raises rate limits) |

No other API keys needed. OpenAlex and arXiv are free and keyless.

---

## Running tests

```bash
# Unit tests — fast, no Ollama needed
pytest tests/unit/ -v

# Integration tests — slow, requires Ollama running with all 3 models
pytest tests/integration/ -v

# All tests
pytest tests/ -v
```

---

## Evaluation benchmark

The tool ships with an evaluation harness that measures macro-F1 across all five verdict classes, hallucination rate (false-positive `supported`), resolution rate, and latency.

```bash
# Run benchmark (requires labeled ground_truth.jsonl and indexed papers)
citverif-eval --ground-truth data/eval/ground_truth.jsonl \
              --loops react,react+reflexion

# Quick smoke-test on the first 10 entries
citverif-eval --limit 10
```

Results are written to `data/eval/results/` as JSONL (per-entry) and JSON (summary metrics). The runner is crash-recoverable — it skips already-completed entries on restart.

### Labeling format

Add entries to `data/eval/ground_truth.jsonl` (one JSON object per line):

```json
{
  "cite_key": "vaswani2017attention",
  "claim": "The Transformer achieves 28.4 BLEU on WMT 2014 English-to-German.",
  "claim_context": "The Transformer achieves 28.4 BLEU on WMT 2014 English-to-German.",
  "expected_verdict": "supported",
  "source": "arxiv",
  "notes": "Table 2, row 1 — exact figure."
}
```

Fields: `cite_key` (required), `claim` (required), `expected_verdict` (required), `claim_context` (optional, defaults to `claim`), `paper_id` (optional, defaults to `cite_key`), `source`, `notes`.

---

## Project structure

```
citation-verifier/
├── src/citverif/
│   ├── cli.py                 # Entry point: verify-citations
│   ├── schema.py              # CitationVerdict, EvidenceSpan, AgentResult
│   ├── parse/
│   │   ├── tex.py             # \cite{} extraction + uncited claim detection
│   │   └── bib.py             # BibTeX parser (DOI, arXiv ID, authors)
│   ├── resolve/
│   │   ├── chain.py           # arXiv → OpenAlex → S2 → Unpaywall fallback
│   │   ├── arxiv.py
│   │   ├── openalex.py
│   │   ├── semantic_scholar.py
│   │   └── unpaywall.py
│   ├── memory/
│   │   ├── paper_cache.py     # Disk cache for PDFs and abstracts
│   │   ├── chunker.py         # PDF → 512-token chunks (section-aware)
│   │   └── vector_store.py    # chromadb wrapper (one collection per paper)
│   ├── extract/
│   │   └── claims.py          # llama3.1:8b claim refinement
│   ├── agent/
│   │   ├── verifier.py        # qwen2.5:14b ReAct agent (main pass)
│   │   ├── reflexion.py       # Counter-evidence pass
│   │   ├── tools.py           # semantic_search, fetch_section, web_search
│   │   └── prompts.py         # System prompts
│   ├── report/
│   │   └── markdown.py        # Report renderer
│   └── eval/
│       ├── dataset.py         # ground_truth.jsonl loader
│       ├── metrics.py         # Macro-F1, hallucination rate, resolution rate
│       └── runner.py          # Benchmark CLI: citverif-eval
├── tests/
│   ├── unit/                  # Fast tests, no Ollama
│   └── integration/           # Slow tests, requires Ollama
├── data/
│   ├── papers/                # Downloaded PDFs (gitignored)
│   ├── chroma/                # Vector store (gitignored)
│   └── eval/
│       └── ground_truth.jsonl # Labeled eval triples
└── .context/
    ├── decisions.md           # Architectural decision log
    ├── known-issues.md        # Deferred items
    └── test-papers/           # Small .tex + .bib pairs for dev testing
```

---

## Limitations

- **qwen2.5:14b requires ~10 GB VRAM** (or ~24 GB RAM for CPU offload). Smaller machines will be slow.
- **Resolution depends on open access**. Papers behind paywalls with no OA PDF will fall back to abstract-only verification (confidence capped at 0.7) or `unverifiable`.
- **Non-English papers** are supported for retrieval (bge-m3 is multilingual), but the report is always in English.
- **Tool call budget is hard-capped**: 4 calls in the main pass, 3 in reflexion. The agent cannot loop indefinitely.

---

## License

MIT

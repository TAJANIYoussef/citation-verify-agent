# Citation Verifier — Claude Code Context

## Project summary
A CLI tool (`verify-citations paper.tex refs.bib --out report.md`) that checks every `\cite{key}` in a LaTeX paper and returns a structured verdict (supported / partially_supported / unsupported / misleading / unverifiable) with evidence spans and a confidence score. It exists because manual citation checking is slow and error-prone, and is built for Youssef's research workflow.

See `CITATION_VERIFIER_PROMPT.md` for the full spec.

---

## Tech stack

| Layer | Choice | Version |
|---|---|---|
| Language | Python | 3.11+ |
| LLM runtime | Ollama (local) | latest |
| Extractor model | `llama3.1:8b` | — |
| Verifier model | `qwen2.5:14b` | — |
| Embedder model | `bge-m3` | — |
| Agent framework | `pydantic-ai` | ≥0.0.14 |
| Vector store | `chromadb` | ≥0.5 |
| TeX parsing | `pylatexenc` + custom regex | ≥2.10 |
| BibTeX parsing | `bibtexparser` | v2 |
| PDF extraction | `pymupdf` (fitz) | ≥1.24 |
| arXiv fetching | `arxiv` | ≥2.1 |
| HTTP client | `httpx` (async) | ≥0.27 |
| CLI | `typer` | ≥0.12 |
| Logging | `rich` | ≥13 |
| Tests | `pytest` + `pytest-asyncio` | — |
| Lint/format | `ruff` | ≥0.4 |

**Do NOT add:** LangChain, LlamaIndex, CrewAI, AutoGen, langchain-ollama, Letta, mem0.

---

## Architecture

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
             │           │  chromadb  │
             │           │ (per paper)│
             │           └──────┬─────┘
             │                  │
             └─────────┬────────┘
                       ▼
      ┌─────────────────────────────────────┐
      │   qwen2.5:14b — verifier agent      │
      │   ReAct loop (max 4 tool calls)     │
      │   tools: semantic_search,           │
      │           fetch_section, web_search │
      └──────────────┬──────────────────────┘
                     │
                     ▼
          confidence < 0.6 → Reflexion pass (max 3 calls)
                     │
                     ▼
            verdict + report.md
```

---

## Build status
All 7 phases implemented. Unit tests cover every module with no Ollama dependency.
Integration tests require `ollama serve` with `llama3.1:8b`, `qwen2.5:14b`, `bge-m3`.

## Commands

```bash
# Install
uv pip install -e ".[dev]"

# Pull models (one-time)
ollama pull llama3.1:8b
ollama pull qwen2.5:14b
ollama pull bge-m3

# Run on a paper
verify-citations paper.tex refs.bib --out report.md

# Run benchmark
citverif-eval --ground-truth data/eval/ground_truth.jsonl --loops react,react+reflexion
# Quick smoke-test (first 10 entries)
citverif-eval --limit 10

# Tests
pytest tests/ -v
pytest tests/unit/ -v          # fast, no LLM calls
pytest tests/integration/ -v   # slow, hits Ollama

# Lint + format
ruff check src/ tests/
ruff format src/ tests/
```

---

## Do NOT do this

- No multi-agent orchestration — one agent, one ReAct loop.
- No LangChain / LlamaIndex / CrewAI / AutoGen.
- Don't expose parsers/resolver/embedder as agent tools — they run before the agent.
- Hard cap: 4 tool calls (main pass) + 3 (reflexion). No unbounded loops.
- Don't use `langchain-ollama`; use Ollama's OpenAI-compat endpoint via pydantic-ai.
- Don't hardcode API keys. S2 key from env; Unpaywall email from `UNPAYWALL_EMAIL` env var.
- Don't invent ground-truth eval labels — Youssef labels them.
- Don't assume English-only papers — bge-m3 is multilingual.
- Don't silently swallow resolution failures — log, count, and surface them.
- Don't put PDFs in git (`data/` is gitignored).
- Default verdict in genuine uncertainty: `unverifiable`, never `supported`.

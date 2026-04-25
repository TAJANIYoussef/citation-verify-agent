# Architectural Decisions

## 2026-04-25 — §12 open questions resolved

**Uncited claims — in scope for v1.** Missing-citation findings are emitted as a **separate section** in the markdown report. The 5-verdict `CitationVerdict` schema is unchanged — no 6th verdict type.

**Report language — English only.**

**Eval set — hundreds of triples from third-party papers** (not Youssef's own papers). Supersedes §10's reference to AutoSMOTE-NC / MTSOT. Eval runner must handle large JSONL files.

**No `--fast` mode.** Reflexion always runs; accuracy over latency.

---

## 2026-04-25 — Multi-model pipeline architecture

Adopted multi-model pipeline architecture per CITATION_VERIFIER_PROMPT.md §2.

**Decision:** Use a sequential pipeline (deterministic parsing → llama3.1:8b claim extraction → reference resolution → bge-m3 embedding → qwen2.5:14b verifier agent) rather than a multi-agent framework.

**Rationale:** The verifier is the only component that needs to loop over tools. Everything else is either deterministic code or a single-shot LLM call. Adding agent-coordination frameworks (CrewAI, AutoGen, LangGraph multi-agent) would add dependency weight and complexity with no benefit for this use case.

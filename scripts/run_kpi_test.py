#!/usr/bin/env python3
"""
One-step KPI test runner.

Steps:
  1. Resolve + download + chunk + index all papers in eval_paper.tex/bib
  2. Run the verifier agent against ground_truth.jsonl (both react and react+reflexion)
  3. Print a full KPI dashboard

Usage:
    python scripts/run_kpi_test.py
    python scripts/run_kpi_test.py --limit 10        # quick smoke-test
    python scripts/run_kpi_test.py --skip-index       # if papers already indexed
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
TEX  = ROOT / ".context/test-papers/eval_paper.tex"
BIB  = ROOT / ".context/test-papers/eval_paper.bib"
GT   = ROOT / "data/eval/ground_truth.jsonl"
CACHE_DIR = ROOT / "data/papers"
STORE_DIR = ROOT / "data/chroma"
OUT_DIR   = ROOT / "data/eval/results"

MODES = ["react", "react+reflexion"]
VERDICT_ORDER = ["unsupported", "misleading", "partially_supported", "supported", "unverifiable"]
COLOR = {
    "unsupported":       "\033[91m",   # red
    "misleading":        "\033[91m",   # red
    "partially_supported": "\033[93m", # yellow
    "supported":         "\033[92m",   # green
    "unverifiable":      "\033[90m",   # grey
    "RESET":             "\033[0m",
    "BOLD":              "\033[1m",
    "CYAN":              "\033[96m",
    "DIM":               "\033[2m",
}

def c(text: str, *keys: str) -> str:
    codes = "".join(COLOR[k] for k in keys)
    return f"{codes}{text}{COLOR['RESET']}"


# ── Step 1: Index ─────────────────────────────────────────────────────────────

def run_index() -> None:
    print(c("\n▶ Step 1 — Resolving and indexing papers", "BOLD", "CYAN"))
    print(f"  tex : {TEX}")
    print(f"  bib : {BIB}")

    from citverif.parse.tex import parse_tex
    from citverif.parse.bib import parse_bib
    from citverif.resolve.chain import resolve_all
    from citverif.memory.paper_cache import PaperCache
    from citverif.memory.chunker import chunk_pdf
    from citverif.memory.vector_store import VectorStore

    citations, uncited = parse_tex(TEX)
    bib_entries = parse_bib(BIB)
    print(f"  {len(citations)} cited claims  |  {len(bib_entries)} bib entries  |  "
          f"{len(uncited)} uncited candidates")

    print("  Resolving references (fetching PDFs / abstracts)...")
    cache = PaperCache(CACHE_DIR)
    results = asyncio.run(resolve_all(bib_entries, cache))

    resolved = [r for r in results if r.pdf_path or r.abstract]
    failed   = [r for r in results if not r.pdf_path and not r.abstract]
    rate = len(resolved) / len(results) if results else 0.0

    for r in results:
        if r.pdf_path:
            status = c("PDF", "BOLD", "supported")
        elif r.abstract:
            status = c("abstract", "DIM")
        else:
            status = c("FAILED", "unsupported")
        print(f"    {r.cite_key:<35} [{r.source or '—':12}]  {status}")

    print(f"\n  Resolution rate: {c(f'{len(resolved)}/{len(results)} ({rate:.0%})', 'BOLD')}")
    if failed:
        print(f"  {c('Unresolved:', 'unsupported')} {[r.cite_key for r in failed]}")

    print("\n  Chunking and embedding PDFs into chromadb...")
    store = VectorStore(STORE_DIR)
    indexed = 0
    for ref in results:
        if not ref.pdf_path:
            continue
        if store.has_paper(ref.cite_key):
            print(f"    {ref.cite_key:<35} already indexed, skip")
            continue
        chunks = chunk_pdf(ref.pdf_path, ref.cite_key)
        if chunks:
            store.index_chunks(chunks)
            indexed += 1
            print(f"    {ref.cite_key:<35} {c(f'{len(chunks)} chunks', 'supported')}")
        else:
            print(f"    {ref.cite_key:<35} {c('no chunks extracted', 'DIM')}")

    print(f"\n  {c(f'{indexed} papers newly indexed', 'BOLD')}")


# ── Step 2: Eval ──────────────────────────────────────────────────────────────

async def _run_entry(entry, store, disable_reflexion: bool):
    from citverif.agent.verifier import verify_citation
    from citverif.extract.claims import RefinedClaim
    from citverif.parse.tex import CitationContext
    from citverif.resolve.chain import ResolvedRef

    ctx = CitationContext(
        cite_key=entry.cite_key,
        raw_text=entry.claim_context,
        claim_context=entry.claim_context,
    )
    refined = RefinedClaim(claim=entry.claim, is_verifiable=True)
    ref = ResolvedRef(
        cite_key=entry.cite_key,
        pdf_path=Path("/indexed"),
        abstract=None,
        source=entry.source,
        abstract_only=False,
    )

    if not store.has_paper(entry.paper_id):
        return "unverifiable", 0.0, False

    t0 = time.perf_counter()
    if disable_reflexion:
        with patch("citverif.agent.reflexion.needs_reflexion", return_value=False):
            verdict = await verify_citation(ctx, refined, ref, store)
    else:
        verdict = await verify_citation(ctx, refined, ref, store)
    elapsed = time.perf_counter() - t0

    return verdict.verdict, elapsed, True


async def run_eval_mode(entries, store, mode: str) -> dict:
    disable_reflexion = (mode == "react")
    expected, predicted, latencies, resolved_flags = [], [], [], []

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUT_DIR / f"results_{mode}.jsonl"

    # Crash recovery
    done: set[str] = set()
    if result_path.exists():
        with result_path.open() as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done.add(obj["cite_key"] + "|" + obj["claim"][:30])
                    expected.append(obj["expected"])
                    predicted.append(obj["predicted"])
                    latencies.append(obj["latency_s"])
                    resolved_flags.append(obj["resolved"])
                except Exception:
                    pass

    todo = [e for e in entries
            if (e.cite_key + "|" + e.claim[:30]) not in done]

    if done:
        print(f"  Resuming: {len(done)} done, {len(todo)} remaining")

    for i, entry in enumerate(todo):
        label = f"[{i+1}/{len(todo)}] {entry.cite_key}"
        print(f"  {c(label, 'DIM')} ...", end="", flush=True)

        pred, latency, resolved = await _run_entry(entry, store, disable_reflexion)

        correct = pred == entry.expected_verdict
        mark = c("✓", "supported") if correct else c("✗", "unsupported")
        print(f"\r  {mark} {label:<50} "
              f"expected={entry.expected_verdict:<20} "
              f"got={c(pred, 'BOLD'):<20} "
              f"({latency:.1f}s)")

        record = {
            "cite_key": entry.cite_key,
            "claim": entry.claim,
            "expected": entry.expected_verdict,
            "predicted": pred,
            "latency_s": round(latency, 2),
            "resolved": resolved,
            "mode": mode,
        }
        with result_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        expected.append(entry.expected_verdict)
        predicted.append(pred)
        latencies.append(latency)
        resolved_flags.append(resolved)

    return {
        "mode": mode,
        "expected": expected,
        "predicted": predicted,
        "latencies": latencies,
        "resolved_flags": resolved_flags,
    }


# ── Step 3: KPI dashboard ─────────────────────────────────────────────────────

def print_kpi(run_results: list[dict]) -> None:
    from citverif.eval.metrics import compute_metrics, ALL_VERDICTS

    print(c("\n" + "═" * 72, "BOLD"))
    print(c("  KPI DASHBOARD", "BOLD", "CYAN"))
    print(c("═" * 72, "BOLD"))

    # Per-mode table
    for run in run_results:
        mode = run["mode"]
        m = compute_metrics(
            run["expected"], run["predicted"],
            run["latencies"], run["resolved_flags"]
        )

        print(f"\n  {c(f'Mode: {mode}', 'BOLD')}")
        print(f"  {'─' * 60}")
        print(f"  {'Verdict':<22} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>5}")
        print(f"  {'─' * 60}")
        for v in ALL_VERDICTS:
            vm = m.per_class.get(v)
            if not vm:
                continue
            col = COLOR.get(v, "")
            reset = COLOR["RESET"]
            print(f"  {col}{v:<22}{reset} {vm.precision:>6.3f} {vm.recall:>6.3f} "
                  f"{vm.f1:>6.3f} {vm.support:>5}")
        print(f"  {'─' * 60}")

        macro_col = "supported" if m.macro_f1 >= 0.6 else "unsupported"
        hall_col  = "unsupported" if m.hallucination_rate > 0.1 else "supported"
        res_col   = "unsupported" if m.resolution_rate < 0.7 else "supported"

        print(f"  {'Macro-F1':<22} {c(f'{m.macro_f1:.3f}', macro_col, 'BOLD'):>40}")
        print(f"  {'Accuracy':<22} {m.accuracy:.1%}  ({m.correct}/{m.total})")
        print(f"  {'Hallucination rate':<22} {c(f'{m.hallucination_rate:.1%}', hall_col)}")
        print(f"  {'Resolution rate':<22} {c(f'{m.resolution_rate:.1%}', res_col)}")
        print(f"  {'Mean latency':<22} {m.mean_latency_s:.1f}s / citation")

    # Side-by-side macro-F1 comparison
    if len(run_results) == 2:
        print(c("\n  Side-by-side comparison", "BOLD"))
        print(f"  {'─' * 50}")
        metrics = []
        for run in run_results:
            from citverif.eval.metrics import compute_metrics
            metrics.append((
                run["mode"],
                compute_metrics(run["expected"], run["predicted"],
                                run["latencies"], run["resolved_flags"])
            ))
        for mode, m in metrics:
            print(f"  {mode:<25} macro-F1={m.macro_f1:.3f}  "
                  f"acc={m.accuracy:.0%}  halluc={m.hallucination_rate:.0%}  "
                  f"latency={m.mean_latency_s:.1f}s")

        # Reflexion delta
        base_f1 = metrics[0][1].macro_f1
        reflex_f1 = metrics[1][1].macro_f1
        delta = reflex_f1 - base_f1
        sign = "+" if delta >= 0 else ""
        col = "supported" if delta >= 0 else "unsupported"
        print(f"\n  Reflexion delta: {c(f'{sign}{delta:.3f} macro-F1', col, 'BOLD')}")

    print(c("\n" + "═" * 72, "BOLD"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="One-step KPI test runner")
    parser.add_argument("--skip-index", action="store_true",
                        help="Skip paper download/indexing (papers already in store)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap eval entries for a quick smoke-test (0 = all)")
    parser.add_argument("--modes", default="react,react+reflexion",
                        help="Comma-separated loop modes")
    args = parser.parse_args()

    print(c("\n  Citation Verifier — KPI Test Suite", "BOLD", "CYAN"))
    print(f"  ground truth : {GT}")
    print(f"  store dir    : {STORE_DIR}")

    # ── Step 1: index ─────────────────────────────────────────────────────────
    if not args.skip_index:
        run_index()
    else:
        print(c("\n▶ Step 1 — Skipped (--skip-index)", "DIM"))

    # ── Step 2: eval ──────────────────────────────────────────────────────────
    from citverif.eval.dataset import load_dataset
    from citverif.memory.vector_store import VectorStore

    entries = load_dataset(GT)
    if args.limit > 0:
        entries = entries[:args.limit]

    store = VectorStore(STORE_DIR)
    modes = [m.strip() for m in args.modes.split(",")]
    run_results = []

    for mode in modes:
        print(c(f"\n▶ Step 2 — Evaluating [{mode}]  ({len(entries)} entries)", "BOLD", "CYAN"))
        result = asyncio.run(run_eval_mode(entries, store, mode))
        run_results.append(result)

    # ── Step 3: KPI dashboard ─────────────────────────────────────────────────
    print_kpi(run_results)


if __name__ == "__main__":
    main()

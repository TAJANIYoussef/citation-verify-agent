"""
Benchmark runner.

Usage:
    python -m citverif.eval.runner \\
        --models qwen2.5:14b \\
        --loops react,react+reflexion \\
        --ground-truth data/eval/ground_truth.jsonl \\
        --store-dir data/chroma \\
        --out data/eval/results
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from unittest.mock import patch

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from citverif.agent.verifier import verify_citation
from citverif.eval.dataset import EvalEntry, load_dataset, save_result_line
from citverif.eval.metrics import compute_metrics, format_metrics_table
from citverif.extract.claims import RefinedClaim
from citverif.memory.vector_store import VectorStore
from citverif.parse.tex import CitationContext
from citverif.resolve.chain import ResolvedRef
from citverif.schema import Verdict

log = logging.getLogger(__name__)
console = Console()

app = typer.Typer(add_completion=False)


def _entry_to_inputs(entry: EvalEntry) -> tuple[CitationContext, RefinedClaim, ResolvedRef]:
    ctx = CitationContext(
        cite_key=entry.cite_key,
        raw_text=entry.claim_context,
        claim_context=entry.claim_context,
    )
    refined = RefinedClaim(claim=entry.claim, is_verifiable=True)
    # ResolvedRef: the paper must already be indexed in the vector store.
    # We set pdf_path to a sentinel — the runner doesn't re-download.
    ref = ResolvedRef(
        cite_key=entry.cite_key,
        pdf_path=Path("/indexed"),   # sentinel — store already has this paper
        abstract=None,
        source=entry.source,
        abstract_only=False,
    )
    return ctx, refined, ref


async def _run_one(
    entry: EvalEntry,
    store: VectorStore,
    disable_reflexion: bool,
) -> tuple[Verdict, float, float]:
    """
    Run verification for one entry.
    Returns (predicted_verdict, latency_seconds, resolved_flag).
    """
    ctx, refined, ref = _entry_to_inputs(entry)
    resolved = store.has_paper(entry.paper_id)

    if not resolved:
        return "unverifiable", 0.0, False

    t0 = time.perf_counter()

    if disable_reflexion:
        # Monkey-patch needs_reflexion to always return False for this run
        with patch("citverif.agent.verifier.needs_reflexion", return_value=False):
            verdict = await verify_citation(ctx, refined, ref, store)
    else:
        verdict = await verify_citation(ctx, refined, ref, store)

    elapsed = time.perf_counter() - t0
    return verdict.verdict, elapsed, True


async def _run_loop(
    entries: list[EvalEntry],
    store: VectorStore,
    mode: str,
    out_dir: Path,
) -> None:
    disable_reflexion = mode == "react"
    result_path = out_dir / f"results_{mode}.jsonl"

    # Skip already-computed entries (crash recovery)
    done_keys: set[str] = set()
    if result_path.exists():
        with result_path.open() as f:
            for line in f:
                try:
                    done_keys.add(json.loads(line)["cite_key"])
                except Exception:
                    pass
    todo = [e for e in entries if e.cite_key not in done_keys]
    if done_keys:
        console.print(f"  [dim]Resuming — {len(done_keys)} already done, {len(todo)} remaining[/dim]")

    expected: list[Verdict] = []
    predicted: list[Verdict] = []
    latencies: list[float] = []
    resolved_flags: list[bool] = []

    # Pre-load already-done results for final metrics
    if result_path.exists():
        with result_path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    expected.append(r["expected"])
                    predicted.append(r["predicted"])
                    latencies.append(r["latency_s"])
                    resolved_flags.append(r["resolved"])
                except Exception:
                    pass

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"[mode={mode}]", total=len(todo))

        for entry in todo:
            progress.update(task, description=f"[mode={mode}] {entry.cite_key}")
            pred, latency, resolved = await _run_one(entry, store, disable_reflexion)

            record = {
                "cite_key": entry.cite_key,
                "claim": entry.claim,
                "expected": entry.expected_verdict,
                "predicted": pred,
                "latency_s": round(latency, 2),
                "resolved": resolved,
                "mode": mode,
            }
            save_result_line(result_path, record)

            expected.append(entry.expected_verdict)
            predicted.append(pred)
            latencies.append(latency)
            resolved_flags.append(resolved)

            progress.advance(task)

    metrics = compute_metrics(expected, predicted, latencies, resolved_flags)
    table = format_metrics_table(metrics, mode=mode)
    console.print(table)

    # Save metrics summary
    summary_path = out_dir / f"metrics_{mode}.json"
    summary_path.write_text(json.dumps({
        "mode": mode,
        "macro_f1": metrics.macro_f1,
        "accuracy": metrics.accuracy,
        "hallucination_rate": metrics.hallucination_rate,
        "resolution_rate": metrics.resolution_rate,
        "mean_latency_s": metrics.mean_latency_s,
        "total": metrics.total,
        "per_class": {
            v: {"precision": m.precision, "recall": m.recall, "f1": m.f1, "support": m.support}
            for v, m in metrics.per_class.items()
        },
    }, indent=2), encoding="utf-8")
    console.print(f"Metrics saved to [cyan]{summary_path}[/cyan]")


@app.command()
def main(
    ground_truth: Path = typer.Option(
        Path("data/eval/ground_truth.jsonl"), "--ground-truth", "-g"
    ),
    store_dir: Path = typer.Option(Path("data/chroma"), "--store-dir"),
    out_dir: Path = typer.Option(Path("data/eval/results"), "--out"),
    models: str = typer.Option("qwen2.5:14b", "--models", help="Comma-separated model names (informational)"),
    loops: str = typer.Option("react,react+reflexion", "--loops", help="Comma-separated loop modes"),
    limit: int = typer.Option(0, "--limit", help="Cap entries for a quick smoke-test (0 = all)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the citation verification benchmark and print macro-F1 metrics."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING)

    entries = load_dataset(ground_truth)
    if limit > 0:
        entries = entries[:limit]

    console.print(f"Loaded [bold]{len(entries)}[/bold] eval entries from {ground_truth}")

    store = VectorStore(store_dir)
    modes = [m.strip() for m in loops.split(",")]
    out_dir.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        if mode not in ("react", "react+reflexion"):
            console.print(f"[red]Unknown loop mode: {mode}[/red] — skip")
            continue
        console.rule(f"Mode: {mode}")
        asyncio.run(_run_loop(entries, store, mode, out_dir))


if __name__ == "__main__":
    app()

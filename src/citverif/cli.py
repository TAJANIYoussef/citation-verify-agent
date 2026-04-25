from __future__ import annotations

import asyncio
import logging
from collections import Counter
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(name="verify-citations", add_completion=False)
console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, markup=True)],
    )


def _step(label: str) -> None:
    console.print(f"\n[bold cyan]▶ {label}[/bold cyan]")


@app.command()
def main(
    tex: Path = typer.Argument(..., help="Input .tex file"),
    bib: Path = typer.Argument(..., help="Input .bib file"),
    out: Path = typer.Option(Path("report.md"), "--out", "-o", help="Output report path"),
    cache_dir: Path = typer.Option(Path("data/papers"), "--cache-dir"),
    store_dir: Path = typer.Option(Path("data/chroma"), "--store-dir"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Verify every \\cite{} in a LaTeX paper against the referenced sources."""
    _setup_logging(verbose)

    for p, label in [(tex, "tex"), (bib, "bib")]:
        if not p.exists():
            console.print(f"[red]Error:[/red] {label} file not found: {p}")
            raise typer.Exit(1)

    from citverif.parse.tex import parse_tex
    from citverif.parse.bib import parse_bib
    from citverif.resolve.chain import resolve_all
    from citverif.memory.paper_cache import PaperCache
    from citverif.memory.chunker import chunk_pdf
    from citverif.memory.vector_store import VectorStore
    from citverif.extract.claims import refine_all
    from citverif.agent.verifier import verify_all
    from citverif.report.markdown import render_report

    console.rule("[bold]Citation Verifier[/bold]")

    # ── 1. Parse ──────────────────────────────────────────────────────────
    _step("Parsing")
    citations, uncited = parse_tex(tex)
    bib_entries = parse_bib(bib)

    console.print(
        f"  {len(citations)} cited claims across {len(set(c.cite_key for c in citations))} "
        f"unique keys  |  {len(bib_entries)} bib entries  |  "
        f"{len(uncited)} uncited claim candidates"
    )

    # ── 2. Resolve ────────────────────────────────────────────────────────
    _step("Resolving references")
    cache = PaperCache(cache_dir)
    results = asyncio.run(resolve_all(bib_entries, cache))
    refs_by_key = {r.cite_key: r for r in results}

    resolved = [r for r in results if r.pdf_path or r.abstract]
    abstract_only = [r for r in resolved if r.abstract_only]
    failed = [r for r in results if not r.pdf_path and not r.abstract]
    rate = len(resolved) / len(results) if results else 0.0

    res_table = Table(show_header=True, header_style="bold", show_lines=False, box=None)
    res_table.add_column("cite_key", style="cyan", no_wrap=True)
    res_table.add_column("source")
    res_table.add_column("status")
    for r in results:
        if r.pdf_path:
            status = "[green]PDF[/green]"
        elif r.abstract:
            status = "[yellow]abstract[/yellow]"
        else:
            status = "[red]FAILED[/red]"
        res_table.add_row(r.cite_key, r.source or "—", status)
    console.print(res_table)
    console.print(
        f"  Rate: [bold]{len(resolved)}/{len(results)}[/bold] ({rate:.0%})"
        + (f"  |  {len(abstract_only)} abstract-only" if abstract_only else "")
        + (f"  |  [red]{len(failed)} failed[/red]" if failed else "")
    )
    if failed:
        logging.getLogger(__name__).warning("Unresolved: %s", [r.cite_key for r in failed])

    # ── 3. Chunk + index ──────────────────────────────────────────────────
    _step("Chunking and indexing PDFs")
    store = VectorStore(store_dir)
    indexed = 0
    for ref in results:
        if not ref.pdf_path:
            continue
        if store.has_paper(ref.cite_key):
            continue
        chunks = chunk_pdf(ref.pdf_path, ref.cite_key)
        if chunks:
            store.index_chunks(chunks)
            indexed += 1
    console.print(f"  {indexed} new papers indexed  |  bge-m3 embeddings via Ollama")

    # ── 4. Claim refinement ───────────────────────────────────────────────
    _step("Refining claims  [llama3.1:8b]")
    refined_pairs = asyncio.run(refine_all(citations))
    verifiable = [(ctx, rc) for ctx, rc in refined_pairs if rc.is_verifiable]
    skipped = len(refined_pairs) - len(verifiable)
    console.print(
        f"  {len(verifiable)} verifiable  |  {skipped} skipped (definitions / method refs)"
    )

    # ── 5 + 6. Verify (ReAct + Reflexion) ────────────────────────────────
    _step("Verifying citations  [qwen2.5:14b — ReAct + Reflexion]")
    console.print(f"  {len(verifiable)} claims to verify — this may take a while...")
    verdicts = asyncio.run(verify_all(verifiable, refs_by_key, store))

    counts = Counter(v.verdict for v in verdicts)
    v_table = Table(show_header=False, box=None)
    v_table.add_column("verdict", style="bold")
    v_table.add_column("count")
    color_map = {
        "unsupported": "red", "misleading": "red",
        "partially_supported": "yellow",
        "supported": "green", "unverifiable": "dim",
    }
    for label in ("unsupported", "misleading", "partially_supported", "supported", "unverifiable"):
        n = counts.get(label, 0)
        if n:
            col = color_map[label]
            v_table.add_row(f"[{col}]{label}[/{col}]", str(n))
    console.print(v_table)

    # ── 7. Render report ──────────────────────────────────────────────────
    _step("Writing report")
    report_md = render_report(
        verdicts=verdicts,
        uncited=uncited,
        tex_path=tex,
        resolution_rate=rate,
        abstract_only_count=len(abstract_only),
        failed_count=len(failed),
    )
    out.write_text(report_md, encoding="utf-8")

    critical = counts.get("unsupported", 0) + counts.get("misleading", 0)
    summary = (
        f"[bold green]Done.[/bold green] Report written to [cyan]{out}[/cyan]\n"
        f"{len(verdicts)} citations verified  |  "
        f"{len(uncited)} uncited candidates"
    )
    if critical:
        summary += f"\n[bold red]{critical} critical issue(s) require attention.[/bold red]"

    console.print(Panel(summary, expand=False))


if __name__ == "__main__":
    app()

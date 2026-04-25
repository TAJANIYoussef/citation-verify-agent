Run the citation verifier end-to-end on a single paper.

Usage: /verify-paper <path/to/paper.tex> <path/to/refs.bib> [--out report.md]

Steps:
1. Run `verify-citations $ARGUMENTS --out report.md` (or the --out path if provided).
2. Print the summary table from the generated report (verdict counts + resolution rate).
3. Flag any `unsupported` or `misleading` verdicts immediately.

If Ollama is not running, start it with `ollama serve` first.

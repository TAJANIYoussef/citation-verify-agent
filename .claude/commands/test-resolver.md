Run only the reference resolution step on a .bib file — useful for debugging fetch failures.

Usage: /test-resolver <path/to/refs.bib>

Steps:
1. Run: `python -c "
import asyncio, sys
from citverif.parse.bib import parse_bib
from citverif.resolve.chain import resolve_all

entries = parse_bib(sys.argv[1])
results = asyncio.run(resolve_all(entries))
for r in results:
    status = 'OK' if r.pdf_path or r.abstract else 'FAIL'
    print(f'{status:4} [{r.source or \"none\":12}] {r.cite_key}')
resolved = sum(1 for r in results if r.pdf_path or r.abstract)
print(f'\nResolution rate: {resolved}/{len(results)} = {resolved/len(results):.0%}')
" $ARGUMENTS`
2. Print a summary of failures grouped by failure reason (no DOI, no arXiv ID, fetch error, fuzzy-match below threshold).

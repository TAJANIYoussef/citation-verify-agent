# Known Issues / Deferred Items

## Open (as of 2026-04-25)

- Ground-truth eval set (`data/eval/ground_truth.jsonl`) not yet created — awaiting Youssef's manual labels from AutoSMOTE-NC and MTSOT papers (Phase 6).
- "Missing citation" detector outputs a separate report section (not a CitationVerdict). Needs its own Pydantic model and report renderer.
- Eval set sourcing: Youssef will label hundreds of triples from third-party papers; tooling to load large JSONL must be validated before Phase 6.

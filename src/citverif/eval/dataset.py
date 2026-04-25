from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from citverif.schema import Verdict


@dataclass
class EvalEntry:
    cite_key: str
    claim: str
    claim_context: str
    expected_verdict: Verdict
    paper_id: str          # may differ from cite_key if same paper cited under different keys
    source: str | None     # "arxiv" | "openalex" | "s2" | "unpaywall" | None
    notes: str = ""        # human rationale for the label (not used by runner)


def load_dataset(path: Path) -> list[EvalEntry]:
    """Load ground_truth.jsonl — one JSON object per line."""
    if not path.exists():
        raise FileNotFoundError(
            f"Ground-truth file not found: {path}\n"
            "Label triples manually and place them at data/eval/ground_truth.jsonl"
        )
    entries: list[EvalEntry] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"ground_truth.jsonl line {lineno}: {exc}") from exc

            entries.append(EvalEntry(
                cite_key=obj["cite_key"],
                claim=obj["claim"],
                claim_context=obj.get("claim_context", obj["claim"]),
                expected_verdict=obj["expected_verdict"],
                paper_id=obj.get("paper_id", obj["cite_key"]),
                source=obj.get("source"),
                notes=obj.get("notes", ""),
            ))
    return entries


def save_result_line(path: Path, record: dict) -> None:
    """Append one result record to a JSONL file (incremental write, crash-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

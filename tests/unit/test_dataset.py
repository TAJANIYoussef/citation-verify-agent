"""Unit tests for eval/dataset.py — file I/O only, no Ollama."""
import json
import pytest
from pathlib import Path
from citverif.eval.dataset import load_dataset, save_result_line, EvalEntry


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_load_dataset_basic(tmp_path):
    p = tmp_path / "gt.jsonl"
    _write_jsonl(p, [
        {"cite_key": "a", "claim": "Claim A.", "expected_verdict": "supported", "source": "arxiv"},
        {"cite_key": "b", "claim": "Claim B.", "expected_verdict": "unsupported", "source": "openalex"},
    ])
    entries = load_dataset(p)
    assert len(entries) == 2
    assert entries[0].cite_key == "a"
    assert entries[0].expected_verdict == "supported"
    assert entries[1].cite_key == "b"


def test_load_dataset_skips_comments(tmp_path):
    p = tmp_path / "gt.jsonl"
    p.write_text(
        '# this is a comment\n'
        '{"cite_key": "c", "claim": "C.", "expected_verdict": "misleading", "source": null}\n'
        '\n'
        '{"cite_key": "d", "claim": "D.", "expected_verdict": "unverifiable", "source": null}\n'
    )
    entries = load_dataset(p)
    assert len(entries) == 2


def test_load_dataset_claim_context_defaults_to_claim(tmp_path):
    p = tmp_path / "gt.jsonl"
    _write_jsonl(p, [
        {"cite_key": "x", "claim": "Claim X.", "expected_verdict": "supported", "source": None},
    ])
    entries = load_dataset(p)
    assert entries[0].claim_context == "Claim X."


def test_load_dataset_paper_id_defaults_to_cite_key(tmp_path):
    p = tmp_path / "gt.jsonl"
    _write_jsonl(p, [
        {"cite_key": "y", "claim": "Y.", "expected_verdict": "unsupported", "source": None},
    ])
    entries = load_dataset(p)
    assert entries[0].paper_id == "y"


def test_load_dataset_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_dataset(Path("/nonexistent/ground_truth.jsonl"))


def test_load_dataset_bad_json_raises(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"cite_key": "ok", "claim": "ok.", "expected_verdict": "supported", "source": null}\n'
                 'NOT VALID JSON\n')
    with pytest.raises(ValueError, match="line 2"):
        load_dataset(p)


def test_save_result_line_appends(tmp_path):
    p = tmp_path / "out" / "results.jsonl"
    save_result_line(p, {"cite_key": "a", "predicted": "supported"})
    save_result_line(p, {"cite_key": "b", "predicted": "unsupported"})
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["cite_key"] == "a"
    assert json.loads(lines[1])["predicted"] == "unsupported"


def test_save_result_line_creates_parent_dirs(tmp_path):
    p = tmp_path / "deep" / "nested" / "results.jsonl"
    save_result_line(p, {"x": 1})
    assert p.exists()

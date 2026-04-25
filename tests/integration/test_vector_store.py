"""
Integration tests for chunking + embedding + retrieval.
Requires: Ollama running with bge-m3 pulled.
Run with: pytest tests/integration/test_vector_store.py -v
"""
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from citverif.memory.chunker import Chunk
from citverif.memory.vector_store import VectorStore


PAPER_ID = "test-attention-paper"

GOLD_PASSAGE = (
    "The Transformer model relies entirely on an attention mechanism to draw global "
    "dependencies between input and output, dispensing with recurrence entirely."
)

DECOY_PASSAGES = [
    "Convolutional neural networks use local receptive fields and weight sharing to "
    "process grid-structured data such as images.",
    "Recurrent networks process sequential data by maintaining a hidden state that "
    "is updated at each time step.",
    "Dropout regularization randomly zeros activations during training to reduce "
    "overfitting in deep neural networks.",
]


def _make_chunks(paper_id: str) -> list[Chunk]:
    all_texts = [GOLD_PASSAGE] + DECOY_PASSAGES
    return [
        Chunk(text=t, page=i + 1, section="Method", chunk_idx=i, paper_id=paper_id)
        for i, t in enumerate(all_texts)
    ]


@pytest.fixture
def store(tmp_path):
    return VectorStore(tmp_path)


def test_gold_passage_in_top5(store):
    chunks = _make_chunks(PAPER_ID)
    store.index_chunks(chunks)

    query = "attention mechanism replaces recurrence for sequence modelling"
    results = store.search(query, PAPER_ID, k=5)

    assert results, "Search returned no results"
    top_texts = [r.text for r in results]
    assert any(GOLD_PASSAGE[:40] in t for t in top_texts), (
        f"Gold passage not in top-5.\nTop results:\n" +
        "\n---\n".join(top_texts)
    )


def test_search_returns_metadata(store):
    chunks = _make_chunks(PAPER_ID + "-meta")
    store.index_chunks(chunks)
    results = store.search("attention", PAPER_ID + "-meta", k=3)
    assert all(r.page > 0 for r in results)
    assert all(r.paper_id == PAPER_ID + "-meta" for r in results)


def test_missing_paper_returns_empty(store):
    results = store.search("anything", "nonexistent-paper-id", k=5)
    assert results == []


def test_has_paper_after_index(store):
    chunks = _make_chunks(PAPER_ID + "-exists")
    assert not store.has_paper(PAPER_ID + "-exists")
    store.index_chunks(chunks)
    assert store.has_paper(PAPER_ID + "-exists")

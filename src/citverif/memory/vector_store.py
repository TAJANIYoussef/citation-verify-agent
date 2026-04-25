from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings

from citverif.memory.chunker import Chunk

log = logging.getLogger(__name__)

_EMBED_MODEL = "bge-m3"
_OLLAMA_BASE = "http://localhost:11434"


@dataclass
class SearchResult:
    text: str
    page: int
    section: str | None
    paper_id: str
    score: float   # cosine distance — lower is more similar


class VectorStore:
    """One chromadb collection per cited paper, keyed by paper_id."""

    def __init__(self, store_dir: Path) -> None:
        store_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(store_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._embedder = _OllamaEmbedder(_EMBED_MODEL)

    # ── collection helpers ────────────────────────────────────────────────

    def _collection_name(self, paper_id: str) -> str:
        # chromadb collection names must be 3-63 chars, alphanumeric + hyphens
        slug = hashlib.md5(paper_id.encode()).hexdigest()[:16]
        return f"paper-{slug}"

    def has_paper(self, paper_id: str) -> bool:
        name = self._collection_name(paper_id)
        existing = [c.name for c in self._client.list_collections()]
        return name in existing

    # ── indexing ──────────────────────────────────────────────────────────

    def index_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        paper_id = chunks[0].paper_id
        name = self._collection_name(paper_id)

        col = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed(texts)
        ids = [f"{paper_id}-{c.chunk_idx}" for c in chunks]
        metadatas = [
            {"page": c.page, "section": c.section or "", "paper_id": c.paper_id}
            for c in chunks
        ]

        # Upsert in batches of 100 to avoid chromadb size limits
        batch = 100
        for i in range(0, len(ids), batch):
            col.upsert(
                ids=ids[i:i + batch],
                embeddings=embeddings[i:i + batch],
                documents=texts[i:i + batch],
                metadatas=metadatas[i:i + batch],
            )
        log.info("Indexed %d chunks for paper %s", len(chunks), paper_id)

    # ── retrieval ─────────────────────────────────────────────────────────

    def search(self, query: str, paper_id: str, k: int = 5) -> list[SearchResult]:
        name = self._collection_name(paper_id)
        try:
            col = self._client.get_collection(name)
        except Exception:
            log.warning("No collection found for paper %s", paper_id)
            return []

        q_emb = self._embedder.embed([query])
        results = col.query(
            query_embeddings=q_emb,
            n_results=min(k, col.count()),
            include=["documents", "metadatas", "distances"],
        )

        out: list[SearchResult] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            out.append(SearchResult(
                text=doc,
                page=meta.get("page", 0),
                section=meta.get("section") or None,
                paper_id=meta.get("paper_id", paper_id),
                score=dist,
            ))
        return out


class _OllamaEmbedder:
    """Calls Ollama's OpenAI-compatible embeddings endpoint."""

    def __init__(self, model: str) -> None:
        self.model = model
        # Import here so the module loads even before openai is installed
        from openai import OpenAI
        self._client = OpenAI(
            base_url=f"{_OLLAMA_BASE}/v1",
            api_key="ollama",   # Ollama ignores the key
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Ollama bge-m3 supports batching
        response = self._client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]

import hashlib
from pathlib import Path


class PaperCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, paper_id: str) -> str:
        return hashlib.md5(paper_id.encode()).hexdigest()[:12]

    def pdf_path(self, paper_id: str) -> Path:
        return self.cache_dir / f"{self._key(paper_id)}.pdf"

    def abstract_path(self, paper_id: str) -> Path:
        return self.cache_dir / f"{self._key(paper_id)}.txt"

    def get_pdf(self, paper_id: str) -> Path | None:
        p = self.pdf_path(paper_id)
        return p if p.exists() else None

    def get_abstract(self, paper_id: str) -> str | None:
        p = self.abstract_path(paper_id)
        return p.read_text(encoding="utf-8") if p.exists() else None

    def save_abstract(self, paper_id: str, text: str) -> None:
        self.abstract_path(paper_id).write_text(text, encoding="utf-8")

    def save_pdf_bytes(self, paper_id: str, data: bytes) -> Path:
        p = self.pdf_path(paper_id)
        p.write_bytes(data)
        return p

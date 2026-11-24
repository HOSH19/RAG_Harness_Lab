from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import get_settings


class LocalVectorStore:
    """Minimal persistent vector store (JSON metadata + npz embeddings)."""

    def __init__(self, path: Path | None = None) -> None:
        settings = get_settings()
        self.root = path or (Path(settings.chroma_path).parent / "store")
        if not self.root.is_absolute():
            from .config import ROOT

            self.root = ROOT / self.root
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root / f"{settings.collection_name}.json"
        self.emb_path = self.root / f"{settings.collection_name}.npz"
        self._rows: list[dict[str, Any]] = []
        self._embeddings: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        if self.meta_path.exists():
            self._rows = json.loads(self.meta_path.read_text(encoding="utf-8"))
        else:
            self._rows = []
        if self.emb_path.exists() and self._rows:
            self._embeddings = np.load(self.emb_path)["embeddings"]
        else:
            self._embeddings = None

    def _save(self) -> None:
        self.meta_path.write_text(json.dumps(self._rows, indent=2), encoding="utf-8")
        if self._embeddings is not None and len(self._rows) > 0:
            np.savez_compressed(self.emb_path, embeddings=self._embeddings)
        elif self.emb_path.exists():
            self.emb_path.unlink()

    def reset(self) -> None:
        self._rows = []
        self._embeddings = None
        if self.meta_path.exists():
            self.meta_path.unlink()
        if self.emb_path.exists():
            self.emb_path.unlink()

    def count(self) -> int:
        return len(self._rows)

    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        index = {row["id"]: i for i, row in enumerate(self._rows)}
        dim = len(embeddings[0]) if embeddings else 384
        if self._embeddings is None:
            self._embeddings = np.zeros((0, dim), dtype=np.float32)

        for i, chunk_id in enumerate(ids):
            row = {
                "id": chunk_id,
                "document": documents[i],
                "metadata": metadatas[i],
            }
            vec = np.asarray(embeddings[i], dtype=np.float32).reshape(1, -1)
            if chunk_id in index:
                pos = index[chunk_id]
                self._rows[pos] = row
                self._embeddings[pos] = vec[0]
            else:
                self._rows.append(row)
                self._embeddings = np.vstack([self._embeddings, vec])
                index[chunk_id] = len(self._rows) - 1
        self._save()

    def query(self, query_embedding: list[float], n_results: int) -> list[dict[str, Any]]:
        if not self._rows or self._embeddings is None:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        doc_norms = np.linalg.norm(self._embeddings, axis=1)
        doc_norms[doc_norms == 0] = 1.0
        sims = (self._embeddings @ q) / (doc_norms * q_norm)
        distances = 1.0 - sims
        order = np.argsort(distances)[:n_results]
        out: list[dict[str, Any]] = []
        for idx in order:
            row = self._rows[int(idx)]
            out.append(
                {
                    "id": row["id"],
                    "document": row["document"],
                    "metadata": row["metadata"],
                    "distance": float(distances[int(idx)]),
                }
            )
        return out

    def get(self, chunk_id: str) -> dict[str, Any] | None:
        for row in self._rows:
            if row["id"] == chunk_id:
                return {
                    "id": row["id"],
                    "document": row["document"],
                    "metadata": row["metadata"],
                }
        return None

    def all_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "id": row["id"],
                "document": row["document"],
                "metadata": row["metadata"],
            }
            for row in self._rows
        ]


_STORE: LocalVectorStore | None = None


def get_store(*, reset: bool = False) -> LocalVectorStore:
    global _STORE
    if _STORE is None:
        _STORE = LocalVectorStore()
    if reset:
        _STORE.reset()
    return _STORE


def reset_store_singleton() -> None:
    global _STORE
    _STORE = None
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from rank_bm25 import BM25Okapi

from .config import load_rag_config
from .embeddings import build_embedder
from .store import get_store


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_./+-]+", text.lower()) if t]


@lru_cache
def get_embedder():
    from .config import get_settings

    return build_embedder(get_settings().embedding_model)


def _row_to_chunk(
    doc_id: str,
    metadata: dict[str, Any],
    document: str,
    distance: float | None,
) -> dict[str, Any]:
    score = None if distance is None else max(0.0, 1.0 - float(distance))
    return {
        "id": doc_id,
        "doc_id": metadata.get("doc_id", doc_id.rsplit("#", 1)[0]),
        "chunk_index": int(metadata.get("chunk_index", 0)),
        "source": metadata.get("source", ""),
        "text": document,
        "score": score,
        "distance": distance,
    }


def dense_search(query: str, top_k: int) -> list[dict[str, Any]]:
    store = get_store()
    if store.count() == 0:
        return []
    query_vec = get_embedder().embed([query])[0]
    rows = store.query(query_vec, n_results=min(top_k, store.count()))
    return [
        _row_to_chunk(r["id"], r.get("metadata") or {}, r.get("document") or "", r.get("distance"))
        for r in rows
    ]


def _all_docs() -> list[dict[str, Any]]:
    rows = get_store().all_rows()
    return [
        _row_to_chunk(r["id"], r.get("metadata") or {}, r.get("document") or "", None)
        for r in rows
    ]


def sparse_search(query: str, top_k: int) -> list[dict[str, Any]]:
    rows = _all_docs()
    if not rows:
        return []
    corpus = [_tokenize(r["text"]) for r in rows]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(rows, scores, strict=True), key=lambda x: x[1], reverse=True)
    out: list[dict[str, Any]] = []
    for row, score in ranked[:top_k]:
        item = dict(row)
        item["score"] = float(score)
        out.append(item)
    return out


def _rrf_fuse(
    dense: list[dict[str, Any]],
    sparse: list[dict[str, Any]],
    *,
    top_k: int,
    dense_weight: float,
    sparse_weight: float,
    k: int = 60,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for rank, item in enumerate(dense):
        cid = item["id"]
        payloads[cid] = item
        scores[cid] = scores.get(cid, 0.0) + dense_weight * (1.0 / (k + rank + 1))
    for rank, item in enumerate(sparse):
        cid = item["id"]
        payloads[cid] = item
        scores[cid] = scores.get(cid, 0.0) + sparse_weight * (1.0 / (k + rank + 1))
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    fused: list[dict[str, Any]] = []
    for cid, score in ordered:
        row = dict(payloads[cid])
        row["score"] = score
        fused.append(row)
    return fused


def search(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    cfg = load_rag_config().get("retrieval", {})
    k = top_k or int(cfg.get("top_k", 5))
    dense = dense_search(query, top_k=max(k, 8))
    if not cfg.get("hybrid", True):
        return dense[:k]
    sparse = sparse_search(query, top_k=max(k, 8))
    return _rrf_fuse(
        dense,
        sparse,
        top_k=k,
        dense_weight=float(cfg.get("dense_weight", 0.7)),
        sparse_weight=float(cfg.get("sparse_weight", 0.3)),
    )


def get_chunk(chunk_id: str) -> dict[str, Any] | None:
    row = get_store().get(chunk_id)
    if row is None:
        return None
    return _row_to_chunk(row["id"], row.get("metadata") or {}, row.get("document") or "", None)


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embedder().embed(texts)


def collection_count() -> int:
    return get_store().count()
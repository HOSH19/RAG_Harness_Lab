from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import numpy as np


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_./+-]+", text.lower())


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Deterministic bag-of-tokens hashing embedder (no torch/onnx).

    Good enough for a local practice RAG corpus. Swap to SentenceTransformers
    when torch works on the host (`EMBEDDING_MODEL=<hf-id>`).
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            tokens = _tokenize(text)
            if not tokens:
                out.append(vec.tolist())
                continue
            for tok in tokens:
                digest = hashlib.sha1(tok.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[idx] += sign
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
            out.append(vec.tolist())
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]


def build_embedder(model_name: str) -> Embedder:
    name = (model_name or "").strip()
    if name in {"", "default", "hashing", "all-MiniLM-L6-v2", "onnx-minilm"}:
        return HashingEmbedder(dim=384)
    try:
        return SentenceTransformerEmbedder(name)
    except Exception:
        return HashingEmbedder(dim=384)


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
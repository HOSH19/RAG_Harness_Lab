from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from .config import ROOT, get_settings, load_rag_config
from .retrieve import embed_texts
from .store import get_store, reset_store_singleton


SUPPORTED_SUFFIXES = {".md", ".txt", ".markdown"}


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def _doc_id_for(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    return Path(rel).with_suffix("").as_posix().replace("/", "-")


def load_documents(path: Path) -> list[dict[str, Any]]:
    path = path.resolve()
    if path.is_file():
        root = path.parent
        files = [path]
    else:
        root = path
        files = sorted(
            p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        )

    docs: list[dict[str, Any]] = []
    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        docs.append(
            {
                "doc_id": _doc_id_for(file_path, root),
                "source": str(file_path.relative_to(root).as_posix()),
                "text": text,
            }
        )
    return docs


def ingest_path(path: Path | str, *, reset: bool = False) -> dict[str, Any]:
    settings = get_settings()
    cfg = load_rag_config().get("chunking", {})
    chunk_size = int(cfg.get("chunk_size", 512))
    chunk_overlap = int(cfg.get("chunk_overlap", 64))

    docs = load_documents(Path(path))
    if not docs:
        return {"documents": 0, "chunks": 0, "collection": settings.collection_name}

    if reset:
        reset_store_singleton()
        store = get_store(reset=True)
    else:
        store = get_store()

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for doc in docs:
        pieces = chunk_text(doc["text"], chunk_size, chunk_overlap)
        for idx, piece in enumerate(pieces):
            chunk_id = f"{doc['doc_id']}#{idx}"
            digest = hashlib.sha1(f"{chunk_id}:{piece}".encode()).hexdigest()[:12]
            ids.append(f"{chunk_id}-{digest}")
            documents.append(piece)
            metadatas.append(
                {
                    "doc_id": doc["doc_id"],
                    "chunk_index": idx,
                    "source": doc["source"],
                }
            )

    batch = 64
    for i in range(0, len(ids), batch):
        batch_docs = documents[i : i + batch]
        store.upsert(
            ids=ids[i : i + batch],
            documents=batch_docs,
            metadatas=metadatas[i : i + batch],
            embeddings=embed_texts(batch_docs),
        )

    return {
        "documents": len(docs),
        "chunks": len(ids),
        "collection": settings.collection_name,
        "path": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the local vector store")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(ROOT / "sample_docs"),
        help="File or directory to ingest",
    )
    parser.add_argument("--reset", action="store_true", help="Drop and recreate the store")
    args = parser.parse_args()
    result = ingest_path(args.path, reset=args.reset)
    print(result)


if __name__ == "__main__":
    main()
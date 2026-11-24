from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .backend import backend_status, effective_model_id
from .config import ROOT, get_settings, load_rag_config
from .ingest import ingest_path
from .llm import chat, chat_stream, ping
from .prompts import build_user_prompt, format_context, system_prompt
from .retrieve import collection_count, get_chunk, search

app = FastAPI(title="vLLM RAG Practice API", version="0.1.0")


class IngestRequest(BaseModel):
    path: str = Field(default="sample_docs")
    reset: bool = False


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    model: str | None = None


class ChunkRequest(BaseModel):
    chunk_id: str


@app.get("/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    try:
        chunks = collection_count()
        collection_error = None
    except Exception as exc:  # noqa: BLE001
        chunks = 0
        collection_error = str(exc)

    vllm_status: dict[str, Any]
    try:
        vllm_status = ping(timeout=1.5)
    except Exception as exc:  # noqa: BLE001
        vllm_status = {"ok": False, "error": str(exc)}

    return {
        # Store readiness is enough for retrieval; vLLM may be down during setup.
        "ok": collection_error is None,
        "collection": settings.collection_name,
        "chunks": chunks,
        "collection_error": collection_error,
        "backend": backend_status(),
        "vllm": vllm_status,
    }


@app.post("/ingest")
def ingest(body: IngestRequest) -> dict[str, Any]:
    path = Path(body.path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"path not found: {path}")
    return ingest_path(path, reset=body.reset)


@app.post("/ingest/upload")
async def ingest_upload(file: UploadFile = File(...), reset: bool = False) -> dict[str, Any]:
    suffix = Path(file.filename or "upload.txt").suffix.lower()
    if suffix not in {".md", ".txt", ".markdown"}:
        raise HTTPException(status_code=400, detail="only .md/.txt supported")
    upload_dir = ROOT / ".data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / (file.filename or "upload.txt")
    dest.write_bytes(await file.read())
    return ingest_path(dest, reset=reset)


@app.post("/search")
def search_endpoint(body: SearchRequest) -> dict[str, Any]:
    chunks = search(body.query.strip(), top_k=body.top_k)
    return {"query": body.query, "results": chunks}


@app.post("/chunk")
def chunk_endpoint(body: ChunkRequest) -> dict[str, Any]:
    item = get_chunk(body.chunk_id)
    if item is None:
        raise HTTPException(status_code=404, detail="chunk not found")
    return item


def _rag_messages(query: str, top_k: int | None) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    chunks = search(query.strip(), top_k=top_k)
    context = format_context(chunks) if chunks else "(no passages retrieved)"
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": build_user_prompt(query.strip(), context)},
    ]
    return messages, chunks


@app.post("/chat")
def chat_endpoint(body: ChatRequest) -> dict[str, Any]:
    messages, chunks = _rag_messages(body.query, body.top_k)
    try:
        completion = chat(messages, model=body.model)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"vLLM error: {exc}") from exc
    answer = completion.choices[0].message.content or ""
    return {
        "answer": answer,
        "sources": chunks,
        "model": completion.model,
    }


@app.post("/chat/stream")
def chat_stream_endpoint(body: ChatRequest) -> StreamingResponse:
    messages, chunks = _rag_messages(body.query, body.top_k)

    def event_gen():
        meta = {"type": "sources", "sources": chunks}
        yield f"data: {json.dumps(meta)}\n\n"
        try:
            for token in chat_stream(messages, model=body.model):
                yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
            return
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/config")
def config_endpoint() -> dict[str, Any]:
    settings = get_settings()
    return {
        "model_id": effective_model_id(),
        "model_id_explicit": settings.model_id_explicit,
        "vllm_base_url": settings.vllm_base_url,
        "backend": backend_status(),
        "rag": load_rag_config(),
    }


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "rag.api:app",
        host=settings.rag_api_host,
        port=settings.rag_api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
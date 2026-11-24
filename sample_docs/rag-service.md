# AcmeOps RAG Service

The RAG service exposes:

- `POST /ingest` — index markdown/text documents
- `POST /search` — retrieve ranked chunks
- `POST /chat` — grounded answer with citations
- `POST /chat/stream` — same as chat with token streaming
- `GET /health` — API and upstream vLLM status

## Citation format

Answers must cite passages as `[doc_id#chunk_index]`, for example `[api-auth#0]`.

## Refusal policy

If retrieved context does not support the answer, the model must say it does not know.
Do not invent AcmeOps flags, error codes, or regions.

## Embedding model

Default embedding backend is a local hashing embedder (`EMBEDDING_MODEL=hashing`).
Query and document embeddings must use the same model.
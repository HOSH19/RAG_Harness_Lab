# vLLM RAG + Agent Harness Practice

Local practice stack for **vLLM** serving, a **RAG** chat API, a **Streamlit** UI, and a **hillclimb-style agent harness** that tools into the same knowledge base.

Works on **NVIDIA CUDA** hosts and **Apple Silicon** (M2/M3…) via an `LLM_BACKEND` switch.

Embeddings + retrieval use a **pure-NumPy hashing embedder** and a file-backed store under `.data/store/` (no torch/Chroma required). Optional SentenceTransformer models via `pip install -e ".[st]"` if torch works on your machine.

## Architecture

```
Streamlit UI ──► FastAPI RAG API ──► Local vector store (NumPy)
                      │
                      └──► vLLM OpenAI API (/v1)
Agent harness ────────┘
```

## Prerequisites

- Python 3.11+ (3.12 tested)
- One LLM path:
  - **cuda** — NVIDIA GPU + Docker Desktop GPU support **or** local CUDA `vllm`
  - **metal** — Apple Silicon + macOS 15+ + [vllm-metal](https://github.com/vllm-project/vllm-metal)
  - **remote** — any OpenAI-compatible `/v1` URL
  - **none** — retrieval-only (no generation)

## Backend switch (`LLM_BACKEND`)

Set in `.env` (or pass `--backend` to serve/doctor):

| Value | Meaning |
| --- | --- |
| `auto` (default) | `metal` on Apple Silicon, otherwise `cuda` |
| `cuda` | NVIDIA Docker Compose / native vLLM |
| `metal` | Apple Silicon via vllm-metal (not the NVIDIA image) |
| `remote` | Use `VLLM_BASE_URL` only — no local serve |
| `none` | RAG/retrieval only |

Check what this machine will use:

```bash
python -m rag.doctor
python -m rag.doctor --backend metal
python scripts/serve_llm.py --dry-run
```

### Apple Silicon (M2 MacBook Air) quick path

```bash
cp .env.example .env
# LLM_BACKEND=auto  # resolves to metal on Apple Silicon
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m rag.doctor
brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
brew install vllm-project/vllm-metal/vllm-metal
python scripts/serve_llm.py          # uses Qwen2.5-3B by default on metal
```

On 8GB Airs prefer a small/quantized model and keep `GPU_MEMORY_UTILIZATION` around `0.4–0.5`. Or set `LLM_BACKEND=remote` / `none`.

## Quick start

### 1. Install the app (host Python)

```powershell
cd c:\Users\UserAdmin\Documents\Practice\vLLM
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
copy .env.example .env
python -m rag.doctor
```

### 2. Start the LLM server

Unified entrypoint (respects `LLM_BACKEND`):

```powershell
python scripts\serve_llm.py
# or
.\scripts\serve_vllm.ps1
```

CUDA Docker still works directly:

```powershell
docker compose --env-file .env up vllm
```

Smoke the OpenAI-compatible endpoint:

```powershell
python scripts\smoke_chat.py
```

### 3. Ingest sample docs

```powershell
python -m rag.ingest sample_docs --reset
```

### 4. Run the RAG API

```powershell
python -m rag.api
```

Health check: [http://localhost:8080/health](http://localhost:8080/health)

### 5. Open the UI

```powershell
streamlit run src\rag\ui.py
```

### 6. Run the agent harness

```powershell
.\scripts\run_evals.ps1 smoke
# or
python -m harness.runner --suite smoke --out results\smoke.json
python -m harness.runner --suite full --out results\full-a.json
```

Hillclimb compare after changing `configs/rag.yaml` (e.g. `top_k` or prompts):

```powershell
python -m harness.runner --suite full --out results\full-b.json
python -m harness.hillclimb results\full-a.json results\full-b.json
```

## Config knobs

| File | What to tweak |
| --- | --- |
| `.env` `LLM_BACKEND` | `auto` / `cuda` / `metal` / `remote` / `none` |
| `.env` / `configs/models.yaml` | `MODEL_ID`, `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION` (optional overrides) |
| `.env` `EMBEDDING_MODEL` | `hashing` (default) or a SentenceTransformer id |
| `configs/rag.yaml` | chunk size/overlap, `top_k`, hybrid weights, prompts, agent rounds |

### Model notes

| Model | Notes |
| --- | --- |
| `Qwen/Qwen2.5-7B-Instruct` | cuda profile default — good on ~16GB NVIDIA |
| `Qwen/Qwen2.5-3B-Instruct` | metal profile default — better for M2/M3 Air |
| `meta-llama/Llama-3.1-8B-Instruct` | Needs HF license + `HF_TOKEN` |

If you OOM:

1. Lower `GPU_MEMORY_UTILIZATION` to `0.75`
2. Lower `MAX_MODEL_LEN` to `2048`
3. Switch to the 3B model

## API surface

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | API + vLLM reachability |
| POST | `/ingest` | Index a path (`sample_docs` by default) |
| POST | `/ingest/upload` | Upload `.md`/`.txt` |
| POST | `/search` | Retrieve ranked chunks |
| POST | `/chunk` | Fetch one chunk by id |
| POST | `/chat` | Grounded answer + sources |
| POST | `/chat/stream` | SSE token stream |
| GET | `/config` | Effective RAG config |

## Harness practice loop

1. Run `--suite full` → save JSON
2. Change **one** knob in `configs/rag.yaml` (chunk size, `top_k`, or system prompt)
3. Re-ingest if chunking changed (`python -m rag.ingest sample_docs --reset`)
4. Re-run suite → `python -m harness.hillclimb a.json b.json`
5. Keep only changes that improve aggregates (`hit@5`, `overall`, refusal cases)

Modes:

- `--mode agent` — tool-calling loop against vLLM (`search_docs` / `get_chunk`)
- `--mode retrieval` — score dense+hybrid retrieval only (no generation)

## Optional stretch

```powershell
python scripts\bench_latency.py --n 8 --concurrency 2
```

## Project layout

```
configs/          models.yaml, rag.yaml
sample_docs/      AcmeOps seed corpus
src/rag/          ingest, retrieve, llm, FastAPI, Streamlit UI
harness/          agent, tools, cases, scorers, runner, hillclimb
scripts/          serve, smoke_chat, run_evals, bench_latency
```

## Troubleshooting

- **`smoke_chat` connection error** — vLLM not up yet; first model download can take several minutes.
- **`/health` shows `vllm.ok: false`** — expected until vLLM is running; RAG `/search` still works.
- **Docker cannot see GPU** — enable NVIDIA/WSL GPU in Docker Desktop, or `python scripts/serve_llm.py --via native`.
- **M2 clone + NVIDIA Compose fails** — expected; set `LLM_BACKEND=metal` (or `auto`) and use vllm-metal, not `docker compose` CUDA.
- **`serve_llm` exits with backend error** — run `python -m rag.doctor` and match `LLM_BACKEND` to the machine.
- **Empty `/search` results** — run `python -m rag.ingest sample_docs --reset`.
- **Tool calls fail in harness** — keep `python -m rag.api` running; agent mode tools call `RAG_API_BASE_URL`.
- **Retrieval-only evals without vLLM** — `LLM_BACKEND=none` or `python -m harness.runner --suite smoke --mode retrieval`.
- **Windows note** — this stack avoids torch/Chroma native crashes by using NumPy hashing + a file store.

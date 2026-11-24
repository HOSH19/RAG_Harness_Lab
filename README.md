# vLLM RAG + Agent Harness

Local RAG + OpenAI-compatible LLM + hillclimb evals. Backend switch: `LLM_BACKEND=auto|cuda|metal|remote|none`.

## Stack

```mermaid
flowchart LR
  UI[Streamlit] --> API[FastAPI]
  Harness[AgentHarness] --> API
  Harness --> LLM[OpenAI_v1]
  API --> Store[NumPyStore]
  API --> LLM
  Docs[sample_docs] --> Ingest[Ingest]
  Ingest --> Store
```

## Query path

```mermaid
sequenceDiagram
  participant U as User
  participant API as FastAPI
  participant S as Store
  participant L as vLLM
  U->>API: POST /chat
  API->>S: dense+BM25 search
  S-->>API: top_k chunks
  API->>L: grounded prompt
  L-->>API: answer + citations
  API-->>U: answer + sources
```

## Backends

```mermaid
flowchart TD
  Auto[LLM_BACKEND_auto] -->|AppleSilicon| Metal[metal_vllm-metal]
  Auto -->|else| Cuda[cuda_Docker_or_native]
  Remote[remote] --> URL[VLLM_BASE_URL]
  None[none] --> RetOnly[retrieval_only]
```

| Value | Default model | Serve |
| --- | --- | --- |
| `cuda` | Qwen2.5-7B | `docker compose up vllm` / native |
| `metal` | Qwen2.5-3B | [vllm-metal](https://github.com/vllm-project/vllm-metal) |
| `remote` | — | set `VLLM_BASE_URL` |
| `none` | — | no LLM |

```bash
python -m rag.doctor
python scripts/serve_llm.py --dry-run
```

## Run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -e .
cp .env.example .env                                  # Windows: copy .env.example .env
python -m rag.doctor
python scripts/serve_llm.py                           # skip if remote/none
python -m rag.ingest sample_docs --reset
python -m rag.api                                     # :8080
streamlit run src/rag/ui.py
python -m harness.runner --suite smoke --mode retrieval
python -m harness.runner --suite smoke                # needs LLM + API
```

## Harness loop

```mermaid
flowchart LR
  A[run_suite] --> B[results_JSON]
  B --> C[tweak_rag.yaml]
  C --> D[re_ingest_if_chunking]
  D --> A
  B --> E[hillclimb_compare]
```

```bash
python -m harness.runner --suite full --out results/a.json
# edit configs/rag.yaml (one knob)
python -m harness.runner --suite full --out results/b.json
python -m harness.hillclimb results/a.json results/b.json
```

## API

`GET /health` · `GET /config` · `POST /ingest` · `POST /search` · `POST /chunk` · `POST /chat` · `POST /chat/stream`

## Layout

`configs/` · `sample_docs/` · `src/rag/` · `harness/` · `scripts/`

## Tips

- OOM → lower `GPU_MEMORY_UTILIZATION` / `MAX_MODEL_LEN`, or use 3B / `remote` / `none`
- M2 → `auto`→`metal`; do **not** use NVIDIA Compose
- Empty search → re-ingest · `vllm.ok:false` until LLM is up is fine for `/search`

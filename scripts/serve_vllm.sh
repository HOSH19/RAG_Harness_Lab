#!/usr/bin/env bash
# Native (non-Docker) vLLM serve fallback.
# Prefer: docker compose --env-file .env up vllm
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

echo "Starting vLLM: model=${MODEL_ID} port=${VLLM_PORT} max_model_len=${MAX_MODEL_LEN}"

exec vllm serve "${MODEL_ID}" \
  --host 0.0.0.0 \
  --port "${VLLM_PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --dtype auto
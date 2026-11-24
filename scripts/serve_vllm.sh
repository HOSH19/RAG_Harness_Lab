#!/usr/bin/env bash
# Thin wrapper — prefers scripts/serve_llm.py backend switch.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -x .venv/bin/python ]]; then
  PY=.venv/bin/python
else
  PY=python3
fi

exec "$PY" scripts/serve_llm.py "$@"
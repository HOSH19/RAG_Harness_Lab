#!/usr/bin/env python3
"""Start a local OpenAI-compatible LLM server for the selected backend."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

load_dotenv(ROOT / ".env")


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve LLM for the configured backend")
    parser.add_argument(
        "--backend",
        default=os.getenv("LLM_BACKEND", "auto"),
        help="auto|cuda|metal|remote|none (default: LLM_BACKEND or auto)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print command only")
    parser.add_argument(
        "--via",
        choices=["auto", "docker", "native"],
        default="auto",
        help="cuda only: docker compose vs native vllm",
    )
    args = parser.parse_args()

    from rag.backend import resolve_backend, validate_serve_request
    from rag.config import get_settings, reload_configs

    reload_configs()
    settings = get_settings()
    backend = resolve_backend(args.backend)
    try:
        profile = validate_serve_request(backend)
    except RuntimeError as exc:
        print(f"serve_llm: {exc}", file=sys.stderr)
        return 2

    model = settings.model_id.strip() or profile.model_id
    max_len = settings.max_model_len or profile.max_model_len
    gpu_util = settings.gpu_memory_utilization or profile.gpu_memory_utilization
    host = settings.vllm_host
    port = str(settings.vllm_port)

    print(f"backend={backend} model={model} max_model_len={max_len} gpu_util={gpu_util}")

    if backend == "metal":
        if shutil.which("vllm") is None and not args.dry_run:
            print(
                "vllm CLI not found. Install vllm-metal first:\n"
                f"  {profile.install_hint}",
                file=sys.stderr,
            )
            return 1
        cmd = [
            "vllm",
            "serve",
            model,
            "--host",
            host,
            "--port",
            port,
            "--max-model-len",
            str(max_len),
            "--gpu-memory-utilization",
            str(gpu_util),
        ]
        if args.dry_run:
            print(" ".join(cmd))
            return 0
        print("Running:", " ".join(cmd))
        return subprocess.call(cmd, cwd=ROOT)

    # cuda
    if args.via == "docker":
        use_docker = True
    elif args.via == "native":
        use_docker = False
    else:
        use_docker = shutil.which("docker") is not None

    if use_docker:
        env = os.environ.copy()
        env["MODEL_ID"] = model
        env["MAX_MODEL_LEN"] = str(max_len)
        env["GPU_MEMORY_UTILIZATION"] = str(gpu_util)
        env["VLLM_PORT"] = port
        cmd = ["docker", "compose", "--env-file", ".env", "up", "vllm"]
        print("Using Docker Compose for CUDA backend")
        if args.dry_run:
            print(" ".join(cmd))
            print(f"# with MODEL_ID={model} MAX_MODEL_LEN={max_len} GPU_MEMORY_UTILIZATION={gpu_util}")
            return 0
        return subprocess.call(cmd, cwd=ROOT, env=env)

    if shutil.which("vllm") is None and not args.dry_run:
        print(
            "vllm CLI not found. Install CUDA vLLM or use Docker:\n"
            "  docker compose --env-file .env up vllm",
            file=sys.stderr,
        )
        return 1
    cmd = [
        "vllm",
        "serve",
        model,
        "--host",
        host,
        "--port",
        port,
        "--max-model-len",
        str(max_len),
        "--gpu-memory-utilization",
        str(gpu_util),
        "--dtype",
        "auto",
    ]
    if args.dry_run:
        print(" ".join(cmd))
        return 0
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
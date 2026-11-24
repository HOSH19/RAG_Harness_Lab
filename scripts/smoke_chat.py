#!/usr/bin/env python3
"""Smoke-test vLLM OpenAI-compatible /v1 endpoints."""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Smoke chat against local vLLM")
    parser.add_argument(
        "--base-url",
        default=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
    )
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"),
    )
    parser.add_argument(
        "--prompt",
        default="Say hello in one short sentence.",
    )
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    print(f"Listing models at {args.base_url} ...")
    models = client.models.list()
    ids = [m.id for m in models.data]
    print("models:", ", ".join(ids) if ids else "(none)")

    model = args.model if args.model in ids or not ids else ids[0]
    print(f"Chat completion with model={model!r} ...")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": args.prompt}],
        max_tokens=64,
        temperature=0.2,
    )
    content = resp.choices[0].message.content or ""
    print("assistant:", content.strip())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"smoke_chat failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
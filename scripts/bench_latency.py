#!/usr/bin/env python3
"""Tiny latency/throughput smoke against local vLLM."""

from __future__ import annotations

import argparse
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI


def one_call(client: OpenAI, model: str, prompt: str, max_tokens: int) -> float:
    start = time.perf_counter()
    client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return time.perf_counter() - start


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt", default="List three colors in one line.")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    latencies: list[float] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(one_call, client, args.model, args.prompt, args.max_tokens)
            for _ in range(args.n)
        ]
        for fut in as_completed(futures):
            latencies.append(fut.result())
    wall = time.perf_counter() - t0

    print(f"requests={args.n} concurrency={args.concurrency}")
    print(f"wall_s={wall:.2f} throughput_rps={args.n / wall:.2f}")
    print(
        "latency_s "
        f"mean={statistics.mean(latencies):.3f} "
        f"p50={statistics.median(latencies):.3f} "
        f"max={max(latencies):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
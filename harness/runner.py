from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml

from rag.config import ROOT, get_settings

from .agent import run_agent
from .scorers import score_case

CASES_DIR = Path(__file__).resolve().parent / "cases"


def load_cases(suite: str) -> list[dict[str, Any]]:
    path = CASES_DIR / f"{suite}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"case suite not found: {path}")
    with path.open(encoding="utf-8") as fh:
        cases = yaml.safe_load(fh) or []
    return list(cases)


def _ensure_api() -> None:
    base = os.getenv("RAG_API_BASE_URL", "http://localhost:8080")
    try:
        resp = httpx.get(f"{base}/health", timeout=5.0)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"RAG API not reachable at {base}. Start it with: python -m rag.api"
        ) from exc


def run_suite(suite: str, *, mode: str = "agent") -> dict[str, Any]:
    cases = load_cases(suite)
    results: list[dict[str, Any]] = []

    if mode == "agent":
        _ensure_api()

    for case in cases:
        if mode == "retrieval":
            # In-process retrieval — no API/vLLM required.
            from rag.retrieve import search as local_search

            hits = local_search(case["question"], top_k=5)
            doc_ids = [r.get("doc_id") for r in hits if r.get("doc_id")]
            payload = {
                "answer": "",
                "tool_trace": [{"tool": "search_docs", "arguments": {"query": case["question"]}}],
                "retrieved_doc_ids": doc_ids,
                "search_doc_ids": doc_ids,
            }
        else:
            payload = run_agent(case["question"])

        scores = score_case(case, payload)
        results.append(
            {
                "id": case["id"],
                "suite": suite,
                "question": case["question"],
                "expected_doc_ids": case.get("expected_doc_ids", []),
                "expect_refuse": bool(case.get("expect_refuse")),
                "result": {
                    "answer": payload.get("answer"),
                    "retrieved_doc_ids": payload.get("retrieved_doc_ids"),
                    "tool_trace": [
                        {"tool": t.get("tool"), "arguments": t.get("arguments")}
                        for t in payload.get("tool_trace", [])
                    ],
                    "model": payload.get("model"),
                },
                "scores": scores,
            }
        )

    aggregates: dict[str, float] = {}
    keys = sorted({k for r in results for k in r["scores"]})
    for key in keys:
        vals = [r["scores"][key] for r in results if key in r["scores"]]
        aggregates[key] = sum(vals) / len(vals) if vals else 0.0

    return {
        "suite": suite,
        "mode": mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": get_settings().model_id,
        "aggregates": aggregates,
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run harness eval suite")
    parser.add_argument("--suite", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--mode", choices=["agent", "retrieval"], default="agent")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    report = run_suite(args.suite, mode=args.mode)

    out = args.out
    if not out:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = str(ROOT / "results" / f"{args.suite}-{stamp}.json")

    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["aggregates"], indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import re
from typing import Any


def hit_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    if not expected:
        return 1.0
    top = retrieved[:k]
    return 1.0 if any(doc in top for doc in expected) else 0.0


def mrr(retrieved: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0
    for idx, doc in enumerate(retrieved):
        if doc in expected:
            return 1.0 / (idx + 1)
    return 0.0


def called_tool(tool_trace: list[dict[str, Any]], name: str = "search_docs") -> float:
    return 1.0 if any(t.get("tool") == name for t in tool_trace) else 0.0


def refused_when_unknown(answer: str) -> float:
    text = answer.lower()
    markers = [
        "do not know",
        "don't know",
        "doesn't know",
        "not sure",
        "cannot find",
        "can't find",
        "no information",
        "not in the",
        "not covered",
        "unable to answer",
    ]
    return 1.0 if any(m in text for m in markers) else 0.0


def citation_present(answer: str) -> float:
    return 1.0 if re.search(r"\[[a-z0-9_-]+#\d+\]", answer, flags=re.I) else 0.0


def contains_any(answer: str, needles: list[str]) -> float:
    if not needles:
        return 1.0
    text = answer.lower()
    return 1.0 if any(n.lower() in text for n in needles) else 0.0


def score_case(case: dict[str, Any], result: dict[str, Any]) -> dict[str, float]:
    retrieved = result.get("retrieved_doc_ids") or result.get("search_doc_ids") or []
    expected = case.get("expected_doc_ids") or []
    answer = result.get("answer") or ""
    tool_trace = result.get("tool_trace") or []
    retrieval_only = not answer

    scores: dict[str, float] = {
        "hit@1": hit_at_k(retrieved, expected, 1),
        "hit@5": hit_at_k(retrieved, expected, 5),
        "mrr": mrr(retrieved, expected),
        "called_tool": called_tool(tool_trace),
    }

    if not retrieval_only:
        scores["citation"] = citation_present(answer)
        if case.get("expect_refuse"):
            scores["refused"] = refused_when_unknown(answer)
        else:
            scores["answer_contains"] = contains_any(
                answer, case.get("expected_answer_contains") or []
            )

    # Weighted overall (retrieval-only focuses on ranking metrics)
    weights = {
        "hit@5": 0.45 if retrieval_only else 0.35,
        "mrr": 0.25 if retrieval_only else 0.15,
        "hit@1": 0.30 if retrieval_only else 0.0,
        "called_tool": 0.0 if retrieval_only else 0.15,
        "citation": 0.10,
        "answer_contains": 0.25,
        "refused": 0.35,
    }
    total = 0.0
    weight_sum = 0.0
    for key, weight in weights.items():
        if key in scores and weight > 0:
            total += scores[key] * weight
            weight_sum += weight
    scores["overall"] = total / weight_sum if weight_sum else 0.0
    return scores
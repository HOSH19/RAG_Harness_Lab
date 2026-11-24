from __future__ import annotations

import json
from typing import Any

from rag.config import load_rag_config
from rag.llm import chat

from .tools import TOOL_SCHEMAS, execute_tool

AGENT_SYSTEM = """You are an AcmeOps research agent.
Use tools to search the knowledge base before answering.
Cite sources as [doc_id#chunk_index] when possible.
If tools do not provide enough evidence, say you do not know. Do not invent facts.
"""


def run_agent(question: str, *, model: str | None = None) -> dict[str, Any]:
    cfg = load_rag_config().get("agent", {})
    max_rounds = int(cfg.get("max_tool_rounds", 4))
    temperature = float(cfg.get("temperature", 0.1))
    max_tokens = int(cfg.get("max_tokens", 768))

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": question},
    ]
    tool_trace: list[dict[str, Any]] = []
    retrieved_doc_ids: list[str] = []

    for _ in range(max_rounds):
        completion = chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )
        message = completion.choices[0].message
        tool_calls = message.tool_calls or []

        if not tool_calls:
            answer = message.content or ""
            return {
                "answer": answer,
                "tool_trace": tool_trace,
                "retrieved_doc_ids": sorted(set(retrieved_doc_ids)),
                "model": completion.model,
                "rounds": len(tool_trace),
            }

        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            payload = execute_tool(name, args)
            tool_trace.append({"tool": name, "arguments": args, "result": payload})

            if name == "search_docs":
                try:
                    parsed = json.loads(payload)
                    for item in parsed.get("results", []):
                        doc_id = item.get("doc_id")
                        if doc_id:
                            retrieved_doc_ids.append(str(doc_id))
                except json.JSONDecodeError:
                    pass

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": payload,
                }
            )

    # Force a final answer without tools
    messages.append(
        {
            "role": "user",
            "content": "Please give your final answer now based on the tool results.",
        }
    )
    completion = chat(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return {
        "answer": completion.choices[0].message.content or "",
        "tool_trace": tool_trace,
        "retrieved_doc_ids": sorted(set(retrieved_doc_ids)),
        "model": completion.model,
        "rounds": len(tool_trace),
    }
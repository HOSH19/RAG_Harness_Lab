from __future__ import annotations

import json
import os
from typing import Any

import httpx

API_BASE = os.getenv("RAG_API_BASE_URL", "http://localhost:8080")

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the AcmeOps knowledge base for relevant passages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of passages to return",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chunk",
            "description": "Fetch a single chunk by its id from search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "Chunk id returned by search_docs",
                    }
                },
                "required": ["chunk_id"],
            },
        },
    },
]


def search_docs(query: str, top_k: int = 5) -> dict[str, Any]:
    resp = httpx.post(
        f"{API_BASE}/search",
        json={"query": query, "top_k": top_k},
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()


def get_chunk(chunk_id: str) -> dict[str, Any]:
    resp = httpx.post(
        f"{API_BASE}/chunk",
        json={"chunk_id": chunk_id},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    if name == "search_docs":
        result = search_docs(
            query=str(arguments.get("query", "")),
            top_k=int(arguments.get("top_k", 5)),
        )
    elif name == "get_chunk":
        result = get_chunk(str(arguments.get("chunk_id", "")))
    else:
        result = {"error": f"unknown tool: {name}"}
    return json.dumps(result, ensure_ascii=True)
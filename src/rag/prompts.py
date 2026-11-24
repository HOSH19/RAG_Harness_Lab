from __future__ import annotations

from .config import load_rag_config


def system_prompt() -> str:
    cfg = load_rag_config()
    return str(cfg.get("prompts", {}).get("system", "")).strip()


def build_user_prompt(question: str, context: str) -> str:
    cfg = load_rag_config()
    template = str(
        cfg.get("prompts", {}).get(
            "user_template",
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        )
    )
    return template.format(context=context, question=question)


def format_context(chunks: list[dict]) -> str:
    parts: list[str] = []
    for item in chunks:
        doc_id = item.get("doc_id", "unknown")
        chunk_index = item.get("chunk_index", 0)
        text = item.get("text", "").strip()
        parts.append(f"[{doc_id}#{chunk_index}]\n{text}")
    return "\n\n".join(parts)
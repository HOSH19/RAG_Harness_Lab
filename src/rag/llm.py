from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from .config import get_settings, load_rag_config


def get_client(*, timeout: float = 120.0) -> OpenAI:
    settings = get_settings()
    return OpenAI(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        timeout=timeout,
    )


def resolve_model(model: str | None = None) -> str:
    return model or get_settings().model_id


def chat(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> Any:
    cfg = load_rag_config().get("generation", {})
    kwargs: dict[str, Any] = {
        "model": resolve_model(model),
        "messages": messages,
        "temperature": temperature if temperature is not None else cfg.get("temperature", 0.2),
        "max_tokens": max_tokens if max_tokens is not None else cfg.get("max_tokens", 512),
    }
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    return get_client().chat.completions.create(**kwargs)


def chat_stream(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Iterator[str]:
    cfg = load_rag_config().get("generation", {})
    stream = get_client().chat.completions.create(
        model=resolve_model(model),
        messages=messages,
        temperature=temperature if temperature is not None else cfg.get("temperature", 0.2),
        max_tokens=max_tokens if max_tokens is not None else cfg.get("max_tokens", 512),
        stream=True,
    )
    for event in stream:
        delta = event.choices[0].delta.content if event.choices else None
        if delta:
            yield delta


def ping(timeout: float = 1.5) -> dict[str, Any]:
    """Lightweight reachability check (httpx, not the OpenAI SDK)."""
    import httpx

    settings = get_settings()
    base = settings.vllm_base_url.rstrip("/")
    models_url = f"{base}/models"
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(models_url, headers={"Authorization": f"Bearer {settings.vllm_api_key}"})
        resp.raise_for_status()
        payload = resp.json()
    ids = [m.get("id", "") for m in payload.get("data", [])]
    return {
        "ok": True,
        "base_url": settings.vllm_base_url,
        "models": ids,
    }
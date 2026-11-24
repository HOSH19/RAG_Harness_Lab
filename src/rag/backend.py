from __future__ import annotations

import platform
import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import yaml

from .config import ROOT, get_settings

BackendName = Literal["auto", "cuda", "metal", "remote", "none"]
ResolvedBackend = Literal["cuda", "metal", "remote", "none"]


@dataclass(frozen=True)
class BackendProfile:
    name: ResolvedBackend
    model_id: str
    max_model_len: int
    gpu_memory_utilization: float
    serve_via: str
    notes: str
    install_hint: str


def _load_models_config() -> dict[str, Any]:
    settings = get_settings()
    path = ROOT / settings.models_config
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def detect_platform() -> dict[str, Any]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    is_macos = system == "darwin"
    is_apple_silicon = is_macos and machine in {"arm64", "aarch64"}
    is_windows = system == "windows"
    is_linux = system == "linux"
    return {
        "system": system,
        "machine": machine,
        "python": sys.version.split()[0],
        "is_macos": is_macos,
        "is_apple_silicon": is_apple_silicon,
        "is_windows": is_windows,
        "is_linux": is_linux,
        "has_nvidia_smi": shutil.which("nvidia-smi") is not None,
        "has_docker": shutil.which("docker") is not None,
        "has_vllm": shutil.which("vllm") is not None,
    }


def recommend_backend(plat: dict[str, Any] | None = None) -> ResolvedBackend:
    info = plat or detect_platform()
    if info["is_apple_silicon"]:
        return "metal"
    if info["has_nvidia_smi"] or info["is_windows"] or info["is_linux"]:
        return "cuda"
    return "none"


def resolve_backend(requested: str | None = None) -> ResolvedBackend:
    settings = get_settings()
    raw = (requested or settings.llm_backend or "auto").strip().lower()
    if raw not in {"auto", "cuda", "metal", "remote", "none"}:
        raise ValueError(
            f"Unknown LLM_BACKEND={raw!r}. Use auto|cuda|metal|remote|none."
        )
    if raw == "auto":
        return recommend_backend()
    return raw  # type: ignore[return-value]


def profile_for(backend: ResolvedBackend | None = None) -> BackendProfile:
    resolved = backend or resolve_backend()
    cfg = _load_models_config()
    backends = cfg.get("backends", {})
    entry = backends.get(resolved, {}) if isinstance(backends, dict) else {}

    defaults: dict[str, dict[str, Any]] = {
        "cuda": {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.85,
            "serve_via": "docker-compose",
            "notes": "NVIDIA CUDA — Docker Compose or native vllm serve.",
            "install_hint": "docker compose --env-file .env up vllm",
        },
        "metal": {
            "model_id": "Qwen/Qwen2.5-3B-Instruct",
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.5,
            "serve_via": "vllm-metal",
            "notes": "Apple Silicon via vllm-metal (macOS 15+). Avoid NVIDIA Docker image.",
            "install_hint": (
                "brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal && "
                "brew install vllm-project/vllm-metal/vllm-metal"
            ),
        },
        "remote": {
            "model_id": get_settings().model_id,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.0,
            "serve_via": "remote",
            "notes": "Use an existing OpenAI-compatible endpoint (desktop GPU, cloud, etc.).",
            "install_hint": "Set VLLM_BASE_URL to your remote /v1 endpoint; no local serve.",
        },
        "none": {
            "model_id": get_settings().model_id,
            "max_model_len": 0,
            "gpu_memory_utilization": 0.0,
            "serve_via": "none",
            "notes": "RAG/retrieval only — no local LLM serving.",
            "install_hint": "python -m harness.runner --suite smoke --mode retrieval",
        },
    }
    base = defaults[resolved]
    return BackendProfile(
        name=resolved,
        model_id=str(entry.get("model_id", base["model_id"])),
        max_model_len=int(entry.get("max_model_len", base["max_model_len"])),
        gpu_memory_utilization=float(
            entry.get("gpu_memory_utilization", base["gpu_memory_utilization"])
        ),
        serve_via=str(entry.get("serve_via", base["serve_via"])),
        notes=str(entry.get("notes", base["notes"])),
        install_hint=str(entry.get("install_hint", base["install_hint"])),
    )


def effective_model_id() -> str:
    settings = get_settings()
    # Explicit MODEL_ID always wins; otherwise use backend profile default.
    if settings.model_id_explicit:
        return settings.model_id
    return profile_for().model_id


@lru_cache
def backend_status() -> dict[str, Any]:
    settings = get_settings()
    plat = detect_platform()
    requested = settings.llm_backend.strip().lower() or "auto"
    resolved = resolve_backend(requested)
    profile = profile_for(resolved)
    warnings: list[str] = []

    if requested == "cuda" and plat["is_apple_silicon"]:
        warnings.append(
            "LLM_BACKEND=cuda on Apple Silicon will not use the NVIDIA Docker image. "
            "Prefer LLM_BACKEND=metal or remote."
        )
    if requested == "metal" and not plat["is_apple_silicon"]:
        warnings.append(
            "LLM_BACKEND=metal is intended for Apple Silicon. "
            "On this machine prefer cuda or remote."
        )
    if resolved == "cuda" and plat["is_apple_silicon"] and requested == "auto":
        # Shouldn't happen because recommend_backend picks metal, but keep defensive.
        warnings.append("Auto-detected Apple Silicon; use metal backend.")

    return {
        "requested": requested,
        "resolved": resolved,
        "platform": plat,
        "profile": {
            "model_id": profile.model_id,
            "max_model_len": profile.max_model_len,
            "gpu_memory_utilization": profile.gpu_memory_utilization,
            "serve_via": profile.serve_via,
            "notes": profile.notes,
            "install_hint": profile.install_hint,
        },
        "effective_model_id": effective_model_id(),
        "vllm_base_url": settings.vllm_base_url,
        "warnings": warnings,
    }


def reload_backend() -> None:
    backend_status.cache_clear()
    get_settings.cache_clear()


def validate_serve_request(backend: ResolvedBackend | None = None) -> BackendProfile:
    """Raise RuntimeError with a clear message if local serve is not appropriate."""
    resolved = backend or resolve_backend()
    profile = profile_for(resolved)
    plat = detect_platform()

    if resolved == "none":
        raise RuntimeError(
            "LLM_BACKEND=none — local LLM serving is disabled. "
            "Use retrieval-only mode or set LLM_BACKEND=remote/cuda/metal."
        )
    if resolved == "remote":
        raise RuntimeError(
            "LLM_BACKEND=remote — do not start a local server. "
            f"Point VLLM_BASE_URL at your endpoint (current: {get_settings().vllm_base_url})."
        )
    if resolved == "metal" and not plat["is_apple_silicon"]:
        raise RuntimeError(
            "LLM_BACKEND=metal requires Apple Silicon. "
            "On this host use LLM_BACKEND=cuda or remote."
        )
    if resolved == "cuda" and plat["is_apple_silicon"]:
        raise RuntimeError(
            "LLM_BACKEND=cuda is for NVIDIA hosts. "
            "On Apple Silicon set LLM_BACKEND=metal (vllm-metal) or remote."
        )
    return profile
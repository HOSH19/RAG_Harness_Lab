from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # auto|cuda|metal|remote|none — see rag.backend
    llm_backend: str = Field(default="auto", alias="LLM_BACKEND")

    # Empty => use backend profile default model
    model_id: str = Field(default="", alias="MODEL_ID")
    max_model_len: int | None = Field(default=None, alias="MAX_MODEL_LEN")
    gpu_memory_utilization: float | None = Field(
        default=None,
        alias="GPU_MEMORY_UTILIZATION",
    )

    vllm_base_url: str = Field(default="http://localhost:8000/v1", alias="VLLM_BASE_URL")
    vllm_api_key: str = Field(default="EMPTY", alias="VLLM_API_KEY")
    vllm_host: str = Field(default="0.0.0.0", alias="VLLM_HOST")
    vllm_port: int = Field(default=8000, alias="VLLM_PORT")

    rag_api_host: str = Field(default="0.0.0.0", alias="RAG_API_HOST")
    rag_api_port: int = Field(default=8080, alias="RAG_API_PORT")
    rag_api_base_url: str = Field(default="http://localhost:8080", alias="RAG_API_BASE_URL")

    chroma_path: str = Field(default=".data/chroma", alias="CHROMA_PATH")
    collection_name: str = Field(default="practice_docs", alias="COLLECTION_NAME")
    embedding_model: str = Field(
        default="hashing",
        alias="EMBEDDING_MODEL",
    )

    models_config: str = Field(default="configs/models.yaml", alias="MODELS_CONFIG")
    rag_config: str = Field(default="configs/rag.yaml", alias="RAG_CONFIG")

    @field_validator("model_id", "llm_backend", mode="before")
    @classmethod
    def strip_str(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("max_model_len", "gpu_memory_utilization", mode="before")
    @classmethod
    def empty_to_none(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @property
    def model_id_explicit(self) -> bool:
        return bool(self.model_id.strip())

    @property
    def chroma_dir(self) -> Path:
        path = Path(self.chroma_path)
        if not path.is_absolute():
            path = ROOT / path
        return path

    @property
    def rag_config_path(self) -> Path:
        path = Path(self.rag_config)
        if not path.is_absolute():
            path = ROOT / path
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def load_rag_config() -> dict[str, Any]:
    settings = get_settings()
    with settings.rag_config_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def reload_configs() -> None:
    get_settings.cache_clear()
    load_rag_config.cache_clear()
    try:
        from .backend import reload_backend

        reload_backend()
    except Exception:  # noqa: BLE001
        pass
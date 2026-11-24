from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_id: str = Field(default="Qwen/Qwen2.5-7B-Instruct", alias="MODEL_ID")
    vllm_base_url: str = Field(default="http://localhost:8000/v1", alias="VLLM_BASE_URL")
    vllm_api_key: str = Field(default="EMPTY", alias="VLLM_API_KEY")

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
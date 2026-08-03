from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    streamlit_port: int = 8501
    data_dir: Path = Path("data")

    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: str | None = None

    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_chat_model: str = "deepseek-v4-flash"
    deepseek_extraction_model: str = "deepseek-v4-flash"
    deepseek_thinking_enabled: bool = False
    llm_request_timeout_seconds: int = 120
    llm_max_retries: int = 3
    llm_max_concurrency: int = 2

    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_model_id: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dimension: int = 1024
    embedding_device: str = "auto"
    embedding_batch_size: int = 8

    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "auto"

    parser_ocr_enabled: bool = False
    max_upload_size_mb: int = 50
    chunk_target_tokens: int = 700
    chunk_overlap_tokens: int = 100

    lightrag_working_dir: Path = Path("data/lightrag")
    lightrag_workspace: str = "demo_v1"
    lightrag_query_mode: Literal["naive", "local", "global", "hybrid", "mix"] = "mix"
    lightrag_top_k: int = 20
    lightrag_chunk_top_k: int = 10
    lightrag_max_total_tokens: int = 24_000
    lightrag_cosine_threshold: float = 0.2
    lightrag_summary_language: str = "Chinese"
    entity_extraction_use_json: bool = True
    graph_index_concurrency: int = Field(default=1, ge=1, le=4)
    event_extraction_enabled: bool = True
    event_extraction_chunk_chars: int = Field(default=12_000, ge=2_000, le=50_000)
    event_extraction_max_events: int = Field(default=40, ge=1, le=200)

    @property
    def required_data_dirs(self) -> tuple[Path, ...]:
        return tuple(
            self.data_dir / name
            for name in (
                "raw",
                "parsed",
                "registry",
                "events",
                "lightrag",
                "exports",
                "logs",
            )
        )

    def ensure_data_dirs(self) -> None:
        for path in self.required_data_dirs:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()

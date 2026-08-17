from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
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
    llm_max_concurrency: int = Field(default=4, ge=1, le=16)
    agentic_max_retrieval_rounds: int = Field(default=2, ge=1, le=5)
    agentic_rrf_k: int = Field(default=60, ge=1)

    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_model_id: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dimension: int = 1024
    embedding_device: str = "auto"
    embedding_batch_size: int = Field(default=8, ge=1, le=128)
    embedding_max_concurrency: int = Field(default=2, ge=1, le=8)
    embedding_provider: Literal["local", "bailian", "zhipu"] = "local"
    bailian_api_key: str | None = None
    bailian_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    bailian_embedding_model: str = "text-embedding-v4"
    bailian_embedding_dimensions_enabled: bool = True
    zhipu_api_key: str | None = None
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    zhipu_embedding_model: str = "embedding-3"
    zhipu_embedding_dimensions_enabled: bool = True
    embedding_request_timeout_seconds: int = 60
    embedding_max_retries: int = 3

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
    light_entity_prompt_file: Path = Path("prompts/entity_type/light_entity.yaml")
    light_entity_max_gleaning: int = Field(default=0, ge=0, le=3)
    light_entity_max_entities: int = Field(default=4, ge=1, le=40)
    light_entity_max_relationships: int = Field(default=4, ge=0, le=60)
    light_entity_max_records: int = Field(default=8, ge=1, le=100)
    graph_index_concurrency: int = Field(default=2, ge=1, le=4)
    event_extraction_enabled: bool = True
    event_extraction_chunk_chars: int = Field(default=12_000, ge=2_000, le=50_000)
    event_extraction_max_events: int = Field(default=12, ge=1, le=200)
    event_extraction_max_events_per_part: int = Field(default=5, ge=1, le=30)
    event_index_concurrency: int = Field(default=3, ge=1, le=8)
    tkg_enabled: bool = True
    tkg_schema_mode: Literal["ontology", "open"] = "ontology"
    tkg_ontology_version: str = "0.1.0"
    tkg_ontology_dir: Path = Path("ontology")
    tkg_registry_path: Path | None = None
    tkg_validate_shacl: bool = True
    neo4j_enabled: bool = True
    neo4j_required: bool = True
    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "rag-graph-demo-change-me"
    neo4j_database: str = "neo4j"
    neo4j_workspace: str = "default"
    entity_type_mapping_file: Path = Path("ontology/mappings/entity_types.json")
    entity_resolution_enabled: bool = True
    entity_resolution_candidate_limit: int = Field(default=5, ge=1, le=20)
    entity_resolution_embedding_threshold: float = Field(default=0.72, ge=0, le=1)
    entity_resolution_auto_match_threshold: float = Field(default=0.94, ge=0, le=1)
    entity_resolution_llm_threshold: float = Field(default=0.78, ge=0, le=1)

    @field_validator("tkg_registry_path", mode="before")
    @classmethod
    def empty_tkg_path_uses_default(cls, value: object) -> object:
        return None if value == "" else value

    @property
    def effective_embedding_concurrency(self) -> int:
        """Keep local MPS/CUDA model access serialized; parallelize remote APIs only."""
        return 1 if self.embedding_provider == "local" else self.embedding_max_concurrency

    @property
    def required_data_dirs(self) -> tuple[Path, ...]:
        return tuple(
            self.data_dir / name
            for name in (
                "raw",
                "parsed",
                "registry",
                "events",
                "tkg",
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

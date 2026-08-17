from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class SearchApiConfig(BaseModel):
    """Provider-neutral JSON search endpoint.

    ``result_path`` is a dotted path to the result list. Field names inside
    each result are configurable so the collector is not tied to one vendor.
    """

    endpoint: str
    query_param: str = "q"
    page_param: str | None = None
    page_start: int = 1
    max_pages: int = Field(default=1, ge=1, le=20)
    result_path: str = "results"
    url_field: str = "url"
    title_field: str = "title"
    summary_field: str = "description"
    api_key_env: str | None = None
    api_key_header: str = "Authorization"
    api_key_prefix: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    extra_params: dict[str, str] = Field(default_factory=dict)


class TopicProfile(BaseModel):
    topic_id: str
    question: str
    keywords: list[str]
    exclude_keywords: list[str] = Field(default_factory=list)
    feeds: list[str] = Field(default_factory=list)
    sitemaps: list[str] = Field(default_factory=list)
    seed_urls: list[str] = Field(default_factory=list)
    search: SearchApiConfig | None = None
    language: str = "auto"
    max_rounds: int = Field(default=2, ge=1, le=5)
    max_candidates_per_round: int = Field(default=100, ge=1, le=100_000)
    max_fetches: int = Field(default=500, ge=1, le=1_000_000)
    max_queries_per_round: int = Field(default=12, ge=1, le=100)
    max_expansion_terms: int = Field(default=12, ge=0, le=100)
    rate_limit_seconds: float = Field(default=1.5, ge=0)
    max_bytes: int = Field(default=25 * 1024 * 1024, ge=1024)
    use_llm: bool = True

    @field_validator("topic_id")
    @classmethod
    def valid_topic_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{2,80}", cleaned):
            raise ValueError("topic_id must contain only letters, digits, '_' or '-'")
        return cleaned

    @field_validator("keywords", "exclude_keywords")
    @classmethod
    def clean_terms(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = " ".join(str(value).split()).strip()
            key = cleaned.casefold()
            if cleaned and key not in seen:
                result.append(cleaned)
                seen.add(key)
        return result

    @model_validator(mode="after")
    def has_discovery_entry(self) -> TopicProfile:
        if not (self.feeds or self.sitemaps or self.seed_urls or self.search):
            raise ValueError("at least one feed, sitemap, seed_url, or search endpoint is required")
        return self

    @classmethod
    def load(cls, path: Path | str) -> TopicProfile:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(value)


class TopicRoundReport(BaseModel):
    round_index: int
    terms: list[str] = Field(default_factory=list)
    discovered: int = 0
    new_candidates: int = 0
    fetched: int = 0
    published: int = 0
    new_records: int = 0
    expansion_terms: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    stop_reason: str | None = None
    run_id: str | None = None
    finished_at: datetime = Field(default_factory=utc_now)


class TopicState(BaseModel):
    topic_id: str
    profile_hash: str
    terms: list[str] = Field(default_factory=list)
    seen_urls: list[str] = Field(default_factory=list)
    completed_rounds: int = 0
    fetched_count: int = 0
    reports: list[TopicRoundReport] = Field(default_factory=list)
    planner_output: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=utc_now)

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class DocumentStatus(StrEnum):
    UPLOADED = "uploaded"
    PARSING = "parsing"
    PARSED = "parsed"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"
    DELETING = "deleting"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    title: str
    suffix: str
    mime_type: str | None = None
    size_bytes: int
    checksum: str
    raw_path: str
    parsed_path: str | None = None
    metadata_path: str | None = None
    page_count: int | None = None
    status: DocumentStatus = DocumentStatus.UPLOADED
    lightrag_track_id: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class JobRecord(BaseModel):
    job_id: str
    document_id: str
    operation: Literal["index", "delete", "reindex"]
    status: JobStatus = JobStatus.PENDING
    message: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class ParsedDocument(BaseModel):
    document_id: str
    title: str
    text: str
    page_count: int | None = None
    source_spans: list[dict[str, Any]] = Field(default_factory=list)


QueryMode = Literal["naive", "local", "global", "hybrid", "mix"]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    mode: QueryMode = "mix"
    top_k: int = Field(default=20, ge=1, le=100)
    chunk_top_k: int = Field(default=10, ge=1, le=50)
    max_total_tokens: int = Field(default=24_000, ge=1_000, le=200_000)
    enable_rerank: bool | None = None
    response_type: str = "Multiple Paragraphs"
    conversation_history: list[dict[str, str]] = Field(default_factory=list)


class CompareQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    modes: list[QueryMode] = Field(
        default_factory=lambda: ["naive", "local", "global", "mix"]
    )
    top_k: int = Field(default=20, ge=1, le=100)
    chunk_top_k: int = Field(default=10, ge=1, le=50)
    enable_rerank: bool | None = None


class Citation(BaseModel):
    reference_id: str
    document_id: str | None = None
    title: str | None = None
    filename: str | None = None
    file_path: str
    page_number: int | None = None
    section: str | None = None
    excerpt: str | None = None


class QueryResponse(BaseModel):
    status: str
    message: str
    mode: QueryMode
    answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    entities: list[dict[str, Any]] = Field(default_factory=list)
    relationships: list[dict[str, Any]] = Field(default_factory=list)
    chunks: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    elapsed_ms: int

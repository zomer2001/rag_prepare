from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkDocument:
    document_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def metadata_text(self) -> str:
        values = [f"Title: {self.title}"]
        for key in ("source", "published_at", "author", "category"):
            value = self.metadata.get(key)
            if value not in (None, ""):
                values.append(f"{key.replace('_', ' ').title()}: {value}")
        return "\n".join(values)


@dataclass(frozen=True)
class GoldEvidence:
    text: str | None = None
    document_id: str | None = None


@dataclass(frozen=True)
class BenchmarkQuery:
    query_id: str
    question: str
    question_type: str
    answer: str | None
    gold_evidence: tuple[GoldEvidence, ...]


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    documents: tuple[BenchmarkDocument, ...]
    queries: tuple[BenchmarkQuery, ...]


@dataclass(frozen=True)
class BenchmarkChunk:
    chunk_id: str
    document_id: str
    text: str
    order: int


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str | None
    text: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

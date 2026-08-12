from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from rag_graph_demo.domain.events import EventQueryResponse

RetrievalChannel = Literal["vector", "graph", "event"]


class QueryPlan(BaseModel):
    """One query analysis shared by the graph and event pathways."""

    low_level_keywords: list[str] = Field(default_factory=list)
    high_level_keywords: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    event_types: list[str] = Field(default_factory=list)
    start_time: str | None = None
    end_time: str | None = None
    state_intents: list[str] = Field(default_factory=list)
    expand_event_chain: bool = False


@dataclass(frozen=True)
class EvidenceCandidate:
    evidence_id: str
    channel: RetrievalChannel
    rank: int
    document_id: str | None
    chunk_id: str | None
    text: str
    file_path: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceBundle:
    document_id: str
    score: float
    channels: tuple[RetrievalChannel, ...]
    passages: tuple[EvidenceCandidate, ...]


@dataclass(frozen=True)
class UnifiedRetrievalResult:
    plan: QueryPlan
    bundles: tuple[EvidenceBundle, ...]
    candidates: dict[RetrievalChannel, tuple[EvidenceCandidate, ...]]
    graph_data: dict[str, Any] = field(default_factory=dict)
    event_result: EventQueryResponse | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

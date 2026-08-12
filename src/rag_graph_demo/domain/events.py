from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from rag_graph_demo.domain.models import utc_now


class EventStatus(StrEnum):
    CANDIDATE = "candidate"
    SINGLE_SOURCE = "single_source"
    MULTI_SOURCE = "multi_source"
    DISPUTED = "disputed"


class TimePrecision(StrEnum):
    UNKNOWN = "unknown"
    YEAR = "year"
    MONTH = "month"
    DAY = "day"


EventStance = Literal["affirmed", "denied", "uncertain"]


class EventCandidate(BaseModel):
    event_type: str = Field(min_length=1, max_length=100)
    summary: str = Field(min_length=1, max_length=1_000)
    action: str = Field(default="", max_length=200)
    actors: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    start_time: str | None = None
    end_time: str | None = None
    time_precision: TimePrecision = TimePrecision.UNKNOWN
    location: str | None = None
    result: str | None = None
    trigger: str | None = None
    evidence: str = Field(min_length=1, max_length=4_000)
    stance: EventStance = "affirmed"
    confidence: float = Field(default=0.65, ge=0, le=1)
    source_segment_id: str | None = None
    source_segment_start: int | None = Field(default=None, ge=0)
    source_segment_end: int | None = Field(default=None, ge=0)
    evidence_start: int | None = Field(default=None, ge=0)
    evidence_end: int | None = Field(default=None, ge=0)

    @field_validator("actors", "objects", mode="after")
    @classmethod
    def clean_entities(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            cleaned = " ".join(value.split()).strip(" ,，;；")
            if cleaned and cleaned not in result:
                result.append(cleaned)
        return result


class EventMention(BaseModel):
    mention_id: str
    event_id: str
    document_id: str
    document_title: str
    trigger: str | None = None
    evidence: str
    source_segment_id: str | None = None
    source_segment_start: int | None = Field(default=None, ge=0)
    source_segment_end: int | None = Field(default=None, ge=0)
    evidence_start: int | None = Field(default=None, ge=0)
    evidence_end: int | None = Field(default=None, ge=0)
    stance: EventStance = "affirmed"
    confidence: float = Field(ge=0, le=1)
    recorded_at: datetime = Field(default_factory=utc_now)


class EventRecord(BaseModel):
    event_id: str
    canonical_key: str
    event_type: str
    summary: str
    action: str = ""
    actors: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    start_time: str | None = None
    end_time: str | None = None
    time_precision: TimePrecision = TimePrecision.UNKNOWN
    location: str | None = None
    result: str | None = None
    status: EventStatus = EventStatus.CANDIDATE
    confidence: float = Field(default=0.5, ge=0, le=1)
    source_document_ids: list[str] = Field(default_factory=list)
    mention_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class EventRelation(BaseModel):
    relation_id: str
    source_event_id: str
    target_event_id: str
    relation_type: Literal["precedes", "same_time", "related"]
    shared_entities: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.7, ge=0, le=1)


class EventConflict(BaseModel):
    conflict_id: str
    event_id: str
    conflict_type: Literal["stance"] = "stance"
    description: str
    mention_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class EventQueryRequest(BaseModel):
    question: str = Field(default="", max_length=10_000)
    entity: str | None = None
    entities: list[str] = Field(default_factory=list)
    event_types: list[str] = Field(default_factory=list)
    start_time: str | None = None
    end_time: str | None = None
    min_confidence: float = Field(default=0, ge=0, le=1)
    limit: int = Field(default=20, ge=1, le=100)
    include_disputed: bool = True
    sort_by: Literal["relevance", "time"] = "relevance"
    state_intents: list[str] = Field(default_factory=list)
    expand_event_chain: bool = False


class EventRanking(BaseModel):
    """Query-local, explainable ranking signals for one event occurrence."""

    event_id: str
    rank: int = Field(ge=1)
    score: float = Field(ge=0)
    lexical_score: float = Field(ge=0, le=1)
    entity_score: float = Field(ge=0, le=1)
    action_score: float = Field(ge=0, le=1)
    temporal_score: float = Field(ge=0, le=1)
    state_score: float = Field(ge=0, le=1)
    confidence_score: float = Field(ge=0, le=1)
    source_score: float = Field(ge=0, le=1)
    expanded: bool = False


class EventQueryResponse(BaseModel):
    events: list[EventRecord] = Field(default_factory=list)
    rankings: list[EventRanking] = Field(default_factory=list)
    mentions: list[EventMention] = Field(default_factory=list)
    relations: list[EventRelation] = Field(default_factory=list)
    conflicts: list[EventConflict] = Field(default_factory=list)
    inferred_start_time: str | None = None
    inferred_end_time: str | None = None

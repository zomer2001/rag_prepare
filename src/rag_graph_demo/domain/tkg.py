from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

from rag_graph_demo.domain.models import utc_now


class MentionStance(StrEnum):
    ASSERTED = "asserted"
    DENIED = "denied"
    UNCERTAIN = "uncertain"
    CORRECTED = "corrected"
    RETRACTED = "retracted"


class AssertionStatus(StrEnum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DISPUTED = "disputed"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"


class EventEffectType(StrEnum):
    CREATES = "creates"
    UPDATES = "updates"
    TERMINATES = "terminates"
    SUPERSEDES = "supersedes"
    DENIES = "denies"
    CONFIRMS = "confirms"


class ResolutionAction(StrEnum):
    MATCH_EXISTING = "match_existing"
    CREATE_NEW = "create_new"
    AMBIGUOUS = "ambiguous"
    REJECT = "reject"


class ConflictType(StrEnum):
    STANCE = "stance"
    TEMPORAL_VALUE = "temporal_value"
    ENTITY_IDENTITY = "entity_identity"


class SemanticEntityCandidate(BaseModel):
    name: str = Field(min_length=1, max_length=500)
    entity_type: str = Field(default="Concept", max_length=100)
    description: str = Field(default="", max_length=4_000)
    aliases: list[str] = Field(default_factory=list)
    source_segment_id: str | None = None

    @field_validator("name", mode="after")
    @classmethod
    def clean_name(cls, value: str) -> str:
        return " ".join(value.split()).strip(" ,，;；")


class SemanticRelationshipCandidate(BaseModel):
    source: str = Field(min_length=1, max_length=500)
    target: str = Field(min_length=1, max_length=500)
    predicate: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=4_000)
    confidence: float = Field(default=0.65, ge=0, le=1)
    source_segment_id: str | None = None


class SemanticExtractionBatch(BaseModel):
    entities: list[SemanticEntityCandidate] = Field(default_factory=list)
    relationships: list[SemanticRelationshipCandidate] = Field(default_factory=list)


class EntityRecord(BaseModel):
    entity_id: str
    canonical_name: str = Field(min_length=1, max_length=500)
    entity_type: str = Field(min_length=1, max_length=100)
    aliases: list[str] = Field(default_factory=list)
    ontology_version: str
    mention_ids: list[str] = Field(default_factory=list)
    source_document_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class EntityMention(BaseModel):
    mention_id: str
    entity_id: str
    document_id: str
    document_title: str = ""
    surface_form: str = Field(min_length=1, max_length=500)
    evidence_text: str = Field(min_length=1, max_length=4_000)
    source_segment_id: str | None = None
    evidence_start: int | None = Field(default=None, ge=0)
    evidence_end: int | None = Field(default=None, ge=0)
    extractor_confidence: float = Field(default=0.65, ge=0, le=1)
    recorded_at: datetime = Field(default_factory=utc_now)


class AssertionRecord(BaseModel):
    assertion_id: str
    subject_id: str
    predicate_id: str = Field(min_length=1, max_length=200)
    object_id: str
    valid_from: str | None = None
    valid_to: str | None = None
    stated_valid_to: str | None = None
    recorded_at: datetime = Field(default_factory=utc_now)
    invalidated_at: datetime | None = None
    status: AssertionStatus = AssertionStatus.CANDIDATE
    confidence: float = Field(default=0.5, ge=0, le=1)
    ontology_version: str
    mention_ids: list[str] = Field(default_factory=list)
    event_effect_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_interval(self) -> AssertionRecord:
        if self.valid_from and self.valid_to and self.valid_from > self.valid_to:
            raise ValueError("valid_from must not be later than valid_to")
        return self


class AssertionMention(BaseModel):
    mention_id: str
    assertion_id: str
    document_id: str
    document_title: str = ""
    evidence_text: str = Field(min_length=1, max_length=4_000)
    source_segment_id: str | None = None
    stance: MentionStance = MentionStance.ASSERTED
    extractor_confidence: float = Field(default=0.65, ge=0, le=1)
    published_at: str | None = None
    recorded_at: datetime = Field(default_factory=utc_now)


class EventEntityLink(BaseModel):
    link_id: str
    event_id: str
    entity_id: str
    role: str = Field(pattern="^(actor|object|location)$")
    surface_form: str
    document_id: str


class EventEffect(BaseModel):
    effect_id: str
    event_id: str
    assertion_id: str
    effect_type: EventEffectType
    stance: MentionStance = MentionStance.ASSERTED
    document_id: str
    effective_time: str | None = None
    reason: str = ""
    created_at: datetime = Field(default_factory=utc_now)


class ConflictCase(BaseModel):
    conflict_id: str
    conflict_type: ConflictType
    assertion_ids: list[str] = Field(min_length=1)
    mention_ids: list[str] = Field(default_factory=list)
    description: str
    status: str = "open"
    created_at: datetime = Field(default_factory=utc_now)


class ResolutionDecision(BaseModel):
    decision_id: str
    document_id: str
    candidate_name: str
    candidate_type: str
    action: ResolutionAction
    selected_entity_id: str | None = None
    candidate_entity_ids: list[str] = Field(default_factory=list)
    reason: str
    similarity: float | None = Field(default=None, ge=0, le=1)
    decided_by: str = "rule"
    ontology_version: str
    created_at: datetime = Field(default_factory=utc_now)


class TKGSnapshot(BaseModel):
    schema_version: str = "1"
    ontology_version: str
    entities: list[EntityRecord] = Field(default_factory=list)
    entity_mentions: list[EntityMention] = Field(default_factory=list)
    assertions: list[AssertionRecord] = Field(default_factory=list)
    assertion_mentions: list[AssertionMention] = Field(default_factory=list)
    event_entity_links: list[EventEntityLink] = Field(default_factory=list)
    event_effects: list[EventEffect] = Field(default_factory=list)
    conflicts: list[ConflictCase] = Field(default_factory=list)
    resolution_decisions: list[ResolutionDecision] = Field(default_factory=list)


class TKGIndexResult(BaseModel):
    document_id: str
    entity_count: int = 0
    assertion_count: int = 0
    event_link_count: int = 0
    event_effect_count: int = 0
    conflict_count: int = 0


class TKGQueryRequest(BaseModel):
    entity_id: str | None = None
    entity_name: str | None = None
    predicate_id: str | None = None
    at_time: str | None = None
    include_inactive: bool = False
    include_disputed: bool = True
    limit: int = Field(default=100, ge=1, le=500)


class TKGQueryResponse(BaseModel):
    entities: list[EntityRecord] = Field(default_factory=list)
    assertions: list[AssertionRecord] = Field(default_factory=list)
    mentions: list[AssertionMention] = Field(default_factory=list)
    effects: list[EventEffect] = Field(default_factory=list)
    conflicts: list[ConflictCase] = Field(default_factory=list)

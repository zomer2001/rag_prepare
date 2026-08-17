from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field, field_validator

from rag_graph_demo.evaluation.models import RetrievedChunk


class EvidenceAssessment(BaseModel):
    relevant: bool
    sufficient: bool
    reason: str = ""
    missing_information: str = ""


class QueryRewrite(BaseModel):
    query: str = Field(min_length=1)
    rationale: str = ""


class GroundedAnswer(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    insufficient_evidence: bool = False


KnowledgeType = Literal[
    "text_evidence",
    "relation_evidence",
    "evolution_evidence",
]
NeedStatus = Literal["missing", "partial", "satisfied"]
NeedRoute = Literal["vector", "graph", "event"]


class NeedItem(BaseModel):
    """One auditable knowledge gap and the route-specific action that can close it."""

    need_id: str = Field(default="", coerce_numbers_to_str=True)
    knowledge_type: KnowledgeType
    status: NeedStatus = "missing"
    query: str = Field(min_length=1)
    route: NeedRoute
    rationale: str = ""
    entities: list[str] = Field(default_factory=list)
    relations: list[str] = Field(default_factory=list)
    event_types: list[str] = Field(default_factory=list)
    start_time: str | None = None
    end_time: str | None = None
    missing_slots: list[str] = Field(default_factory=list)
    supported_evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("relations", mode="before")
    @classmethod
    def _coerce_relation_patterns(cls, value: Any) -> Any:
        """Accept either relation strings or structured relation paths from an LLM."""
        if not isinstance(value, list):
            return value

        normalized: list[Any] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                parts = [str(part).strip() for part in item if str(part).strip()]
                normalized.append(" -> ".join(parts))
            else:
                normalized.append(item)
        return normalized


class EstablishedFinding(BaseModel):
    finding: str
    evidence_ids: list[str] = Field(default_factory=list)
    knowledge_types: list[KnowledgeType] = Field(default_factory=list)


class KnowledgeNeedState(BaseModel):
    """The analysis state updated after every evidence-acquisition round."""

    analytic_goal: str
    sufficient: bool = False
    needs: list[NeedItem] = Field(default_factory=list)
    established_findings: list[EstablishedFinding] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    stop_reason: str = ""

    @field_validator("established_findings", mode="before")
    @classmethod
    def _coerce_finding_strings(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [
            {"finding": item, "evidence_ids": [], "knowledge_types": []}
            if isinstance(item, str)
            else item
            for item in value
        ]


T = TypeVar("T")


@dataclass(frozen=True)
class ModelCall(Generic[T]):
    value: T
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class AgenticRun:
    question: str
    answer: str
    citations: tuple[str, ...]
    evidence: tuple[RetrievedChunk, ...]
    trace: tuple[dict[str, Any], ...]
    retrieval_rounds: int
    llm_calls: int
    input_tokens: int
    output_tokens: int
    insufficient_evidence: bool

    def trajectory(self) -> dict[str, Any]:
        return {
            "retrieval_rounds": self.retrieval_rounds,
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "events": list(self.trace),
        }

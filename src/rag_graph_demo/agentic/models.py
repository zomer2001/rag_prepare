from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

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

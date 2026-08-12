from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from rag_graph_demo.agentic.model import AgentModel
from rag_graph_demo.agentic.models import (
    AgenticRun,
    EvidenceAssessment,
    GroundedAnswer,
)
from rag_graph_demo.agentic.prompts import evidence_id
from rag_graph_demo.evaluation.models import RetrievedChunk


class Retriever(Protocol):
    async def __call__(self, question: str, top_k: int) -> list[RetrievedChunk]: ...


class AgentState(TypedDict, total=False):
    question: str
    current_query: str
    top_k: int
    context_k: int
    max_retrieval_rounds: int
    rrf_k: int
    retrieval_round: int
    retrieval_batches: list[list[RetrievedChunk]]
    evidence: list[RetrievedChunk]
    assessment: EvidenceAssessment
    answer: GroundedAnswer
    trace: list[dict[str, Any]]
    llm_calls: int
    input_tokens: int
    output_tokens: int


class AgenticRAGGraph:
    def __init__(self, retriever: Retriever, model: AgentModel) -> None:
        self.retriever = retriever
        self.model = model
        builder = StateGraph(AgentState)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("assess", self._assess)
        builder.add_node("rewrite", self._rewrite)
        builder.add_node("answer", self._answer)
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "assess")
        builder.add_conditional_edges(
            "assess",
            self._route_after_assessment,
            {"rewrite": "rewrite", "answer": "answer"},
        )
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("answer", END)
        self.compiled = builder.compile()

    async def run(
        self,
        question: str,
        *,
        top_k: int,
        context_k: int,
        max_retrieval_rounds: int,
        rrf_k: int,
    ) -> AgenticRun:
        state = await self.compiled.ainvoke(
            {
                "question": question,
                "current_query": question,
                "top_k": top_k,
                "context_k": context_k,
                "max_retrieval_rounds": max_retrieval_rounds,
                "rrf_k": rrf_k,
                "retrieval_round": 0,
                "retrieval_batches": [],
                "evidence": [],
                "trace": [],
                "llm_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }
        )
        answer = state["answer"]
        return AgenticRun(
            question=question,
            answer=answer.answer,
            citations=tuple(answer.citations),
            evidence=tuple(state["evidence"]),
            trace=tuple(state["trace"]),
            retrieval_rounds=state["retrieval_round"],
            llm_calls=state["llm_calls"],
            input_tokens=state["input_tokens"],
            output_tokens=state["output_tokens"],
            insufficient_evidence=answer.insufficient_evidence,
        )

    async def _retrieve(self, state: AgentState) -> dict[str, Any]:
        query = state["current_query"]
        batch = await self.retriever(query, state["top_k"])
        batches = [*state["retrieval_batches"], batch]
        evidence = reciprocal_rank_fusion_rounds(
            batches,
            rrf_k=state["rrf_k"],
            limit=state["context_k"],
        )
        retrieval_round = state["retrieval_round"] + 1
        return {
            "retrieval_batches": batches,
            "evidence": evidence,
            "retrieval_round": retrieval_round,
            "trace": [
                *state["trace"],
                {
                    "node": "retrieve",
                    "round": retrieval_round,
                    "query": query,
                    "result_count": len(batch),
                    "evidence_ids": [evidence_id(item) for item in batch],
                },
            ],
        }

    async def _assess(self, state: AgentState) -> dict[str, Any]:
        if not state["evidence"]:
            assessment = EvidenceAssessment(
                relevant=False,
                sufficient=False,
                reason="No evidence was retrieved.",
                missing_information="Evidence that directly addresses the question.",
            )
            usage = {}
        else:
            call = await self.model.assess(
                state["question"],
                state["evidence"],
                retrieval_round=state["retrieval_round"],
            )
            assessment = call.value
            usage = _usage_update(state, call.input_tokens, call.output_tokens)
        return {
            "assessment": assessment,
            "trace": [
                *state["trace"],
                {
                    "node": "assess",
                    "round": state["retrieval_round"],
                    **assessment.model_dump(),
                },
            ],
            **usage,
        }

    def _route_after_assessment(self, state: AgentState) -> str:
        if state["assessment"].sufficient:
            return "answer"
        if state["retrieval_round"] >= state["max_retrieval_rounds"]:
            return "answer"
        return "rewrite"

    async def _rewrite(self, state: AgentState) -> dict[str, Any]:
        call = await self.model.rewrite(
            state["question"],
            state["current_query"],
            state["evidence"],
            state["assessment"],
        )
        query = call.value.query.strip()
        if query.casefold() == state["current_query"].strip().casefold():
            query = f"{query} {state['assessment'].missing_information}".strip()
        return {
            "current_query": query,
            "trace": [
                *state["trace"],
                {
                    "node": "rewrite",
                    "round": state["retrieval_round"],
                    "query": query,
                    "rationale": call.value.rationale,
                },
            ],
            **_usage_update(state, call.input_tokens, call.output_tokens),
        }

    async def _answer(self, state: AgentState) -> dict[str, Any]:
        call = await self.model.answer(
            state["question"],
            state["evidence"],
            state.get("assessment"),
        )
        valid_ids = {evidence_id(item) for item in state["evidence"]}
        citations = list(dict.fromkeys(item for item in call.value.citations if item in valid_ids))
        answer = call.value.model_copy(update={"citations": citations})
        return {
            "answer": answer,
            "trace": [
                *state["trace"],
                {
                    "node": "answer",
                    "citations": citations,
                    "insufficient_evidence": answer.insufficient_evidence,
                },
            ],
            **_usage_update(state, call.input_tokens, call.output_tokens),
        }


def reciprocal_rank_fusion_rounds(
    batches: list[list[RetrievedChunk]],
    *,
    rrf_k: int,
    limit: int,
) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    selected: dict[str, RetrievedChunk] = {}
    round_hits: dict[str, list[int]] = {}
    for round_number, batch in enumerate(batches, start=1):
        seen: set[str] = set()
        for rank, item in enumerate(batch, start=1):
            key = evidence_id(item)
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            selected.setdefault(key, item)
            round_hits.setdefault(key, []).append(round_number)
    ordered = sorted(scores, key=lambda key: (scores[key], key), reverse=True)[:limit]
    return [
        replace(
            selected[key],
            score=scores[key],
            metadata={
                **selected[key].metadata,
                "fusion": "agentic_round_rrf",
                "agentic_round_hits": round_hits[key],
            },
        )
        for key in ordered
    ]


def _usage_update(state: AgentState, input_tokens: int, output_tokens: int) -> dict[str, int]:
    return {
        "llm_calls": state["llm_calls"] + 1,
        "input_tokens": state["input_tokens"] + input_tokens,
        "output_tokens": state["output_tokens"] + output_tokens,
    }

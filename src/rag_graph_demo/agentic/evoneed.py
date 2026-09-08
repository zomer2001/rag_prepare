from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from rag_graph_demo.agentic.models import (
    AgenticRun,
    EvidenceAssessment,
    GroundedAnswer,
    KnowledgeNeedState,
    ModelCall,
    NeedItem,
)
from rag_graph_demo.agentic.prompts import evidence_id
from rag_graph_demo.evaluation.models import RetrievedChunk


class EvoNeedRetriever(Protocol):
    async def __call__(
        self,
        needs: list[NeedItem],
        budgets: list[int],
    ) -> list[list[RetrievedChunk]]: ...


class EvoNeedModel(Protocol):
    async def plan_needs(
        self,
        question: str,
        *,
        total_budget: int,
    ) -> ModelCall[KnowledgeNeedState]: ...

    async def assess_needs(
        self,
        question: str,
        need_state: KnowledgeNeedState,
        evidence: list[RetrievedChunk],
        *,
        retrieval_round: int,
    ) -> ModelCall[KnowledgeNeedState]: ...

    async def answer(
        self,
        question: str,
        evidence: list[RetrievedChunk],
        assessment: EvidenceAssessment | None,
    ) -> ModelCall[GroundedAnswer]: ...


class EvoNeedState(TypedDict, total=False):
    question: str
    top_k: int
    retrieval_budget: int
    context_k: int
    max_retrieval_rounds: int
    rrf_k: int
    retrieval_round: int
    need_state: KnowledgeNeedState
    retrieval_batches: list[list[RetrievedChunk]]
    evidence: list[RetrievedChunk]
    answer: GroundedAnswer
    trace: list[dict[str, Any]]
    llm_calls: int
    input_tokens: int
    output_tokens: int
    retrieval_only: bool


class EvoNeedRAGGraph:
    """Training-free typed-gap planning over selective EvoLink retrieval actions."""

    def __init__(
        self,
        retriever: EvoNeedRetriever,
        model: EvoNeedModel,
        *,
        initial_retrieval: Literal["question_anchor", "planned_needs"] = "question_anchor",
        route_budget_mode: Literal[
            "per_action", "shared_total", "equal_total", "planned_total"
        ] = "per_action",
        minimum_route_budget: int = 1,
    ) -> None:
        if initial_retrieval not in {"question_anchor", "planned_needs"}:
            raise ValueError(f"Unsupported initial retrieval mode: {initial_retrieval}")
        if route_budget_mode not in {
            "per_action",
            "shared_total",
            "equal_total",
            "planned_total",
        }:
            raise ValueError(f"Unsupported route budget mode: {route_budget_mode}")
        if minimum_route_budget < 1:
            raise ValueError("minimum_route_budget must be positive")
        self.retriever = retriever
        self.model = model
        self.initial_retrieval = initial_retrieval
        self.route_budget_mode = route_budget_mode
        self.minimum_route_budget = minimum_route_budget
        builder = StateGraph(EvoNeedState)
        builder.add_node("plan", self._plan)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("assess", self._assess)
        builder.add_node("answer", self._answer)
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "retrieve")
        builder.add_conditional_edges(
            "retrieve",
            self._route_after_retrieval,
            {"assess": "assess", "end": END},
        )
        builder.add_conditional_edges(
            "assess",
            self._route_after_assessment,
            {"retrieve": "retrieve", "answer": "answer"},
        )
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
        retrieval_budget: int | None = None,
        retrieval_only: bool = False,
    ) -> AgenticRun:
        state = await self.compiled.ainvoke(
            {
                "question": question,
                "top_k": top_k,
                "retrieval_budget": retrieval_budget or top_k,
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
                "retrieval_only": retrieval_only,
            }
        )
        answer = state.get("answer") or GroundedAnswer(
            answer="",
            citations=[],
            insufficient_evidence=False,
        )
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

    @staticmethod
    def _route_after_retrieval(state: EvoNeedState) -> str:
        return "end" if state.get("retrieval_only", False) else "assess"

    async def _plan(self, state: EvoNeedState) -> dict[str, Any]:
        call = await self.model.plan_needs(
            state["question"],
            total_budget=state["retrieval_budget"],
        )
        need_state = normalize_need_state(
            call.value,
            question=state["question"],
            initial=True,
        )
        return {
            "need_state": need_state,
            "trace": [
                *state["trace"],
                {
                    "node": "plan_needs",
                    "round": 0,
                    "need_state": need_state.model_dump(mode="json"),
                },
            ],
            **_usage_update(state, call.input_tokens, call.output_tokens),
        }

    async def _retrieve(self, state: EvoNeedState) -> dict[str, Any]:
        if (
            state["retrieval_round"] == 0
            and self.initial_retrieval == "question_anchor"
        ):
            open_needs = [
                NeedItem(
                    need_id="need_question_anchor",
                    knowledge_type="text_evidence",
                    query=state["question"],
                    route="vector",
                    rationale=(
                        "Lossless three-path acquisition used to establish the initial "
                        "evidence pool before typed gap filling."
                    ),
                )
            ]
        else:
            # In planned-needs mode, the first acquisition executes the
            # planner's route-specific subqueries. All actions still run in
            # parallel below and therefore count as one retrieval round.
            open_needs = [
                item
                for item in state["need_state"].needs
                if item.status != "satisfied"
            ][:3]
        if not open_needs:
            open_needs = [
                NeedItem(
                    need_id="need_text_fallback",
                    knowledge_type="text_evidence",
                    query=state["question"],
                    route="vector",
                    rationale="Fallback grounding search when no open need was emitted.",
                )
            ]
        if self.route_budget_mode in {"shared_total", "equal_total", "planned_total"}:
            # Every route receives at least one slot without exceeding the
            # query-level acquisition budget.
            open_needs = open_needs[: max(1, state["retrieval_budget"])]
        shared_total = self.route_budget_mode in {
            "shared_total",
            "equal_total",
            "planned_total",
        }
        planned_budgets = (
            None
            if self.route_budget_mode == "equal_total"
            else [item.budget for item in open_needs]
        )
        budgets = allocate_route_budgets(
            state["retrieval_budget"],
            len(open_needs),
            shared_total=shared_total,
            planned_budgets=planned_budgets,
            minimum_per_action=(
                self.minimum_route_budget
                if self.route_budget_mode in {"shared_total", "planned_total"}
                else 1
            ),
        )
        results = await self.retriever(open_needs, budgets)
        if len(results) != len(open_needs):
            raise RuntimeError("EvoNeed retriever returned the wrong number of action batches")
        retrieval_round = state["retrieval_round"] + 1
        action_batches: list[list[RetrievedChunk]] = []
        round_result_count = 0
        actions: list[dict[str, Any]] = []
        for need, budget, values in zip(open_needs, budgets, results, strict=True):
            actions.append(
                {
                    "need_id": need.need_id,
                    "knowledge_type": need.knowledge_type,
                    "route": need.route,
                    "query": need.query,
                    "planned_budget": need.budget,
                    "budget": budget,
                    "result_count": len(values),
                }
            )
            action_batch: list[RetrievedChunk] = []
            for item in values:
                prior_need_ids = list(item.metadata.get("supported_need_ids") or [])
                prior_types = list(item.metadata.get("knowledge_types") or [])
                action_batch.append(
                    replace(
                        item,
                        metadata={
                            **item.metadata,
                            "supported_need_ids": list(
                                dict.fromkeys([*prior_need_ids, need.need_id])
                            ),
                            "knowledge_types": list(
                                dict.fromkeys([*prior_types, need.knowledge_type])
                            ),
                            "evoneed_route": need.route,
                            "evoneed_action_id": need.need_id,
                            "evoneed_round": retrieval_round,
                        },
                    )
                )
                round_result_count += 1
            action_batches.append(action_batch)
        batches = [*state["retrieval_batches"], *action_batches]
        evidence = need_coverage_fusion(
            batches,
            rrf_k=state["rrf_k"],
            limit=state["context_k"],
        )
        return {
            "retrieval_batches": batches,
            "evidence": evidence,
            "retrieval_round": retrieval_round,
            "trace": [
                *state["trace"],
                {
                    "node": "retrieve_needs",
                    "round": retrieval_round,
                    "initial_retrieval": self.initial_retrieval,
                    "route_budget_mode": self.route_budget_mode,
                    "total_candidate_budget": sum(budgets),
                    "actions": actions,
                    "result_count": round_result_count,
                    "evidence_ids": [evidence_id(item) for item in evidence],
                },
            ],
        }

    async def _assess(self, state: EvoNeedState) -> dict[str, Any]:
        call = await self.model.assess_needs(
            state["question"],
            state["need_state"],
            state["evidence"],
            retrieval_round=state["retrieval_round"],
        )
        valid_ids = {evidence_id(item) for item in state["evidence"]}
        need_state = normalize_need_state(
            call.value,
            question=state["question"],
            valid_evidence_ids=valid_ids,
            previous_state=state["need_state"],
        )
        evidence = prioritize_supported_evidence(state["evidence"], need_state)
        return {
            "need_state": need_state,
            "evidence": evidence,
            "trace": [
                *state["trace"],
                {
                    "node": "assess_needs",
                    "round": state["retrieval_round"],
                    "need_state": need_state.model_dump(mode="json"),
                    "prioritized_evidence_ids": [
                        evidence_id(item) for item in evidence
                    ],
                },
            ],
            **_usage_update(state, call.input_tokens, call.output_tokens),
        }

    def _route_after_assessment(self, state: EvoNeedState) -> str:
        if state["need_state"].sufficient:
            return "answer"
        if state["retrieval_round"] >= state["max_retrieval_rounds"]:
            return "answer"
        return "retrieve"

    async def _answer(self, state: EvoNeedState) -> dict[str, Any]:
        open_needs = [
            item for item in state["need_state"].needs if item.status != "satisfied"
        ]
        diagnosis = EvidenceAssessment(
            relevant=bool(state["evidence"]),
            sufficient=state["need_state"].sufficient,
            reason=(
                state["need_state"].stop_reason
                or "Structured knowledge-need assessment completed."
            ),
            missing_information="; ".join(item.query for item in open_needs),
        )
        call = await self.model.answer(
            state["question"],
            state["evidence"],
            diagnosis,
        )
        valid_ids = {evidence_id(item) for item in state["evidence"]}
        citations = list(
            dict.fromkeys(item for item in call.value.citations if item in valid_ids)
        )
        answer = call.value.model_copy(
            update={
                "citations": citations,
                "insufficient_evidence": (
                    call.value.insufficient_evidence
                    or not state["need_state"].sufficient
                ),
            }
        )
        return {
            "answer": answer,
            "trace": [
                *state["trace"],
                {
                    "node": "answer",
                    "citations": citations,
                    "insufficient_evidence": answer.insufficient_evidence,
                    "final_need_state": state["need_state"].model_dump(mode="json"),
                },
            ],
            **_usage_update(state, call.input_tokens, call.output_tokens),
        }


def allocate_route_budgets(
    top_k: int,
    action_count: int,
    *,
    shared_total: bool = False,
    planned_budgets: list[int | None] | None = None,
    minimum_per_action: int = 1,
) -> list[int]:
    action_count = max(1, action_count)
    if shared_total:
        minimum_per_action = max(1, minimum_per_action)
        if action_count * minimum_per_action > top_k:
            raise ValueError(
                "shared route budget requires total >= actions * minimum"
            )
        if planned_budgets is None:
            planned_budgets = [1] * action_count
        if len(planned_budgets) != action_count:
            raise ValueError("planned budget count must match action count")
        weights = [max(1, int(value or 1)) for value in planned_budgets]
        if sum(weights) == top_k and all(
            value >= minimum_per_action for value in weights
        ):
            return weights
        remaining = top_k - action_count * minimum_per_action
        raw = [remaining * weight / sum(weights) for weight in weights]
        budgets = [minimum_per_action + int(value) for value in raw]
        missing = top_k - sum(budgets)
        order = sorted(
            range(action_count),
            key=lambda index: (-(raw[index] - int(raw[index])), index),
        )
        for index in order[:missing]:
            budgets[index] += 1
        return budgets
    return [max(1, top_k)] * action_count


def normalize_need_state(
    value: KnowledgeNeedState,
    *,
    question: str,
    initial: bool = False,
    valid_evidence_ids: set[str] | None = None,
    previous_state: KnowledgeNeedState | None = None,
) -> KnowledgeNeedState:
    route_by_type = {
        "text_evidence": "vector",
        "relation_evidence": "graph",
        "evolution_evidence": "event",
    }
    previous_by_id = (
        {item.need_id.casefold(): item for item in previous_state.needs}
        if previous_state is not None
        else {}
    )
    needs: list[NeedItem] = []
    seen: set[str] = set()
    for index, item in enumerate(value.needs[:6], start=1):
        need_id = item.need_id.strip() or f"need_{index}"
        base_id = need_id
        suffix = 2
        while need_id.casefold() in seen:
            need_id = f"{base_id}_{suffix}"
            suffix += 1
        seen.add(need_id.casefold())
        supported = list(dict.fromkeys(item.supported_evidence_ids))
        if valid_evidence_ids is not None:
            supported = [item for item in supported if item in valid_evidence_ids]
        status = "missing" if initial else item.status
        if status == "satisfied" and not supported:
            status = "partial"
        previous = previous_by_id.get(need_id.casefold())
        query = item.query.strip()
        if not query:
            query = (
                previous.query.strip()
                if previous is not None and previous.query.strip()
                else question
            )
        needs.append(
            item.model_copy(
                update={
                    "need_id": need_id,
                    "query": query,
                    "route": route_by_type[item.knowledge_type],
                    "status": status,
                    "budget": item.budget or (previous.budget if previous else None),
                    "low_level_keywords": list(
                        dict.fromkeys(item.low_level_keywords)
                    )[:12],
                    "high_level_keywords": list(
                        dict.fromkeys(item.high_level_keywords)
                    )[:12],
                    "entities": list(dict.fromkeys(item.entities))[:6],
                    "relations": list(dict.fromkeys(item.relations))[:6],
                    "event_types": list(dict.fromkeys(item.event_types))[:6],
                    "state_intents": [
                        intent
                        for intent in dict.fromkeys(item.state_intents)
                        if intent
                        in {
                            "latest",
                            "change",
                            "conflict",
                            "denial",
                            "correction",
                            "replacement",
                        }
                    ][:6],
                    "missing_slots": list(dict.fromkeys(item.missing_slots))[:6],
                    "supported_evidence_ids": supported,
                }
            )
        )
    if initial and _requires_evolution_evidence(question) and not any(
        item.knowledge_type == "evolution_evidence" for item in needs
    ):
        evolution_need = _evolution_guard_need(question, needs)
        if len(needs) >= 3:
            needs[-1] = evolution_need
        else:
            needs.append(evolution_need)
    if initial:
        # The controlled experiment permits at most one action per retrieval route.
        unique_routes: dict[str, NeedItem] = {}
        for item in needs:
            unique_routes.setdefault(item.route, item)
        needs = list(unique_routes.values())[:3]
    if previous_state is not None:
        needs = [
            _progress_repeated_query(item, previous_by_id.get(item.need_id.casefold()), question)
            for item in needs
        ]
    if not needs:
        needs = [
            NeedItem(
                need_id="need_1",
                knowledge_type="text_evidence",
                query=question,
                route="vector",
                rationale="Ground the analytic goal in source text.",
            )
        ]
    valid_findings = []
    for finding in value.established_findings:
        ids = list(dict.fromkeys(finding.evidence_ids))
        if valid_evidence_ids is not None:
            ids = [item for item in ids if item in valid_evidence_ids]
        if ids:
            valid_findings.append(finding.model_copy(update={"evidence_ids": ids}))
    sufficient = (
        not initial
        and value.sufficient
        and all(item.status == "satisfied" for item in needs)
    )
    return value.model_copy(
        update={
            "analytic_goal": value.analytic_goal.strip() or question,
            "sufficient": sufficient,
            "needs": needs,
            "established_findings": valid_findings,
        }
    )


def need_coverage_fusion(
    batches: list[list[RetrievedChunk]],
    *,
    rrf_k: int,
    limit: int,
) -> list[RetrievedChunk]:
    best_scores: dict[str, float] = {}
    selected: dict[str, RetrievedChunk] = {}
    need_ids: dict[str, set[str]] = {}
    scores_by_need: dict[str, dict[str, float]] = {}
    channels_by_id: dict[str, set[str]] = {}
    round_hits: dict[str, list[int]] = {}
    for batch_number, batch in enumerate(batches, start=1):
        seen: set[str] = set()
        for rank, item in enumerate(batch, start=1):
            key = evidence_id(item)
            if not key or key in seen:
                continue
            seen.add(key)
            rank_score = 1.0 / (rrf_k + rank)
            best_scores[key] = max(best_scores.get(key, 0.0), rank_score)
            selected.setdefault(key, item)
            need_ids.setdefault(key, set()).update(
                str(value) for value in item.metadata.get("supported_need_ids") or []
            )
            channels_by_id.setdefault(key, set()).update(
                str(value) for value in item.metadata.get("channels") or []
            )
            action_id = str(item.metadata.get("evoneed_action_id") or "unassigned")
            prior_action_score = scores_by_need.setdefault(key, {}).get(action_id, 0.0)
            scores_by_need[key][action_id] = max(prior_action_score, rank_score)
            retrieval_round = int(item.metadata.get("evoneed_round") or batch_number)
            round_hits.setdefault(key, []).append(retrieval_round)

    result: list[RetrievedChunk] = []
    covered: set[str] = set()
    remaining = set(best_scores)
    while remaining and len(result) < limit:
        def utility(key: str) -> tuple[float, float, str]:
            uncovered_scores = [
                score
                for need_id, score in scores_by_need[key].items()
                if need_id not in covered
            ]
            coverage_bonus = 0.002 if uncovered_scores else 0.0
            anchor_bonus = (
                0.003
                if "need_question_anchor" in scores_by_need[key]
                else 0.0
            )
            bridge_bonus = 0.0005 if len(channels_by_id[key]) > 1 else 0.0
            consensus_bonus = min(len(scores_by_need[key]) - 1, 3) * 0.0002
            score = (
                best_scores[key]
                + coverage_bonus
                + anchor_bonus
                + bridge_bonus
                + consensus_bonus
            )
            return (score, best_scores[key], key)

        key = max(remaining, key=utility)
        remaining.remove(key)
        uncovered = {
            need_id: score
            for need_id, score in scores_by_need[key].items()
            if need_id not in covered
        }
        if uncovered:
            covered.add(max(uncovered, key=uncovered.get))
        result.append(
            replace(
                selected[key],
                score=best_scores[key],
                metadata={
                    **selected[key].metadata,
                    "fusion": "evoneed_coverage_rrf",
                    "agentic_round_hits": sorted(set(round_hits[key])),
                    "supported_need_ids": sorted(need_ids[key]),
                    "channels": sorted(channels_by_id[key]),
                },
            )
        )
    return result


def prioritize_supported_evidence(
    evidence: list[RetrievedChunk],
    need_state: KnowledgeNeedState,
) -> list[RetrievedChunk]:
    support_scores: dict[str, int] = {}
    status_weight = {"missing": 1, "partial": 2, "satisfied": 3}
    for need in need_state.needs:
        for item_id in need.supported_evidence_ids:
            support_scores[item_id] = support_scores.get(item_id, 0) + status_weight[
                need.status
            ]
    for finding in need_state.established_findings:
        for item_id in finding.evidence_ids:
            support_scores[item_id] = support_scores.get(item_id, 0) + 2
    indexed = list(enumerate(evidence))
    indexed.sort(
        key=lambda pair: (
            -support_scores.get(evidence_id(pair[1]), 0),
            pair[0],
        )
    )
    return [item for _, item in indexed]


_EVOLUTION_CUES = re.compile(
    r"\b(change(?:d|s)?|changing|evol(?:ve|ved|ution)|shift(?:ed|s)?|"
    r"remain(?:ed|s)?|consistent|before\s+and\s+after|over\s+time|"
    r"from\s+.+?\s+to\s+)\b",
    flags=re.IGNORECASE,
)


def _requires_evolution_evidence(question: str) -> bool:
    return bool(_EVOLUTION_CUES.search(question))


def _evolution_guard_need(
    question: str,
    existing_needs: list[NeedItem],
) -> NeedItem:
    entities = list(
        dict.fromkeys(entity for item in existing_needs for entity in item.entities)
    )[:6]
    relations = list(
        dict.fromkeys(relation for item in existing_needs for relation in item.relations)
    )[:6]
    starts = [item.start_time for item in existing_needs if item.start_time]
    ends = [item.end_time for item in existing_needs if item.end_time]
    return NeedItem(
        need_id="need_evolution_guard",
        knowledge_type="evolution_evidence",
        query=question,
        route="event",
        rationale=(
            "The question explicitly asks about change or consistency across time."
        ),
        entities=entities,
        relations=relations,
        event_types=["state_or_stance_change"],
        start_time=min(starts) if starts else None,
        end_time=max(ends) if ends else None,
        missing_slots=[
            "state_before",
            "change_event",
            "effective_time",
            "state_after",
            "stance_status",
            "supporting_source",
        ],
    )


def _same_query(left: str, right: str) -> bool:
    def normalize(value: str) -> str:
        return " ".join(value.casefold().split()).strip(" ?.,;:")

    return normalize(left) == normalize(right)


def _progress_repeated_query(
    current: NeedItem,
    previous: NeedItem | None,
    question: str,
) -> NeedItem:
    if (
        previous is None
        or current.status == "satisfied"
        or not _same_query(current.query, previous.query)
    ):
        return current
    return current.model_copy(
        update={
            "query": f"{current.query}. Full question context: {question}",
            "rationale": (
                f"{current.rationale} The prior query was unchanged, so the lossless "
                "question context was restored for the next retrieval round."
            ).strip(),
        }
    )


def _usage_update(
    state: EvoNeedState,
    input_tokens: int,
    output_tokens: int,
) -> dict[str, int]:
    return {
        "llm_calls": state["llm_calls"] + 1,
        "input_tokens": state["input_tokens"] + input_tokens,
        "output_tokens": state["output_tokens"] + output_tokens,
    }

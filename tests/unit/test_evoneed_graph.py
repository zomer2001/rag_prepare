from __future__ import annotations

import pytest

from rag_graph_demo.agentic.evoneed import (
    EvoNeedRAGGraph,
    allocate_route_budgets,
    need_coverage_fusion,
    normalize_need_state,
    prioritize_supported_evidence,
)
from rag_graph_demo.agentic.models import (
    EstablishedFinding,
    GroundedAnswer,
    KnowledgeNeedState,
    ModelCall,
    NeedItem,
)
from rag_graph_demo.evaluation.models import RetrievedChunk


def test_need_id_accepts_numeric_model_output() -> None:
    need = NeedItem.model_validate(
        {
            "need_id": 1,
            "knowledge_type": "text_evidence",
            "query": "source wording",
            "route": "vector",
        }
    )

    assert need.need_id == "1"


def test_satisfied_need_accepts_zero_remaining_budget() -> None:
    need = NeedItem(
        need_id="done",
        knowledge_type="text_evidence",
        status="satisfied",
        query="completed evidence need",
        route="vector",
        budget=0,
    )

    assert need.budget == 0


def test_need_relations_accept_structured_relation_paths() -> None:
    need = NeedItem.model_validate(
        {
            "need_id": "relation-1",
            "knowledge_type": "relation_evidence",
            "query": "Which companies did TechCrunch report on?",
            "route": "graph",
            "relations": [
                ["TechCrunch", "reports_on", "car companies"],
                "automakers compete_with technology companies",
            ],
        }
    )

    assert need.relations == [
        "TechCrunch -> reports_on -> car companies",
        "automakers compete_with technology companies",
    ]


def test_need_state_accepts_finding_strings_without_treating_them_as_cited() -> None:
    state = KnowledgeNeedState.model_validate(
        {
            "analytic_goal": "goal",
            "established_findings": ["An uncited model summary."],
        }
    )

    assert state.established_findings[0].finding == "An uncited model summary."
    assert state.established_findings[0].evidence_ids == []


class FakeEvoNeedModel:
    async def plan_needs(self, _question, *, total_budget):
        assert total_budget > 0
        return ModelCall(
            KnowledgeNeedState(
                analytic_goal="Explain the relationship change with source support.",
                needs=[
                    NeedItem(
                        need_id="text",
                        knowledge_type="text_evidence",
                        query="Acme exact announcement",
                        route="vector",
                        budget=7,
                    ),
                    NeedItem(
                        need_id="evolution",
                        knowledge_type="evolution_evidence",
                        query="Acme relationship change timeline",
                        route="event",
                        budget=4,
                        missing_slots=["change_event", "state_after"],
                    ),
                ],
            ),
            input_tokens=10,
            output_tokens=5,
        )

    async def assess_needs(
        self,
        _question,
        _need_state,
        _evidence,
        *,
        retrieval_round,
    ):
        if retrieval_round == 1:
            state = KnowledgeNeedState(
                analytic_goal="Explain the relationship change with source support.",
                needs=[
                    NeedItem(
                        need_id="text",
                        knowledge_type="text_evidence",
                        query="Acme exact announcement",
                        route="vector",
                        status="satisfied",
                        supported_evidence_ids=["doc-vector"],
                    ),
                    NeedItem(
                        need_id="evolution",
                        knowledge_type="evolution_evidence",
                        query="Acme relationship change timeline",
                        route="event",
                        status="satisfied",
                        supported_evidence_ids=["doc-event"],
                    ),
                    NeedItem(
                        need_id="relation",
                        knowledge_type="relation_evidence",
                        query="Acme relation to Orbit after the change",
                        route="graph",
                    ),
                ],
            )
        else:
            state = KnowledgeNeedState(
                analytic_goal="Explain the relationship change with source support.",
                sufficient=True,
                needs=[
                    NeedItem(
                        need_id="text",
                        knowledge_type="text_evidence",
                        query="Acme exact announcement",
                        route="vector",
                        status="satisfied",
                        supported_evidence_ids=["doc-vector"],
                    ),
                    NeedItem(
                        need_id="evolution",
                        knowledge_type="evolution_evidence",
                        query="Acme relationship change timeline",
                        route="event",
                        status="satisfied",
                        supported_evidence_ids=["doc-event"],
                    ),
                    NeedItem(
                        need_id="relation",
                        knowledge_type="relation_evidence",
                        query="Acme relation to Orbit after the change",
                        route="graph",
                        status="satisfied",
                        supported_evidence_ids=["doc-graph"],
                    ),
                ],
                established_findings=[
                    EstablishedFinding(
                        finding="The relationship changed after the event.",
                        evidence_ids=["doc-event", "doc-graph"],
                        knowledge_types=[
                            "evolution_evidence",
                            "relation_evidence",
                        ],
                    )
                ],
                stop_reason="All typed needs are supported.",
            )
        return ModelCall(state, input_tokens=20, output_tokens=10)

    async def answer(self, _question, _evidence, _assessment):
        return ModelCall(
            GroundedAnswer(
                answer="The evidence supports the change.",
                citations=["doc-vector", "doc-event", "doc-graph", "unknown"],
            ),
            input_tokens=12,
            output_tokens=6,
        )


@pytest.mark.asyncio
async def test_evoneed_retrieves_only_routes_for_open_typed_needs() -> None:
    calls: list[tuple[str, str, int]] = []

    async def retrieve(
        needs: list[NeedItem], budgets: list[int]
    ) -> list[list[RetrievedChunk]]:
        batches = []
        for need, budget in zip(needs, budgets, strict=True):
            calls.append((need.need_id, need.route, budget))
            batches.append(
                [
                    RetrievedChunk(
                        chunk_id=f"chunk-{need.route}",
                        document_id=f"doc-{need.route}",
                        text=f"{need.route} evidence",
                        metadata={"channels": [need.route]},
                    )
                ]
            )
        return batches

    run = await EvoNeedRAGGraph(retrieve, FakeEvoNeedModel()).run(
        "How did Acme's relationship change?",
        top_k=10,
        context_k=10,
        max_retrieval_rounds=2,
        rrf_k=60,
    )

    assert calls == [
        ("need_question_anchor", "vector", 10),
        ("evolution", "event", 10),
        ("relation", "graph", 10),
    ]
    assert run.retrieval_rounds == 2
    assert run.llm_calls == 4
    assert run.citations == ("doc-vector", "doc-event", "doc-graph")
    assert run.insufficient_evidence is False
    assert {item.document_id for item in run.evidence} == {
        "doc-vector",
        "doc-event",
        "doc-graph",
    }
    assert run.trace[-1]["final_need_state"]["sufficient"] is True


@pytest.mark.asyncio
async def test_planned_evoneed_uses_planner_actions_in_first_round() -> None:
    calls: list[tuple[str, str, str, int]] = []

    async def retrieve(
        needs: list[NeedItem], budgets: list[int]
    ) -> list[list[RetrievedChunk]]:
        batches = []
        for need, budget in zip(needs, budgets, strict=True):
            calls.append((need.need_id, need.route, need.query, budget))
            batches.append(
                [
                    RetrievedChunk(
                        chunk_id=f"chunk-{need.route}",
                        document_id=f"doc-{need.route}",
                        text=f"{need.route} evidence",
                        metadata={"channels": [need.route]},
                    )
                ]
            )
        return batches

    run = await EvoNeedRAGGraph(
        retrieve,
        FakeEvoNeedModel(),
        initial_retrieval="planned_needs",
    ).run(
        "How did Acme's relationship change?",
        top_k=10,
        context_k=10,
        max_retrieval_rounds=1,
        rrf_k=60,
    )

    assert calls == [
        ("text", "vector", "Acme exact announcement", 10),
        ("evolution", "event", "Acme relationship change timeline", 10),
    ]
    assert run.retrieval_rounds == 1
    assert run.llm_calls == 3
    retrieve_trace = next(
        item for item in run.trace if item["node"] == "retrieve_needs"
    )
    assert retrieve_trace["initial_retrieval"] == "planned_needs"
    assert {item.document_id for item in run.evidence} == {
        "doc-vector",
        "doc-event",
    }


@pytest.mark.asyncio
async def test_planned_evoneed_can_share_one_budget_across_actions() -> None:
    calls: list[tuple[str, int]] = []

    async def retrieve(
        needs: list[NeedItem], budgets: list[int]
    ) -> list[list[RetrievedChunk]]:
        batches = []
        for need, budget in zip(needs, budgets, strict=True):
            calls.append((need.route, budget))
            batches.append(
                [
                    RetrievedChunk(
                        chunk_id=f"chunk-{need.route}",
                        document_id=f"doc-{need.route}",
                        text=f"{need.route} evidence",
                    )
                ]
            )
        return batches

    run = await EvoNeedRAGGraph(
        retrieve,
        FakeEvoNeedModel(),
        initial_retrieval="planned_needs",
        route_budget_mode="shared_total",
    ).run(
        "How did Acme's relationship change?",
        top_k=11,
        context_k=11,
        max_retrieval_rounds=1,
        rrf_k=60,
    )

    assert calls == [("vector", 7), ("event", 4)]
    retrieve_trace = next(
        item for item in run.trace if item["node"] == "retrieve_needs"
    )
    assert retrieve_trace["route_budget_mode"] == "shared_total"
    assert retrieve_trace["total_candidate_budget"] == 11


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected"),
    [("equal_total", [30, 30]), ("planned_total", [37, 23])],
)
async def test_controlled_evoneed_uses_one_fixed_total_budget(mode, expected) -> None:
    observed: list[int] = []

    async def retrieve(
        needs: list[NeedItem], budgets: list[int]
    ) -> list[list[RetrievedChunk]]:
        observed.extend(budgets)
        return [
            [
                RetrievedChunk(
                    chunk_id=f"chunk-{need.route}",
                    document_id=f"doc-{need.route}",
                    text=f"{need.route} evidence",
                )
            ]
            for need in needs
        ]

    run = await EvoNeedRAGGraph(
        retrieve,
        FakeEvoNeedModel(),
        initial_retrieval="planned_needs",
        route_budget_mode=mode,
        minimum_route_budget=5,
    ).run(
        "How did Acme's relationship change?",
        top_k=20,
        context_k=20,
        retrieval_budget=60,
        max_retrieval_rounds=1,
        rrf_k=60,
        retrieval_only=True,
    )

    assert observed == expected
    assert sum(observed) == 60
    assert run.llm_calls == 1
    assert [item["node"] for item in run.trace] == [
        "plan_needs",
        "retrieve_needs",
    ]


def test_need_coverage_fusion_prefers_new_need_coverage() -> None:
    repeated = RetrievedChunk(
        chunk_id="repeated",
        document_id="doc-repeated",
        text="repeated",
        metadata={"supported_need_ids": ["need-a"]},
    )
    new_need = RetrievedChunk(
        chunk_id="new",
        document_id="doc-new",
        text="new",
        metadata={"supported_need_ids": ["need-b"]},
    )

    result = need_coverage_fusion(
        [[repeated, new_need], [repeated]],
        rrf_k=60,
        limit=2,
    )

    assert [item.document_id for item in result] == ["doc-repeated", "doc-new"]
    assert result[0].metadata["fusion"] == "evoneed_coverage_rrf"


def test_need_actions_enter_fusion_as_independent_ranked_lists() -> None:
    first_action = [
        RetrievedChunk(
            chunk_id=f"a-{index}",
            document_id=f"doc-a-{index}",
            text="first action",
            metadata={"supported_need_ids": ["need-a"]},
        )
        for index in range(1, 6)
    ]
    second_action_top = RetrievedChunk(
        chunk_id="b-1",
        document_id="doc-b-1",
        text="second action top result",
        metadata={"supported_need_ids": ["need-b"]},
    )

    result = need_coverage_fusion(
        [first_action, [second_action_top]],
        rrf_k=60,
        limit=2,
    )

    assert {item.document_id for item in result} == {"doc-a-1", "doc-b-1"}


def test_repeated_hub_document_does_not_accumulate_unbounded_rrf_score() -> None:
    hub = RetrievedChunk(
        chunk_id="hub",
        document_id="doc-hub",
        text="generic hub",
        metadata={"evoneed_action_id": "need-a"},
    )

    result = need_coverage_fusion(
        [[hub], [hub], [hub]],
        rrf_k=60,
        limit=1,
    )

    assert result[0].score == pytest.approx(1 / 61)


def test_question_anchor_preserves_its_top_four_candidates() -> None:
    anchor = [
        RetrievedChunk(
            chunk_id=f"anchor-{index}",
            document_id=f"doc-anchor-{index}",
            text="lossless question result",
            metadata={"evoneed_action_id": "need_question_anchor"},
        )
        for index in range(1, 5)
    ]
    focused_top = RetrievedChunk(
        chunk_id="focused-1",
        document_id="doc-focused-1",
        text="focused query result",
        metadata={"evoneed_action_id": "need-focused"},
    )

    result = need_coverage_fusion(
        [anchor, [focused_top]],
        rrf_k=60,
        limit=4,
    )

    assert [item.document_id for item in result] == [
        "doc-anchor-1",
        "doc-anchor-2",
        "doc-anchor-3",
        "doc-anchor-4",
    ]


def test_route_candidate_depth_is_not_split_across_actions() -> None:
    assert allocate_route_budgets(top_k=10, action_count=3) == [10, 10, 10]


def test_route_candidate_depth_can_be_shared_across_actions() -> None:
    assert allocate_route_budgets(
        top_k=20,
        action_count=3,
        shared_total=True,
        planned_budgets=[12, 5, 3],
    ) == [12, 5, 3]


def test_invalid_planner_budgets_are_normalized_to_total() -> None:
    budgets = allocate_route_budgets(
        top_k=20,
        action_count=3,
        shared_total=True,
        planned_budgets=[100, 20, 10],
    )
    assert sum(budgets) == 20
    assert budgets[0] > budgets[1] > budgets[2]


def test_temporal_change_question_gets_evolution_guard() -> None:
    initial = KnowledgeNeedState(
        analytic_goal="Compare two reports.",
        needs=[
            NeedItem(
                need_id=str(index),
                knowledge_type="text_evidence",
                query=f"source {index}",
                route="vector",
            )
            for index in range(1, 4)
        ],
    )

    normalized = normalize_need_state(
        initial,
        question="Did the portrayal change between October and December?",
        initial=True,
    )

    assert len(normalized.needs) == 2
    assert normalized.needs[-1].knowledge_type == "evolution_evidence"
    assert normalized.needs[-1].route == "event"
    assert normalized.needs[-1].query.startswith("Did the portrayal change")


def test_repeated_open_query_restores_full_question_context() -> None:
    previous = KnowledgeNeedState(
        analytic_goal="goal",
        needs=[
            NeedItem(
                need_id="source",
                knowledge_type="text_evidence",
                query="TechCrunch report",
                route="vector",
            )
        ],
    )

    normalized = normalize_need_state(
        previous,
        question="What changed between the two TechCrunch reports?",
        previous_state=previous,
    )

    assert normalized.needs[0].query == (
        "TechCrunch report. Full question context: "
        "What changed between the two TechCrunch reports?"
    )


def test_blank_assessment_query_reuses_previous_query() -> None:
    previous = KnowledgeNeedState(
        analytic_goal="goal",
        needs=[
            NeedItem(
                need_id="source",
                knowledge_type="text_evidence",
                query="TechCrunch report",
                route="vector",
            )
        ],
    )
    assessed = KnowledgeNeedState(
        analytic_goal="goal",
        needs=[
            NeedItem(
                need_id="source",
                knowledge_type="text_evidence",
                status="partial",
                query="",
                route="vector",
            )
        ],
    )

    normalized = normalize_need_state(
        assessed,
        question="What changed between the two TechCrunch reports?",
        previous_state=previous,
    )

    assert normalized.needs[0].query.startswith("TechCrunch report")


def test_need_item_accepts_slot_detail_mapping() -> None:
    need = NeedItem(
        need_id="evolution",
        knowledge_type="evolution_evidence",
        query="What changed?",
        route="event",
        missing_slots={"state_before": "old state", "change_event": "event"},
    )

    assert need.missing_slots == ["state_before", "change_event"]


def test_supported_evidence_is_promoted_without_reordering_unsupported_items() -> None:
    evidence = [
        RetrievedChunk(
            chunk_id=item_id,
            document_id=item_id,
            text=item_id,
        )
        for item_id in ("doc-a", "doc-b", "doc-c", "doc-d")
    ]
    state = KnowledgeNeedState(
        analytic_goal="goal",
        needs=[
            NeedItem(
                need_id="need-c",
                knowledge_type="text_evidence",
                query="query c",
                route="vector",
                status="partial",
                supported_evidence_ids=["doc-c"],
            ),
            NeedItem(
                need_id="need-b",
                knowledge_type="relation_evidence",
                query="query b",
                route="graph",
                status="satisfied",
                supported_evidence_ids=["doc-b"],
            ),
        ],
    )

    result = prioritize_supported_evidence(evidence, state)

    assert [item.document_id for item in result] == [
        "doc-b",
        "doc-c",
        "doc-a",
        "doc-d",
    ]

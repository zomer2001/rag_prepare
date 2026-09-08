from __future__ import annotations

import pytest

from rag_graph_demo.domain.tkg import (
    AssertionMention,
    AssertionRecord,
    EntityRecord,
    EventEffect,
    TKGExpansionResponse,
    TKGQueryResponse,
)
from rag_graph_demo.retrieval.evolink import EvoLinkCoordinator
from rag_graph_demo.retrieval.models import EvidenceCandidate, QueryPlan


class FakeCoordinationStore:
    def __init__(self) -> None:
        self.organization = EntityRecord(
            entity_id="entity-acme",
            canonical_name="Acme",
            entity_type="Organization",
            ontology_version="0.1.0",
        )
        self.product = EntityRecord(
            entity_id="entity-orbit",
            canonical_name="Orbit",
            entity_type="Product",
            ontology_version="0.1.0",
        )
        self.assertion = AssertionRecord(
            assertion_id="assertion-launch",
            subject_id=self.organization.entity_id,
            predicate_id="develops",
            object_id=self.product.entity_id,
            status="active",
            confidence=0.9,
            ontology_version="0.1.0",
        )
        self.last_expansion = None

    async def query(self, request):
        assert request.entity_name == "Acme"
        return TKGQueryResponse(
            entities=[self.organization, self.product],
            assertions=[self.assertion],
        )

    async def expand(self, request):
        self.last_expansion = request
        return TKGExpansionResponse(
            entities=[self.organization, self.product],
            assertions=[self.assertion],
            mentions=[
                AssertionMention(
                    mention_id="mention-assertion",
                    assertion_id=self.assertion.assertion_id,
                    document_id="doc-semantic",
                    source_segment_id="chunk-semantic",
                    evidence_text="Acme develops Orbit.",
                )
            ],
            events=[
                {
                    "event_id": "event-launch",
                    "event_type": "launch",
                    "summary": "Acme launched Orbit.",
                    "start_time": "2025",
                    "status": "single_source",
                    "confidence": 0.8,
                }
            ],
            event_mentions=[
                {
                    "mention_id": "mention-event",
                    "event_id": "event-launch",
                    "document_id": "doc-event",
                    "source_segment_id": "chunk-event",
                    "evidence": "Acme launched Orbit in 2025.",
                }
            ],
            event_links=[
                {
                    "event_id": "event-launch",
                    "entity_id": "entity-acme",
                    "role": "actor",
                }
            ],
            effects=[
                EventEffect(
                    effect_id="effect-launch",
                    event_id="event-launch",
                    assertion_id="assertion-launch",
                    effect_type="creates",
                    document_id="doc-event",
                    effective_time="2025",
                )
            ],
        )


@pytest.mark.asyncio
async def test_evolink_connects_event_entity_assertion_and_source_evidence() -> None:
    store = FakeCoordinationStore()
    coordinator = EvoLinkCoordinator(store)
    event_seed = EvidenceCandidate(
        evidence_id="event:seed",
        channel="event",
        rank=1,
        document_id="doc-event",
        chunk_id="chunk-event",
        text="Acme launched Orbit in 2025.",
        metadata={"event_id": "event-launch"},
    )

    result = await coordinator.coordinate(
        QueryPlan(entities=["Acme"], expand_event_chain=True),
        event_candidates=[event_seed],
        limit=10,
    )

    assert store.last_expansion.entity_ids == ["entity-acme", "entity-orbit"]
    assert store.last_expansion.event_ids == ["event-launch"]
    assert store.last_expansion.rank_by_seed_score is True
    assert store.last_expansion.entity_scores == {
        "entity-acme": 1.0,
        "entity-orbit": 0.5,
    }
    assert store.last_expansion.assertion_scores == {"assertion-launch": 1.0}
    assert store.last_expansion.event_scores == {"event-launch": 1.0}
    assert result.graph_candidates[0].document_id == "doc-semantic"
    assert result.event_candidates[0].document_id == "doc-event"
    assert {item["type"] for item in result.evidence_graph["nodes"]} == {
        "entity",
        "assertion",
        "event",
    }
    assert {
        (item["source"], item["target"], item["type"])
        for item in result.evidence_graph["edges"]
    } >= {
        ("entity-acme", "assertion-launch", "SUBJECT"),
        ("assertion-launch", "entity-orbit", "OBJECT"),
        ("event-launch", "entity-acme", "ACTOR"),
        ("event-launch", "assertion-launch", "CREATES"),
    }
    assert result.metadata["linked_effect_count"] == 1
    assert result.metadata["expansion_limit"] == 10


class BroadSeedStore:
    def __init__(self) -> None:
        self.queries = []
        self.last_expansion = None

    async def query(self, request):
        self.queries.append(request)
        return TKGQueryResponse(
            entities=[
                EntityRecord(
                    entity_id=f"{request.entity_name}-{index}",
                    canonical_name=f"{request.entity_name} {index}",
                    entity_type="Concept",
                    ontology_version="0.1.0",
                )
                for index in range(1, 4)
            ]
        )

    async def expand(self, request):
        self.last_expansion = request
        return TKGExpansionResponse()


@pytest.mark.asyncio
async def test_evolink_filters_generic_names_and_round_robins_seed_budget() -> None:
    store = BroadSeedStore()
    coordinator = EvoLinkCoordinator(
        store,
        seed_limit_per_name=3,
        max_entity_names=8,
    )

    result = await coordinator.coordinate(
        QueryPlan(entities=["company", "Acme"]),
        entity_names=["Orbit"],
        limit=4,
    )

    assert [request.entity_name for request in store.queries] == ["Acme", "Orbit"]
    assert all(request.limit == 3 for request in store.queries)
    assert store.last_expansion.entity_ids == [
        "Acme-1",
        "Orbit-1",
        "Acme-2",
        "Orbit-2",
    ]
    assert store.last_expansion.rank_by_seed_score is True
    assert result.metadata["entity_names"] == ["Acme", "Orbit"]

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag_graph_demo.domain.tkg import (
    AssertionMention,
    AssertionRecord,
    EntityRecord,
    TKGExpansionResponse,
)
from rag_graph_demo.retrieval.models import QueryPlan
from rag_graph_demo.retrieval.tkg_retriever import UnifiedTKGRetriever


class FakePlanner:
    async def plan(self, _question: str) -> QueryPlan:
        return QueryPlan(
            entities=["Acme"],
            relations=["develops"],
            event_types=["发布"],
            expand_event_chain=True,
        )


class FakeVectorIndex:
    async def search(self, kind: str, _text: str, *, limit: int):
        del limit
        values = {
            "chunk": [
                {
                    "node_id": "chunk-1",
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "content": "Acme develops Lattice.",
                }
            ],
            "entity": [{"node_id": "ent-acme"}],
            "predicate": [{"node_id": "develops"}],
            "assertion": [{"node_id": "ast-1"}],
            "event": [{"node_id": "evt-1"}],
        }
        return values[kind]


class DynamicVectorIndex:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def search(self, kind: str, _text: str, *, limit: int):
        self.calls.append((kind, limit))
        if kind == "entity":
            return [
                {"node_id": "ent-acme", "score": 0.91},
                {"node_id": "ent-noise", "score": 0.62},
            ]
        if kind == "assertion":
            return [
                {"node_id": "ast-1", "score": 0.86},
                {"node_id": "ast-noise", "score": 0.55},
            ]
        if kind == "chunk":
            return [
                {
                    "node_id": "chunk-1",
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "content": "Acme develops Lattice.",
                    "score": 0.83,
                },
                {
                    "node_id": "chunk-noise",
                    "chunk_id": "chunk-noise",
                    "document_id": "doc-noise",
                    "content": "Noise.",
                    "score": 0.41,
                },
            ]
        raise AssertionError(f"disabled dynamic route was searched: {kind}")


class FakeGraphStore:
    def __init__(self) -> None:
        self.request = None

    async def expand(self, request):
        self.request = request
        return TKGExpansionResponse(
            entities=[
                EntityRecord(
                    entity_id="ent-acme",
                    canonical_name="Acme",
                    entity_type="Organization",
                    ontology_version="0.1.0",
                ),
                EntityRecord(
                    entity_id="ent-model",
                    canonical_name="Lattice",
                    entity_type="Model",
                    ontology_version="0.1.0",
                ),
            ],
            assertions=[
                AssertionRecord(
                    assertion_id="ast-1",
                    subject_id="ent-acme",
                    predicate_id="develops",
                    object_id="ent-model",
                    status="active",
                    ontology_version="0.1.0",
                )
            ],
            mentions=[
                AssertionMention(
                    mention_id="mention-1",
                    assertion_id="ast-1",
                    document_id="doc-1",
                    evidence_text="Acme develops Lattice.",
                    recorded_at=datetime.now(UTC),
                )
            ],
        )


@pytest.mark.asyncio
async def test_tkg_retriever_uses_all_seed_types_and_returns_source_evidence() -> None:
    store = FakeGraphStore()
    retriever = UnifiedTKGRetriever(FakePlanner(), FakeVectorIndex(), store)

    result = await retriever.retrieve("What does Acme develop?", 4)

    assert store.request.entity_ids == ["ent-acme"]
    assert store.request.predicate_ids == ["develops"]
    assert store.request.assertion_ids == ["ast-1"]
    assert store.request.event_ids == ["evt-1"]
    assert result[0].document_id == "doc-1"
    assert result[0].metadata["channels"] == ["graph", "vector"]
    assert result[0].metadata["fusion"] == "unified_tkg_rrf"


@pytest.mark.asyncio
async def test_dynamic_tkg_retriever_selects_routes_and_prunes_weak_seeds() -> None:
    class FactPlanner:
        async def plan(self, _question: str) -> QueryPlan:
            return QueryPlan(entities=["Acme"])

    vector = DynamicVectorIndex()
    store = FakeGraphStore()
    retriever = UnifiedTKGRetriever(FactPlanner(), vector, store, dynamic=True)

    result = await retriever.retrieve("What is known about Acme?", 4)

    assert {kind for kind, _ in vector.calls} == {"chunk", "entity", "assertion"}
    assert store.request.entity_ids == ["ent-acme"]
    assert store.request.assertion_ids == ["ast-1"]
    assert store.request.predicate_ids == []
    assert store.request.event_ids == []
    assert store.request.rank_by_seed_score is True
    assert result[0].metadata["fusion"] == "dynamic_tkg_coverage"
    assert result[0].metadata["tkg_dynamic_plan"]["enabled"]["event"] is False

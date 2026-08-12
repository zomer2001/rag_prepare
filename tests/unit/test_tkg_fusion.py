import json

import pytest

from rag_graph_demo.domain.events import EventFactCandidate, EventRecord
from rag_graph_demo.domain.tkg import (
    AssertionStatus,
    EventEffectType,
    MentionStance,
    SemanticEntityCandidate,
    SemanticExtractionBatch,
    SemanticRelationshipCandidate,
    TKGQueryRequest,
)
from rag_graph_demo.services.tkg_fusion_service import TKGFusionService
from rag_graph_demo.storage.tkg_registry import TKGRegistry


def build_service(tmp_path):
    registry = TKGRegistry(tmp_path / "tkg.json", ontology_version="0.1.0")
    return registry, TKGFusionService(registry, ontology_version="0.1.0")


def semantic_batch(description: str = "Acme develops Lattice."):
    return SemanticExtractionBatch(
        entities=[
            SemanticEntityCandidate(
                name="Acme",
                entity_type="Organization",
                description="Acme is a company.",
                aliases=["Acme Corp"],
            ),
            SemanticEntityCandidate(
                name="Lattice",
                entity_type="Model",
                description="Lattice is a model.",
            ),
        ],
        relationships=[
            SemanticRelationshipCandidate(
                source="Acme",
                target="Lattice",
                predicate="development, production",
                description=description,
                confidence=0.8,
            )
        ],
    )


@pytest.mark.asyncio
async def test_semantic_relation_becomes_source_aware_assertion(tmp_path) -> None:
    registry, service = build_service(tmp_path)

    result = await service.index_document("doc-1", "Report", semantic_batch(), [])
    snapshot = await registry.snapshot()

    assert result.entity_count == 2
    assert result.assertion_count == 1
    assert len(snapshot.entities) == 2
    assert snapshot.assertions[0].predicate_id == "develops"
    assert snapshot.assertions[0].status == AssertionStatus.ACTIVE
    assert snapshot.assertion_mentions[0].document_id == "doc-1"
    assert snapshot.assertion_mentions[0].evidence_text == "Acme develops Lattice."


@pytest.mark.asyncio
async def test_alias_and_event_roles_share_one_entity_identity(tmp_path) -> None:
    registry, service = build_service(tmp_path)
    event = EventRecord(
        event_id="evt-1",
        canonical_key="release",
        event_type="发布",
        summary="Acme Corp released Lattice.",
        actors=["Acme Corp"],
        objects=["Lattice"],
    )

    await service.index_document("doc-1", "Report", semantic_batch(), [event])
    snapshot = await registry.snapshot()
    acme = next(item for item in snapshot.entities if item.canonical_name == "Acme")

    assert next(
        item for item in snapshot.event_entity_links if item.surface_form == "Acme Corp"
    ).entity_id == acme.entity_id


@pytest.mark.asyncio
async def test_event_terminates_assertion_and_time_query_honors_validity(tmp_path) -> None:
    registry, service = build_service(tmp_path)
    await service.index_document("doc-1", "Initial", semantic_batch(), [])
    event = EventRecord(
        event_id="evt-stop",
        canonical_key="stop",
        event_type="终止",
        summary="Acme stopped developing Lattice in 2026.",
        actors=["Acme"],
        objects=["Lattice"],
        start_time="2026-03-01",
        fact_effects=[
            EventFactCandidate(
                subject="Acme",
                predicate="develops",
                object="Lattice",
                effect_type=EventEffectType.TERMINATES,
                reason="The programme was formally terminated.",
            )
        ],
    )
    await service.index_document("doc-2", "Termination", SemanticExtractionBatch(), [event])

    current = await registry.query(TKGQueryRequest(entity_name="Acme"))
    historical = await registry.query(
        TKGQueryRequest(entity_name="Acme", at_time="2025-12-31", include_inactive=True)
    )
    snapshot = await registry.snapshot()

    assert current.assertions == []
    assert historical.assertions[0].valid_to == "2026-03-01"
    assert historical.assertions[0].status == AssertionStatus.INACTIVE
    assert snapshot.event_effects[0].effect_type == EventEffectType.TERMINATES


@pytest.mark.asyncio
async def test_denial_preserves_both_mentions_and_opens_conflict(tmp_path) -> None:
    registry, service = build_service(tmp_path)
    await service.index_document("doc-1", "Claim", semantic_batch(), [])
    denial = EventRecord(
        event_id="evt-deny",
        canonical_key="deny",
        event_type="否认",
        summary="Acme denied developing Lattice.",
        actors=["Acme"],
        objects=["Lattice"],
        fact_effects=[
            EventFactCandidate(
                subject="Acme",
                predicate="develops",
                object="Lattice",
                effect_type=EventEffectType.DENIES,
                reason="Acme issued a denial.",
            )
        ],
    )
    await service.index_document("doc-2", "Denial", SemanticExtractionBatch(), [denial])
    snapshot = await registry.snapshot()

    assert snapshot.assertions[0].status == AssertionStatus.DISPUTED
    assert {item.stance for item in snapshot.assertion_mentions} == {
        MentionStance.ASSERTED,
        MentionStance.DENIED,
    }
    assert len(snapshot.conflicts) == 1

    await service.delete_document("doc-2")
    resolved = await registry.snapshot()
    assert resolved.assertions[0].status == AssertionStatus.ACTIVE
    assert resolved.assertions[0].valid_to is None
    assert resolved.conflicts == []


@pytest.mark.asyncio
async def test_reindex_is_idempotent_for_document_provenance(tmp_path) -> None:
    registry, service = build_service(tmp_path)
    await service.index_document("doc-1", "Report", semantic_batch(), [])
    await service.index_document("doc-1", "Report", semantic_batch(), [])
    snapshot = await registry.snapshot()

    assert len(snapshot.entities) == 2
    assert len(snapshot.entity_mentions) == 2
    assert len(snapshot.assertions) == 1
    assert len(snapshot.assertion_mentions) == 1


@pytest.mark.asyncio
async def test_uncertain_event_does_not_mutate_assertion_lifecycle(tmp_path) -> None:
    registry, service = build_service(tmp_path)
    await service.index_document("doc-1", "Claim", semantic_batch(), [])
    uncertain = EventRecord(
        event_id="evt-rumour",
        canonical_key="rumour",
        event_type="传闻终止",
        summary="A report speculated that Acme might stop developing Lattice.",
        stance="uncertain",
        fact_effects=[
            EventFactCandidate(
                subject="Acme",
                predicate="develops",
                object="Lattice",
                effect_type=EventEffectType.TERMINATES,
            )
        ],
    )

    await service.index_document(
        "doc-2", "Rumour", SemanticExtractionBatch(), [uncertain]
    )
    snapshot = await registry.snapshot()

    assert snapshot.assertions[0].status == AssertionStatus.ACTIVE
    assert snapshot.assertions[0].valid_to is None
    assert snapshot.event_effects[0].stance == MentionStance.UNCERTAIN


@pytest.mark.asyncio
async def test_registry_refuses_implicit_ontology_version_upgrade(tmp_path) -> None:
    path = tmp_path / "tkg.json"
    path.write_text(
        json.dumps({"schema_version": "1", "ontology_version": "0.0.9"}),
        encoding="utf-8",
    )
    registry = TKGRegistry(path, ontology_version="0.1.0")

    with pytest.raises(ValueError, match="explicit ontology migration"):
        await registry.snapshot()


@pytest.mark.asyncio
async def test_update_event_supersedes_old_value_and_activates_new_value(tmp_path) -> None:
    registry, service = build_service(tmp_path)
    await service.index_document("doc-1", "Old", semantic_batch(), [])
    update = EventRecord(
        event_id="evt-update",
        canonical_key="update",
        event_type="变更",
        summary="Acme replaced Lattice with Lattice 2.",
        start_time="2026-04-01",
        fact_effects=[
            EventFactCandidate(
                subject="Acme",
                predicate="develops",
                object="Lattice 2",
                effect_type=EventEffectType.UPDATES,
            )
        ],
    )

    await service.index_document("doc-2", "Update", SemanticExtractionBatch(), [update])
    snapshot = await registry.snapshot()
    entities = {item.entity_id: item for item in snapshot.entities}
    facts = {
        entities[item.object_id].canonical_name: item for item in snapshot.assertions
    }

    assert facts["Lattice"].status == AssertionStatus.SUPERSEDED
    assert facts["Lattice"].valid_to == "2026-04-01"
    assert facts["Lattice 2"].status == AssertionStatus.ACTIVE
    assert {item.effect_type for item in snapshot.event_effects} == {
        EventEffectType.UPDATES,
        EventEffectType.SUPERSEDES,
    }


@pytest.mark.asyncio
async def test_recreated_fact_starts_new_assertion_after_termination(tmp_path) -> None:
    registry, service = build_service(tmp_path)
    await service.index_document("doc-1", "Initial", semantic_batch(), [])
    for document_id, event_id, event_type, effect_type, event_time in (
        ("doc-2", "evt-stop", "终止", EventEffectType.TERMINATES, "2026-03-01"),
        ("doc-3", "evt-restart", "重启", EventEffectType.CREATES, "2027-01-01"),
    ):
        event = EventRecord(
            event_id=event_id,
            canonical_key=event_id,
            event_type=event_type,
            summary=f"Acme {event_type} Lattice.",
            start_time=event_time,
            fact_effects=[
                EventFactCandidate(
                    subject="Acme",
                    predicate="develops",
                    object="Lattice",
                    effect_type=effect_type,
                )
            ],
        )
        await service.index_document(
            document_id, event_type, SemanticExtractionBatch(), [event]
        )

    snapshot = await registry.snapshot()
    facts = sorted(snapshot.assertions, key=lambda item: item.valid_from or "")

    assert len(facts) == 2
    assert facts[0].status == AssertionStatus.INACTIVE
    assert facts[0].valid_to == "2026-03-01"
    assert facts[1].status == AssertionStatus.ACTIVE
    assert facts[1].valid_from == "2027-01-01"

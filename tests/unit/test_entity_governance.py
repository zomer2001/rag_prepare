import pytest

from rag_graph_demo.domain.tkg import (
    AssertionMention,
    AssertionRecord,
    EntityCandidateScore,
    EntityMention,
    EntityRecord,
    EntityReviewCase,
    EventEntityLink,
    GovernanceStatus,
    ResolutionAction,
    TKGSnapshot,
)
from rag_graph_demo.services.entity_governance_service import EntityGovernanceService
from rag_graph_demo.storage.tkg_registry import TKGRegistry


@pytest.mark.asyncio
async def test_review_merge_moves_all_references_and_can_rollback(tmp_path) -> None:
    registry = TKGRegistry(tmp_path / "tkg.json", ontology_version="0.1.0")

    def seed(snapshot: TKGSnapshot):
        source = EntityRecord(
            entity_id="source",
            canonical_name="Acme Inc",
            entity_type="Organization",
            ontology_version="0.1.0",
        )
        target = EntityRecord(
            entity_id="target",
            canonical_name="Acme",
            entity_type="Organization",
            ontology_version="0.1.0",
        )
        obj = EntityRecord(
            entity_id="object",
            canonical_name="Lattice",
            entity_type="Model",
            ontology_version="0.1.0",
        )
        snapshot.entities.extend([source, target, obj])
        snapshot.entity_mentions.append(
            EntityMention(
                mention_id="mention-source",
                entity_id="source",
                document_id="doc-1",
                surface_form="Acme Inc",
                evidence_text="Acme Inc develops Lattice.",
            )
        )
        snapshot.assertions.append(
            AssertionRecord(
                assertion_id="assertion-1",
                subject_id="source",
                predicate_id="develops",
                object_id="object",
                ontology_version="0.1.0",
            )
        )
        snapshot.assertions.append(
            AssertionRecord(
                assertion_id="assertion-existing",
                subject_id="target",
                predicate_id="develops",
                object_id="object",
                ontology_version="0.1.0",
            )
        )
        snapshot.assertion_mentions.append(
            AssertionMention(
                mention_id="assertion-mention",
                assertion_id="assertion-1",
                document_id="doc-1",
                evidence_text="Acme Inc develops Lattice.",
            )
        )
        snapshot.assertion_mentions.append(
            AssertionMention(
                mention_id="assertion-mention-existing",
                assertion_id="assertion-existing",
                document_id="doc-2",
                evidence_text="Acme develops Lattice.",
            )
        )
        snapshot.event_entity_links.append(
            EventEntityLink(
                link_id="link-1",
                event_id="event-1",
                entity_id="source",
                role="actor",
                surface_form="Acme Inc",
                document_id="doc-1",
            )
        )
        snapshot.review_cases.append(
            EntityReviewCase(
                review_id="review-1",
                document_id="doc-1",
                provisional_entity_id="source",
                candidate_name="Acme Inc",
                candidate_type="Organization",
                candidates=[
                    EntityCandidateScore(
                        entity_id="target",
                        canonical_name="Acme",
                        entity_type="Organization",
                        lexical_score=0.9,
                    )
                ],
                reason="needs review",
            )
        )

    await registry.mutate(seed)
    service = EntityGovernanceService(registry)

    review = await service.resolve_review(
        "review-1",
        action=ResolutionAction.MATCH_EXISTING,
        selected_entity_id="target",
        reviewer="tester",
        reason="same company",
    )
    merged = await registry.snapshot()
    source = next(item for item in merged.entities if item.entity_id == "source")

    assert review.status == GovernanceStatus.APPROVED
    assert source.status == "merged"
    assert merged.entity_mentions[0].entity_id == "target"
    assert len(merged.assertions) == 1
    assert merged.assertions[0].subject_id == "target"
    assert {item.assertion_id for item in merged.assertion_mentions} == {
        merged.assertions[0].assertion_id
    }
    assert merged.event_entity_links[0].entity_id == "target"

    merge = merged.merge_records[0]
    await service.rollback_merge(merge.merge_id, actor="tester")
    restored = await registry.snapshot()

    assert next(item for item in restored.entities if item.entity_id == "source").status == "active"
    assert restored.entity_mentions[0].entity_id == "source"
    assert len(restored.assertions) == 2
    assert next(
        item for item in restored.assertions if item.assertion_id == "assertion-1"
    ).subject_id == "source"
    assert next(
        item
        for item in restored.assertion_mentions
        if item.mention_id == "assertion-mention"
    ).assertion_id == "assertion-1"
    assert restored.event_entity_links[0].entity_id == "source"
    assert restored.merge_records[0].status == GovernanceStatus.ROLLED_BACK

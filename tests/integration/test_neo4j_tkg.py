import os

import pytest

from rag_graph_demo.domain.tkg import (
    AssertionMention,
    AssertionRecord,
    EntityMention,
    EntityRecord,
    TKGExpansionRequest,
    TKGQueryRequest,
    TKGSnapshot,
)
from rag_graph_demo.storage.neo4j_tkg_store import Neo4jTKGStore

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_NEO4J_INTEGRATION") != "1",
    reason="set RUN_NEO4J_INTEGRATION=1 with the project Neo4j container running",
)


@pytest.mark.asyncio
async def test_real_neo4j_snapshot_query_and_graph() -> None:
    store = Neo4jTKGStore(
        os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
        os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASSWORD", "rag-graph-demo-change-me"),
        workspace_id="integration_test",
    )
    try:
        await store.initialize_schema()
        organization = EntityRecord(
            entity_id="integration-org",
            canonical_name="Integration Acme",
            entity_type="Organization",
            ontology_version="0.1.0",
        )
        model = EntityRecord(
            entity_id="integration-model",
            canonical_name="Integration Lattice",
            entity_type="Model",
            ontology_version="0.1.0",
        )
        assertion = AssertionRecord(
            assertion_id="integration-assertion",
            subject_id=organization.entity_id,
            predicate_id="develops",
            object_id=model.entity_id,
            status="active",
            ontology_version="0.1.0",
        )
        await store.replace_snapshot(
            TKGSnapshot(
                ontology_version="0.1.0",
                entities=[organization, model],
                entity_mentions=[
                    EntityMention(
                        mention_id="integration-entity-mention",
                        entity_id=organization.entity_id,
                        document_id="integration-document",
                        surface_form="Integration Acme",
                        evidence_text="Integration Acme develops Integration Lattice.",
                    )
                ],
                assertions=[assertion],
                assertion_mentions=[
                    AssertionMention(
                        mention_id="integration-assertion-mention",
                        assertion_id=assertion.assertion_id,
                        document_id="integration-document",
                        evidence_text="Integration Acme develops Integration Lattice.",
                    )
                ],
            )
        )

        result = await store.query(
            TKGQueryRequest(entity_name="Integration Acme")
        )
        graph = await store.query_graph(entity_name="Integration Acme")
        expansion = await store.expand(
            TKGExpansionRequest(
                entity_ids=[organization.entity_id],
                entity_scores={organization.entity_id: 0.92},
                assertion_scores={assertion.assertion_id: 0.88},
                rank_by_seed_score=True,
                limit=10,
            )
        )

        assert result.assertions[0].assertion_id == "integration-assertion"
        assert {item.canonical_name for item in result.entities} == {
            "Integration Acme",
            "Integration Lattice",
        }
        assert len(graph["nodes"]) == 3
        assert len(graph["edges"]) == 2
        assert expansion.assertions[0].assertion_id == "integration-assertion"
        assert expansion.mentions[0].document_id == "integration-document"
    finally:
        await store.close()

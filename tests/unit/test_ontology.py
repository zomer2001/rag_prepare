from pathlib import Path

import pytest
from rdflib import RDF, Graph, Literal, Namespace, URIRef

from rag_graph_demo.domain.tkg import (
    EntityMention,
    EntityRecord,
    TKGSnapshot,
)
from rag_graph_demo.services.ontology_service import (
    OntologyService,
    OntologyValidationError,
)

TKG = Namespace("https://zomer2001.github.io/rag_prepare/ontology/tkg#")


def ontology_dir() -> Path:
    return Path(__file__).parents[2] / "ontology"


def test_ontology_bundle_parses_and_accepts_valid_entity() -> None:
    service = OntologyService(ontology_dir())
    schema = service.load_schema()
    assert len(schema) > 50

    entity = URIRef("urn:test:entity:1")
    data = Graph()
    data.add((entity, RDF.type, TKG.Entity))
    data.add((entity, TKG.canonicalName, Literal("LightRAG")))
    data.add((entity, TKG.ontologyVersion, Literal("0.1.0")))

    service.validate_graph(data)


def test_ontology_rejects_entity_without_canonical_name() -> None:
    entity = URIRef("urn:test:entity:missing-name")
    data = Graph()
    data.add((entity, RDF.type, TKG.Entity))
    data.add((entity, TKG.ontologyVersion, Literal("0.1.0")))

    with pytest.raises(OntologyValidationError, match="canonicalName"):
        OntologyService(ontology_dir()).validate_graph(data)


def test_persisted_snapshot_is_validated_before_write() -> None:
    service = OntologyService(ontology_dir())
    entity = EntityRecord(
        entity_id="ent-1",
        canonical_name="LightRAG",
        entity_type="Model",
        ontology_version="0.1.0",
    )
    snapshot = TKGSnapshot(
        ontology_version="0.1.0",
        entities=[entity],
        entity_mentions=[
            EntityMention(
                mention_id="mention-1",
                entity_id="ent-1",
                document_id="doc-1",
                surface_form="LightRAG",
                evidence_text="LightRAG is evaluated in the report.",
            )
        ],
    )

    service.validate_snapshot(snapshot)

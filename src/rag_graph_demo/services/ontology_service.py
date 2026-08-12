from __future__ import annotations

from pathlib import Path

from pyshacl import validate
from rdflib import RDF, Graph, Literal, Namespace, URIRef

from rag_graph_demo.domain.tkg import TKGSnapshot

TKG = Namespace("https://zomer2001.github.io/rag_prepare/ontology/tkg#")


class OntologyValidationError(ValueError):
    pass


class OntologyService:
    """Offline parser and SHACL validator for the versioned ontology bundle."""

    def __init__(self, ontology_dir: Path) -> None:
        self._ontology_dir = ontology_dir

    def load_schema(self) -> Graph:
        graph = Graph()
        for relative in ("core/tkg-core.ttl", "domain/research-technology.ttl"):
            graph.parse(self._ontology_dir / relative, format="turtle")
        return graph

    def validate_graph(self, data_graph: Graph) -> None:
        shapes = Graph().parse(
            self._ontology_dir / "shapes" / "tkg-shapes.ttl", format="turtle"
        )
        conforms, _, report = validate(
            data_graph,
            shacl_graph=shapes,
            ont_graph=self.load_schema(),
            inference="rdfs",
            abort_on_first=False,
        )
        if not conforms:
            raise OntologyValidationError(str(report))

    def validate_snapshot(self, snapshot: TKGSnapshot) -> None:
        """Validate the persisted contract, not the LLM candidate payload."""
        graph = Graph()
        documents: set[str] = set()

        for entity in snapshot.entities:
            node = URIRef(f"urn:tkg:{entity.entity_id}")
            graph.add((node, RDF.type, TKG.Entity))
            graph.add((node, TKG.canonicalName, Literal(entity.canonical_name)))
            graph.add((node, TKG.ontologyVersion, Literal(entity.ontology_version)))
        for assertion in snapshot.assertions:
            node = URIRef(f"urn:tkg:{assertion.assertion_id}")
            graph.add((node, RDF.type, TKG.Assertion))
            graph.add((node, TKG.subject, URIRef(f"urn:tkg:{assertion.subject_id}")))
            graph.add((node, TKG.predicate, URIRef(f"urn:tkg:predicate:{assertion.predicate_id}")))
            graph.add((node, TKG.object, URIRef(f"urn:tkg:{assertion.object_id}")))
            graph.add((node, TKG.status, Literal(assertion.status.value)))
            graph.add((node, TKG.ontologyVersion, Literal(assertion.ontology_version)))
        for mention in snapshot.entity_mentions:
            node = URIRef(f"urn:tkg:{mention.mention_id}")
            document = URIRef(f"urn:tkg:document:{mention.document_id}")
            documents.add(mention.document_id)
            graph.add((node, RDF.type, TKG.EntityMention))
            graph.add((node, TKG.mentionsEntity, URIRef(f"urn:tkg:{mention.entity_id}")))
            graph.add((node, TKG.fromDocument, document))
            graph.add((node, TKG.evidenceText, Literal(mention.evidence_text)))
        for mention in snapshot.assertion_mentions:
            node = URIRef(f"urn:tkg:{mention.mention_id}")
            document = URIRef(f"urn:tkg:document:{mention.document_id}")
            documents.add(mention.document_id)
            graph.add((node, RDF.type, TKG.AssertionMention))
            graph.add((node, TKG.supportsAssertion, URIRef(f"urn:tkg:{mention.assertion_id}")))
            graph.add((node, TKG.fromDocument, document))
            graph.add((node, TKG.evidenceText, Literal(mention.evidence_text)))
            graph.add((node, TKG.stance, Literal(mention.stance.value)))
        for effect in snapshot.event_effects:
            node = URIRef(f"urn:tkg:{effect.effect_id}")
            graph.add((node, RDF.type, TKG.EventEffect))
            graph.add((node, TKG.affectsAssertion, URIRef(f"urn:tkg:{effect.assertion_id}")))
            graph.add((node, TKG.effectType, Literal(effect.effect_type.value)))
            graph.add((node, TKG.stance, Literal(effect.stance.value)))
        for document_id in documents:
            graph.add(
                (URIRef(f"urn:tkg:document:{document_id}"), RDF.type, TKG.SourceDocument)
            )
        self.validate_graph(graph)

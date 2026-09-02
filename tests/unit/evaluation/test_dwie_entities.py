from rag_graph_demo.domain.tkg import (
    EntityRecord,
    SemanticEntityCandidate,
    SemanticExtractionBatch,
    TKGSnapshot,
)
from rag_graph_demo.evaluation.dwie_entities import (
    coarsen_gold,
    exact_alias_prediction,
    exact_match_prediction,
    gold_alias_candidates,
    has_plausible_candidate,
    semantic_prediction,
    singleton_prediction,
)


def _gold_document() -> dict:
    content = "International Business Machines (IBM) met Alice in Paris. IBM agreed."
    return {
        "id": "d1",
        "content": content,
        "tags": ["test"],
        "concepts": [
            {
                "concept": 0,
                "text": "International Business Machines",
                "tags": ["type::entity", "type::organization"],
            },
            {
                "concept": 1,
                "text": "Alice",
                "tags": ["type::entity", "type::person"],
            },
            {
                "concept": 2,
                "text": "agreement",
                "tags": ["type::event"],
            },
            {
                "concept": 3,
                "text": "Paris",
                "tags": ["type::entity", "type::location"],
            },
        ],
        "mentions": [
            {"begin": 0, "end": 31, "text": content[0:31], "concept": 0},
            {"begin": 33, "end": 36, "text": "IBM", "concept": 0},
            {"begin": 42, "end": 47, "text": "Alice", "concept": 1},
            {"begin": 51, "end": 56, "text": "Paris", "concept": 3},
            {"begin": 58, "end": 61, "text": "IBM", "concept": 0},
        ],
        "relations": [],
    }


def test_coarsen_gold_keeps_only_shared_types_and_remaps_concepts() -> None:
    scoped = coarsen_gold(_gold_document())

    assert [item["tags"] for item in scoped["concepts"]] == [
        ["Organization"],
        ["Person"],
        ["Location"],
    ]
    assert {item["concept"] for item in scoped["mentions"]} == {0, 1, 2}


def test_exact_alias_groups_acronym_with_full_name() -> None:
    prediction = exact_alias_prediction(_gold_document())
    organization_mentions = [
        item for item in prediction["mentions"] if item["tags"] == ["Organization"]
    ]

    assert len({item["concept"] for item in organization_mentions}) == 1


def test_singleton_and_exact_match_form_expected_gold_clusters() -> None:
    singleton = singleton_prediction(_gold_document())
    exact = exact_match_prediction(_gold_document())

    assert len(singleton["concepts"]) == len(singleton["mentions"])
    ibm_mentions = [item for item in exact["mentions"] if item["text"] == "IBM"]
    assert len({item["concept"] for item in ibm_mentions}) == 1
    full_name = next(
        item
        for item in exact["mentions"]
        if item["text"] == "International Business Machines"
    )
    assert full_name["concept"] != ibm_mentions[0]["concept"]


def test_gold_resolution_preclusters_aliases_before_remote_candidates() -> None:
    candidates, mention_clusters = gold_alias_candidates(_gold_document())

    assert len(candidates) == 3
    organization = next(item for item in candidates if item.entity_type == "Organization")
    assert organization.name == "International Business Machines"
    assert organization.aliases == ["IBM"]
    assert mention_clusters[0] == mention_clusters[1] == mention_clusters[4]


def test_lexical_gate_skips_unrelated_same_type_names() -> None:
    snapshot = TKGSnapshot(
        ontology_version="test",
        entities=[
            EntityRecord(
                entity_id="alice",
                canonical_name="Alice Johnson",
                entity_type="Person",
                ontology_version="test",
            )
        ],
    )

    assert has_plausible_candidate(
        snapshot,
        SemanticEntityCandidate(name="Alice", entity_type="Person", description=""),
    )
    assert not has_plausible_candidate(
        snapshot,
        SemanticEntityCandidate(name="Robert Smith", entity_type="Person", description=""),
    )


def test_semantic_prediction_aligns_literal_mentions_and_resolved_ids() -> None:
    document = _gold_document()
    semantic = SemanticExtractionBatch(
        entities=[
            SemanticEntityCandidate(
                name="IBM",
                entity_type="Organization",
                description="Company",
            ),
            SemanticEntityCandidate(
                name="Alice",
                entity_type="Person",
                description="Person",
            ),
            SemanticEntityCandidate(
                name="agreement",
                entity_type="Event",
                description="Event",
            ),
        ]
    )
    prediction = semantic_prediction(
        document,
        semantic,
        resolved_entity_ids={"ibm|organization": "entity-ibm"},
    )

    assert [(item["begin"], item["end"]) for item in prediction["mentions"]] == [
        (33, 36),
        (42, 47),
        (58, 61),
    ]
    ibm_mentions = [item for item in prediction["mentions"] if item["text"] == "IBM"]
    assert len({item["concept"] for item in ibm_mentions}) == 1

from __future__ import annotations

import json

from rag_graph_demo.evaluation.answering import (
    AnswerContextBuilder,
    EvidenceHydrator,
    TokenCounter,
    answer_exact_match,
    answer_token_f1,
)
from rag_graph_demo.evaluation.native_context import (
    NativeContextRegistry,
    NativeGraphEdge,
    NativeGraphNode,
    SourceAlignedNativeGraph,
)


def test_short_answer_metrics_use_standard_normalization() -> None:
    assert answer_exact_match("The OpenAI.", "OpenAI") == 1.0
    assert answer_exact_match("No", "Yes") == 0.0
    assert answer_token_f1("OpenAI organization", "OpenAI") == 2 / 3


def test_context_builder_hydrates_and_filters_structured_evidence(tmp_path) -> None:
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            [
                {
                    "chunk_id": "chunk-a",
                    "document_id": "doc-a",
                    "text": "Title: A\n" + "full source evidence " * 20,
                }
            ]
        ),
        encoding="utf-8",
    )
    light_path = tmp_path / "light.json"
    light_path.write_text(
        json.dumps(
            {
                "chunk-a-chunk-000": {
                    "content": "Title: A\n" + "full source evidence " * 20
                }
            }
        ),
        encoding="utf-8",
    )
    event_path = tmp_path / "events.json"
    event_path.write_text(
        json.dumps(
            {
                "mentions": {
                    "m1": {
                        "event_id": "evt-a",
                        "document_id": "doc-a",
                        "source_segment_id": "doc-a::event-part-1",
                        "evidence": "The policy changed in 2024.",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    graph = {
        "nodes": [
            {"id": "person", "type": "entity", "label": "Company", "entity_type": "Organization"},
            {"id": "object", "type": "entity", "label": "Policy", "entity_type": "Concept"},
            {
                "id": "assertion",
                "type": "assertion",
                "label": "uses policy",
                "predicate": "uses",
                "status": "active",
                "source_document_ids": ["doc-a"],
            },
            {
                "id": "event",
                "type": "event",
                "label": "Company changed the policy",
                "event_type": "policy change",
                "start_time": "2024",
                "source_document_ids": ["doc-a"],
            },
            {
                "id": "unrelated",
                "type": "assertion",
                "label": "unrelated fact",
                "source_document_ids": ["doc-b"],
            },
        ],
        "edges": [
            {"source": "person", "target": "assertion", "type": "SUBJECT"},
            {"source": "assertion", "target": "object", "type": "OBJECT"},
            {"source": "event", "target": "person", "type": "ACTOR"},
            {
                "source": "event",
                "target": "assertion",
                "type": "UPDATES",
                "effective_time": "2024",
                "document_id": "doc-a",
            },
        ],
    }
    row = {
        "retrieved": [
            {
                "chunk_id": "chunk-a-chunk-000",
                "document_id": "doc-a",
                "text": "truncated",
                "metadata": {
                    "passages": [
                        {"channel": "vector", "chunk_id": "chunk-a-chunk-000"},
                        {
                            "channel": "event",
                            "chunk_id": "doc-a::event-part-1",
                            "metadata": {"event_id": "evt-a"},
                        },
                    ],
                    "evidence_graph": graph,
                },
            }
        ]
    }
    hydrator = EvidenceHydrator(
        external_corpus_path=corpus_path,
        light_chunk_store_path=light_path,
        event_store_path=event_path,
    )
    hydrated = hydrator.hydrate_row(row, top_k=20)
    assert "full source evidence" in hydrated[0].text
    assert "The policy changed in 2024." in hydrated[0].text

    builder = AnswerContextBuilder(
        hydrator,
        token_counter=TokenCounter("unicode_chars_v1"),
        max_context_tokens=1_000,
        max_evidence_tokens=100,
        structured_token_share=0.7,
    )
    graph_context = builder.build(row, variant="graph", top_k=20)
    evolution_context = builder.build(row, variant="evolution", top_k=20)
    assert "Relations and assertions" in graph_context.text
    assert "Evolution effects" not in graph_context.text
    assert "UPDATES" in evolution_context.text
    assert "unrelated fact" not in evolution_context.text
    assert evolution_context.token_count <= 1_000


def test_native_context_uses_only_sources_from_retrieved_evidence(tmp_path) -> None:
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            [
                {
                    "chunk_id": "chunk-a",
                    "document_id": "doc-a",
                    "text": "Evidence A",
                }
            ]
        ),
        encoding="utf-8",
    )
    view = SourceAlignedNativeGraph(
        label="Native graph",
        nodes=[
            NativeGraphNode("a", "Company A", "entity", "selected", ("chunk-a",)),
            NativeGraphNode("b", "Policy B", "entity", "selected", ("chunk-a",)),
            NativeGraphNode("x", "Unrelated X", "entity", "unrelated", ("chunk-x",)),
        ],
        edges=[
            NativeGraphEdge("a", "b", "updates", "selected relation", ("chunk-a",)),
            NativeGraphEdge("b", "x", "mentions", "unrelated relation", ("chunk-x",)),
        ],
    )
    registry = NativeContextRegistry()
    registry.add("baseline", view)
    builder = AnswerContextBuilder(
        EvidenceHydrator(external_corpus_path=corpus_path),
        token_counter=TokenCounter("unicode_chars_v1"),
        max_context_tokens=1_000,
        structured_token_share=0.5,
        native_contexts=registry,
    )
    context = builder.build(
        {
            "retrieved": [
                {
                    "chunk_id": "chunk-a",
                    "document_id": "doc-a",
                    "text": "Evidence A",
                }
            ]
        },
        method="baseline",
        variant="native",
        top_k=20,
    )
    assert "Company A" in context.text
    assert "selected relation" in context.text
    assert "Unrelated X" not in context.text
    assert "E1" in context.text
    assert context.structured_context_origin == "reconstructed_from_index_by_retrieved_sources"

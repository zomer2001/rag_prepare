from __future__ import annotations

import json

from rag_graph_demo.evaluation.datasets import load_bundle


def test_load_bundle_accepts_multihop_rag_shape(tmp_path) -> None:
    corpus = [
        {
            "title": "Article A",
            "source": "Example News",
            "published_at": "2025-01-01",
            "body": "Alpha evidence appears here.",
        }
    ]
    queries = [
        {
            "query": "Where is the evidence?",
            "question_type": "inference_query",
            "answer": "Here",
            "evidence_list": [
                {
                    "title": "Article A",
                    "source": "Example News",
                    "published_at": "2025-01-01",
                    "fact": "Alpha evidence appears here.",
                }
            ],
        }
    ]
    (tmp_path / "corpus.json").write_text(json.dumps(corpus), encoding="utf-8")
    (tmp_path / "MultiHopRAG.json").write_text(json.dumps(queries), encoding="utf-8")

    dataset = load_bundle(tmp_path, name="fixture")

    assert dataset.documents[0].document_id == dataset.queries[0].gold_evidence[0].document_id
    assert dataset.queries[0].gold_evidence[0].text == "Alpha evidence appears here."

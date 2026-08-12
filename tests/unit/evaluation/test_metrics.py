from __future__ import annotations

import pytest

from rag_graph_demo.evaluation.metrics import aggregate_results, query_metrics
from rag_graph_demo.evaluation.models import BenchmarkQuery, GoldEvidence, RetrievedChunk


def test_retrieval_metrics_measure_complete_evidence_sets() -> None:
    query = BenchmarkQuery(
        query_id="q1",
        question="question",
        question_type="temporal_query",
        answer="answer",
        gold_evidence=(
            GoldEvidence(text="first fact", document_id="doc-1"),
            GoldEvidence(text="second fact", document_id="doc-2"),
        ),
    )
    retrieved = [
        RetrievedChunk("c2", "doc-2", "second fact"),
        RetrievedChunk("noise", "doc-x", "unrelated"),
        RetrievedChunk("c1", "doc-1", "first fact"),
    ]

    metrics = query_metrics(query, retrieved, cutoffs=(1, 3))

    assert metrics["hits@1"] == 1
    assert metrics["evidence_recall@1"] == 0.5
    assert metrics["complete_recall@1"] == 0
    assert metrics["complete_recall@3"] == 1
    assert metrics["mrr@3"] == 1
    assert 0 < metrics["ndcg@3"] <= 1


def test_aggregate_results_reports_latency_and_type_slices() -> None:
    summary = aggregate_results(
        [
            {
                "question_type": "temporal",
                "elapsed_ms": 10,
                "metrics": {"hits@10": 1.0},
            },
            {
                "question_type": "temporal",
                "elapsed_ms": 30,
                "metrics": {"hits@10": 0.0},
            },
        ]
    )

    assert summary["metrics"]["hits@10"] == pytest.approx(0.5)
    assert summary["metrics"]["latency_p50_ms"] == pytest.approx(20)
    assert summary["by_question_type"]["temporal"]["query_count"] == 2

from __future__ import annotations

import math
import re
import statistics
from collections import defaultdict
from typing import Any

from rag_graph_demo.evaluation.models import BenchmarkQuery, GoldEvidence, RetrievedChunk


def normalize_evidence_text(value: str) -> str:
    return re.sub(r"\W+", "", value.casefold(), flags=re.UNICODE)


def matched_evidence_indices(
    retrieved: list[RetrievedChunk],
    gold: tuple[GoldEvidence, ...],
) -> list[set[int]]:
    gold_text = [normalize_evidence_text(item.text or "") for item in gold]
    matches: list[set[int]] = []
    for chunk in retrieved:
        normalized = normalize_evidence_text(chunk.text)
        current: set[int] = set()
        for index, item in enumerate(gold):
            document_match = bool(item.document_id and chunk.document_id == item.document_id)
            text_match = bool(gold_text[index] and gold_text[index] in normalized)
            if document_match or text_match:
                current.add(index)
        matches.append(current)
    return matches


def query_metrics(
    query: BenchmarkQuery,
    retrieved: list[RetrievedChunk],
    *,
    cutoffs: tuple[int, ...],
) -> dict[str, float]:
    gold_count = len(query.gold_evidence)
    if not gold_count:
        return {}
    matches = matched_evidence_indices(retrieved, query.gold_evidence)
    values: dict[str, float] = {}
    for cutoff in cutoffs:
        seen: set[int] = set()
        relevant_ranks: list[int] = []
        average_precision = 0.0
        dcg = 0.0
        for rank, current in enumerate(matches[:cutoff], start=1):
            new = current - seen
            if not new:
                continue
            seen.update(new)
            relevant_ranks.append(rank)
            average_precision += len(new) * min(1.0, len(seen) / rank)
            dcg += 1.0 / math.log2(rank + 1)
        document_ids = {item.document_id for item in query.gold_evidence if item.document_id}
        ideal_units = len(document_ids) if len(document_ids) == gold_count else gold_count
        ideal_relevant_chunks = min(ideal_units, cutoff)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant_chunks + 1))
        values[f"hits@{cutoff}"] = float(bool(seen))
        values[f"evidence_recall@{cutoff}"] = len(seen) / gold_count
        values[f"complete_recall@{cutoff}"] = float(len(seen) == gold_count)
        values[f"mrr@{cutoff}"] = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0
        values[f"map@{cutoff}"] = average_precision / gold_count
        values[f"ndcg@{cutoff}"] = dcg / idcg if idcg else 0.0
    return values


def aggregate_results(per_query: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [item for item in per_query if not item.get("error")]
    metric_rows = [item["metrics"] for item in successful if item.get("metrics")]
    aggregate = _average_metric_rows(metric_rows)
    latencies = [float(item["elapsed_ms"]) for item in successful]
    aggregate.update(_latency_metrics(latencies))
    by_type: dict[str, list[dict[str, float]]] = defaultdict(list)
    for item in successful:
        if item.get("metrics"):
            by_type[str(item["question_type"])].append(item["metrics"])
    return {
        "query_count": len(per_query),
        "evaluated_query_count": len(metric_rows),
        "error_count": len(per_query) - len(successful),
        "metrics": aggregate,
        "by_question_type": {
            key: {
                "query_count": len(rows),
                "metrics": _average_metric_rows(rows),
            }
            for key, rows in sorted(by_type.items())
        },
    }


def _average_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: sum(row.get(key, 0.0) for row in rows) / len(rows) for key in keys} if rows else {}


def _latency_metrics(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "latency_mean_ms": statistics.fmean(values),
        "latency_p50_ms": statistics.median(values),
        "latency_p95_ms": ordered[p95_index],
    }

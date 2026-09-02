#!/usr/bin/env python3
"""Score raw HippoRAG rankings with this project's retrieval metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rag_graph_demo.evaluation.datasets import load_bundle
from rag_graph_demo.evaluation.metrics import aggregate_results, query_metrics
from rag_graph_demo.evaluation.models import RetrievedChunk


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--raw-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name")
    parser.add_argument("--cutoffs", default="4,10")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = load_bundle(
        args.dataset_dir.resolve(),
        name=args.dataset_name or args.dataset_dir.name,
    )
    query_by_id = {query.query_id: query for query in dataset.queries}
    cutoffs = tuple(sorted({int(value) for value in args.cutoffs.split(",") if value.strip()}))
    if not cutoffs:
        raise ValueError("At least one cutoff is required")

    scored_rows = []
    for raw in load_jsonl(args.raw_results.resolve()):
        query_id = str(raw["query_id"])
        query = query_by_id.get(query_id)
        if query is None:
            raise ValueError(f"Unknown query_id in HippoRAG output: {query_id}")
        retrieved = [
            RetrievedChunk(
                chunk_id=str(item["chunk_id"]),
                document_id=(
                    str(item["document_id"])
                    if item.get("document_id") not in (None, "")
                    else None
                ),
                text=str(item.get("text") or ""),
                score=float(item["score"]) if item.get("score") is not None else None,
                metadata=dict(item.get("metadata") or {}),
            )
            for item in raw.get("retrieved") or []
        ]
        scored_rows.append(
            {
                "query_id": query.query_id,
                "question": query.question,
                "question_type": query.question_type,
                "answer": query.answer,
                "gold_evidence_count": len(query.gold_evidence),
                "elapsed_ms": float(raw.get("elapsed_ms") or 0),
                "metrics": query_metrics(query, retrieved, cutoffs=cutoffs),
                "retrieved": [
                    {**asdict(item), "text": item.text[:1_500]} for item in retrieved
                ],
            }
        )

    summary = aggregate_results(scored_rows)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "hipporag.jsonl"
    result_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored_rows),
        encoding="utf-8",
    )
    (output_dir / "hipporag_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    preferred = [
        f"hits@{max(cutoffs)}",
        f"evidence_recall@{max(cutoffs)}",
        f"complete_recall@{max(cutoffs)}",
        f"ndcg@{max(cutoffs)}",
        f"mrr@{max(cutoffs)}",
        "latency_p95_ms",
    ]
    metrics = summary.get("metrics", {})
    lines = [
        f"# HippoRAG retrieval evaluation: {dataset.name}",
        "",
        f"- Evaluated queries: {summary['evaluated_query_count']}",
        f"- Errors: {summary['error_count']}",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in preferred:
        if key in metrics:
            lines.append(f"| {key} | {metrics[key]:.4f} |")
    (output_dir / "hipporag_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Scored HippoRAG results written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

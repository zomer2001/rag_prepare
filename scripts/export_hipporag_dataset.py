#!/usr/bin/env python3
"""Convert this project's JSONL retrieval benchmark into HippoRAG's JSON format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag_graph_demo.evaluation.chunking import build_chunks
from rag_graph_demo.evaluation.datasets import load_bundle


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def export_dataset(
    dataset_dir: Path,
    output_dir: Path,
    name: str,
    *,
    chunking_strategy: str = "document",
    chunk_size: int = 700,
    chunk_overlap: int = 100,
    query_limit: int | None = None,
    distractor_documents: int | None = None,
) -> dict[str, int | str]:
    source_corpus = load_jsonl(dataset_dir / "corpus.jsonl")
    source_queries = load_jsonl(dataset_dir / "queries.jsonl")

    if query_limit is not None:
        source_queries = source_queries[:query_limit]

    corpus_by_url_index = {
        str(source["url"]): index
        for index, source in enumerate(source_corpus)
        if source.get("url")
    }
    corpus_by_title_index = {
        str(source["title"]): index
        for index, source in enumerate(source_corpus)
        if source.get("title")
    }
    selected_indices = set(range(len(source_corpus)))
    if distractor_documents is not None:
        selected_indices = set()
        for source in source_queries:
            for evidence in source.get("evidence_list", []):
                index = corpus_by_url_index.get(str(evidence.get("url") or ""))
                if index is None:
                    index = corpus_by_title_index.get(str(evidence.get("title") or ""))
                if index is not None:
                    selected_indices.add(index)
        distractors_added = 0
        for index in range(len(source_corpus)):
            if index in selected_indices:
                continue
            selected_indices.add(index)
            distractors_added += 1
            if distractors_added >= distractor_documents:
                break
        source_corpus = [
            source for index, source in enumerate(source_corpus) if index in selected_indices
        ]

    corpus: list[dict[str, Any]] = []
    corpus_by_url: dict[str, dict[str, Any]] = {}
    corpus_by_title: dict[str, dict[str, Any]] = {}
    bundle = load_bundle(dataset_dir, name=name)
    documents_by_title = {document.title: document for document in bundle.documents}
    selected_documents = []
    for source in source_corpus:
        title = str(source.get("title") or "")
        document = documents_by_title.get(title)
        if document is None:
            raise ValueError(f"Cannot map corpus title to benchmark document: {title!r}")
        selected_documents.append(document)

    if chunking_strategy == "document":
        for index, document in enumerate(selected_documents):
            item = {
                "title": document.title,
                "text": document.text,
                "idx": index,
                "document_id": document.document_id,
                "chunk_id": f"document-{document.document_id}",
                "order": 0,
            }
            corpus.append(item)
    else:
        chunks = build_chunks(
            tuple(selected_documents),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=chunking_strategy,
        )
        title_by_id = {document.document_id: document.title for document in selected_documents}
        for index, chunk in enumerate(chunks):
            corpus.append(
                {
                    "title": title_by_id[chunk.document_id],
                    "text": chunk.text,
                    "idx": index,
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "order": chunk.order,
                }
            )

    first_chunk_by_document: dict[str, dict[str, Any]] = {}
    for item in corpus:
        first_chunk_by_document.setdefault(str(item["document_id"]), item)

    for source, document in zip(source_corpus, selected_documents, strict=True):
        item = first_chunk_by_document[document.document_id]
        if source.get("url"):
            corpus_by_url[source["url"]] = item
        if source.get("title"):
            corpus_by_title[source["title"]] = item

    missing_evidence = 0
    samples: list[dict[str, Any]] = []
    for index, source in enumerate(source_queries):
        paragraphs = []
        for evidence in source.get("evidence_list", []):
            document = corpus_by_url.get(evidence.get("url", ""))
            if document is None:
                document = corpus_by_title.get(evidence.get("title", ""))
            if document is None:
                missing_evidence += 1
                continue
            paragraphs.append({**document, "is_supporting": True})

        samples.append(
            {
                "id": source.get("query_id") or f"{name}-{index:05d}",
                "question": source.get("query") or source.get("question") or "",
                "answer": source.get("answer", ""),
                "answerable": bool(source.get("answer")),
                "paragraphs": paragraphs,
                "question_type": source.get("question_type"),
            }
        )

    if missing_evidence:
        raise ValueError(
            f"{missing_evidence} gold evidence records do not match the corpus by URL or title"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}_corpus.json").write_text(
        json.dumps(corpus, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / f"{name}.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "chunking_strategy": chunking_strategy,
        "source_documents": len(selected_documents),
        "corpus_documents": len(corpus),
        "queries": len(samples),
        "gold_evidence": sum(len(sample["paragraphs"]) for sample in samples),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--chunking-strategy",
        choices=("document", "fixed", "recursive"),
        default="document",
    )
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument(
        "--distractor-documents",
        type=int,
        help="Keep every selected query's evidence documents plus this many distractors.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = export_dataset(
        args.dataset_dir.resolve(),
        args.output_dir.resolve(),
        args.name,
        chunking_strategy=args.chunking_strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        query_limit=args.query_limit,
        distractor_documents=args.distractor_documents,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

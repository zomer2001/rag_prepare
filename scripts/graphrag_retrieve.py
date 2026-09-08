#!/usr/bin/env python3
"""Retrieve ranked source chunks from a prepared GraphRAG Local Search index."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from graphrag.cli.query import _resolve_output_files
from graphrag.config.embeddings import entity_description_embedding
from graphrag.config.load_config import load_config
from graphrag.query.factory import get_local_search_engine
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.utils.api import get_embedding_store, load_search_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--community-level", type=int, default=2)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    root = args.runtime_dir.resolve()
    config = load_config(root_dir=root)
    frames = _resolve_output_files(
        config=config,
        output_list=["communities", "community_reports", "text_units", "relationships", "entities"],
        optional_list=["covariates"],
    )
    entities = read_indexer_entities(
        frames["entities"], frames["communities"], args.community_level
    )
    reports = read_indexer_reports(
        frames["community_reports"], frames["communities"], args.community_level
    )
    text_units = read_indexer_text_units(frames["text_units"])
    relationships = read_indexer_relationships(frames["relationships"])
    covariates = (
        read_indexer_covariates(frames["covariates"]) if frames["covariates"] is not None else []
    )
    embedding_store = get_embedding_store(
        config=config.vector_store,
        embedding_name=entity_description_embedding,
    )
    engine = get_local_search_engine(
        config=config,
        reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        covariates={"claims": covariates},
        description_embedding_store=embedding_store,
        response_type="single sentence",
        system_prompt=load_search_prompt(config.local_search.prompt),
    )
    unit_by_short_id = {str(unit.short_id): unit for unit in text_units}
    source_map: dict[str, dict[str, Any]] = _load_json(root / "source_map.json")
    queries: list[dict[str, Any]] = _load_json(root / "queries.json")
    if args.limit is not None:
        queries = queries[: args.limit]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output.is_file():
        with output.open(encoding="utf-8") as handle:
            completed = {str(json.loads(line)["query_id"]) for line in handle if line.strip()}

    with output.open("a", encoding="utf-8") as handle:
        for query in queries:
            query_id = str(query["id"])
            if query_id in completed:
                continue
            started = time.perf_counter()
            context = engine.context_builder.build_context(
                query=str(query["question"]),
                **engine.context_builder_params,
            )
            sources = context.context_records.get("sources")
            retrieved = []
            seen_documents: set[str] = set()
            if sources is not None:
                for _, row in sources.iterrows():
                    unit = unit_by_short_id.get(str(row["id"]))
                    if unit is None or unit.document_id is None:
                        continue
                    mapped = source_map.get(str(unit.document_id))
                    if mapped is None:
                        continue
                    document_id = str(mapped["document_id"])
                    if document_id in seen_documents:
                        continue
                    seen_documents.add(document_id)
                    rank = len(retrieved) + 1
                    retrieved.append(
                        {
                            "chunk_id": str(mapped["chunk_id"]),
                            "document_id": document_id,
                            "text": str(mapped["text"]),
                            "score": 1.0 / rank,
                            "metadata": {
                                "title": mapped.get("title"),
                                "source": "graphrag_local_context",
                            },
                        }
                    )
                    if len(retrieved) >= args.top_k:
                        break
            row = {
                "query_id": query_id,
                "question": query["question"],
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "retrieved": retrieved,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
    print(f"Raw GraphRAG results written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

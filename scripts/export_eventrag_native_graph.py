#!/usr/bin/env python3
"""Export a read-only EventRAG Neo4j graph snapshot for offline answer contexts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default="bolt://127.0.0.1:7689")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_value(item) for item in value]
    return str(value)


def node_name(labels: list[str], entity_type: str, node_id: str) -> str:
    candidates = [label for label in labels if label.casefold() != entity_type.casefold()]
    if candidates:
        return sorted(candidates, key=lambda value: (value == '"UNKNOWN"', value))[0]
    return node_id


def main() -> None:
    args = parse_args()
    if not args.password:
        raise RuntimeError("Set NEO4J_PASSWORD or pass --password")
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    with (
        GraphDatabase.driver(args.uri, auth=(args.user, args.password)) as driver,
        driver.session() as session,
    ):
        records = session.run(
            "MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, "
            "properties(n) AS props ORDER BY id"
        )
        for record in records:
            node_id = str(record["id"])
            labels = [str(value) for value in record["labels"]]
            props = dict(record["props"] or {})
            entity_type = str(props.get("entity_type") or "entity")
            nodes.append(
                {
                    "id": node_id,
                    "name": node_name(labels, entity_type, node_id),
                    "type": entity_type,
                    "description": str(props.get("description") or ""),
                    "source_ids": str(props.get("source_id") or ""),
                }
            )
        records = session.run(
            "MATCH (a)-[r]->(b) RETURN elementId(a) AS source, "
            "elementId(b) AS target, type(r) AS relation, properties(r) AS props "
            "ORDER BY source, target, relation"
        )
        for record in records:
            props = dict(record["props"] or {})
            edges.append(
                {
                    "source": str(record["source"]),
                    "target": str(record["target"]),
                    "relation": str(record["relation"]),
                    "description": str(props.get("description") or ""),
                    "keywords": str(props.get("keywords") or ""),
                    "weight": json_value(props.get("weight")),
                    "source_ids": str(props.get("source_id") or ""),
                }
            )
    output = {
        "format": "eventrag_native_graph_snapshot_v1",
        "source": args.uri,
        "nodes": nodes,
        "edges": edges,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Exported {len(nodes)} nodes and {len(edges)} edges to {args.output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import asyncio
import re
import shutil
from pathlib import Path

from qdrant_client import QdrantClient

from rag_graph_demo.config import Settings, get_settings
from rag_graph_demo.domain.tkg import TKGSnapshot
from rag_graph_demo.lightrag_engine.local_models import embedding_model_name
from rag_graph_demo.storage.neo4j_tkg_store import Neo4jTKGStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete all local demo documents and LightRAG indexes."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm destructive reset.",
    )
    return parser.parse_args()


def clear_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def main() -> None:
    args = parse_args()
    if not args.yes:
        raise SystemExit("Reset cancelled. Re-run with --yes to confirm.")

    settings = get_settings()
    for group in (
        "raw",
        "parsed",
        "registry",
        "events",
        "tkg",
        "lightrag",
        "exports",
        "logs",
    ):
        clear_directory(settings.data_dir / group)

    suffix = re.sub(
        r"[^a-zA-Z0-9_]",
        "_",
        embedding_model_name(settings).strip().lower(),
    )
    suffix = f"{suffix}_{settings.embedding_dimension}d"
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        trust_env=False,
        check_compatibility=False,
    )
    for namespace in ("entities", "relationships", "chunks"):
        collection = f"lightrag_vdb_{namespace}_{suffix}"
        if client.collection_exists(collection):
            client.delete_collection(collection)
            print(f"Deleted Qdrant collection: {collection}")
    if settings.neo4j_enabled:
        asyncio.run(_clear_neo4j(settings))
        print("Deleted project RagTKG nodes from Neo4j.")
    print("Local document registry, events, TKG, and LightRAG state reset.")


async def _clear_neo4j(settings: Settings) -> None:
    store = Neo4jTKGStore(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        database=settings.neo4j_database,
    )
    try:
        await store.replace_snapshot(
            TKGSnapshot(ontology_version=settings.tkg_ontology_version)
        )
    finally:
        await store.close()


if __name__ == "__main__":
    main()

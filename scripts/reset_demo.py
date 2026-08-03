from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from qdrant_client import QdrantClient

from rag_graph_demo.config import get_settings


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
    for group in ("raw", "parsed", "registry", "lightrag", "exports", "logs"):
        clear_directory(settings.data_dir / group)

    suffix = re.sub(
        r"[^a-zA-Z0-9_]",
        "_",
        settings.embedding_model_id.strip().lower(),
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
    print("Local document registry and LightRAG state reset.")


if __name__ == "__main__":
    main()

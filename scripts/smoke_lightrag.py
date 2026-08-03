from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from lightrag.utils import EmbeddingFunc
from qdrant_client import QdrantClient

from rag_graph_demo.config import Settings
from rag_graph_demo.ingestion.source_mapper import SourceMapper
from rag_graph_demo.lightrag_engine.engine import LightRAGEngine
from rag_graph_demo.storage.document_registry import DocumentRegistry

COLLECTIONS = (
    "lightrag_vdb_entities_smoke_embedding_8d",
    "lightrag_vdb_relationships_smoke_embedding_8d",
    "lightrag_vdb_chunks_smoke_embedding_8d",
)


async def fake_embedding(
    texts: list[str],
    context: str = "document",
) -> np.ndarray:
    del context
    vectors = np.zeros((len(texts), 8), dtype=np.float32)
    vectors[:, 0] = 1
    return vectors


def fake_llm_builder(_: str):
    async def complete(
        prompt: str,
        **__: Any,
    ) -> str:
        del prompt
        return "ok"

    return complete


async def run() -> int:
    with tempfile.TemporaryDirectory(prefix="rag-lightrag-smoke-") as directory:
        root = Path(directory)
        settings = Settings(
            _env_file=None,
            data_dir=root,
            lightrag_working_dir=root / "lightrag",
            lightrag_workspace="smoke_v1",
            embedding_dimension=8,
            embedding_model="smoke-embedding",
            reranker_enabled=False,
        )
        settings.ensure_data_dirs()
        mapper = SourceMapper(
            DocumentRegistry(root / "registry" / "documents.json")
        )
        embedding = EmbeddingFunc(
            embedding_dim=8,
            max_token_size=128,
            model_name="smoke-embedding",
            supports_asymmetric=True,
            func=fake_embedding,
        )
        engine = LightRAGEngine(
            settings,
            mapper,
            embedding_func=embedding,
            llm_func_builder=fake_llm_builder,
        )
        try:
            await engine.initialize()
            labels = await engine.get_graph_labels()
            print(f"LightRAG/Qdrant smoke test passed; graph labels: {labels}")
        finally:
            await engine.finalize()

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        trust_env=False,
        check_compatibility=False,
    )
    for collection in COLLECTIONS:
        if client.collection_exists(collection):
            client.delete_collection(collection)
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()

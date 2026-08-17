from __future__ import annotations

import numpy as np
import pytest
from qdrant_client import AsyncQdrantClient

from rag_graph_demo.domain.events import EventRecord
from rag_graph_demo.domain.tkg import (
    AssertionRecord,
    EntityRecord,
    TKGSnapshot,
)
from rag_graph_demo.evaluation.models import BenchmarkChunk
from rag_graph_demo.retrieval.tkg_vector import TKGVectorIndex


@pytest.mark.asyncio
async def test_tkg_vector_indexes_all_five_seed_kinds(tmp_path) -> None:
    async def embed(texts: list[str], context: str) -> np.ndarray:
        del context
        return np.asarray(
            [[float("Acme" in text), float("发布" in text), 1.0] for text in texts],
            dtype=np.float32,
        )

    client = AsyncQdrantClient(":memory:")
    index = TKGVectorIndex(
        qdrant_url="http://127.0.0.1:6333",
        qdrant_api_key=None,
        workspace_id="unit-test",
        dimension=3,
        embed=embed,
        manifest_path=tmp_path / "manifest.json",
        client=client,
    )
    snapshot = TKGSnapshot(
        ontology_version="0.1.0",
        entities=[
            EntityRecord(
                entity_id="entity-1",
                canonical_name="Acme",
                entity_type="Organization",
                ontology_version="0.1.0",
            ),
            EntityRecord(
                entity_id="entity-2",
                canonical_name="Lattice",
                entity_type="Model",
                ontology_version="0.1.0",
            ),
        ],
        assertions=[
            AssertionRecord(
                assertion_id="assertion-1",
                subject_id="entity-1",
                predicate_id="develops",
                object_id="entity-2",
                ontology_version="0.1.0",
            )
        ],
    )
    events = [
        EventRecord(
            event_id="event-1",
            canonical_key="release",
            event_type="发布",
            summary="Acme 发布 Lattice。",
        )
    ]
    chunks = [BenchmarkChunk("chunk-1", "doc-1", "Acme 发布 Lattice。", 0)]

    await index.initialize(snapshot, events, chunks, fingerprint="fixture")

    for kind in index.KINDS:
        hits = await index.search(kind, "Acme 发布", limit=5)
        assert hits, kind
        assert hits[0]["node_type"] == kind
    await client.close()

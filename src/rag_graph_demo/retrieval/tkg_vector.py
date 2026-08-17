from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
from qdrant_client import AsyncQdrantClient, models
from tqdm.auto import tqdm

from rag_graph_demo.domain.events import EventRecord
from rag_graph_demo.domain.tkg import TKGSnapshot
from rag_graph_demo.evaluation.models import BenchmarkChunk
from rag_graph_demo.retrieval.event_vector import event_embedding_text
from rag_graph_demo.storage.atomic_json import read_json, write_json_atomic

EmbeddingCallable = Callable[[list[str], str], Awaitable[Any]]
TKGVectorKind = Literal["chunk", "entity", "predicate", "assertion", "event"]
TKG_VECTOR_PROFILE_VERSION = "tkg_vector_v1"


class TKGVectorIndex:
    """Qdrant seed index for the unified TKG; Neo4j remains the structure store."""

    KINDS: tuple[TKGVectorKind, ...] = (
        "chunk",
        "entity",
        "predicate",
        "assertion",
        "event",
    )

    def __init__(
        self,
        *,
        qdrant_url: str,
        qdrant_api_key: str | None,
        workspace_id: str,
        dimension: int,
        embed: EmbeddingCallable,
        manifest_path: Path,
        batch_size: int = 8,
        max_concurrency: int = 1,
        show_progress: bool = False,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        bypass = "localhost,127.0.0.1,::1"
        existing_no_proxy = os.environ.get("NO_PROXY", "")
        merged_no_proxy = ",".join(
            dict.fromkeys(item for item in f"{existing_no_proxy},{bypass}".split(",") if item)
        )
        os.environ["NO_PROXY"] = merged_no_proxy
        os.environ["no_proxy"] = merged_no_proxy
        self._client = client or AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self._owns_client = client is None
        self._workspace_id = workspace_id
        self._dimension = dimension
        self._embed = embed
        self._manifest_path = manifest_path
        self._batch_size = max(1, batch_size)
        self._max_concurrency = max(1, max_concurrency)
        self._show_progress = show_progress
        suffix = hashlib.sha256(workspace_id.encode()).hexdigest()[:12]
        self.collections = {
            kind: f"rag_tkg_{suffix}_{kind}s" for kind in self.KINDS
        }

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()

    async def initialize(
        self,
        snapshot: TKGSnapshot,
        events: list[EventRecord],
        chunks: list[BenchmarkChunk],
        *,
        fingerprint: str,
    ) -> None:
        expected = {
            "fingerprint": fingerprint,
            "profile": TKG_VECTOR_PROFILE_VERSION,
            "workspace_id": self._workspace_id,
            "dimension": self._dimension,
            "collections": self.collections,
        }
        stored = read_json(self._manifest_path, default={})
        if stored == expected and all(
            [await self._client.collection_exists(name) for name in self.collections.values()]
        ):
            return
        rows = self._rows(snapshot, events, chunks)
        for kind in self.KINDS:
            collection = self.collections[kind]
            if await self._client.collection_exists(collection):
                await self._client.delete_collection(collection)
            await self._client.create_collection(
                collection,
                vectors_config=models.VectorParams(
                    size=self._dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            await self._index_rows(collection, kind, rows[kind])
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self._manifest_path, expected)

    async def search(
        self,
        kind: TKGVectorKind,
        text: str,
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        if not text.strip() or limit <= 0:
            return []
        vector = np.asarray(await self._embed([text], "query"), dtype=np.float32)[0]
        response = await self._client.query_points(
            self.collections[kind],
            query=vector.tolist(),
            limit=limit,
            with_payload=True,
        )
        return [
            {**dict(point.payload or {}), "score": float(point.score)}
            for point in response.points
        ]

    async def _index_rows(
        self,
        collection: str,
        kind: TKGVectorKind,
        rows: list[dict[str, Any]],
    ) -> None:
        semaphore = asyncio.Semaphore(self._max_concurrency)
        progress = tqdm(
            total=len(rows),
            desc=f"tkg-{kind}:index",
            unit=kind,
            dynamic_ncols=True,
            disable=not self._show_progress,
        )

        async def index_batch(batch: list[dict[str, Any]]) -> None:
            async with semaphore:
                vectors = np.asarray(
                    await self._embed([row["text"] for row in batch], "document"),
                    dtype=np.float32,
                )
                await self._client.upsert(
                    collection,
                    points=[
                        models.PointStruct(
                            id=str(
                                uuid.uuid5(
                                    uuid.NAMESPACE_URL,
                                    f"{collection}:{row['node_id']}",
                                )
                            ),
                            vector=vector.tolist(),
                            payload={key: value for key, value in row.items() if key != "text"},
                        )
                        for row, vector in zip(batch, vectors, strict=True)
                    ],
                    wait=True,
                )
            progress.update(len(batch))

        try:
            await asyncio.gather(
                *(
                    index_batch(rows[start : start + self._batch_size])
                    for start in range(0, len(rows), self._batch_size)
                )
            )
        finally:
            progress.close()

    def _rows(
        self,
        snapshot: TKGSnapshot,
        events: list[EventRecord],
        chunks: list[BenchmarkChunk],
    ) -> dict[TKGVectorKind, list[dict[str, Any]]]:
        entities = {item.entity_id: item for item in snapshot.entities}
        predicate_ids = sorted({item.predicate_id for item in snapshot.assertions})
        return {
            "chunk": [
                {
                    "node_id": item.chunk_id,
                    "node_type": "chunk",
                    "document_id": item.document_id,
                    "chunk_id": item.chunk_id,
                    "content": item.text,
                    "text": item.text,
                }
                for item in chunks
            ],
            "entity": [
                {
                    "node_id": item.entity_id,
                    "node_type": "entity",
                    "entity_type": item.entity_type,
                    "text": (
                        f"名称：{item.canonical_name}\n类型：{item.entity_type}\n"
                        f"别名：{', '.join(item.aliases)}\n{item.description}"
                    ),
                }
                for item in snapshot.entities
                if item.status == "active"
            ],
            "predicate": [
                {
                    "node_id": predicate_id,
                    "node_type": "predicate",
                    "predicate_id": predicate_id,
                    "text": self._predicate_text(predicate_id),
                }
                for predicate_id in predicate_ids
            ],
            "assertion": [
                {
                    "node_id": item.assertion_id,
                    "node_type": "assertion",
                    "subject_id": item.subject_id,
                    "predicate_id": item.predicate_id,
                    "object_id": item.object_id,
                    "valid_from": item.valid_from or "",
                    "valid_to": item.valid_to or "",
                    "status": item.status.value,
                    "text": self._assertion_text(item, entities),
                }
                for item in snapshot.assertions
            ],
            "event": [
                {
                    "node_id": item.event_id,
                    "node_type": "event",
                    "event_type": item.event_type,
                    "valid_from": item.start_time or "",
                    "valid_to": item.end_time or "",
                    "status": item.status.value,
                    "text": event_embedding_text(item),
                }
                for item in events
            ],
        }

    @staticmethod
    def _assertion_text(item: Any, entities: dict[str, Any]) -> str:
        subject = entities.get(item.subject_id)
        obj = entities.get(item.object_id)
        interval = " ".join(value for value in (item.valid_from, item.valid_to) if value)
        return "｜".join(
            value
            for value in (
                subject.canonical_name if subject else item.subject_id,
                item.predicate_id,
                obj.canonical_name if obj else item.object_id,
                interval,
                f"状态：{item.status.value}",
            )
            if value
        )

    @staticmethod
    def _predicate_text(predicate_id: str) -> str:
        words = re.sub(r"(?<!^)(?=[A-Z])", " ", predicate_id).replace("_", " ")
        return f"关系：{predicate_id}\n含义：{words}"


def tkg_vector_fingerprint(
    base_fingerprint: str,
    snapshot: TKGSnapshot,
    events: list[EventRecord],
) -> str:
    digest = hashlib.sha256(
        f"{base_fingerprint}\x1f{TKG_VECTOR_PROFILE_VERSION}".encode()
    )
    digest.update(
        json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False, sort_keys=True).encode()
    )
    for event in sorted(events, key=lambda item: item.event_id):
        digest.update(event.model_dump_json().encode())
    return digest.hexdigest()

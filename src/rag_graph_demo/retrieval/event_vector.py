from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from rag_graph_demo.domain.events import EventRecord

EmbeddingCallable = Callable[[list[str], str], Awaitable[np.ndarray]]


def event_embedding_text(event: EventRecord) -> str:
    return "｜".join(
        value
        for value in (
            event.start_time or event.end_time or "时间未知",
            event.event_type,
            ", ".join(event.actors),
            event.action,
            ", ".join(event.objects),
            event.result or "",
            event.summary,
            f"状态：{event.status.value}",
        )
        if value
    )


class EventVectorIndex:
    """Small persistent event vector index; storage can later move to Qdrant."""

    def __init__(
        self,
        path: Path,
        embed: EmbeddingCallable,
        *,
        embedding_identity: str,
        batch_size: int = 8,
        max_concurrency: int = 1,
        show_progress: bool = False,
    ) -> None:
        self._path = path
        self._embed = embed
        self._embedding_identity = embedding_identity
        self._batch_size = max(1, batch_size)
        self._max_concurrency = max(1, max_concurrency)
        self._show_progress = show_progress
        self._lock = asyncio.Lock()
        self._fingerprint: str | None = None
        self._event_ids: list[str] = []
        self._vectors: np.ndarray | None = None

    async def search(
        self,
        question: str,
        events: list[EventRecord],
        *,
        limit: int,
    ) -> list[tuple[str, float]]:
        await self._ensure(events)
        if self._vectors is None or not self._event_ids:
            return []
        query = await self._embed([question], "query")
        query_vector = self._normalize(np.asarray(query, dtype=np.float32))[0]
        scores = self._vectors @ query_vector
        order = np.argsort(scores)[::-1][:limit]
        return [(self._event_ids[index], float(scores[index])) for index in order]

    async def _ensure(self, events: list[EventRecord]) -> None:
        fingerprint = self._event_fingerprint(events)
        if self._fingerprint == fingerprint:
            return
        async with self._lock:
            if self._fingerprint == fingerprint:
                return
            if self._load(fingerprint):
                return
            ordered = sorted(events, key=lambda item: item.event_id)
            self._event_ids = [item.event_id for item in ordered]
            if ordered:
                texts = [event_embedding_text(item) for item in ordered]
                semaphore = asyncio.Semaphore(self._max_concurrency)
                progress = tqdm(
                    total=len(texts),
                    desc="event-vector:index",
                    unit="event",
                    dynamic_ncols=True,
                    disable=not self._show_progress,
                )

                async def embed_batch(batch: list[str]) -> np.ndarray:
                    async with semaphore:
                        values = np.asarray(
                            await self._embed(batch, "document"),
                            dtype=np.float32,
                        )
                    progress.update(len(batch))
                    return values

                try:
                    batches = await asyncio.gather(
                        *(
                            embed_batch(texts[start : start + self._batch_size])
                            for start in range(0, len(texts), self._batch_size)
                        )
                    )
                finally:
                    progress.close()
                self._vectors = self._normalize(np.vstack(batches))
            else:
                self._vectors = np.empty((0, 0), dtype=np.float32)
            self._fingerprint = fingerprint
            self._path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self._path,
                fingerprint=np.asarray([fingerprint]),
                event_ids=np.asarray(self._event_ids),
                vectors=self._vectors,
            )

    def _load(self, fingerprint: str) -> bool:
        if not self._path.is_file():
            return False
        try:
            with np.load(self._path, allow_pickle=False) as stored:
                saved = str(stored["fingerprint"][0])
                if saved != fingerprint:
                    return False
                self._event_ids = [str(item) for item in stored["event_ids"]]
                self._vectors = np.asarray(stored["vectors"], dtype=np.float32)
        except (OSError, KeyError, ValueError):
            return False
        self._fingerprint = fingerprint
        return True

    def _event_fingerprint(self, events: list[EventRecord]) -> str:
        digest = hashlib.sha256(self._embedding_identity.encode())
        for event in sorted(events, key=lambda item: item.event_id):
            digest.update(event.event_id.encode())
            digest.update(event.updated_at.isoformat().encode())
            digest.update(event_embedding_text(event).encode())
        return digest.hexdigest()

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return values / norms

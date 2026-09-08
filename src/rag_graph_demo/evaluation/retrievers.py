from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from tqdm.auto import tqdm

from rag_graph_demo.evaluation.models import BenchmarkChunk, RetrievedChunk


class Retriever(Protocol):
    async def retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]: ...


EmbeddingCallable = Callable[[list[str], str], Awaitable[np.ndarray]]


class DenseRetriever:
    def __init__(
        self,
        chunks: list[BenchmarkChunk],
        embedding_func: EmbeddingCallable,
        *,
        cache_dir: Path,
        batch_size: int = 16,
        max_concurrency: int = 1,
        show_progress: bool = True,
    ) -> None:
        self.chunks = chunks
        self.embedding_func = embedding_func
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.max_concurrency = max(1, max_concurrency)
        self.show_progress = show_progress
        self._embeddings: np.ndarray | None = None

    async def initialize(self, fingerprint: str) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.cache_dir / "manifest.json"
        embeddings_path = self.cache_dir / "embeddings.npy"
        if manifest_path.is_file() and embeddings_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("fingerprint") == fingerprint:
                cached = np.load(embeddings_path)
                if len(cached) == len(self.chunks):
                    self._embeddings = cached
                    print(f"[dense:index] loaded {len(cached)} cached chunks")
                    return
        batch_dir = (
            self.cache_dir
            / "batches"
            / hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
        )
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_specs: list[tuple[int, int, Path]] = []
        for start in range(0, len(self.chunks), self.batch_size):
            end = min(start + self.batch_size, len(self.chunks))
            batch_path = batch_dir / f"{start:08d}-{end:08d}.npy"
            batch_specs.append((start, end, batch_path))

        semaphore = asyncio.Semaphore(self.max_concurrency)
        output_lock = asyncio.Lock()
        progress = tqdm(
            total=len(self.chunks),
            desc="dense:index",
            unit="chunk",
            dynamic_ncols=True,
            disable=not self.show_progress,
        )

        async def prepare_batch(start: int, end: int, batch_path: Path) -> np.ndarray:
            batch = _load_embedding_batch(batch_path, expected_rows=end - start)
            source = "cached"
            if batch is None:
                texts = [item.text for item in self.chunks[start:end]]
                async with semaphore:
                    batch = np.asarray(
                        await self.embedding_func(texts, "document"),
                        dtype=np.float32,
                    )
                if batch.ndim != 2 or len(batch) != end - start:
                    raise ValueError(
                        "Embedding batch has an unexpected shape: "
                        f"expected {end - start} rows, got {batch.shape}"
                    )
                _save_numpy(batch_path, batch)
                source = "embedded"
            async with output_lock:
                progress.update(end - start)
                progress.set_postfix_str(source, refresh=False)
            return batch

        try:
            batches = await asyncio.gather(
                *(prepare_batch(start, end, path) for start, end, path in batch_specs)
            )
        finally:
            progress.close()
        self._embeddings = np.vstack(batches).astype(np.float32)
        _save_numpy(embeddings_path, self._embeddings)
        manifest_path.write_text(
            json.dumps(
                {"fingerprint": fingerprint, "chunk_count": len(self.chunks)},
                indent=2,
            ),
            encoding="utf-8",
        )

    async def retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]:
        if self._embeddings is None:
            raise RuntimeError("DenseRetriever.initialize() must be called first")
        query = await self.embedding_func([question], "query")
        scores = self._embeddings @ query[0]
        indices = top_indices(scores, top_k)
        return [to_retrieved(self.chunks[index], float(scores[index])) for index in indices]


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        *,
        candidate_multiplier: int = 1,
        rrf_k: int = 60,
    ) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("Hybrid evaluation needs: uv sync --extra evaluation") from exc
        self.dense = dense
        self.candidate_multiplier = candidate_multiplier
        self.rrf_k = rrf_k
        self._bm25 = BM25Okapi([tokenize(item.text) for item in dense.chunks])

    async def retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]:
        candidate_k = min(
            len(self.dense.chunks),
            max(top_k * self.candidate_multiplier, top_k),
        )
        dense_results = await self.dense.retrieve(question, candidate_k)
        scores = np.asarray(self._bm25.get_scores(tokenize(question)), dtype=np.float32)
        bm25_indices = top_indices(scores, candidate_k)
        ranks: dict[str, float] = defaultdict(float)
        chunks: dict[str, RetrievedChunk] = {}
        for rank, item in enumerate(dense_results, start=1):
            ranks[item.chunk_id] += 1.0 / (self.rrf_k + rank)
            chunks[item.chunk_id] = item
        for rank, index in enumerate(bm25_indices, start=1):
            item = to_retrieved(self.dense.chunks[index], float(scores[index]))
            ranks[item.chunk_id] += 1.0 / (self.rrf_k + rank)
            chunks[item.chunk_id] = item
        ordered = sorted(ranks, key=ranks.get, reverse=True)[:top_k]
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=chunks[chunk_id].document_id,
                text=chunks[chunk_id].text,
                score=ranks[chunk_id],
                metadata={"fusion": "dense_bm25_rrf"},
            )
            for chunk_id in ordered
        ]


def tokenize(value: str) -> list[str]:
    lowered = value.casefold()
    tokens = re.findall(r"[a-z0-9]+", lowered)
    for run in re.findall(r"[\u4e00-\u9fff]+", lowered):
        tokens.extend(run[index : index + 2] for index in range(max(1, len(run) - 1)))
    return tokens


def top_indices(scores: np.ndarray, top_k: int) -> list[int]:
    count = min(max(top_k, 0), len(scores))
    if count == 0:
        return []
    if count == len(scores):
        return np.argsort(scores)[::-1].tolist()
    candidates = np.argpartition(scores, -count)[-count:]
    return candidates[np.argsort(scores[candidates])[::-1]].tolist()


def _load_embedding_batch(path: Path, *, expected_rows: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        value = np.load(path)
    except (OSError, ValueError):
        return None
    if value.ndim != 2 or len(value) != expected_rows:
        return None
    return np.asarray(value, dtype=np.float32)


def _save_numpy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    temporary.replace(path)


def to_retrieved(chunk: BenchmarkChunk, score: float | None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        text=chunk.text,
        score=score,
    )


def document_id_from_file_path(
    file_path: str,
    file_path_to_document: dict[str, str],
) -> str | None:
    if file_path in file_path_to_document:
        return file_path_to_document[file_path]
    return file_path_to_document.get(Path(file_path).name)


def light_data_to_chunks(
    raw_data: dict[str, Any],
    file_path_to_document: dict[str, str],
) -> list[RetrievedChunk]:
    data = raw_data.get("data") or {}
    result: list[RetrievedChunk] = []
    for index, value in enumerate(data.get("chunks") or []):
        file_path = str(value.get("file_path") or "")
        result.append(
            RetrievedChunk(
                chunk_id=str(value.get("chunk_id") or f"light-{index}"),
                document_id=document_id_from_file_path(file_path, file_path_to_document),
                text=str(value.get("content") or ""),
                metadata={"file_path": file_path, "source": "lightrag"},
            )
        )
    return result

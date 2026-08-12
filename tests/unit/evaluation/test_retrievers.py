from __future__ import annotations

import asyncio

import numpy as np
import pytest

from rag_graph_demo.evaluation.models import BenchmarkChunk, RetrievedChunk
from rag_graph_demo.evaluation.retrievers import DenseRetriever, top_indices
from rag_graph_demo.evaluation.runner import benchmark_chunk_file_path, rrf_by_document


def test_top_indices_returns_descending_scores() -> None:
    assert top_indices(np.asarray([0.1, 0.9, 0.4]), 2) == [1, 2]


@pytest.mark.asyncio
async def test_dense_retriever_uses_asymmetric_context(tmp_path) -> None:
    contexts: list[str] = []

    async def embed(texts: list[str], context: str) -> np.ndarray:
        contexts.append(context)
        return np.asarray(
            [[1.0, 0.0] if "alpha" in text.casefold() else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )

    retriever = DenseRetriever(
        [
            BenchmarkChunk("c1", "d1", "alpha", 0),
            BenchmarkChunk("c2", "d2", "beta", 0),
        ],
        embed,
        cache_dir=tmp_path,
    )
    await retriever.initialize("fingerprint")

    result = await retriever.retrieve("alpha question", 1)

    assert result[0].document_id == "d1"
    assert contexts == ["document", "query"]


@pytest.mark.asyncio
async def test_dense_retriever_resumes_completed_embedding_batches(tmp_path) -> None:
    chunks = [
        BenchmarkChunk("c1", "d1", "alpha", 0),
        BenchmarkChunk("c2", "d2", "beta", 0),
        BenchmarkChunk("c3", "d3", "gamma", 0),
    ]
    first_calls = 0

    async def interrupted_embed(texts: list[str], _context: str) -> np.ndarray:
        nonlocal first_calls
        first_calls += 1
        if first_calls == 2:
            raise RuntimeError("interrupted")
        return np.ones((len(texts), 2), dtype=np.float32)

    interrupted = DenseRetriever(
        chunks,
        interrupted_embed,
        cache_dir=tmp_path,
        batch_size=2,
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        await interrupted.initialize("fingerprint")

    resumed_batches: list[list[str]] = []

    async def resumed_embed(texts: list[str], _context: str) -> np.ndarray:
        resumed_batches.append(texts)
        return np.ones((len(texts), 2), dtype=np.float32)

    resumed = DenseRetriever(
        chunks,
        resumed_embed,
        cache_dir=tmp_path,
        batch_size=2,
    )
    await resumed.initialize("fingerprint")

    assert resumed_batches == [["gamma"]]


@pytest.mark.asyncio
async def test_dense_retriever_limits_parallel_api_batches(tmp_path) -> None:
    active = 0
    maximum_active = 0
    two_started = asyncio.Event()
    release = asyncio.Event()

    async def embed(texts: list[str], _context: str) -> np.ndarray:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        if maximum_active == 2:
            two_started.set()
        await release.wait()
        active -= 1
        return np.ones((len(texts), 2), dtype=np.float32)

    retriever = DenseRetriever(
        [
            BenchmarkChunk(f"c{index}", f"d{index}", f"text {index}", 0)
            for index in range(4)
        ],
        embed,
        cache_dir=tmp_path,
        batch_size=1,
        max_concurrency=2,
    )
    task = asyncio.create_task(retriever.initialize("parallel-fingerprint"))
    await asyncio.wait_for(two_started.wait(), timeout=1)

    assert maximum_active == 2
    release.set()
    await task
    assert maximum_active == 2


def test_event_rrf_prefers_cross_path_document_and_event_evidence() -> None:
    result = rrf_by_document(
        [
            RetrievedChunk("l1", "doc-a", "light A"),
            RetrievedChunk("l2", "doc-b", "light B"),
        ],
        [
            RetrievedChunk("e1", "doc-b", "event B"),
            RetrievedChunk("e2", "doc-c", "event C"),
        ],
        top_k=2,
    )

    assert result[0].document_id == "doc-b"
    assert result[0].text == "event B"
    assert len(result) == 2


def test_event_sources_share_their_event_level_fusion_rank() -> None:
    result = rrf_by_document(
        [],
        [
            RetrievedChunk(
                "e1",
                "doc-a",
                "source A",
                metadata={"fusion_rank": 1},
            ),
            RetrievedChunk(
                "e2",
                "doc-b",
                "source B",
                metadata={"fusion_rank": 1},
            ),
        ],
        top_k=2,
    )

    assert result[0].score == result[1].score == pytest.approx(1 / 61)


def test_benchmark_chunk_file_path_is_unique_per_chunk() -> None:
    first = benchmark_chunk_file_path("doc-a", "chunk-1")
    second = benchmark_chunk_file_path("doc-a", "chunk-2")

    assert first != second
    assert first.startswith("doc-a__")

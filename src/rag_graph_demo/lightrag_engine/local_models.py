from __future__ import annotations

import asyncio
from functools import partial
from typing import Any

import numpy as np
import torch
from lightrag.utils import EmbeddingFunc

from rag_graph_demo.config import Settings

QWEN_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query: "
)


def resolve_device(configured: str) -> str:
    if configured != "auto":
        return configured
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class LocalEmbeddingModel:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any | None = None
        self._load_lock = asyncio.Lock()

    async def encode(self, texts: list[str], context: str = "document") -> np.ndarray:
        model = await self._get_model()
        prepared = (
            [f"{QWEN_QUERY_PREFIX}{text}" for text in texts]
            if context == "query"
            else texts
        )
        embeddings = await asyncio.to_thread(
            model.encode,
            prepared,
            batch_size=self._settings.embedding_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        result = np.asarray(embeddings, dtype=np.float32)
        if result.ndim != 2 or result.shape[1] != self._settings.embedding_dimension:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {self._settings.embedding_dimension}, got {result.shape}"
            )
        return result

    async def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load_model)
        return self._model

    def _load_model(self) -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            self._settings.embedding_model,
            device=resolve_device(self._settings.embedding_device),
            trust_remote_code=True,
        )


class LocalReranker:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any | None = None
        self._load_lock = asyncio.Lock()

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        **_: Any,
    ) -> list[dict[str, float | int]]:
        if not documents:
            return []
        model = await self._get_model()
        pairs = [(query, document) for document in documents]
        scores = await asyncio.to_thread(model.predict, pairs, show_progress_bar=False)
        ranked = sorted(
            enumerate(float(score) for score in scores),
            key=lambda item: item[1],
            reverse=True,
        )
        if top_n is not None:
            ranked = ranked[:top_n]
        return [
            {"index": index, "relevance_score": score} for index, score in ranked
        ]

    async def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load_model)
        return self._model

    def _load_model(self) -> Any:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(
            self._settings.reranker_model,
            device=resolve_device(self._settings.reranker_device),
            trust_remote_code=True,
        )


def build_embedding_func(settings: Settings) -> EmbeddingFunc:
    model = LocalEmbeddingModel(settings)
    return EmbeddingFunc(
        embedding_dim=settings.embedding_dimension,
        max_token_size=8192,
        model_name=settings.embedding_model_id,
        supports_asymmetric=True,
        func=partial(model.encode),
    )


def build_rerank_func(settings: Settings):
    if not settings.reranker_enabled:
        return None
    return LocalReranker(settings).rerank

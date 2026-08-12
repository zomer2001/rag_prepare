from __future__ import annotations

import asyncio
from functools import partial
from typing import Any

import numpy as np
import torch
from lightrag.utils import EmbeddingFunc
from openai import AsyncOpenAI

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


class BailianEmbeddingModel:
    """Alibaba Cloud Model Studio embeddings through its OpenAI-compatible API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncOpenAI | None = None

    async def encode(self, texts: list[str], context: str = "document") -> np.ndarray:
        del context  # Bailian text-embedding models use the same encoding for both sides.
        if not texts:
            return np.empty((0, self._settings.embedding_dimension), dtype=np.float32)
        request: dict[str, Any] = {
            "model": self._settings.bailian_embedding_model,
            "input": texts,
            "encoding_format": "float",
        }
        if self._settings.bailian_embedding_dimensions_enabled:
            request["dimensions"] = self._settings.embedding_dimension
        response = await self._get_client().embeddings.create(**request)
        ordered = sorted(response.data, key=lambda item: item.index)
        result = np.asarray([item.embedding for item in ordered], dtype=np.float32)
        if result.shape != (len(texts), self._settings.embedding_dimension):
            raise ValueError(
                "Bailian embedding shape mismatch: "
                f"expected {(len(texts), self._settings.embedding_dimension)}, "
                f"got {result.shape}"
            )
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Bailian returned a zero-length embedding vector")
        return result / norms

    def _get_client(self) -> AsyncOpenAI:
        if self._client is not None:
            return self._client
        if not self._settings.bailian_api_key:
            raise ValueError("BAILIAN_API_KEY is required when EMBEDDING_PROVIDER=bailian")
        self._client = AsyncOpenAI(
            api_key=self._settings.bailian_api_key,
            base_url=self._settings.bailian_base_url,
            timeout=self._settings.embedding_request_timeout_seconds,
            max_retries=self._settings.embedding_max_retries,
        )
        return self._client


class ZhipuEmbeddingModel:
    """Zhipu GLM embeddings through its OpenAI-compatible API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncOpenAI | None = None

    async def encode(self, texts: list[str], context: str = "document") -> np.ndarray:
        del context
        if not texts:
            return np.empty((0, self._settings.embedding_dimension), dtype=np.float32)
        request: dict[str, Any] = {
            "model": self._settings.zhipu_embedding_model,
            "input": texts,
        }
        if self._settings.zhipu_embedding_dimensions_enabled:
            request["dimensions"] = self._settings.embedding_dimension
        response = await self._get_client().embeddings.create(**request)
        ordered = sorted(response.data, key=lambda item: item.index)
        result = np.asarray([item.embedding for item in ordered], dtype=np.float32)
        if result.shape != (len(texts), self._settings.embedding_dimension):
            raise ValueError(
                "Zhipu embedding shape mismatch: "
                f"expected {(len(texts), self._settings.embedding_dimension)}, "
                f"got {result.shape}"
            )
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Zhipu returned a zero-length embedding vector")
        return result / norms

    def _get_client(self) -> AsyncOpenAI:
        if self._client is not None:
            return self._client
        if not self._settings.zhipu_api_key:
            raise ValueError("ZHIPU_API_KEY is required when EMBEDDING_PROVIDER=zhipu")
        self._client = AsyncOpenAI(
            api_key=self._settings.zhipu_api_key,
            base_url=self._settings.zhipu_base_url,
            timeout=self._settings.embedding_request_timeout_seconds,
            max_retries=self._settings.embedding_max_retries,
        )
        return self._client


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
    if settings.embedding_provider == "bailian":
        model = BailianEmbeddingModel(settings)
        supports_asymmetric = False
    elif settings.embedding_provider == "zhipu":
        model = ZhipuEmbeddingModel(settings)
        supports_asymmetric = False
    else:
        model = LocalEmbeddingModel(settings)
        supports_asymmetric = True
    return EmbeddingFunc(
        embedding_dim=settings.embedding_dimension,
        max_token_size=3072 if settings.embedding_provider == "zhipu" else 8192,
        model_name=embedding_model_name(settings),
        supports_asymmetric=supports_asymmetric,
        func=partial(model.encode),
    )


def embedding_model_name(settings: Settings) -> str:
    if settings.embedding_provider == "bailian":
        return f"bailian/{settings.bailian_embedding_model}"
    if settings.embedding_provider == "zhipu":
        return f"zhipu/{settings.zhipu_embedding_model}"
    return settings.embedding_model_id


def build_rerank_func(settings: Settings):
    if not settings.reranker_enabled:
        return None
    return LocalReranker(settings).rerank

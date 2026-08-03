from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Callable
from time import perf_counter
from typing import Any

from lightrag import LightRAG, QueryParam
from lightrag.base import DocStatus
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

from rag_graph_demo.config import Settings
from rag_graph_demo.domain.exceptions import ConfigurationError, EngineNotReadyError
from rag_graph_demo.domain.models import Citation, QueryRequest, QueryResponse
from rag_graph_demo.ingestion.source_mapper import SourceMapper
from rag_graph_demo.lightrag_engine.local_models import (
    build_embedding_func,
    build_rerank_func,
)

REFERENCE_LINE_PATTERN = re.compile(r"(?m)^(?P<prefix>\s*[-*]\s*\[(?P<id>\d+)\]\s+).*$")
logger = logging.getLogger(__name__)


def select_answer_citations(
    answer: str | None,
    citations: list[Citation],
) -> list[Citation]:
    if not answer:
        return citations
    reference_ids = {
        match.group("id") for match in REFERENCE_LINE_PATTERN.finditer(answer)
    }
    if not reference_ids:
        return citations
    return [
        citation
        for citation in citations
        if citation.reference_id in reference_ids
    ]


def rewrite_reference_titles(
    answer: str | None,
    citations: list[Citation],
) -> str | None:
    if not answer:
        return answer
    titles = {
        citation.reference_id: citation.title or citation.filename
        for citation in citations
    }

    def replace(match: re.Match[str]) -> str:
        title = titles.get(match.group("id"))
        return f"{match.group('prefix')}{title}" if title else match.group(0)

    return REFERENCE_LINE_PATTERN.sub(replace, answer)


class LightRAGEngine:
    def __init__(
        self,
        settings: Settings,
        source_mapper: SourceMapper,
        *,
        embedding_func: EmbeddingFunc | None = None,
        llm_func_builder: Callable[[str], Callable[..., Any]] | None = None,
    ) -> None:
        self._settings = settings
        self._source_mapper = source_mapper
        self._provided_embedding_func = embedding_func
        self._llm_func_builder = llm_func_builder
        self._rag: LightRAG | None = None
        self._init_lock = asyncio.Lock()
        self._index_lock = asyncio.Lock()
        self._embedding_lock = asyncio.Lock()
        self._embedding_ready = False

    @property
    def initialized(self) -> bool:
        return self._rag is not None

    async def initialize(self) -> LightRAG:
        if self._rag is not None:
            return self._rag
        async with self._init_lock:
            if self._rag is not None:
                return self._rag
            if not self._settings.deepseek_api_key and self._llm_func_builder is None:
                raise ConfigurationError(
                    "DEEPSEEK_API_KEY is not configured. Copy .env.example to .env and add the key."
                )

            self._configure_environment()
            embedding_func = self._provided_embedding_func or build_embedding_func(
                self._settings
            )
            build_llm = self._llm_func_builder or self._build_deepseek_func
            query_llm = build_llm(self._settings.deepseek_chat_model)
            extraction_llm = build_llm(self._settings.deepseek_extraction_model)

            rag = LightRAG(
                working_dir=str(self._settings.lightrag_working_dir),
                workspace=self._settings.lightrag_workspace,
                kv_storage="JsonKVStorage",
                vector_storage="QdrantVectorDBStorage",
                graph_storage="NetworkXStorage",
                doc_status_storage="JsonDocStatusStorage",
                embedding_func=embedding_func,
                rerank_model_func=build_rerank_func(self._settings),
                llm_model_func=query_llm,
                llm_model_name=self._settings.deepseek_chat_model,
                role_llm_configs={
                    "extract": {
                        "func": extraction_llm,
                        "max_async": self._settings.llm_max_concurrency,
                    },
                    "keyword": {
                        "func": extraction_llm,
                        "max_async": self._settings.llm_max_concurrency,
                    },
                    "query": {
                        "func": query_llm,
                        "max_async": self._settings.llm_max_concurrency,
                    },
                },
                llm_model_max_async=self._settings.llm_max_concurrency,
                default_llm_timeout=self._settings.llm_request_timeout_seconds,
                embedding_batch_num=self._settings.embedding_batch_size,
                embedding_func_max_async=1,
                chunk_token_size=self._settings.chunk_target_tokens,
                chunk_overlap_token_size=self._settings.chunk_overlap_tokens,
                top_k=self._settings.lightrag_top_k,
                chunk_top_k=self._settings.lightrag_chunk_top_k,
                max_total_tokens=self._settings.lightrag_max_total_tokens,
                max_parallel_insert=self._settings.graph_index_concurrency,
                entity_extraction_use_json=self._settings.entity_extraction_use_json,
                vector_db_storage_cls_kwargs={
                    "cosine_better_than_threshold": self._settings.lightrag_cosine_threshold
                },
                addon_params={"language": self._settings.lightrag_summary_language},
            )
            await rag.initialize_storages()
            self._rag = rag
        return self._rag

    async def finalize(self) -> None:
        if self._rag is not None:
            await self._rag.finalize_storages()
            self._rag = None
            self._embedding_ready = False

    async def index_document(
        self,
        *,
        document_id: str,
        text: str,
        file_path: str,
    ) -> str:
        rag = await self.initialize()
        async with self._index_lock:
            await self.warmup_embedding()
            track_id = await rag.ainsert(
                input=text,
                ids=document_id,
                file_paths=file_path,
            )
            statuses = await rag.aget_docs_by_track_id(track_id)
            status = statuses.get(document_id)
            if status is None:
                raise EngineNotReadyError(
                    f"LightRAG returned no processing status for {document_id}"
                )
            if status.status != DocStatus.PROCESSED:
                detail = status.error_msg or f"unexpected status: {status.status.value}"
                raise EngineNotReadyError(
                    f"LightRAG indexing failed for {document_id}: {detail}"
                )
            return track_id

    async def warmup_embedding(self) -> None:
        if self._embedding_ready:
            return
        rag = await self.initialize()
        async with self._embedding_lock:
            if self._embedding_ready:
                return
            await rag.embedding_func(
                ["LightRAG embedding readiness check"],
                context="document",
            )
            self._embedding_ready = True

    async def delete_document(self, document_id: str) -> dict[str, Any]:
        rag = await self.initialize()
        result = await rag.adelete_by_doc_id(document_id, delete_llm_cache=True)
        # LightRAG records every rejected duplicate as a separate failed
        # doc-status row.  Those rows survive deletion of the original and keep
        # blocking its filename forever.  Remove only duplicate tombstones that
        # point at this stable document id/file path; leave unrelated failures
        # available for manual review.
        duplicate_ids: list[str] = []
        get_many = getattr(rag.doc_status, "get_docs_by_statuses", None)
        if get_many is not None:
            statuses = await get_many(list(DocStatus))
            expected_prefix = f"{document_id}__"
            for status_id, status in statuses.items():
                metadata = status.metadata or {}
                if not metadata.get("is_duplicate"):
                    continue
                if (
                    metadata.get("original_doc_id") == document_id
                    or status.file_path.startswith(expected_prefix)
                ):
                    duplicate_ids.append(status_id)
        if duplicate_ids:
            await rag.doc_status.delete(duplicate_ids)
            await rag.doc_status.index_done_callback()
            logger.info(
                "Purged %s duplicate status tombstone(s) for %s",
                len(duplicate_ids),
                document_id,
            )
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        return dict(result)

    async def query(
        self, request: QueryRequest, *, user_prompt: str | None = None
    ) -> QueryResponse:
        rag = await self.initialize()
        started = perf_counter()
        enable_rerank = (
            self._settings.reranker_enabled
            if request.enable_rerank is None
            else request.enable_rerank
        )
        result = await rag.aquery_llm(
            request.question,
            QueryParam(
                mode=request.mode,
                stream=False,
                top_k=request.top_k,
                chunk_top_k=request.chunk_top_k,
                max_total_tokens=request.max_total_tokens,
                response_type=request.response_type,
                conversation_history=request.conversation_history,
                user_prompt=user_prompt,
                enable_rerank=enable_rerank,
                include_references=True,
            ),
        )
        elapsed_ms = int((perf_counter() - started) * 1000)
        data = result.get("data") or {}
        llm_response = result.get("llm_response") or {}
        citations = await self._source_mapper.resolve(result)
        answer = llm_response.get("content")
        citations = select_answer_citations(answer, citations)
        answer = rewrite_reference_titles(answer, citations)
        return QueryResponse(
            status=str(result.get("status", "failure")),
            message=str(result.get("message", "")),
            mode=request.mode,
            answer=answer,
            citations=citations,
            entities=list(data.get("entities") or []),
            relationships=list(data.get("relationships") or []),
            chunks=list(data.get("chunks") or []),
            metadata=dict(result.get("metadata") or {}),
            elapsed_ms=elapsed_ms,
        )

    async def complete_extraction(self, prompt: str, system_prompt: str) -> str:
        """Run a small structured extraction without initializing vector storage."""
        if not self._settings.deepseek_api_key and self._llm_func_builder is None:
            raise ConfigurationError("DEEPSEEK_API_KEY is required for event extraction")
        build_llm = self._llm_func_builder or self._build_deepseek_func
        result = await build_llm(self._settings.deepseek_extraction_model)(
            prompt,
            system_prompt=system_prompt,
            response_format={"type": "json_object"},
        )
        if isinstance(result, str):
            return result
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)

    async def get_graph(
        self, label: str = "*", max_depth: int = 2, max_nodes: int = 100
    ) -> dict[str, Any]:
        rag = await self.initialize()
        graph = await rag.get_knowledge_graph(label, max_depth=max_depth, max_nodes=max_nodes)
        return graph.model_dump(mode="json")

    async def get_graph_labels(self) -> list[str]:
        rag = await self.initialize()
        labels = await rag.get_graph_labels()
        return list(labels)

    def _configure_environment(self) -> None:
        os.environ["QDRANT_URL"] = self._settings.qdrant_url
        if self._settings.qdrant_api_key:
            os.environ["QDRANT_API_KEY"] = self._settings.qdrant_api_key
        else:
            os.environ.pop("QDRANT_API_KEY", None)
        bypass = "localhost,127.0.0.1,::1"
        os.environ["NO_PROXY"] = bypass
        os.environ["no_proxy"] = bypass

    def _build_deepseek_func(self, model: str) -> Callable[..., Any]:
        settings = self._settings

        async def complete(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> Any:
            for managed_argument in (
                "model",
                "enable_cot",
                "base_url",
                "api_key",
                "timeout",
            ):
                kwargs.pop(managed_argument, None)
            extra_body = dict(kwargs.pop("extra_body", {}) or {})
            extra_body["thinking"] = {
                "type": (
                    "enabled"
                    if settings.deepseek_thinking_enabled
                    else "disabled"
                )
            }
            kwargs["extra_body"] = extra_body
            kwargs.setdefault("temperature", 0)
            # LightRAG already retries a short burst internally.  This outer loop
            # uses the project's configurable retry budget to survive a longer
            # transient DeepSeek/network interruption without retrying permanent
            # authentication or malformed-request failures.
            attempts = max(1, settings.llm_max_retries)
            for attempt in range(1, attempts + 1):
                try:
                    return await openai_complete_if_cache(
                        model,
                        prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages or [],
                        enable_cot=settings.deepseek_thinking_enabled,
                        base_url=settings.deepseek_base_url,
                        api_key=settings.deepseek_api_key,
                        timeout=settings.llm_request_timeout_seconds,
                        **kwargs,
                    )
                except (
                    APIConnectionError,
                    APITimeoutError,
                    InternalServerError,
                    RateLimitError,
                ) as exc:
                    if attempt >= attempts:
                        raise
                    delay = min(2 ** (attempt - 1), 10)
                    logger.warning(
                        "Transient DeepSeek error (%s); retrying in %ss (%s/%s)",
                        type(exc).__name__,
                        delay,
                        attempt,
                        attempts,
                    )
                    await asyncio.sleep(delay)

            raise RuntimeError("unreachable DeepSeek retry state")

        return complete

    def require_initialized(self) -> LightRAG:
        if self._rag is None:
            raise EngineNotReadyError("LightRAG is not initialized")
        return self._rag

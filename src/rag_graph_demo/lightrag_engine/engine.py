from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Callable
from time import perf_counter
from typing import Any, Literal

from lightrag import LightRAG, QueryParam
from lightrag.base import DocStatus
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

from rag_graph_demo.config import Settings
from rag_graph_demo.domain.exceptions import ConfigurationError, EngineNotReadyError
from rag_graph_demo.domain.models import Citation, QueryRequest, QueryResponse
from rag_graph_demo.domain.tkg import (
    SemanticEntityCandidate,
    SemanticExtractionBatch,
    SemanticRelationshipCandidate,
)
from rag_graph_demo.ingestion.source_mapper import SourceMapper
from rag_graph_demo.lightrag_engine.local_models import (
    build_embedding_func,
    build_rerank_func,
)

REFERENCE_LINE_PATTERN = re.compile(r"(?m)^(?P<prefix>\s*[-*]\s*\[(?P<id>\d+)\]\s+).*$")
logger = logging.getLogger(__name__)
ExtractionProfile = Literal["default", "light_entity"]


def _first_source_id(value: Any) -> str | None:
    source = str(value or "").split("<SEP>", maxsplit=1)[0].strip()
    return source or None


def _source_ids(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split("<SEP>") if item.strip()]


def _source_evidence(value: dict[str, Any], chunks: dict[str, str]) -> str:
    return next(
        (
            chunks[source_id][:4_000]
            for source_id in _source_ids(value.get("source_id"))
            if source_id in chunks
        ),
        "",
    )


def _source_segment_id(value: dict[str, Any], chunks: dict[str, str]) -> str | None:
    return next(
        (
            source_id
            for source_id in _source_ids(value.get("source_id"))
            if source_id in chunks
        ),
        None,
    )


def _bounded_relationship_predicate(value: Any) -> str:
    """Convert LightRAG's accumulated keyword bag into a valid predicate hint."""
    return str(value or "relatedFact").strip()[:200] or "relatedFact"


def _doc_status_value(status: Any) -> DocStatus:
    value = status.get("status") if isinstance(status, dict) else status.status
    return value if isinstance(value, DocStatus) else DocStatus(str(value))


def _doc_status_error(status: Any) -> str | None:
    value = (
        status.get("error_msg") if isinstance(status, dict) else getattr(status, "error_msg", None)
    )
    return str(value) if value else None


def limit_controlled_extraction_result(
    value: Any,
    *,
    max_entities: int,
    max_relationships: int,
) -> Any:
    """Hard-cap controlled extraction after the LLM and before LightRAG parsing."""
    was_string = isinstance(value, str)
    if was_string:
        cleaned = re.sub(r"^```(?:json)?\s*", "", value.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            return value
    elif isinstance(value, dict):
        payload = dict(value)
    else:
        return value
    if not isinstance(payload, dict):
        return value

    raw_entities = payload.get("entities")
    raw_relationships = payload.get("relationships")
    if not isinstance(raw_entities, list) or not isinstance(raw_relationships, list):
        return value

    entities = [item for item in raw_entities if isinstance(item, dict)][:max_entities]
    retained_names = {
        str(item.get("name") or "").strip().casefold()
        for item in entities
        if str(item.get("name") or "").strip()
    }
    relationships = [
        item
        for item in raw_relationships
        if isinstance(item, dict)
        and str(item.get("source") or "").strip().casefold() in retained_names
        and str(item.get("target") or "").strip().casefold() in retained_names
    ][:max_relationships]
    payload["entities"] = entities
    payload["relationships"] = relationships
    return json.dumps(payload, ensure_ascii=False) if was_string else payload


def select_answer_citations(
    answer: str | None,
    citations: list[Citation],
) -> list[Citation]:
    if not answer:
        return citations
    reference_ids = {match.group("id") for match in REFERENCE_LINE_PATTERN.finditer(answer)}
    if not reference_ids:
        return citations
    return [citation for citation in citations if citation.reference_id in reference_ids]


def rewrite_reference_titles(
    answer: str | None,
    citations: list[Citation],
) -> str | None:
    if not answer:
        return answer
    titles = {citation.reference_id: citation.title or citation.filename for citation in citations}

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
        extraction_profile: ExtractionProfile = "default",
    ) -> None:
        self._settings = settings
        self._source_mapper = source_mapper
        self._provided_embedding_func = embedding_func
        self._llm_func_builder = llm_func_builder
        self._extraction_profile = extraction_profile
        self._rag: LightRAG | None = None
        self._init_lock = asyncio.Lock()
        self._index_lock = asyncio.Lock()
        self._embedding_lock = asyncio.Lock()
        self._direct_extraction_semaphore = asyncio.Semaphore(
            self._settings.llm_max_concurrency
        )
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
            embedding_func = self._provided_embedding_func or build_embedding_func(self._settings)
            build_llm = self._llm_func_builder or self._build_deepseek_func
            query_llm = build_llm(self._settings.deepseek_chat_model)
            extraction_llm = build_llm(self._settings.deepseek_extraction_model)
            if self._extraction_profile == "light_entity":
                base_extraction_llm = extraction_llm

                async def controlled_extraction_llm(*args: Any, **kwargs: Any) -> Any:
                    result = await base_extraction_llm(*args, **kwargs)
                    return limit_controlled_extraction_result(
                        result,
                        max_entities=self._settings.light_entity_max_entities,
                        max_relationships=(
                            self._settings.light_entity_max_relationships
                        ),
                    )

                extraction_llm = controlled_extraction_llm

            addon_params: dict[str, Any] = {"language": self._settings.lightrag_summary_language}
            extraction_limits: dict[str, int] = {}
            if self._extraction_profile == "light_entity":
                prompt_file = self._settings.light_entity_prompt_file.resolve()
                if not prompt_file.is_file():
                    raise ConfigurationError(
                        f"LIGHT_ENTITY_PROMPT_FILE does not exist: {prompt_file}"
                    )
                if prompt_file.parent.name != "entity_type":
                    raise ConfigurationError(
                        "LIGHT_ENTITY_PROMPT_FILE must be inside a PROMPT_DIR/entity_type "
                        f"directory: {prompt_file}"
                    )
                os.environ["PROMPT_DIR"] = str(prompt_file.parent.parent)
                addon_params["entity_type_prompt_file"] = prompt_file.name
                extraction_limits = {
                    "entity_extract_max_gleaning": self._settings.light_entity_max_gleaning,
                    "entity_extract_max_records": self._settings.light_entity_max_records,
                    "entity_extract_max_entities": self._settings.light_entity_max_entities,
                }

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
                embedding_func_max_async=self._settings.effective_embedding_concurrency,
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
                addon_params=addon_params,
                **extraction_limits,
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
                raise EngineNotReadyError(f"LightRAG indexing failed for {document_id}: {detail}")
            return track_id

    async def index_documents(
        self,
        documents: list[tuple[str, str, str]],
    ) -> str:
        """Index a resumable batch of pre-chunked evaluation documents."""
        if not documents:
            raise ValueError("documents must not be empty")
        rag = await self.initialize()
        async with self._index_lock:
            await self.warmup_embedding()
            pending_documents = documents
            doc_status = getattr(rag, "doc_status", None)
            if doc_status is not None:
                stored = await doc_status.get_by_ids([item[0] for item in documents])
                if any(
                    value is not None and _doc_status_value(value) != DocStatus.PROCESSED
                    for value in stored
                ):
                    await rag.apipeline_process_enqueue_documents()
                    stored = await doc_status.get_by_ids([item[0] for item in documents])
                pending_documents = [
                    document
                    for document, value in zip(documents, stored, strict=True)
                    if value is None
                ]

            track_id = "already-processed"
            if pending_documents:
                track_id = await rag.ainsert(
                    input=[item[1] for item in pending_documents],
                    ids=[item[0] for item in pending_documents],
                    file_paths=[item[2] for item in pending_documents],
                )

            if doc_status is not None:
                status_values = await doc_status.get_by_ids([item[0] for item in documents])
                statuses = dict(
                    zip(
                        [item[0] for item in documents],
                        status_values,
                        strict=True,
                    )
                )
            else:
                statuses = await rag.aget_docs_by_track_id(track_id)
            failures: list[str] = []
            for document_id, _, _ in documents:
                status = statuses.get(document_id)
                if status is None:
                    failures.append(f"{document_id}: missing status")
                elif _doc_status_value(status) != DocStatus.PROCESSED:
                    failures.append(
                        f"{document_id}: "
                        f"{_doc_status_error(status) or _doc_status_value(status).value}"
                    )
            if failures:
                preview = "; ".join(failures[:5])
                raise EngineNotReadyError(
                    f"LightRAG batch indexing failed for {len(failures)} item(s): {preview}"
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
                if metadata.get("original_doc_id") == document_id or status.file_path.startswith(
                    expected_prefix
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

    async def retrieve_data(
        self,
        question: str,
        *,
        mode: str = "mix",
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        max_total_tokens: int | None = None,
        enable_rerank: bool | None = None,
        high_level_keywords: list[str] | None = None,
        low_level_keywords: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return structured LightRAG retrieval data without answer generation."""
        rag = await self.initialize()
        return await rag.aquery_data(
            question,
            QueryParam(
                mode=mode,
                stream=False,
                top_k=top_k or self._settings.lightrag_top_k,
                chunk_top_k=chunk_top_k or self._settings.lightrag_chunk_top_k,
                max_total_tokens=max_total_tokens or self._settings.lightrag_max_total_tokens,
                hl_keywords=high_level_keywords or [],
                ll_keywords=low_level_keywords or [],
                enable_rerank=(
                    self._settings.reranker_enabled if enable_rerank is None else enable_rerank
                ),
            ),
        )

    async def embed_texts(self, texts: list[str], context: str) -> Any:
        """Expose the configured embedding model to sibling retrieval indexes."""
        rag = await self.initialize()
        return await rag.embedding_func(texts, context=context)

    async def get_document_semantic_extraction(
        self, document_id: str
    ) -> SemanticExtractionBatch:
        """Read the entities/relations that LightRAG already extracted for one document.

        This is intentionally post-index: the TKG layer consumes the same semantic
        extraction as the existing graph path and does not make a duplicate LLM call.
        """
        rag = await self.initialize()
        graph = rag.chunk_entity_relation_graph
        file_marker = f"{document_id}__"
        nodes = await graph.get_all_nodes()
        edges = await graph.get_all_edges()
        source_ids = {
            source_id
            for value in [*nodes, *edges]
            for source_id in _source_ids(value.get("source_id"))
        }
        sorted_source_ids = sorted(source_ids)
        chunks = (
            await rag.text_chunks.get_by_ids(sorted_source_ids)
            if sorted_source_ids
            else []
        )
        chunk_content = {
            source_id: str(chunk.get("content") or "")
            for source_id, chunk in zip(sorted_source_ids, chunks, strict=True)
            if chunk
            and file_marker in str(chunk.get("file_path") or "")
            and str(chunk.get("content") or "").strip()
        }
        document_nodes = [
            node for node in nodes if file_marker in str(node.get("file_path") or "")
        ]
        retained_names = {
            str(node.get("id") or node.get("entity_id") or "")
            for node in document_nodes
        }
        entities = [
            SemanticEntityCandidate(
                name=str(node.get("id") or node.get("entity_id") or ""),
                entity_type=str(node.get("entity_type") or "Concept"),
                description=_source_evidence(node, chunk_content)
                or str(node.get("description") or node.get("id") or ""),
                source_segment_id=_source_segment_id(node, chunk_content)
                or _first_source_id(node.get("source_id")),
            )
            for node in document_nodes
            if str(node.get("id") or node.get("entity_id") or "").strip()
        ]
        relationships = []
        for edge in edges:
            if file_marker not in str(edge.get("file_path") or ""):
                continue
            source = str(edge.get("source") or "")
            target = str(edge.get("target") or "")
            if source not in retained_names or target not in retained_names:
                continue
            description = _source_evidence(edge, chunk_content)
            if not description:
                description = str(edge.get("description") or "").strip()
            if not description:
                continue
            relationships.append(
                SemanticRelationshipCandidate(
                    source=source,
                    target=target,
                    # LightRAG stores a comma-separated keyword bag on an edge,
                    # while the governed graph expects one bounded predicate hint.
                    # Preserve enough keywords for canonicalization without letting
                    # an accumulated cross-document bag violate the domain schema.
                    predicate=_bounded_relationship_predicate(edge.get("keywords")),
                    description=description,
                    confidence=min(1.0, max(0.0, float(edge.get("weight") or 0.65))),
                    source_segment_id=_source_segment_id(edge, chunk_content)
                    or _first_source_id(edge.get("source_id")),
                )
            )
        return SemanticExtractionBatch(entities=entities, relationships=relationships)

    async def complete_query(
        self,
        prompt: str,
        *,
        system_prompt: str,
        history_messages: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate from an externally assembled, already-retrieved evidence context."""
        if not self._settings.deepseek_api_key and self._llm_func_builder is None:
            raise ConfigurationError("DEEPSEEK_API_KEY is required for answer generation")
        build_llm = self._llm_func_builder or self._build_deepseek_func
        result = await build_llm(self._settings.deepseek_chat_model)(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
        )
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            return str(result.get("content") or json.dumps(result, ensure_ascii=False))
        return str(result)

    async def complete_extraction(self, prompt: str, system_prompt: str) -> str:
        """Run a small structured extraction without initializing vector storage."""
        if not self._settings.deepseek_api_key and self._llm_func_builder is None:
            raise ConfigurationError("DEEPSEEK_API_KEY is required for event extraction")
        build_llm = self._llm_func_builder or self._build_deepseek_func
        async with self._direct_extraction_semaphore:
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
                "type": ("enabled" if settings.deepseek_thinking_enabled else "disabled")
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

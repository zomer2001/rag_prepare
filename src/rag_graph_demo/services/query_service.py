from __future__ import annotations

import logging
import re
from time import perf_counter

from rag_graph_demo.domain.events import EventQueryResponse
from rag_graph_demo.domain.models import (
    Citation,
    CompareQueryRequest,
    QueryRequest,
    QueryResponse,
)
from rag_graph_demo.lightrag_engine.engine import LightRAGEngine
from rag_graph_demo.retrieval.models import EvidenceBundle, UnifiedRetrievalResult
from rag_graph_demo.retrieval.orchestrator import UnifiedRetrievalOrchestrator
from rag_graph_demo.services.event_service import EventService, event_evidence_labels
from rag_graph_demo.storage.document_registry import DocumentRegistry

logger = logging.getLogger(__name__)
REFERENCE_SECTION_PATTERN = re.compile(
    r"\n#{2,4}\s+References\s*\n.*$", re.IGNORECASE | re.DOTALL
)
UNIFIED_ANSWER_SYSTEM_PROMPT = """你是证据约束型问答助手。只能依据给定的原文证据回答，
不得用自身知识补充事实。每个事实必须使用对应的 [数字] 文档编号引用；证据不足时明确说明。
事件的时间先后不自动表示因果；存在冲突、否认或不确定状态时必须明确呈现。"""


class QueryService:
    def __init__(
        self,
        engine: LightRAGEngine,
        event_service: EventService | None = None,
        documents: DocumentRegistry | None = None,
        orchestrator: UnifiedRetrievalOrchestrator | None = None,
    ) -> None:
        self._engine = engine
        self._event_service = event_service
        self._documents = documents
        self._orchestrator = orchestrator

    async def query(self, request: QueryRequest) -> QueryResponse:
        if request.mode == "mix" and self._orchestrator is not None:
            return await self._query_unified(request)
        event_context = ""
        event_result = None
        if request.use_events and self._event_service is not None:
            try:
                event_context, event_result = await self._event_service.build_query_context(
                    request.question, request.event_limit
                )
            except Exception:
                # Event data is an assertion/validation layer. A damaged or
                # temporarily unavailable event store must not take down base RAG.
                logger.exception("Event pathway unavailable; falling back to base RAG")
        response = await self._engine.query(
            request, user_prompt=event_context or None
        )
        if event_result is not None:
            had_base_citations = bool(response.citations)
            response.events = [
                item.model_dump(mode="json") for item in event_result.events
            ]
            response.event_relations = [
                item.model_dump(mode="json") for item in event_result.relations
            ]
            response.event_conflicts = [
                item.model_dump(mode="json") for item in event_result.conflicts
            ]
            event_citations = await self._event_citations(event_result)
            response.citations.extend(event_citations)
            if event_citations and not had_base_citations and response.answer:
                response.answer = REFERENCE_SECTION_PATTERN.sub("", response.answer).rstrip()
            response.metadata["event_pathway"] = {
                "enabled": True,
                "event_count": len(event_result.events),
                "relation_count": len(event_result.relations),
                "conflict_count": len(event_result.conflicts),
                "inferred_start_time": event_result.inferred_start_time,
                "inferred_end_time": event_result.inferred_end_time,
            }
        else:
            response.metadata["event_pathway"] = {"enabled": False}
        return response

    async def _query_unified(self, request: QueryRequest) -> QueryResponse:
        started = perf_counter()
        retrieval = await self._orchestrator.retrieve(
            request.question,
            top_k=request.chunk_top_k,
            include_events=request.use_events,
        )
        context, citations = await self._build_unified_context(retrieval)
        prompt = (
            f"问题：{request.question}\n\n"
            f"以下为统一三路检索后选出的证据：\n{context}\n\n"
            "请直接回答问题，并在相关事实后标注对应的 [数字] 引用。"
        )
        answer = await self._engine.complete_query(
            prompt,
            system_prompt=UNIFIED_ANSWER_SYSTEM_PROMPT,
            history_messages=request.conversation_history,
        )
        event_result = retrieval.event_result
        return QueryResponse(
            status="success",
            message="",
            mode=request.mode,
            answer=answer,
            citations=citations,
            entities=list(retrieval.graph_data.get("entities") or []),
            relationships=list(retrieval.graph_data.get("relationships") or []),
            chunks=[self._bundle_payload(item) for item in retrieval.bundles],
            events=(
                [item.model_dump(mode="json") for item in event_result.events]
                if event_result
                else []
            ),
            event_relations=(
                [item.model_dump(mode="json") for item in event_result.relations]
                if event_result
                else []
            ),
            event_conflicts=(
                [item.model_dump(mode="json") for item in event_result.conflicts]
                if event_result
                else []
            ),
            metadata={
                **retrieval.metadata,
                "query_plan": retrieval.plan.model_dump(mode="json"),
                "retrieval_channels": [
                    "vector",
                    "graph",
                    *(("event",) if request.use_events else ()),
                ],
                "reranker_enabled": False,
            },
            elapsed_ms=int((perf_counter() - started) * 1_000),
        )

    async def _build_unified_context(
        self,
        retrieval: UnifiedRetrievalResult,
    ) -> tuple[str, list[Citation]]:
        lines: list[str] = []
        citations: list[Citation] = []
        for reference, bundle in enumerate(retrieval.bundles, start=1):
            document = (
                await self._documents.get(bundle.document_id)
                if self._documents is not None
                else None
            )
            title = document.title if document else bundle.document_id
            filename = document.filename if document else bundle.document_id
            file_path = document.raw_path if document else (
                next((item.file_path for item in bundle.passages if item.file_path), "unknown")
            )
            lines.append(
                f"[{reference}] {title}｜命中通路:{','.join(bundle.channels)}"
            )
            excerpts: list[str] = []
            for passage in bundle.passages:
                excerpt = passage.text.strip()[:1_500]
                excerpts.append(excerpt)
                event_description = ""
                if passage.channel == "event":
                    event_description = (
                        f"｜事件:{passage.metadata.get('event_time') or '时间未知'} "
                        f"{passage.metadata.get('event_type') or ''} "
                        f"{passage.metadata.get('event_summary') or ''} "
                        f"状态={passage.metadata.get('event_status') or 'unknown'}"
                    )
                lines.append(
                    f"- {passage.channel}｜chunk={passage.chunk_id or '-'}"
                    f"{event_description}｜{excerpt}"
                )
            citations.append(
                Citation(
                    reference_id=str(reference),
                    document_id=bundle.document_id,
                    title=title,
                    filename=filename,
                    file_path=file_path,
                    excerpt="\n\n".join(excerpts)[:1_000] or None,
                )
            )
        if retrieval.event_result and retrieval.event_result.relations:
            event_by_id = {
                item.event_id: item for item in retrieval.event_result.events
            }
            lines.append("事件链（先后不自动表示因果）：")
            for relation in retrieval.event_result.relations:
                source = event_by_id.get(relation.source_event_id)
                target = event_by_id.get(relation.target_event_id)
                if source and target:
                    lines.append(
                        f"- {source.start_time or '时间未知'} {source.summary} "
                        f"--{relation.relation_type}--> "
                        f"{target.start_time or '时间未知'} {target.summary}"
                    )
        if retrieval.event_result and retrieval.event_result.conflicts:
            lines.append(
                "冲突提示："
                + "；".join(
                    item.description for item in retrieval.event_result.conflicts
                )
            )
        return "\n".join(lines), citations

    @staticmethod
    def _bundle_payload(bundle: EvidenceBundle) -> dict[str, object]:
        return {
            "document_id": bundle.document_id,
            "score": bundle.score,
            "channels": list(bundle.channels),
            "passages": [
                {
                    "channel": item.channel,
                    "rank": item.rank,
                    "chunk_id": item.chunk_id,
                    "content": item.text,
                    "file_path": item.file_path,
                    "metadata": item.metadata,
                }
                for item in bundle.passages
            ],
        }

    async def _event_citations(
        self, event_result: EventQueryResponse
    ) -> list[Citation]:
        if self._documents is None:
            return []
        citations: list[Citation] = []
        for label, mention in event_evidence_labels(event_result):
            document = await self._documents.get(mention.document_id)
            if document is None:
                continue
            citations.append(
                Citation(
                    reference_id=label,
                    document_id=document.document_id,
                    title=document.title,
                    filename=document.filename,
                    file_path=document.raw_path,
                    section="事件证据",
                    excerpt=mention.evidence,
                )
            )
        return citations

    async def compare(self, request: CompareQueryRequest) -> list[QueryResponse]:
        responses: list[QueryResponse] = []
        for mode in request.modes:
            responses.append(
                await self.query(
                    QueryRequest(
                        question=request.question,
                        mode=mode,
                        top_k=request.top_k,
                        chunk_top_k=request.chunk_top_k,
                        enable_rerank=request.enable_rerank,
                    )
                )
            )
        return responses

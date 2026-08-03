from __future__ import annotations

import logging
import re

from rag_graph_demo.domain.events import EventQueryResponse
from rag_graph_demo.domain.models import (
    Citation,
    CompareQueryRequest,
    QueryRequest,
    QueryResponse,
)
from rag_graph_demo.lightrag_engine.engine import LightRAGEngine
from rag_graph_demo.services.event_service import EventService, event_evidence_labels
from rag_graph_demo.storage.document_registry import DocumentRegistry

logger = logging.getLogger(__name__)
REFERENCE_SECTION_PATTERN = re.compile(
    r"\n#{2,4}\s+References\s*\n.*$", re.IGNORECASE | re.DOTALL
)


class QueryService:
    def __init__(
        self,
        engine: LightRAGEngine,
        event_service: EventService | None = None,
        documents: DocumentRegistry | None = None,
    ) -> None:
        self._engine = engine
        self._event_service = event_service
        self._documents = documents

    async def query(self, request: QueryRequest) -> QueryResponse:
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

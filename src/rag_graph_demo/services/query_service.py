from __future__ import annotations

import logging

from rag_graph_demo.domain.models import (
    CompareQueryRequest,
    QueryRequest,
    QueryResponse,
)
from rag_graph_demo.lightrag_engine.engine import LightRAGEngine
from rag_graph_demo.services.event_service import EventService

logger = logging.getLogger(__name__)


class QueryService:
    def __init__(
        self, engine: LightRAGEngine, event_service: EventService | None = None
    ) -> None:
        self._engine = engine
        self._event_service = event_service

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
            response.events = [
                item.model_dump(mode="json") for item in event_result.events
            ]
            response.event_relations = [
                item.model_dump(mode="json") for item in event_result.relations
            ]
            response.event_conflicts = [
                item.model_dump(mode="json") for item in event_result.conflicts
            ]
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

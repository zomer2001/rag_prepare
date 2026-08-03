from __future__ import annotations

from rag_graph_demo.domain.models import (
    CompareQueryRequest,
    QueryRequest,
    QueryResponse,
)
from rag_graph_demo.lightrag_engine.engine import LightRAGEngine


class QueryService:
    def __init__(self, engine: LightRAGEngine) -> None:
        self._engine = engine

    async def query(self, request: QueryRequest) -> QueryResponse:
        return await self._engine.query(request)

    async def compare(self, request: CompareQueryRequest) -> list[QueryResponse]:
        responses: list[QueryResponse] = []
        for mode in request.modes:
            responses.append(
                await self._engine.query(
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


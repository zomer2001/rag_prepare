from __future__ import annotations

from fastapi import APIRouter, Depends

from rag_graph_demo.api.dependencies import get_container
from rag_graph_demo.domain.models import (
    CompareQueryRequest,
    QueryRequest,
    QueryResponse,
)
from rag_graph_demo.services.container import AppContainer

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    container: AppContainer = Depends(get_container),
) -> QueryResponse:
    return await container.query_service.query(request)


@router.post("/query/compare", response_model=list[QueryResponse])
async def compare_query_modes(
    request: CompareQueryRequest,
    container: AppContainer = Depends(get_container),
) -> list[QueryResponse]:
    return await container.query_service.compare(request)


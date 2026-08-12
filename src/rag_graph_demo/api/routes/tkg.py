from fastapi import APIRouter, Depends

from rag_graph_demo.api.dependencies import get_container
from rag_graph_demo.domain.tkg import TKGQueryRequest, TKGQueryResponse, TKGSnapshot
from rag_graph_demo.services.container import AppContainer

router = APIRouter()


@router.get("/tkg", response_model=TKGSnapshot)
async def get_tkg_snapshot(
    container: AppContainer = Depends(get_container),
) -> TKGSnapshot:
    return await container.tkg_registry.snapshot()


@router.post("/tkg/query", response_model=TKGQueryResponse)
async def query_tkg(
    request: TKGQueryRequest,
    container: AppContainer = Depends(get_container),
) -> TKGQueryResponse:
    return await container.tkg_registry.query(request)

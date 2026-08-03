from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from rag_graph_demo.api.dependencies import get_container
from rag_graph_demo.services.container import AppContainer

router = APIRouter()


@router.get("/graph")
async def get_graph(
    label: str = Query(default="*"),
    max_depth: int = Query(default=2, ge=0, le=5),
    max_nodes: int = Query(default=100, ge=1, le=500),
    container: AppContainer = Depends(get_container),
) -> dict[str, object]:
    return await container.engine.get_graph(label, max_depth, max_nodes)


@router.get("/graph/labels")
async def get_graph_labels(
    container: AppContainer = Depends(get_container),
) -> dict[str, list[str]]:
    return {"labels": await container.engine.get_graph_labels()}


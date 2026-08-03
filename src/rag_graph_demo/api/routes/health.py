from __future__ import annotations

from fastapi import APIRouter, Response, status

from rag_graph_demo import __version__
from rag_graph_demo.services.diagnostics import build_diagnostics

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.get("/ready")
async def ready(response: Response) -> dict[str, object]:
    diagnostics = await build_diagnostics(check_qdrant=True)
    is_ready = diagnostics["summary"]["ready"]
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return diagnostics


@router.get("/diagnostics")
async def diagnostics() -> dict[str, object]:
    return await build_diagnostics(check_qdrant=True)


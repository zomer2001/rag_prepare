from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from rag_graph_demo import __version__
from rag_graph_demo.api.routes.documents import router as documents_router
from rag_graph_demo.api.routes.graph import router as graph_router
from rag_graph_demo.api.routes.health import router as health_router
from rag_graph_demo.api.routes.jobs import router as jobs_router
from rag_graph_demo.api.routes.query import router as query_router
from rag_graph_demo.config import get_settings
from rag_graph_demo.domain.exceptions import (
    ApplicationError,
    ConfigurationError,
    DocumentNotFoundError,
    EngineNotReadyError,
    UnsupportedDocumentError,
    UploadTooLargeError,
)
from rag_graph_demo.services.container import AppContainer


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.container = AppContainer(get_settings())
    try:
        yield
    finally:
        await app.state.container.shutdown()


app = FastAPI(
    title="RAG Graph Demo API",
    description="LightRAG, Qdrant, Docling, and DeepSeek local demo",
    version=__version__,
    lifespan=lifespan,
)
app.include_router(health_router, prefix="/api/v1", tags=["system"])
app.include_router(documents_router, prefix="/api/v1", tags=["documents"])
app.include_router(jobs_router, prefix="/api/v1", tags=["jobs"])
app.include_router(query_router, prefix="/api/v1", tags=["query"])
app.include_router(graph_router, prefix="/api/v1", tags=["graph"])


@app.exception_handler(ApplicationError)
async def application_error_handler(_, exc: ApplicationError) -> JSONResponse:
    status_code = status.HTTP_400_BAD_REQUEST
    if isinstance(exc, DocumentNotFoundError):
        status_code = status.HTTP_404_NOT_FOUND
    elif isinstance(exc, UnsupportedDocumentError):
        status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    elif isinstance(exc, UploadTooLargeError):
        status_code = status.HTTP_413_CONTENT_TOO_LARGE
    elif isinstance(exc, (ConfigurationError, EngineNotReadyError)):
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=status_code, content={"detail": str(exc)})


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"name": "rag-graph-demo", "version": __version__, "docs": "/docs"}

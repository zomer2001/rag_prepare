from __future__ import annotations

from rag_graph_demo.config import Settings
from rag_graph_demo.ingestion.parser import DocumentParser
from rag_graph_demo.ingestion.source_mapper import SourceMapper
from rag_graph_demo.lightrag_engine.engine import LightRAGEngine
from rag_graph_demo.services.document_service import DocumentService
from rag_graph_demo.services.query_service import QueryService
from rag_graph_demo.storage.document_registry import DocumentRegistry
from rag_graph_demo.storage.file_store import FileStore
from rag_graph_demo.storage.job_registry import JobRegistry


class AppContainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.ensure_data_dirs()
        self.documents_registry = DocumentRegistry(
            settings.data_dir / "registry" / "documents.json"
        )
        self.jobs_registry = JobRegistry(settings.data_dir / "registry" / "jobs.json")
        self.file_store = FileStore(settings.data_dir)
        self.source_mapper = SourceMapper(self.documents_registry)
        self.engine = LightRAGEngine(settings, self.source_mapper)
        self.document_service = DocumentService(
            self.file_store,
            self.documents_registry,
            self.jobs_registry,
            DocumentParser(ocr_enabled=settings.parser_ocr_enabled),
            self.engine,
        )
        self.query_service = QueryService(self.engine)

    async def shutdown(self) -> None:
        await self.engine.finalize()

from __future__ import annotations

import json
import uuid
from contextlib import suppress
from pathlib import Path

from rag_graph_demo.domain.exceptions import (
    ConfigurationError,
    DocumentNotFoundError,
)
from rag_graph_demo.domain.models import (
    DocumentRecord,
    DocumentStatus,
    JobRecord,
    JobStatus,
)
from rag_graph_demo.ingestion.parser import DocumentParser
from rag_graph_demo.lightrag_engine.engine import LightRAGEngine
from rag_graph_demo.storage.document_registry import DocumentRegistry
from rag_graph_demo.storage.file_store import FileStore, sanitize_filename
from rag_graph_demo.storage.job_registry import JobRegistry


class DocumentService:
    def __init__(
        self,
        file_store: FileStore,
        registry: DocumentRegistry,
        jobs: JobRegistry,
        parser: DocumentParser,
        engine: LightRAGEngine,
    ) -> None:
        self._file_store = file_store
        self._registry = registry
        self._jobs = jobs
        self._parser = parser
        self._engine = engine

    async def create_upload(
        self,
        *,
        filename: str,
        mime_type: str | None,
        content: bytes,
    ) -> tuple[DocumentRecord, bool]:
        checksum = self._file_store.checksum(content)
        duplicate = await self._registry.find_by_checksum(checksum)
        if duplicate is not None:
            return duplicate, True

        document_id = str(uuid.uuid4())
        safe_name = sanitize_filename(filename)
        suffix = self._file_store.validate_suffix(safe_name)
        raw_path = self._file_store.save_upload(document_id, safe_name, content)
        record = DocumentRecord(
            document_id=document_id,
            filename=safe_name,
            title=Path(safe_name).stem,
            suffix=suffix,
            mime_type=mime_type,
            size_bytes=len(content),
            checksum=checksum,
            raw_path=str(raw_path),
        )
        return await self._registry.upsert(record), False

    async def create_job(self, document_id: str, operation: str) -> JobRecord:
        record = JobRecord(
            job_id=str(uuid.uuid4()),
            document_id=document_id,
            operation=operation,
        )
        return await self._jobs.upsert(record)

    async def index_document(self, document_id: str, job_id: str) -> None:
        await self._jobs.update(job_id, JobStatus.RUNNING, "Parsing document")
        record = await self.require_document(document_id)
        try:
            await self._registry.update_status(document_id, DocumentStatus.PARSING)
            parsed = await self._parser.parse(
                document_id,
                Path(record.raw_path),
                record.filename,
            )
            parsed_path, metadata_path = self._file_store.parsed_paths(document_id)
            parsed_path.write_text(parsed.text, encoding="utf-8")
            metadata_path.write_text(
                json.dumps(parsed.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            await self._registry.update_status(
                document_id,
                DocumentStatus.PARSED,
                title=parsed.title,
                parsed_path=str(parsed_path.resolve()),
                metadata_path=str(metadata_path.resolve()),
                page_count=parsed.page_count,
            )
            await self._jobs.update(job_id, JobStatus.RUNNING, "Building LightRAG index")
            await self._registry.update_status(document_id, DocumentStatus.INDEXING)
            track_id = await self._engine.index_document(
                document_id=document_id,
                text=parsed.text,
                file_path=f"{document_id}__{record.filename}",
            )
            await self._registry.update_status(
                document_id,
                DocumentStatus.READY,
                lightrag_track_id=track_id,
            )
            await self._jobs.update(job_id, JobStatus.SUCCEEDED, "Document is ready")
        except ConfigurationError as exc:
            await self._registry.update_status(
                document_id,
                DocumentStatus.PARSED,
                error=str(exc),
            )
            await self._jobs.update(job_id, JobStatus.FAILED, str(exc))
        except Exception as exc:
            await self._registry.update_status(
                document_id,
                DocumentStatus.FAILED,
                error=str(exc),
            )
            await self._jobs.update(job_id, JobStatus.FAILED, str(exc))

    async def reindex_document(self, document_id: str, job_id: str) -> None:
        await self.require_document(document_id)
        # Failed/interrupted LightRAG inserts still leave doc-status and chunk
        # records behind.  If they are not purged, a retry is classified as a
        # duplicate and never processes the document again.  Reindex is an
        # explicit rebuild operation, so always attempt an idempotent cleanup.
        with suppress(Exception):
            await self._engine.delete_document(document_id)
        await self.index_document(document_id, job_id)

    async def delete_document(self, document_id: str) -> dict[str, object]:
        record = await self.require_document(document_id)
        await self._registry.update_status(document_id, DocumentStatus.DELETING)
        engine_result: dict[str, object] = {"status": "not_initialized"}
        if self._engine.initialized or record.status == DocumentStatus.READY:
            engine_result = await self._engine.delete_document(document_id)
        self._file_store.delete_document_files(document_id)
        await self._registry.delete(document_id)
        return {"document_id": document_id, "engine": engine_result}

    async def list_documents(self) -> list[DocumentRecord]:
        return await self._registry.list()

    async def require_document(self, document_id: str) -> DocumentRecord:
        record = await self._registry.get(document_id)
        if record is None:
            raise DocumentNotFoundError(f"Document not found: {document_id}")
        return record

    async def read_parsed_content(self, document_id: str) -> str:
        record = await self.require_document(document_id)
        if not record.parsed_path:
            return ""
        path = Path(record.parsed_path)
        return path.read_text(encoding="utf-8") if path.exists() else ""

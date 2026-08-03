from pathlib import Path

import pytest

from rag_graph_demo.domain.models import DocumentStatus, JobStatus
from rag_graph_demo.ingestion.parser import DocumentParser
from rag_graph_demo.services.document_service import DocumentService
from rag_graph_demo.storage.document_registry import DocumentRegistry
from rag_graph_demo.storage.file_store import FileStore
from rag_graph_demo.storage.job_registry import JobRegistry


class FakeEngine:
    initialized = False

    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def index_document(
        self,
        *,
        document_id: str,
        text: str,
        file_path: str,
    ) -> str:
        assert document_id
        assert "rag-source" in text
        assert file_path == f"{document_id}__报道.md"
        self.initialized = True
        return "track-123"

    async def delete_document(self, document_id: str) -> dict[str, str]:
        self.deleted.append(document_id)
        return {"status": "success", "document_id": document_id}


def build_service(tmp_path) -> tuple[DocumentService, DocumentRegistry, JobRegistry]:
    registry = DocumentRegistry(tmp_path / "registry" / "documents.json")
    jobs = JobRegistry(tmp_path / "registry" / "jobs.json")
    service = DocumentService(
        FileStore(tmp_path),
        registry,
        jobs,
        DocumentParser(),
        FakeEngine(),
    )
    return service, registry, jobs


@pytest.mark.asyncio
async def test_upload_deduplicates_and_indexes_markdown(tmp_path) -> None:
    service, registry, jobs = build_service(tmp_path)
    content = "# 标题\n\n新闻正文。".encode()

    document, duplicate = await service.create_upload(
        filename="报道.md",
        mime_type="text/markdown",
        content=content,
    )
    job = await service.create_job(document.document_id, "index")
    await service.index_document(document.document_id, job.job_id)

    indexed = await registry.get(document.document_id)
    finished_job = await jobs.get(job.job_id)
    duplicate_document, is_duplicate = await service.create_upload(
        filename="copy.md",
        mime_type="text/markdown",
        content=content,
    )

    assert duplicate is False
    assert indexed.status == DocumentStatus.READY
    assert indexed.lightrag_track_id == "track-123"
    assert Path(indexed.parsed_path).exists()
    assert finished_job.status == JobStatus.SUCCEEDED
    assert is_duplicate is True
    assert duplicate_document.document_id == document.document_id


@pytest.mark.asyncio
async def test_delete_removes_registry_and_files(tmp_path) -> None:
    service, registry, _ = build_service(tmp_path)
    document, _ = await service.create_upload(
        filename="delete.md",
        mime_type="text/markdown",
        content=b"# Delete",
    )

    result = await service.delete_document(document.document_id)

    assert result["document_id"] == document.document_id
    assert await registry.get(document.document_id) is None
    assert not Path(document.raw_path).exists()


@pytest.mark.asyncio
async def test_reindex_purges_failed_lightrag_state_before_retry(tmp_path) -> None:
    service, registry, _ = build_service(tmp_path)
    document, _ = await service.create_upload(
        filename="报道.md",
        mime_type="text/markdown",
        content="# 标题\n\n正文。".encode(),
    )
    await registry.update_status(document.document_id, DocumentStatus.FAILED)
    job = await service.create_job(document.document_id, "reindex")

    await service.reindex_document(document.document_id, job.job_id)

    assert service._engine.deleted == [document.document_id]
    assert (await registry.get(document.document_id)).status == DocumentStatus.READY

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, UploadFile

from rag_graph_demo.api.dependencies import get_container
from rag_graph_demo.domain.exceptions import UploadTooLargeError
from rag_graph_demo.domain.models import DocumentRecord, JobRecord
from rag_graph_demo.services.container import AppContainer

router = APIRouter()


@router.post("/documents")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    auto_index: bool = Query(default=True),
    container: AppContainer = Depends(get_container),
) -> dict[str, object]:
    content = await file.read()
    if len(content) > container.settings.max_upload_size_mb * 1024 * 1024:
        raise UploadTooLargeError(
            f"File exceeds {container.settings.max_upload_size_mb}MB upload limit"
        )
    document, duplicate = await container.document_service.create_upload(
        filename=file.filename or "document",
        mime_type=file.content_type,
        content=content,
    )
    job: JobRecord | None = None
    if auto_index and not duplicate:
        job = await container.document_service.create_job(document.document_id, "index")
        background_tasks.add_task(
            container.document_service.index_document,
            document.document_id,
            job.job_id,
        )
    return {
        "document": document.model_dump(mode="json"),
        "job": job.model_dump(mode="json") if job else None,
        "duplicate": duplicate,
    }


@router.get("/documents", response_model=list[DocumentRecord])
async def list_documents(
    container: AppContainer = Depends(get_container),
) -> list[DocumentRecord]:
    return await container.document_service.list_documents()


@router.get("/documents/{document_id}", response_model=DocumentRecord)
async def get_document(
    document_id: str,
    container: AppContainer = Depends(get_container),
) -> DocumentRecord:
    return await container.document_service.require_document(document_id)


@router.get("/documents/{document_id}/content")
async def get_document_content(
    document_id: str,
    container: AppContainer = Depends(get_container),
) -> dict[str, str]:
    return {
        "document_id": document_id,
        "content": await container.document_service.read_parsed_content(document_id),
    }


@router.post("/documents/{document_id}/reindex")
async def reindex_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    container: AppContainer = Depends(get_container),
) -> dict[str, object]:
    await container.document_service.require_document(document_id)
    job = await container.document_service.create_job(document_id, "reindex")
    background_tasks.add_task(
        container.document_service.reindex_document,
        document_id,
        job.job_id,
    )
    return job.model_dump(mode="json")


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    container: AppContainer = Depends(get_container),
) -> dict[str, object]:
    return await container.document_service.delete_document(document_id)

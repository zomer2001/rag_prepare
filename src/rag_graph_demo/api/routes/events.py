from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from rag_graph_demo.api.dependencies import get_container
from rag_graph_demo.domain.events import EventQueryRequest, EventQueryResponse, EventRecord
from rag_graph_demo.services.container import AppContainer

router = APIRouter()


@router.get("/events", response_model=list[EventRecord])
async def list_events(
    container: AppContainer = Depends(get_container),
) -> list[EventRecord]:
    return await container.events_registry.list_events()


@router.get("/events/{event_id}", response_model=EventRecord)
async def get_event(
    event_id: str,
    container: AppContainer = Depends(get_container),
) -> EventRecord:
    event = await container.event_service.get(event_id)
    if event is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
    return event


@router.post("/events/query", response_model=EventQueryResponse)
async def query_events(
    request: EventQueryRequest,
    container: AppContainer = Depends(get_container),
) -> EventQueryResponse:
    return await container.event_service.search(request)


@router.post("/documents/{document_id}/events/rebuild")
async def rebuild_document_events(
    document_id: str,
    background_tasks: BackgroundTasks,
    container: AppContainer = Depends(get_container),
) -> dict[str, object]:
    await container.document_service.require_document(document_id)
    job = await container.document_service.create_job(document_id, "events")
    background_tasks.add_task(
        container.document_service.rebuild_events, document_id, job.job_id
    )
    return job.model_dump(mode="json")

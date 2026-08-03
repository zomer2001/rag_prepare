from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from rag_graph_demo.api.dependencies import get_container
from rag_graph_demo.domain.models import JobRecord
from rag_graph_demo.services.container import AppContainer

router = APIRouter()


@router.get("/jobs", response_model=list[JobRecord])
async def list_jobs(
    container: AppContainer = Depends(get_container),
) -> list[JobRecord]:
    return await container.jobs_registry.list()


@router.get("/jobs/{job_id}", response_model=JobRecord)
async def get_job(
    job_id: str,
    container: AppContainer = Depends(get_container),
) -> JobRecord:
    job = await container.jobs_registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


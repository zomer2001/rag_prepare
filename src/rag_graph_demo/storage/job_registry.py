from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from rag_graph_demo.domain.models import JobRecord, JobStatus
from rag_graph_demo.storage.atomic_json import read_json, write_json_atomic


class JobRegistry:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()

    async def list(self) -> list[JobRecord]:
        async with self._lock:
            records = self._read_records()
        return sorted(records.values(), key=lambda item: item.created_at, reverse=True)

    async def get(self, job_id: str) -> JobRecord | None:
        async with self._lock:
            return self._read_records().get(job_id)

    async def upsert(self, record: JobRecord) -> JobRecord:
        async with self._lock:
            records = self._read_records()
            records[record.job_id] = record
            self._write_records(records)
            return record

    async def update(
        self, job_id: str, status: JobStatus, message: str = ""
    ) -> JobRecord:
        async with self._lock:
            records = self._read_records()
            record = records[job_id]
            payload = record.model_dump()
            payload["status"] = status
            payload["message"] = message
            now = datetime.now(UTC)
            if status == JobStatus.RUNNING:
                payload["started_at"] = now
            if status in {JobStatus.SUCCEEDED, JobStatus.FAILED}:
                payload["finished_at"] = now
            updated = JobRecord.model_validate(payload)
            records[job_id] = updated
            self._write_records(records)
            return updated

    def _read_records(self) -> dict[str, JobRecord]:
        raw = read_json(self._path, default={})
        return {key: JobRecord.model_validate(value) for key, value in raw.items()}

    def _write_records(self, records: dict[str, JobRecord]) -> None:
        write_json_atomic(
            self._path,
            {key: value.model_dump(mode="json") for key, value in records.items()},
        )

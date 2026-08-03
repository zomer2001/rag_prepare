from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from rag_graph_demo.domain.models import DocumentRecord, DocumentStatus
from rag_graph_demo.storage.atomic_json import read_json, write_json_atomic


class DocumentRegistry:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()

    async def list(self) -> list[DocumentRecord]:
        async with self._lock:
            records = self._read_records()
        return sorted(records.values(), key=lambda item: item.created_at, reverse=True)

    async def get(self, document_id: str) -> DocumentRecord | None:
        async with self._lock:
            return self._read_records().get(document_id)

    async def find_by_checksum(self, checksum: str) -> DocumentRecord | None:
        async with self._lock:
            return next(
                (record for record in self._read_records().values() if record.checksum == checksum),
                None,
            )

    async def find_by_path(self, file_path: str) -> DocumentRecord | None:
        target = str(Path(file_path).resolve())
        async with self._lock:
            return next(
                (
                    record
                    for record in self._read_records().values()
                    if str(Path(record.raw_path).resolve()) == target
                ),
                None,
            )

    async def upsert(self, record: DocumentRecord) -> DocumentRecord:
        async with self._lock:
            records = self._read_records()
            record.updated_at = datetime.now(UTC)
            records[record.document_id] = record
            self._write_records(records)
            return record

    async def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        *,
        error: str | None = None,
        **updates: object,
    ) -> DocumentRecord:
        async with self._lock:
            records = self._read_records()
            record = records[document_id]
            payload = record.model_dump()
            payload.update(updates)
            payload["status"] = status
            payload["error"] = error
            payload["updated_at"] = datetime.now(UTC)
            updated = DocumentRecord.model_validate(payload)
            records[document_id] = updated
            self._write_records(records)
            return updated

    async def delete(self, document_id: str) -> None:
        async with self._lock:
            records = self._read_records()
            records.pop(document_id, None)
            self._write_records(records)

    def _read_records(self) -> dict[str, DocumentRecord]:
        raw = read_json(self._path, default={})
        return {key: DocumentRecord.model_validate(value) for key, value in raw.items()}

    def _write_records(self, records: dict[str, DocumentRecord]) -> None:
        write_json_atomic(
            self._path,
            {key: value.model_dump(mode="json") for key, value in records.items()},
        )

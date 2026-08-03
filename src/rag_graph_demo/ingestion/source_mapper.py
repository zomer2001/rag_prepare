from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rag_graph_demo.domain.models import Citation
from rag_graph_demo.storage.document_registry import DocumentRegistry

SOURCE_PATTERN = re.compile(
    r"<!--\s*rag-source\s+document_id=(?P<document_id>[\w-]+)\s+page=(?P<page>\d+)\s*-->"
)
HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+?)\s*$", flags=re.MULTILINE)


class SourceMapper:
    def __init__(self, registry: DocumentRegistry) -> None:
        self._registry = registry

    async def resolve(self, raw_data: dict[str, Any]) -> list[Citation]:
        data = raw_data.get("data") or {}
        chunks = data.get("chunks") or []
        references = data.get("references") or []
        chunk_by_reference = {
            str(chunk.get("reference_id")): chunk for chunk in chunks if chunk.get("reference_id")
        }
        citations: list[Citation] = []

        for reference in references:
            reference_id = str(reference.get("reference_id", ""))
            file_path = str(reference.get("file_path", "unknown_source"))
            chunk = chunk_by_reference.get(reference_id, {})
            content = str(chunk.get("content", ""))
            marker = SOURCE_PATTERN.search(content)
            document_id = marker.group("document_id") if marker else None
            page_number = (
                int(marker.group("page"))
                if marker and marker.group("page") != "0"
                else None
            )
            record = await self._registry.get(document_id) if document_id else None
            if record is None and file_path != "unknown_source":
                record = await self._registry.find_by_path(file_path)
            heading = HEADING_PATTERN.search(content)
            citations.append(
                Citation(
                    reference_id=reference_id,
                    document_id=record.document_id if record else document_id,
                    title=record.title if record else None,
                    filename=record.filename if record else Path(file_path).name,
                    file_path=record.raw_path if record else file_path,
                    page_number=page_number,
                    section=heading.group(1).strip() if heading else None,
                    excerpt=content[:600].strip() or None,
                )
            )

        return citations

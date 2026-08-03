import pytest

from rag_graph_demo.domain.models import DocumentRecord, DocumentStatus
from rag_graph_demo.storage.document_registry import DocumentRegistry


@pytest.mark.asyncio
async def test_document_registry_round_trip_and_lookup(tmp_path) -> None:
    source = tmp_path / "raw" / "doc-1" / "original.md"
    source.parent.mkdir(parents=True)
    source.write_text("content", encoding="utf-8")
    registry = DocumentRegistry(tmp_path / "registry" / "documents.json")
    record = DocumentRecord(
        document_id="doc-1",
        filename="news.md",
        title="News",
        suffix=".md",
        size_bytes=7,
        checksum="checksum",
        raw_path=str(source),
    )

    await registry.upsert(record)
    updated = await registry.update_status(
        "doc-1",
        DocumentStatus.READY,
        lightrag_track_id="track-1",
    )

    assert updated.status == DocumentStatus.READY
    assert updated.lightrag_track_id == "track-1"
    assert (await registry.find_by_checksum("checksum")).document_id == "doc-1"
    assert (await registry.find_by_path(str(source))).document_id == "doc-1"

    reloaded = DocumentRegistry(tmp_path / "registry" / "documents.json")
    assert (await reloaded.get("doc-1")).status == DocumentStatus.READY

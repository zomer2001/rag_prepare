import pytest

from rag_graph_demo.domain.models import DocumentRecord
from rag_graph_demo.ingestion.source_mapper import SourceMapper
from rag_graph_demo.storage.document_registry import DocumentRegistry


@pytest.mark.asyncio
async def test_source_mapper_resolves_page_heading_and_document(tmp_path) -> None:
    source = tmp_path / "raw" / "doc-1" / "original.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"pdf")
    registry = DocumentRegistry(tmp_path / "documents.json")
    await registry.upsert(
        DocumentRecord(
            document_id="doc-1",
            filename="report.pdf",
            title="季度报道",
            suffix=".pdf",
            size_bytes=3,
            checksum="abc",
            raw_path=str(source),
        )
    )
    mapper = SourceMapper(registry)

    citations = await mapper.resolve(
        {
            "data": {
                "references": [
                    {"reference_id": "1", "file_path": str(source)},
                ],
                "chunks": [
                    {
                        "reference_id": "1",
                        "content": (
                            "<!-- rag-source document_id=doc-1 page=3 -->\n\n"
                            "## 市场变化\n\n原文证据。"
                        ),
                    }
                ],
            }
        }
    )

    assert len(citations) == 1
    assert citations[0].document_id == "doc-1"
    assert citations[0].title == "季度报道"
    assert citations[0].page_number == 3
    assert citations[0].section == "市场变化"
    assert citations[0].excerpt.endswith("原文证据。")


@pytest.mark.asyncio
async def test_source_mapper_falls_back_to_registered_file_path(tmp_path) -> None:
    source = tmp_path / "original.md"
    source.write_text("body", encoding="utf-8")
    registry = DocumentRegistry(tmp_path / "documents.json")
    await registry.upsert(
        DocumentRecord(
            document_id="doc-2",
            filename="brief.md",
            title="Brief",
            suffix=".md",
            size_bytes=4,
            checksum="def",
            raw_path=str(source),
        )
    )

    citations = await SourceMapper(registry).resolve(
        {
            "data": {
                "references": [
                    {"reference_id": "2", "file_path": str(source)},
                ],
                "chunks": [{"reference_id": "2", "content": "A cited excerpt."}],
            }
        }
    )

    assert citations[0].document_id == "doc-2"
    assert citations[0].filename == "brief.md"

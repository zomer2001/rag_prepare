import pytest

from rag_graph_demo.ingestion.parser import DocumentParser


@pytest.mark.asyncio
async def test_markdown_parser_preserves_source_and_title(tmp_path) -> None:
    source = tmp_path / "news.md"
    source.write_text("# 中文标题\n\nA mixed English 中文 paragraph.", encoding="utf-8")

    parsed = await DocumentParser().parse("doc-123", source, source.name)

    assert parsed.title == "中文标题"
    assert "<!-- rag-source document_id=doc-123 page=0 -->" in parsed.text
    assert "A mixed English 中文 paragraph." in parsed.text
    assert parsed.source_spans == [
        {
            "page": None,
            "marker": "<!-- rag-source document_id=doc-123 page=0 -->",
        }
    ]


def test_title_parser_accepts_docling_level_two_heading() -> None:
    assert DocumentParser._title_from_markdown("## DOCX 标题\n\n正文") == "DOCX 标题"

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from rag_graph_demo.domain.models import ParsedDocument

SOURCE_MARKER = "<!-- rag-source document_id={document_id} page={page} -->"


class DocumentParser:
    def __init__(self, *, ocr_enabled: bool = False) -> None:
        self._ocr_enabled = ocr_enabled

    async def parse(self, document_id: str, path: Path, filename: str) -> ParsedDocument:
        if path.suffix.lower() in {".md", ".markdown"}:
            return await asyncio.to_thread(self._parse_markdown, document_id, path, filename)
        return await asyncio.to_thread(self._parse_docling, document_id, path, filename)

    def _parse_markdown(
        self, document_id: str, path: Path, filename: str
    ) -> ParsedDocument:
        content = path.read_text(encoding="utf-8")
        title = self._title_from_markdown(content) or Path(filename).stem
        marker = SOURCE_MARKER.format(document_id=document_id, page=0)
        return ParsedDocument(
            document_id=document_id,
            title=title,
            text=f"{marker}\n\n{content.strip()}\n",
            page_count=None,
            source_spans=[{"page": None, "marker": marker}],
        )

    def _parse_docling(
        self, document_id: str, path: Path, filename: str
    ) -> ParsedDocument:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        format_options = None
        if path.suffix.lower() == ".pdf":
            pipeline_options = PdfPipelineOptions(do_ocr=self._ocr_enabled)
            format_options = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        result = DocumentConverter(format_options=format_options).convert(path)
        document = result.document
        pages = sorted(document.pages)
        source_spans: list[dict[str, object]] = []
        sections: list[str] = []

        if path.suffix.lower() == ".pdf" and pages:
            for page_number in pages:
                markdown = document.export_to_markdown(
                    page_no=page_number,
                    traverse_pictures=True,
                ).strip()
                if not markdown:
                    continue
                marker = SOURCE_MARKER.format(
                    document_id=document_id,
                    page=page_number,
                )
                sections.append(f"{marker}\n\n{markdown}")
                source_spans.append({"page": page_number, "marker": marker})
        else:
            marker = SOURCE_MARKER.format(document_id=document_id, page=0)
            sections.append(
                f"{marker}\n\n{document.export_to_markdown(traverse_pictures=True).strip()}"
            )
            source_spans.append({"page": None, "marker": marker})

        text = "\n\n---\n\n".join(sections).strip()
        title = self._title_from_markdown(text) or document.name or Path(filename).stem
        return ParsedDocument(
            document_id=document_id,
            title=title,
            text=f"{text}\n",
            page_count=len(pages) if pages else None,
            source_spans=source_spans,
        )

    @staticmethod
    def _title_from_markdown(content: str) -> str | None:
        match = re.search(r"^#{1,6}\s+(.+?)\s*$", content, flags=re.MULTILINE)
        return match.group(1).strip() if match else None

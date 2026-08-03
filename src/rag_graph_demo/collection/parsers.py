"""MIME-specific parsers (brief §4.4).

- HTML: stdlib parser with boilerplate stripping, JSON-LD and Next.js
  ``__NEXT_DATA__`` extraction, title/author/date heuristics, table capture.
- JSON: raw preserved; adapters map fields explicitly.
- PDF: lazy Docling conversion for text PDFs; scanned/corrupt files are routed
  to the review queue with a reason (page-level output and OCR are Phase 2).

All external content is treated as untrusted input; nothing here executes
embedded scripts or follows embedded instructions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser
from typing import Any

_DROP_TAGS = {"script", "style", "noscript", "template", "svg", "iframe"}
_BOILERPLATE_TAGS = {"nav", "footer", "aside", "form"}
_BOILERPLATE_HINTS = (
    "cookie",
    "newsletter",
    "subscribe",
    "related",
    "recommended",
    "breadcrumb",
    "share",
    "social",
    "footer",
    "navbar",
    "menu",
    "sidebar",
)


@dataclass
class ParsedContent:
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    published_at_raw: str | None = None
    body_markdown: str = ""
    json_ld: list[dict[str, Any]] = field(default_factory=list)
    next_data: dict[str, Any] | None = None
    tables: list[list[list[str]]] = field(default_factory=list)
    parse_confidence: float = 1.0
    warnings: list[str] = field(default_factory=list)
    needs_review: bool = False
    review_reason: str | None = None


class _HtmlExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_tag_text: list[str] = []
        self.meta: dict[str, str] = {}
        self.json_ld: list[dict[str, Any]] = []
        self.next_data: dict[str, Any] | None = None
        self.blocks: list[str] = []
        self.tables: list[list[list[str]]] = []
        self._drop_depth = 0
        self._boiler_stack: list[str] = []
        self._in_title = False
        self._in_json_ld = False
        self._in_next_data = False
        self._json_buf: list[str] = []
        self._block_buf: list[str] = []
        self._block_tag: str | None = None
        self._in_table = False
        self._table: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    # ------------------------------------------------------------ events
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        if self._drop_depth or tag in _DROP_TAGS:
            if tag == "script":
                script_type = (attr.get("type") or "").lower()
                script_id = attr.get("id") or ""
                if script_type == "application/ld+json":
                    self._in_json_ld = True
                    self._json_buf = []
                elif script_id == "__NEXT_DATA__":
                    self._in_next_data = True
                    self._json_buf = []
            self._drop_depth += 1
            return
        if self._boiler_stack:
            self._boiler_stack.append(tag)
            return
        if tag in _BOILERPLATE_TAGS or self._is_boilerplate(attr):
            self._boiler_stack.append(tag)
            return
        if tag == "meta":
            key = attr.get("property") or attr.get("name") or ""
            content = attr.get("content") or ""
            if key and content:
                self.meta[key.lower()] = content
        elif tag == "title":
            self._in_title = True
        elif tag in ("h1", "h2", "h3", "h4", "p", "li", "blockquote", "pre"):
            self._flush_block()
            self._block_tag = tag
        elif tag == "br" and self._block_tag:
            self._block_buf.append("\n")
        elif tag == "table":
            self._in_table = True
            self._table = []
        elif self._in_table and tag == "tr":
            self._row = []
        elif self._in_table and self._row is not None and tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if self._drop_depth:
            if self._in_json_ld and tag == "script":
                self._in_json_ld = False
                self._capture_json(ld=True)
            elif self._in_next_data and tag == "script":
                self._in_next_data = False
                self._capture_json(ld=False)
            self._drop_depth -= 1
            return
        if self._boiler_stack:
            if tag in self._boiler_stack:
                while self._boiler_stack:
                    popped = self._boiler_stack.pop()
                    if popped == tag:
                        break
            return
        if tag == "title":
            self._in_title = False
        elif tag in ("h1", "h2", "h3", "h4", "p", "li", "blockquote", "pre"):
            self._flush_block()
        elif tag == "table":
            if self._table:
                self.tables.append(self._table)
            self._in_table = False
            self._table = []
        elif self._in_table and tag == "tr" and self._row is not None:
            if self._row:
                self._table.append(self._row)
            self._row = None
        elif self._cell is not None and tag in ("td", "th"):
            self._row = self._row if self._row is not None else []
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._in_json_ld or self._in_next_data:
            self._json_buf.append(data)
            return
        if self._drop_depth or self._boiler_stack:
            return
        if self._in_title:
            self.title_tag_text.append(data)
            return
        if self._cell is not None:
            self._cell.append(data)
        elif self._block_tag:
            self._block_buf.append(data)

    # ------------------------------------------------------------ helpers
    def _is_boilerplate(self, attr: dict[str, str | None]) -> bool:
        marker = " ".join(
            filter(None, [attr.get("class"), attr.get("id"), attr.get("role")])
        ).lower()
        return any(hint in marker for hint in _BOILERPLATE_HINTS)

    def _capture_json(self, ld: bool) -> None:
        raw = "".join(self._json_buf).strip()
        if not raw:
            return
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        if ld:
            if isinstance(data, list):
                self.json_ld.extend(item for item in data if isinstance(item, dict))
            elif isinstance(data, dict):
                self.json_ld.append(data)
        elif isinstance(data, dict):
            self.next_data = data

    def _flush_block(self) -> None:
        if self._block_tag is None:
            return
        text = " ".join("".join(self._block_buf).split())
        if text:
            prefix = {
                "h1": "# ",
                "h2": "## ",
                "h3": "### ",
                "h4": "#### ",
                "li": "- ",
                "blockquote": "> ",
            }.get(self._block_tag, "")
            self.blocks.append(prefix + text if self._block_tag != "pre" else f"```\n{text}\n```")
        self._block_tag = None
        self._block_buf = []


def _markdown_table(table: list[list[str]]) -> str:
    if not table:
        return ""
    header = table[0]
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for row in table[1:]:
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded[: len(header)]) + " |")
    return "\n".join(lines)


def parse_html(html: str) -> ParsedContent:
    extractor = _HtmlExtractor()
    try:
        extractor.feed(html)
        extractor.close()
    except Exception as exc:  # malformed HTML: keep what we have
        extractor._flush_block()
        result = ParsedContent(parse_confidence=0.4, needs_review=True)
        result.warnings.append(f"html parser error: {exc}")
    else:
        extractor._flush_block()
        result = ParsedContent()
    result.json_ld = extractor.json_ld
    result.next_data = extractor.next_data
    result.tables = extractor.tables

    title = " ".join("".join(extractor.title_tag_text).split()) or None
    meta = extractor.meta
    title = meta.get("og:title", title) or title
    result.title = unescape(title) if title else None
    author = meta.get("author") or meta.get("article:author")
    if author:
        result.authors = [a.strip() for a in author.split(",") if a.strip()]
    result.published_at_raw = (
        meta.get("article:published_time")
        or meta.get("date")
        or meta.get("publishdate")
        or meta.get("dc.date")
    )
    # JSON-LD fallback for title/date/author
    for node in result.json_ld:
        node_type = node.get("@type", "")
        if result.published_at_raw is None and isinstance(node, dict):
            result.published_at_raw = node.get("datePublished") or node.get("dateCreated")
        if not result.authors and node_type in ("NewsArticle", "Article", "BlogPosting"):
            raw_author = node.get("author")
            if isinstance(raw_author, dict):
                raw_author = [raw_author]
            if isinstance(raw_author, list):
                result.authors = [
                    str(a.get("name", "")).strip()
                    for a in raw_author
                    if isinstance(a, dict) and a.get("name")
                ]

    body_parts = list(extractor.blocks)
    for table in extractor.tables:
        body_parts.append(_markdown_table(table))
    result.body_markdown = "\n\n".join(part for part in body_parts if part).strip()
    if not result.body_markdown:
        result.parse_confidence = min(result.parse_confidence, 0.3)
        result.needs_review = True
        result.review_reason = "empty body after boilerplate removal"
        result.warnings.append("empty body")
    return result


def parse_json_payload(raw: bytes | str) -> ParsedContent:
    """JSON/API payloads are preserved verbatim; adapters do explicit field
    mapping. Unknown fields must be kept (brief §4.4)."""
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return ParsedContent(
            parse_confidence=0.0,
            needs_review=True,
            review_reason=f"invalid JSON: {exc}",
            warnings=["invalid JSON"],
        )
    return ParsedContent(
        body_markdown="",  # adapters decide what prose (if any) to emit
        parse_confidence=1.0,
        next_data={"payload": data},
    )


def parse_pdf(raw_path: str) -> ParsedContent:
    """Phase 1 PDF handling: try Docling for text PDFs; scanned or damaged
    documents are routed to the review queue. Page-level parsing, tables and
    the OCR queue are Phase 2 scope (brief §9)."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        return ParsedContent(
            parse_confidence=0.0,
            needs_review=True,
            review_reason="docling unavailable; raw PDF preserved only",
        )
    try:
        converted = DocumentConverter().convert(raw_path)
        markdown = converted.document.export_to_markdown()
    except Exception as exc:
        return ParsedContent(
            parse_confidence=0.0,
            needs_review=True,
            review_reason=f"pdf conversion failed (possibly scanned/corrupt): {exc}",
        )
    result = ParsedContent(body_markdown=(markdown or "").strip(), parse_confidence=0.9)
    if not result.body_markdown:
        result.needs_review = True
        result.review_reason = "no extractable text (candidate for OCR queue in Phase 2)"
        result.parse_confidence = 0.2
    return result


def parse_by_mime(
    mime_type: str, *, raw_bytes: bytes | None = None, raw_path: str | None = None
) -> ParsedContent:
    if mime_type in ("text/html", "application/xhtml+xml"):
        text = (raw_bytes or b"").decode("utf-8", errors="replace")
        return parse_html(text)
    if mime_type in ("application/json", "text/json") or mime_type.endswith("+json"):
        return parse_json_payload(raw_bytes or b"")
    if mime_type == "application/pdf":
        if raw_path is None:
            return ParsedContent(
                parse_confidence=0.0, needs_review=True, review_reason="pdf without raw path"
            )
        return parse_pdf(raw_path)
    if mime_type in ("application/xml", "text/xml"):
        text = (raw_bytes or b"").decode("utf-8", errors="replace")
        plain = re.sub(r"<[^>]+>", " ", text)
        return ParsedContent(body_markdown=" ".join(plain.split()), parse_confidence=0.6)
    if mime_type.startswith("text/"):
        text = (raw_bytes or b"").decode("utf-8", errors="replace")
        return ParsedContent(body_markdown=text.strip(), parse_confidence=0.8)
    return ParsedContent(
        parse_confidence=0.0,
        needs_review=True,
        review_reason=f"unsupported mime: {mime_type}",
    )

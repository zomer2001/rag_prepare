"""HTML/JSON parsing: boilerplate removal, JSON-LD, dates, empty bodies."""

from __future__ import annotations

from pathlib import Path

from rag_graph_demo.collection.parsers import parse_by_mime, parse_html, parse_json_payload

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "collection"


class TestHtmlParsing:
    def test_boilerplate_removed_metadata_extracted(self):
        html = (FIXTURES / "anduril_official" / "pages" / "roadrunner_m.html").read_text()
        parsed = parse_html(html)
        assert parsed.title == "Anduril Unveils Roadrunner-M (fixture)"
        assert parsed.published_at_raw == "2026-07-20"
        assert "cookies" not in parsed.body_markdown.lower()
        assert "Recommended" not in parsed.body_markdown
        assert "Roadrunner-M is a high-explosive variant" in parsed.body_markdown
        assert parsed.json_ld and parsed.json_ld[0]["@type"] == "NewsArticle"
        assert parsed.authors  # from meta author

    def test_table_captured_as_markdown(self):
        html = (FIXTURES / "anduril_official" / "pages" / "roadrunner_m.html").read_text()
        parsed = parse_html(html)
        assert parsed.tables
        assert "| Attribute | Value |" in parsed.body_markdown
        assert "Counter-UAS interception" in parsed.body_markdown

    def test_empty_body_goes_to_review(self):
        parsed = parse_html("<html><head><title>x</title></head><body></body></html>")
        assert parsed.needs_review
        assert parsed.parse_confidence < 0.5


class TestJsonParsing:
    def test_payload_preserved(self):
        parsed = parse_json_payload(b'{"a": 1, "unknown_field": "kept"}')
        assert parsed.next_data["payload"]["unknown_field"] == "kept"

    def test_invalid_json_flagged(self):
        parsed = parse_json_payload(b"{broken")
        assert parsed.needs_review and parsed.parse_confidence == 0.0


class TestParseByMime:
    def test_unsupported_mime_flagged(self):
        parsed = parse_by_mime("application/x-msdownload", raw_bytes=b"MZ")
        assert parsed.needs_review

    def test_corrupt_pdf_goes_to_review_not_crash(self, tmp_path):
        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"%PDF-1.4 corrupt bytes without structure")
        parsed = parse_by_mime("application/pdf", raw_bytes=bad.read_bytes(), raw_path=str(bad))
        assert parsed.needs_review

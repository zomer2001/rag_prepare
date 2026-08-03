"""Normalization: date confidence, money strings, clean markdown rules."""

from __future__ import annotations

import json

import pytest

from rag_graph_demo.collection.normalize import (
    RAG_SOURCE_MARKER,
    build_clean_markdown,
    money_string,
    parse_date,
    write_clean_and_sidecar,
)


class TestParseDate:
    @pytest.mark.parametrize(
        ("raw", "expected", "confidence"),
        [
            ("2026-07-20", "2026-07-20", "full"),
            ("2026-07-20T00:00:00.000Z", "2026-07-20", "full"),
            ("2026-07", "2026-07", "month"),
            ("2026", "2026", "year"),
            ("July 20, 2026", "2026-07-20", "full"),
            ("20 July 2026", "2026-07-20", "full"),
            ("not a date", None, "none"),
            (None, None, "none"),
            ("2026-13-01", None, "none"),  # invalid month not fabricated
        ],
    )
    def test_cases(self, raw, expected, confidence):
        assert parse_date(raw) == (expected, confidence)

    def test_year_only_not_inflated(self):
        iso, confidence = parse_date("2024")
        assert iso == "2024" and confidence == "year"
        assert "-01-01" not in iso


class TestMoney:
    def test_exact_decimal_string(self):
        assert money_string(87713060.0) == "87713060"
        assert money_string(5104579.85) == "5104579.85"
        assert money_string(None) is None
        assert money_string("not-money") is None


class TestCleanMarkdown:
    def _record(self, body: str = "Some body text."):
        return {
            "record_id": "src:abc",
            "version_id": "v123",
            "version_index": 1,
            "title": "Title",
            "canonical_url": "https://example.com/x",
            "published_at": "2026-07-20",
            "fetched_at": "2026-08-03T00:00:00Z",
            "body_markdown": body,
            "source": {"publisher": "Pub", "source_id": "src"},
        }

    def test_no_rag_source_marker(self, tmp_path):
        clean_path, sidecar_path = write_clean_and_sidecar(tmp_path, self._record())
        text = clean_path.read_text(encoding="utf-8")
        assert RAG_SOURCE_MARKER not in text
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert sidecar["record_id"] == "src:abc"
        assert "body_markdown" not in sidecar  # body lives in the .md file

    def test_marker_in_body_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            build_clean_markdown(self._record(body=f"evil {RAG_SOURCE_MARKER} x -->"))

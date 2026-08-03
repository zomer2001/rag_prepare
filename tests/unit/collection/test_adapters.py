"""Adapter-level mapping tests using offline fixtures (brief §10):

- contract parent/child relations and the four money kinds never conflated
- patent family members stay separate records
- search snippets never become a verified clean body (no body from discovery)
"""

from __future__ import annotations

from pathlib import Path

from rag_graph_demo.collection.adapters.anduril_site import AndurilSiteAdapter
from rag_graph_demo.collection.adapters.patents import PatentsViewAdapter
from rag_graph_demo.collection.adapters.rss_feed import RssFeedAdapter
from rag_graph_demo.collection.adapters.scholarly import OpenAlexAdapter, reconstruct_abstract
from rag_graph_demo.collection.adapters.sitemap_archive import SitemapArchiveAdapter
from rag_graph_demo.collection.adapters.usaspending import USASpendingAdapter
from rag_graph_demo.collection.http import DomainRateLimiter, HttpFetcher
from rag_graph_demo.collection.models import FetchResult, RecordType
from rag_graph_demo.collection.parsers import ParsedContent, parse_json_payload
from rag_graph_demo.collection.runner import FixtureManifest, _fixture_client

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "collection"


class Ctx:
    live = False
    limit = None
    collection_root = None
    since = None


def _fetcher(tmp_path, source):
    manifest = FixtureManifest(FIXTURES, source)
    return HttpFetcher(
        tmp_path / "raw",
        client=_fixture_client(manifest),
        rate_limiter=DomainRateLimiter(default_interval=0),
        adapter=source.adapter,
    )


class _State:
    cursor: dict = {}
    health: dict = {}
    validators: dict = {}


class TestUSASpending:
    def test_money_kinds_never_conflated(self, registry, tmp_path):
        source = registry.get("usaspending")
        adapter = USASpendingAdapter()
        candidates = adapter.discover(source, _State(), Ctx(), _fetcher(tmp_path, source))
        assert len(candidates) == 2

        detail_path = FIXTURES / "usaspending" / "detail_359204966.json"
        parsed = parse_json_payload(detail_path.read_bytes())
        fetch = FetchResult(
            url=candidates[0].url,
            ok=True,
            sha256="0" * 64,
            raw_path=str(detail_path),
            mime_type="application/json",
        )
        record = adapter.build_record(
            source, candidates[0], fetch, parsed, parsed.next_data["payload"]
        )
        kinds = {a.kind: a.value for a in record.contract.amounts}
        assert kinds["current_award_amount"] == "87713060"
        assert kinds["award_ceiling"] == "175426120"   # potential ≠ spent
        assert kinds["obligation"] == "87713060"
        assert kinds["outlay"] == "12500000"
        fields = {a.source_field for a in record.contract.amounts}
        assert "awards.potential_award_amount" in fields
        # parent award kept as relation, never merged
        assert record.contract.parent_award_id == "W9128Z26DA001"
        assert record.relations[0].relation_type == "parent_award"
        assert record.record_type == RecordType.CONTRACT
        # unknown provider fields preserved
        assert "provider_row" in record.extension

    def test_parent_key_parser(self):
        key = "CONT_AWD_W911W625F0020_9700_W911W624D0008_9700"
        assert USASpendingAdapter._parent_from_key(key) == "W911W624D0008"


class TestAndurilSite:
    def test_sitemap_discovery_with_news_metadata(self, registry, tmp_path):
        source = registry.get("anduril_official")
        candidates = AndurilSiteAdapter().discover(
            source, _State(), Ctx(), _fetcher(tmp_path, source)
        )
        urls = {c.url for c in candidates}
        assert "https://www.anduril.com/news/anduril-unveils-roadrunner-m" in urls
        news = next(c for c in candidates if "/news/" in c.url)
        assert news.title == "Anduril Unveils Roadrunner-M (fixture)"
        assert news.discovered_by == "sitemap"

    def test_since_filters_old_entries(self, registry, tmp_path):
        source = registry.get("anduril_official")
        ctx = Ctx()
        ctx.since = "2026-01-01"
        candidates = AndurilSiteAdapter().discover(
            source, _State(), ctx, _fetcher(tmp_path, source)
        )
        assert all((c.lastmod or "")[:10] >= "2026-01-01" for c in candidates)
        assert "https://www.anduril.com/news/old-2023-post" not in {c.url for c in candidates}


class TestOpenAlex:
    def test_abstract_inverted_index_reconstruction(self):
        idx = {"hello": [0], "world": [1]}
        assert reconstruct_abstract(idx) == "hello world"
        assert reconstruct_abstract(None) is None

    def test_record_mapping_and_year_only_date(self, registry, tmp_path):
        source = registry.get("openalex")
        adapter = OpenAlexAdapter()
        candidates = adapter.discover(source, _State(), Ctx(), _fetcher(tmp_path, source))
        assert len(candidates) == 2
        by_title = {c.title: c for c in candidates}
        memo = by_title["Year-Only Date Fixture Memo on Lattice Sensor Fusion"]
        fetch = FetchResult(
            url=memo.url, ok=True, sha256="0" * 64, raw_path="x", mime_type="application/json"
        )
        record = adapter.build_record(source, memo, fetch, parse_json_payload(b"{}"), {})
        assert record.published_at == "2024"
        assert record.published_at_confidence == "year"
        assert record.paper.openalex_id == "https://openalex.org/W999000222"
        full = by_title[
            "Distributed Command and Control for Attritable Autonomous Systems (fixture)"
        ]
        record2 = adapter.build_record(source, full, fetch, parse_json_payload(b"{}"), {})
        assert record2.paper.abstract.startswith("We present a distributed C2")
        assert record2.paper.affiliations == ["Anduril Industries"]


class TestPatentsView:
    def test_policy_skip_without_api_key(self, registry, tmp_path):
        source = registry.get("patentsview")
        state = _State()
        state.health = {}
        adapter = PatentsViewAdapter(environ={})  # no key
        candidates = adapter.discover(source, state, Ctx(), _fetcher(tmp_path, source))
        assert candidates == []
        assert "policy_skip" in state.health  # recorded, not bypassed

    def test_family_members_stay_separate(self, registry, tmp_path):
        source = registry.get("patentsview")
        adapter = PatentsViewAdapter(environ={"PATENTSVIEW_API_KEY": "fixture-key"})
        candidates = adapter.discover(source, _State(), Ctx(), _fetcher(tmp_path, source))
        assert len(candidates) == 1
        fetch = FetchResult(
            url=candidates[0].url,
            ok=True,
            sha256="0" * 64,
            raw_path="x",
            mime_type="application/json",
        )
        record = adapter.build_record(source, candidates[0], fetch, parse_json_payload(b"{}"), {})
        assert record.patent.publication_number == "US12345678"
        assert record.patent.filing_date == "2023-11-02"
        assert record.patent.grant_date == "2025-09-30"
        assert "Alex Fixture" in record.patent.inventors
        assert record.patent.assignees_current == ["Anduril Industries, Inc."]
        # a same-family sibling publication would yield a DIFFERENT record_id
        from rag_graph_demo.collection.dedupe import make_record_id

        sibling = make_record_id(source.source_id, "US12345679", None)
        assert sibling != record.record_id


class TestRssFeed:
    def test_filters_anduril_and_keeps_summary_discovery_only(self, registry, tmp_path):
        feed = tmp_path / "feed.xml"
        feed.write_text(
            """<?xml version="1.0"?>
            <rss version="2.0"><channel>
              <item><title>Unrelated Navy story</title>
                <link>https://example.com/unrelated</link>
                <description>No matching company.</description></item>
              <item><title>Anduril tests Roadrunner-M</title>
                <link>https://example.com/anduril-test?utm_source=rss</link>
                <description>Feed-only summary that must not become body.</description>
                <pubDate>Sun, 03 Aug 2026 12:00:00 GMT</pubDate>
                <author>Reporter One</author></item>
            </channel></rss>""",
            encoding="utf-8",
        )

        class Fetcher:
            def fetch(self, _url):
                return FetchResult(url=_url, ok=True, raw_path=str(feed), mime_type="text/xml")

        source = registry.get("defense_news")
        candidates = RssFeedAdapter().discover(source, _State(), Ctx(), Fetcher())
        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate.url == "https://example.com/anduril-test"
        assert candidate.summary.startswith("Feed-only summary")

        fetch = FetchResult(
            url=candidate.url,
            ok=True,
            sha256="0" * 64,
            raw_path="article.html",
            mime_type="text/html",
        )
        record = RssFeedAdapter().build_record(
            source,
            candidate,
            fetch,
            ParsedContent(title="Canonical title", body_markdown="Verified article body."),
            None,
        )
        assert record.record_type == RecordType.NEWS
        assert record.body_markdown == "Verified article body."
        assert "Feed-only summary" not in record.body_markdown
        assert record.extension["feed_summary"].startswith("Feed-only summary")


class TestSitemapArchive:
    def test_recurses_index_and_only_selects_keyword_urls(self, registry, tmp_path):
        index = tmp_path / "index.xml"
        child = tmp_path / "child.xml"
        index.write_text(
            """<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <sitemap><loc>https://example.com/child.xml</loc></sitemap></sitemapindex>""",
            encoding="utf-8",
        )
        child.write_text(
            """<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url><loc>https://www.defensenews.com/news/unrelated-story/</loc></url>
            <url><loc>https://www.defensenews.com/news/anduril-roadrunner-test/</loc>
            <lastmod>2025-03-04</lastmod></url></urlset>""",
            encoding="utf-8",
        )

        class Fetcher:
            def fetch(self, url):
                path = index if "sitemap-index" in url else child
                return FetchResult(url=url, ok=True, raw_path=str(path), mime_type="text/xml")

        source = registry.get("defense_news_archive")
        candidates = SitemapArchiveAdapter().discover(source, _State(), Ctx(), Fetcher())
        assert len(candidates) == 1
        assert candidates[0].url.endswith("/anduril-roadrunner-test")
        assert candidates[0].lastmod == "2025-03-04"
        assert candidates[0].extra["matched_keywords"] == ["anduril"]


class TestDiscoverySnippetNeverABody:
    """Brief §0.4: search/discovery summaries must not become verified body."""

    def test_candidates_have_no_body_field(self):
        from rag_graph_demo.collection.models import DiscoveryCandidate

        assert "body_markdown" not in DiscoveryCandidate.model_fields
        assert "content" not in DiscoveryCandidate.model_fields

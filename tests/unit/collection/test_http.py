"""Fetcher behavior: ETag/Last-Modified, 429 Retry-After, redirects, MIME
spoofing, oversize refusal (brief §10). All via httpx.MockTransport — offline."""

from __future__ import annotations

import httpx

from rag_graph_demo.collection.http import DomainRateLimiter, HttpFetcher, sniff_mime


def make_fetcher(tmp_path, handler, **kwargs):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    fetcher = HttpFetcher(
        tmp_path / "raw",
        client=client,
        rate_limiter=DomainRateLimiter(default_interval=0),
        adapter="test",
        adapter_version="0.0.1",
        **kwargs,
    )
    fetcher.sleeper = lambda s: None  # never really sleep in tests
    return fetcher


class TestConditionalRequests:
    def test_sends_validators_and_handles_304(self, tmp_path):
        seen = {}

        def handler(request):
            seen["if_none_match"] = request.headers.get("If-None-Match")
            seen["if_modified_since"] = request.headers.get("If-Modified-Since")
            return httpx.Response(304)

        fetcher = make_fetcher(tmp_path, handler)
        result = fetcher.fetch(
            "https://example.com/page",
            etag='"abc"',
            last_modified="Wed, 01 Jul 2026 00:00:00 GMT",
        )
        assert result.not_modified and result.ok
        assert seen["if_none_match"] == '"abc"'
        assert seen["if_modified_since"].startswith("Wed")

    def test_stores_raw_and_hashes(self, tmp_path):
        def handler(request):
            return httpx.Response(200, content=b"<html>hi</html>", headers={"ETag": '"v1"'})

        fetcher = make_fetcher(tmp_path, handler)
        result = fetcher.fetch("https://example.com/a")
        assert result.ok and result.sha256 and result.raw_path
        assert result.etag == '"v1"'
        with open(result.raw_path, "rb") as fh:
            assert fh.read() == b"<html>hi</html>"

    def test_append_only_reuses_identical_object(self, tmp_path):
        def handler(request):
            return httpx.Response(200, content=b"same body")

        fetcher = make_fetcher(tmp_path, handler)
        first = fetcher.fetch("https://example.com/a")
        second = fetcher.fetch("https://example.com/b")
        assert first.raw_path == second.raw_path  # same sha, never rewritten


class TestRetryAndLimits:
    def test_429_honors_retry_after_then_succeeds(self, tmp_path):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(200, content=b"ok")

        fetcher = make_fetcher(tmp_path, handler)
        result = fetcher.fetch("https://example.com/ratelimited")
        assert result.ok and calls["n"] == 2

    def test_403_is_permanent(self, tmp_path):
        def handler(request):
            return httpx.Response(403)

        result = make_fetcher(tmp_path, handler).fetch("https://example.com/forbidden")
        assert not result.ok and not result.retryable
        assert "403" in result.error

    def test_transport_error_retryable(self, tmp_path):
        def handler(request):
            raise httpx.ConnectError("no route", request=request)

        result = make_fetcher(tmp_path, handler, max_retries=1).fetch("https://example.com/")
        assert not result.ok and result.retryable

    def test_oversize_rejected(self, tmp_path):
        def handler(request):
            return httpx.Response(200, content=b"x" * 2048)

        result = make_fetcher(tmp_path, handler, max_bytes=1024).fetch("https://example.com/big")
        assert not result.ok and "too large" in result.error

    def test_redirect_final_url_captured(self, tmp_path):
        def handler(request):
            if request.url.path == "/old":
                return httpx.Response(
                    301, headers={"Location": "/new"}, request=request
                )
            return httpx.Response(200, content=b"new page", request=request)

        client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
        fetcher = HttpFetcher(
            tmp_path / "raw", client=client, rate_limiter=DomainRateLimiter(default_interval=0)
        )
        fetcher.sleeper = lambda s: None
        result = fetcher.fetch("https://example.com/old")
        assert result.ok and result.final_url.endswith("/new")


class TestMimeSniffing:
    def test_pdf_magic_beats_header(self):
        assert sniff_mime("text/plain", b"%PDF-1.7 ...") == "application/pdf"

    def test_html_detected_without_header(self):
        assert sniff_mime(None, b"<!DOCTYPE html><html>") == "text/html"

    def test_json_detected(self):
        assert sniff_mime(None, b'  {"a": 1}') == "application/json"

    def test_xml_detected(self):
        assert sniff_mime(None, b'<?xml version="1.0"?><urlset>') == "application/xml"

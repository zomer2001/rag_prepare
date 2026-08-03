"""URL canonicalization, SSRF refusal, robots/policy gate (brief §10)."""

from __future__ import annotations

import pytest

from rag_graph_demo.collection.policy import (
    PolicyGate,
    RobotsCache,
    SSRFError,
    assert_ssrf_safe,
    canonicalize_url,
)


class TestCanonicalize:
    def test_strips_tracking_params_keeps_content_params(self):
        url = "https://www.anduril.com/news/x?utm_source=twitter&utm_medium=social&id=42&gclid=abc"
        assert canonicalize_url(url) == "https://www.anduril.com/news/x?id=42"

    def test_strips_fragment_and_sorts_query(self):
        url = "https://example.com/page?b=2&a=1#section"
        assert canonicalize_url(url) == "https://example.com/page?a=1&b=2"

    def test_lowercases_host_and_drops_default_port(self):
        assert canonicalize_url("HTTPS://WWW.ANDURIL.COM:443/News/") == "https://www.anduril.com/News"

    def test_keep_params_override(self):
        url = "https://example.com/p?utm_campaign=keepme"
        assert canonicalize_url(url, keep_params={"utm_campaign"}) == url


class TestSSRF:
    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost/admin",
            "http://127.0.0.1/x",
            "http://10.0.0.5/internal",
            "http://192.168.1.1/router",
            "http://169.254.169.254/latest/meta-data",
            "ftp://example.com/file",
            "file:///etc/passwd",
            "http://0.0.0.0/",
        ],
    )
    def test_rejected(self, url):
        with pytest.raises(SSRFError):
            assert_ssrf_safe(url, resolve_dns=False)

    @pytest.mark.parametrize(
        "url",
        ["https://www.anduril.com/sitemap.xml", "http://8.8.8.8/public"],
    )
    def test_allowed(self, url):
        assert_ssrf_safe(url, resolve_dns=False)


class TestPolicyGate:
    ROBOTS = "User-agent: *\nDisallow: /private/\n"

    def _gate(self) -> PolicyGate:
        robots = RobotsCache(fetcher=lambda url: (200, self.ROBOTS))
        return PolicyGate(robots=robots, resolve_dns=False)

    def test_robots_disallowed_recorded_not_bypassed(self):
        decision = self._gate().check("https://example.com/private/secret")
        assert not decision.allowed
        assert "robots" in decision.reason

    def test_allowed_url_returns_canonical(self):
        decision = self._gate().check("https://example.com/news/x?utm_source=a")
        assert decision.allowed
        assert decision.canonical_url == "https://example.com/news/x"

    def test_ssrf_blocked_at_gate(self):
        decision = self._gate().check("http://127.0.0.1/x")
        assert not decision.allowed
        assert decision.reason.startswith("ssrf:")

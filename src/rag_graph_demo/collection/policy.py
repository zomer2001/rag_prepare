"""Policy gate: URL canonicalization, SSRF defense, robots.txt enforcement.

Nothing in this module bypasses access controls; disallowed URLs are recorded
with a reason and skipped (brief §4.2, §8).
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.parse
import urllib.robotparser
from collections.abc import Callable
from dataclasses import dataclass, field

TRACKING_PARAMS = {
    "fbclid",
    "gclid",
    "gclsrc",
    "dclid",
    "msclkid",
    "mc_cid",
    "mc_eid",
    "igshid",
    "spm",
    "ref_src",
    "ref_url",
}

CLOUD_METADATA_IPS = {"169.254.169.254"}


def canonicalize_url(url: str, keep_params: set[str] | None = None) -> str:
    """Normalize a URL for dedupe: strip tracking params and fragment, sort
    query, lowercase scheme/host, drop default ports. Parameters that affect
    resource content are preserved (brief §4.2)."""
    keep_params = keep_params or set()
    parts = urllib.parse.urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    port = parts.port
    netloc = host
    if port and not (scheme == "http" and port == 80) and not (scheme == "https" and port == 443):
        netloc = f"{host}:{port}"
    path = parts.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    pairs = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    kept = [
        (k, v)
        for k, v in pairs
        if k in keep_params or not (k.lower().startswith("utm_") or k.lower() in TRACKING_PARAMS)
    ]
    kept.sort()
    query = urllib.parse.urlencode(kept, doseq=True)
    return urllib.parse.urlunsplit((scheme, netloc, path, query, ""))


class SSRFError(ValueError):
    pass


def _check_ip(ip_text: str) -> None:
    ip = ipaddress.ip_address(ip_text)
    if ip_text in CLOUD_METADATA_IPS:
        raise SSRFError(f"cloud metadata IP blocked: {ip_text}")
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        raise SSRFError(f"non-public IP blocked: {ip_text}")


def assert_ssrf_safe(url: str, resolve_dns: bool = True) -> None:
    """Reject non-HTTP(S) schemes, localhost, private networks, and cloud
    metadata endpoints (brief §4.2 SSRF guard). ``resolve_dns`` is disabled in
    offline tests; literal-IP checks always run."""
    parts = urllib.parse.urlsplit(url.strip())
    if parts.scheme.lower() not in ("http", "https"):
        raise SSRFError(f"scheme not allowed: {parts.scheme!r}")
    host = parts.hostname
    if not host:
        raise SSRFError("missing host")
    host = host.lower()
    if host in ("localhost", "localhost.localdomain") or host.endswith(".localhost"):
        raise SSRFError(f"localhost blocked: {host}")
    try:
        _check_ip(host)
        return  # literal public IP; no DNS needed
    except ValueError as exc:
        if isinstance(exc, SSRFError):
            raise
    if not resolve_dns:
        return
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise SSRFError(f"DNS resolution failed for {host}: {exc}") from exc
    for info in infos:
        _check_ip(info[4][0])


@dataclass
class RobotsCache:
    """robots.txt checker with an injectable fetcher (offline-testable)."""

    fetcher: Callable[[str], tuple[int, str]] | None = None
    default_allow: bool = True
    _parsers: dict[str, urllib.robotparser.RobotFileParser] = field(default_factory=dict)
    _failures: dict[str, str] = field(default_factory=dict)

    def _load(self, robots_url: str) -> urllib.robotparser.RobotFileParser | None:
        if robots_url in self._parsers:
            return self._parsers[robots_url]
        if robots_url in self._failures:
            return None
        if self.fetcher is None:
            return None
        try:
            status, text = self.fetcher(robots_url)
        except Exception as exc:  # network error -> conservative allow w/ note
            self._failures[robots_url] = f"fetch error: {exc}"
            return None
        if status >= 400:
            # 4xx/5xx robots means "no restrictions" per convention.
            self._parsers[robots_url] = None  # type: ignore[assignment]
            return None
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(text.splitlines())
        self._parsers[robots_url] = parser
        return parser

    def allowed(self, url: str, robots_url: str | None = None, user_agent: str = "*") -> bool:
        if robots_url is None:
            parts = urllib.parse.urlsplit(url)
            robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
        parser = self._load(robots_url)
        if parser is None:
            return self.default_allow
        return parser.can_fetch(user_agent, url)


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str = ""
    canonical_url: str | None = None


class PolicyGate:
    """Combines SSRF, robots, and per-source policy into one decision."""

    def __init__(self, robots: RobotsCache | None = None, resolve_dns: bool = True) -> None:
        self.robots = robots or RobotsCache()
        self.resolve_dns = resolve_dns

    def check(
        self,
        url: str,
        *,
        robots_url: str | None = None,
        keep_params: set[str] | None = None,
        user_agent: str = "*",
    ) -> PolicyDecision:
        canonical = canonicalize_url(url, keep_params=keep_params)
        try:
            assert_ssrf_safe(canonical, resolve_dns=self.resolve_dns)
        except SSRFError as exc:
            return PolicyDecision(False, f"ssrf: {exc}", canonical)
        if not self.robots.allowed(canonical, robots_url=robots_url, user_agent=user_agent):
            return PolicyDecision(False, "robots: disallowed", canonical)
        return PolicyDecision(True, canonical_url=canonical)

"""HTTP fetching with conditional requests, backoff, rate limits and raw storage.

Raw responses are immutable and append-only: a body whose SHA-256 already
exists is never rewritten; failed fetches never overwrite a prior success
(brief §4.3).
"""

from __future__ import annotations

import hashlib
import random
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .models import FetchResult

USER_AGENT = "rag-prepare-osint-collector/0.1 (+research; contact: local)"

_EXT_BY_MIME = {
    "text/html": ".html",
    "application/json": ".json",
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "text/vtt": ".vtt",
    "text/csv": ".csv",
    "application/xml": ".xml",
    "text/xml": ".xml",
}


def sniff_mime(header_content_type: str | None, body_prefix: bytes) -> str:
    """Combine the Content-Type header with magic bytes; magic wins for the
    formats we persist (brief §4.3 MIME sniffing)."""
    if body_prefix.startswith(b"%PDF"):
        return "application/pdf"
    if body_prefix[:2] == b"PK":
        return "application/zip"
    stripped = body_prefix.lstrip()[:200].lower()
    if stripped.startswith(b"<?xml") or stripped.startswith(b"<urlset") or stripped.startswith(
        b"<rss"
    ):
        return "application/xml"
    if stripped.startswith(b"<!doctype html") or stripped.startswith(b"<html"):
        return "text/html"
    if stripped.startswith((b"{", b"[")):
        return "application/json"
    if header_content_type:
        return header_content_type.split(";")[0].strip().lower()
    return "application/octet-stream"


@dataclass
class DomainRateLimiter:
    """Per-domain minimum interval between requests (brief §4.3: default
    concurrency 1–2, per-domain throttling)."""

    default_interval: float = 2.0
    per_domain: dict[str, float] = field(default_factory=dict)
    _last: dict[str, float] = field(default_factory=dict)
    sleeper: callable = time.sleep
    clock: callable = time.monotonic

    def interval_for(self, url: str) -> float:
        host = urllib.parse.urlsplit(url).netloc.lower()
        return self.per_domain.get(host, self.default_interval)

    def wait(self, url: str) -> None:
        host = urllib.parse.urlsplit(url).netloc.lower()
        interval = self.interval_for(url)
        last = self._last.get(host)
        now = self.clock()
        if last is not None and now - last < interval:
            self.sleeper(interval - (now - last))
        self._last[host] = self.clock()


class HttpFetcher:
    def __init__(
        self,
        raw_root: Path | str,
        *,
        client: httpx.Client | None = None,
        rate_limiter: DomainRateLimiter | None = None,
        max_bytes: int = 25 * 1024 * 1024,
        timeout: float = 30.0,
        max_retries: int = 3,
        max_redirects: int = 10,
        adapter: str | None = None,
        adapter_version: str | None = None,
    ) -> None:
        self.raw_root = Path(raw_root)
        self.client = client or httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            max_redirects=max_redirects,
            headers={"User-Agent": USER_AGENT},
        )
        self.rate_limiter = rate_limiter or DomainRateLimiter()
        self.max_bytes = max_bytes
        self.max_retries = max_retries
        self.adapter = adapter
        self.adapter_version = adapter_version
        self.sleeper = time.sleep  # injectable for tests

    # ------------------------------------------------------------------ API

    def fetch(
        self,
        url: str,
        *,
        etag: str | None = None,
        last_modified: str | None = None,
        headers: dict[str, str] | None = None,
        method: str = "GET",
        json_body: dict | None = None,
    ) -> FetchResult:
        req_headers = dict(headers or {})
        if etag:
            req_headers["If-None-Match"] = etag
        if last_modified:
            req_headers["If-Modified-Since"] = last_modified

        result = FetchResult(url=url, adapter=self.adapter, adapter_version=self.adapter_version)
        for attempt in range(self.max_retries + 1):
            self.rate_limiter.wait(url)
            try:
                response = self.client.request(method, url, headers=req_headers, json=json_body)
            except httpx.HTTPError as exc:
                result.error = f"transport: {exc.__class__.__name__}: {exc}"
                result.retryable = True
                if attempt < self.max_retries:
                    self.sleeper(self._backoff(attempt))
                    continue
                return result

            result.status = response.status_code
            result.final_url = str(response.url)
            result.etag = response.headers.get("ETag")
            result.last_modified = response.headers.get("Last-Modified")

            if response.status_code == 304:
                result.not_modified = True
                result.ok = True
                return result
            if response.status_code == 429 or response.status_code >= 500:
                result.error = f"http {response.status_code}"
                result.retryable = True
                if attempt < self.max_retries:
                    retry_after = response.headers.get("Retry-After")
                    self.sleeper(self._retry_after_seconds(retry_after, attempt))
                    continue
                return result
            if response.status_code >= 400:
                result.error = f"http {response.status_code}"
                result.retryable = False
                return result

            body = response.content
            if len(body) > self.max_bytes:
                result.error = f"response too large: {len(body)} > {self.max_bytes}"
                result.retryable = False
                return result
            result.mime_type = sniff_mime(response.headers.get("Content-Type"), body[:512])
            result.size_bytes = len(body)
            result.sha256 = hashlib.sha256(body).hexdigest()
            result.raw_path = str(self._store_raw(result, body))
            result.ok = True
            return result
        return result  # pragma: no cover

    # -------------------------------------------------------------- internals

    def _backoff(self, attempt: int) -> float:
        return min(60.0, 2.0**attempt) + random.uniform(0, 0.5)  # noqa: S311 - jitter only

    def _retry_after_seconds(self, header: str | None, attempt: int) -> float:
        if header:
            try:
                return max(0.0, float(header))
            except ValueError:
                pass  # HTTP-date form: fall back to backoff
        return self._backoff(attempt)

    def _store_raw(self, result: FetchResult, body: bytes) -> Path:
        """Append-only raw store: raw/<adapter>/<YYYY-MM-DD>/<sha256><ext>.
        Existing identical objects are reused, never overwritten."""
        assert result.sha256 is not None
        date_dir = result.fetched_at.strftime("%Y-%m-%d")
        ext = _EXT_BY_MIME.get(result.mime_type or "", ".bin")
        folder = self.raw_root / (self.adapter or "unknown") / date_dir
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{result.sha256}{ext}"
        if not path.exists():
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(body)
            tmp.replace(path)  # atomic publish
        return path

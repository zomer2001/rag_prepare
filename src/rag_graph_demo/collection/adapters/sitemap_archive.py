"""Direct historical backfill from publisher-provided XML sitemaps.

Only URLs whose path explicitly contains a configured company/product keyword
are selected.  This keeps the crawl bounded: sitemap documents are read, but
unrelated article pages are never fetched.  Search pages forbidden by robots
are not used, and no access controls are bypassed.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from ..dedupe import make_record_id, normalized_hash
from ..models import (
    ContentBlock,
    DiscoveryCandidate,
    FetchResult,
    NormalizedRecord,
    Provenance,
    Quality,
    SourceBlock,
    SourceConfig,
    SourceState,
)
from ..normalize import parse_date
from ..parsers import ParsedContent
from ..policy import canonicalize_url
from .base import CollectionContext, make_candidate_id


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


class SitemapArchiveAdapter:
    name = "sitemap_archive"
    version = "1.0.0"

    def discover(
        self,
        source: SourceConfig,
        state: SourceState,
        ctx: CollectionContext,
        fetcher: Any,
    ) -> list[DiscoveryCandidate]:
        roots = [str(v) for v in source.entry.get("sitemaps", [])]
        keywords = [str(v).casefold() for v in source.entry.get("url_keywords", [])]
        include_prefixes = [str(v) for v in source.entry.get("include_prefixes", [])]
        exclude_parts = [str(v).casefold() for v in source.entry.get("exclude_url_parts", [])]
        max_sitemaps = int(source.entry.get("max_sitemaps", 100))
        if not roots or not keywords:
            raise ValueError(f"{source.source_id}: sitemaps and url_keywords are required")

        queue = list(roots)
        visited: set[str] = set()
        candidates: list[DiscoveryCandidate] = []
        seen_urls: set[str] = set()
        while queue and len(visited) < max_sitemaps:
            sitemap_url = queue.pop(0)
            if sitemap_url in visited:
                continue
            visited.add(sitemap_url)
            result = fetcher.fetch(sitemap_url)
            if not result.ok or not result.raw_path:
                raise RuntimeError(
                    f"sitemap fetch failed for {sitemap_url}: {result.error or result.status}"
                )
            root = ET.fromstring(Path(result.raw_path).read_bytes())
            if _local(root.tag) == "sitemapindex":
                for node in root.iter():
                    if _local(node.tag) == "loc" and node.text:
                        child = node.text.strip()
                        if child and child not in visited:
                            queue.append(child)
                continue

            for url_node in root:
                if _local(url_node.tag) != "url":
                    continue
                loc = None
                lastmod = None
                for node in url_node:
                    if _local(node.tag) == "loc" and node.text:
                        loc = node.text.strip()
                    elif _local(node.tag) == "lastmod" and node.text:
                        lastmod = node.text.strip()
                if not loc:
                    continue
                canonical = canonicalize_url(loc)
                if include_prefixes and not any(canonical.startswith(p) for p in include_prefixes):
                    continue
                searchable = unquote(urlsplit(canonical).path).casefold().replace("-", " ")
                matched = sorted({kw for kw in keywords if kw.replace("-", " ") in searchable})
                if not matched or any(part in canonical.casefold() for part in exclude_parts):
                    continue
                if ctx.since and lastmod and lastmod[:10] < ctx.since:
                    continue
                if canonical in seen_urls:
                    continue
                seen_urls.add(canonical)
                candidates.append(
                    DiscoveryCandidate(
                        candidate_id=make_candidate_id(source.source_id, canonical, None),
                        source_id=source.source_id,
                        url=canonical,
                        discovered_by="sitemap",
                        lastmod=lastmod,
                        extra={
                            "sitemap_url": sitemap_url,
                            "matched_keywords": matched,
                        },
                    )
                )
                if ctx.limit and len(candidates) >= ctx.limit:
                    return candidates
        return candidates

    def build_record(
        self,
        source: SourceConfig,
        candidate: DiscoveryCandidate,
        fetch: FetchResult,
        parsed: ParsedContent,
        raw_payload: dict[str, Any] | None,
    ) -> NormalizedRecord:
        canonical = canonicalize_url(candidate.url or fetch.final_url or fetch.url)
        title = parsed.title or canonical.rstrip("/").rsplit("/", 1)[-1].replace("-", " ")
        published_raw = parsed.published_at_raw or candidate.lastmod
        published, confidence = parse_date(published_raw)
        body = parsed.body_markdown
        version_id = normalized_hash(body or title)
        logical_source_id = source.source_id.removesuffix("_archive")
        warnings = list(parsed.warnings)
        needs_review = parsed.needs_review
        if not body:
            warnings.append("historical page body unavailable")
            needs_review = True
        return NormalizedRecord(
            record_id=make_record_id(logical_source_id, None, canonical),
            version_id=version_id,
            record_type=source.record_type,
            source=SourceBlock(
                source_id=logical_source_id,
                publisher=source.publisher,
                authority_tier=source.authority_tier,
            ),
            title=title,
            canonical_url=canonical,
            authors=parsed.authors,
            published_at=published,
            published_at_raw=published_raw,
            published_at_confidence=confidence,  # type: ignore[arg-type]
            fetched_at=fetch.fetched_at,
            body_markdown=body,
            content=ContentBlock(
                raw_sha256=fetch.sha256 or "",
                normalized_sha256=version_id,
                raw_path=fetch.raw_path,
                mime_type=fetch.mime_type or "text/html",
                license=source.policy.content_license,
            ),
            provenance=Provenance(
                discovered_by="sitemap",
                discovery_query="URL path: " + " OR ".join(source.entry.get("url_keywords", [])),
                adapter=self.name,
                adapter_version=self.version,
            ),
            quality=Quality(
                parse_confidence=parsed.parse_confidence,
                needs_review=needs_review,
                warnings=warnings,
            ),
            extension={
                "sitemap_url": candidate.extra.get("sitemap_url"),
                "matched_keywords": candidate.extra.get("matched_keywords", []),
                "historical_backfill": True,
            },
        )

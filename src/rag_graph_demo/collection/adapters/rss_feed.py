"""Keyword-filtered RSS/Atom adapter for P1 news and think-tank sources.

Feeds are discovery mechanisms only.  The runner subsequently fetches the
canonical article/report page and parses that page as the verified body; feed
descriptions remain discovery summaries and never become document content.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

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

_TAG_RE = re.compile(r"<[^>]+>")


def _text(node: ET.Element, names: tuple[str, ...]) -> str | None:
    for child in node.iter():
        local = child.tag.rsplit("}", 1)[-1].lower()
        if local in names and child.text and child.text.strip():
            return child.text.strip()
    return None


def _link(node: ET.Element) -> str | None:
    for child in node.iter():
        if child.tag.rsplit("}", 1)[-1].lower() != "link":
            continue
        href = (child.attrib.get("href") or "").strip()
        rel = (child.attrib.get("rel") or "alternate").lower()
        if href and rel in {"alternate", ""}:
            return href
        if child.text and child.text.strip():
            return child.text.strip()
    return None


def _plain(value: str | None) -> str:
    return " ".join(_TAG_RE.sub(" ", html.unescape(value or "")).split())


class RssFeedAdapter:
    name = "rss_feed"
    version = "1.0.0"

    def discover(
        self,
        source: SourceConfig,
        state: SourceState,
        ctx: CollectionContext,
        fetcher: Any,
    ) -> list[DiscoveryCandidate]:
        feed_urls = source.entry.get("feeds") or []
        keywords = [str(v).casefold() for v in source.entry.get("keywords", [])]
        if not feed_urls:
            raise ValueError(f"{source.source_id}: entry.feeds is empty")
        if not keywords:
            raise ValueError(f"{source.source_id}: entry.keywords is empty")

        candidates: list[DiscoveryCandidate] = []
        seen_urls: set[str] = set()
        for feed_url in feed_urls:
            result = fetcher.fetch(str(feed_url))
            if not result.ok or not result.raw_path:
                raise RuntimeError(
                    f"feed fetch failed for {feed_url}: {result.error or result.status}"
                )
            root = ET.fromstring(Path(result.raw_path).read_bytes())
            entries = [
                node
                for node in root.iter()
                if node.tag.rsplit("}", 1)[-1].lower() in {"item", "entry"}
            ]
            for entry in entries:
                url = _link(entry)
                title = _text(entry, ("title",))
                summary_raw = _text(entry, ("description", "summary", "content", "encoded"))
                published = _text(entry, ("pubdate", "published", "updated", "date"))
                author = _text(entry, ("creator", "author"))
                haystack = " ".join((title or "", _plain(summary_raw))).casefold()
                matched = sorted({kw for kw in keywords if kw in haystack})
                if not url or not matched:
                    continue
                canonical = canonicalize_url(url)
                if canonical in seen_urls:
                    continue
                normalized_date, _ = parse_date(published)
                if ctx.since and normalized_date and normalized_date[:10] < ctx.since:
                    continue
                seen_urls.add(canonical)
                candidates.append(
                    DiscoveryCandidate(
                        candidate_id=make_candidate_id(source.source_id, canonical, None),
                        source_id=source.source_id,
                        url=canonical,
                        title=_plain(title),
                        summary=_plain(summary_raw) or None,
                        discovered_by="rss",
                        lastmod=published,
                        extra={
                            "feed_url": str(feed_url),
                            "matched_keywords": matched,
                            "feed_author": _plain(author) or None,
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
        title = parsed.title or candidate.title or canonical
        published_raw = parsed.published_at_raw or candidate.lastmod
        published, confidence = parse_date(published_raw)
        authors = parsed.authors
        if not authors and candidate.extra.get("feed_author"):
            authors = [str(candidate.extra["feed_author"])]
        body = parsed.body_markdown
        version_id = normalized_hash(body or title)
        warnings = list(parsed.warnings)
        needs_review = parsed.needs_review
        if not body:
            warnings.append(
                "article/report body unavailable; RSS summary retained only in discovery"
            )
            needs_review = True
        return NormalizedRecord(
            record_id=make_record_id(source.source_id, None, canonical),
            version_id=version_id,
            record_type=source.record_type,
            source=SourceBlock(
                source_id=source.source_id,
                publisher=source.publisher,
                authority_tier=source.authority_tier,
            ),
            title=title,
            canonical_url=canonical,
            authors=authors,
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
                discovered_by="rss",
                discovery_query=" OR ".join(source.entry.get("keywords", [])),
                adapter=self.name,
                adapter_version=self.version,
            ),
            quality=Quality(
                parse_confidence=parsed.parse_confidence,
                needs_review=needs_review,
                warnings=warnings,
            ),
            extension={
                "feed_url": candidate.extra.get("feed_url"),
                "matched_keywords": candidate.extra.get("matched_keywords", []),
                "feed_summary": candidate.summary,
            },
        )

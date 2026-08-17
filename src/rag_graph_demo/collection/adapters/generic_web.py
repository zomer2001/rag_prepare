"""Generic canonical-web-page adapter used by topic-driven OSINT runs.

Discovery is handled by the topic layer (RSS, sitemaps, search APIs, or seed
URLs).  This adapter only maps a fetched canonical page into the collection
pipeline's auditable normalized record format.
"""

from __future__ import annotations

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
from .base import CollectionContext


class GenericWebAdapter:
    name = "generic_web"
    version = "1.0.0"

    def discover(
        self,
        source: SourceConfig,
        state: SourceState,
        ctx: CollectionContext,
        fetcher: Any,
    ) -> list[DiscoveryCandidate]:
        del source, state, ctx, fetcher
        return []

    def build_record(
        self,
        source: SourceConfig,
        candidate: DiscoveryCandidate,
        fetch: FetchResult,
        parsed: ParsedContent,
        raw_payload: dict[str, Any] | None,
    ) -> NormalizedRecord:
        del raw_payload
        canonical = canonicalize_url(candidate.url or fetch.final_url or fetch.url)
        title = parsed.title or candidate.title or canonical
        published_raw = parsed.published_at_raw or candidate.lastmod
        published, confidence = parse_date(published_raw)
        body = parsed.body_markdown
        version_id = normalized_hash(body or title)
        warnings = list(parsed.warnings)
        needs_review = parsed.needs_review
        if not body:
            warnings.append("canonical page body unavailable")
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
                discovered_by=candidate.discovered_by,
                discovery_query=candidate.discovery_query,
                adapter=self.name,
                adapter_version=self.version,
            ),
            quality=Quality(
                parse_confidence=parsed.parse_confidence,
                needs_review=needs_review,
                warnings=warnings,
            ),
            extension={
                "topic_id": candidate.extra.get("topic_id"),
                "topic_round": candidate.extra.get("topic_round"),
                "matched_terms": candidate.extra.get("matched_terms", []),
                "discovery_source": candidate.extra.get("discovery_source"),
            },
        )

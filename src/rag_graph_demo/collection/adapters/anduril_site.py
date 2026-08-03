"""Anduril official site adapter (P0).

Verified 2026-08-03: https://www.anduril.com/robots.txt allows all agents and
points to https://www.anduril.com/sitemap.xml, a plain urlset that also
carries the Google News namespace (news:title / news:publication_date) and
per-URL lastmod. Discovery = sitemap walk; the product seed list in
config/entities.yaml is only used to tag candidates, never as a closed list
(brief §2.2).

Extraction strategy: Next.js page — prefer __NEXT_DATA__ JSON-LD/meta when
present, otherwise the generic HTML extractor with boilerplate removal.
Fallback: if the sitemap is unreachable, candidates already logged in
data/collection/discovery/ can be re-run with `retry`.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any

from ..dedupe import make_record_id, normalized_hash
from ..models import (
    AuthorityTier,
    ContentBlock,
    DiscoveryCandidate,
    FetchResult,
    NormalizedRecord,
    Provenance,
    Quality,
    RecordType,
    SourceBlock,
    SourceConfig,
    SourceState,
)
from ..normalize import parse_date
from ..parsers import ParsedContent
from ..policy import canonicalize_url
from .base import CollectionContext, make_candidate_id

NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    "news": "http://www.google.com/schemas/sitemap-news/0.9",
}

_SLUG_PRODUCT_RE = re.compile(r"^[a-z0-9-]+$")


class AndurilSiteAdapter:
    name = "anduril_site"
    version = "1.0.0"

    def discover(
        self,
        source: SourceConfig,
        state: SourceState,
        ctx: CollectionContext,
        fetcher: Any,
    ) -> list[DiscoveryCandidate]:
        sitemap_url = source.entry.get("sitemap", "https://www.anduril.com/sitemap.xml")
        result = fetcher.fetch(sitemap_url)
        if not result.ok or result.raw_path is None:
            raise RuntimeError(f"sitemap fetch failed for {sitemap_url}: {result.error}")
        xml_text = open(result.raw_path, encoding="utf-8").read()  # noqa: SIM115
        root = ET.fromstring(xml_text)

        seeds = {s.lower() for s in (source.entry.get("product_seeds") or [])}
        candidates: list[DiscoveryCandidate] = []
        for url_node in root.findall("sm:url", NS):
            loc = url_node.findtext("sm:loc", default="", namespaces=NS).strip()
            if not loc:
                continue
            lastmod = url_node.findtext("sm:lastmod", default=None, namespaces=NS)
            news_node = url_node.find("news:news", NS)
            title = None
            published = None
            if news_node is not None:
                title = news_node.findtext("news:title", default=None, namespaces=NS)
                published = news_node.findtext(
                    "news:publication_date", default=None, namespaces=NS
                )
            kind = self._classify(loc)
            slug = loc.rstrip("/").rsplit("/", 1)[-1].lower()
            extra: dict[str, Any] = {"page_kind": kind}
            if slug in seeds:
                extra["product_seed_hit"] = slug
            if ctx.since and lastmod and lastmod[:10] < ctx.since:
                continue
            candidates.append(
                DiscoveryCandidate(
                    candidate_id=make_candidate_id(source.source_id, loc, None),
                    source_id=source.source_id,
                    url=loc,
                    title=title,
                    discovered_by="sitemap",
                    lastmod=lastmod or published,
                    extra=extra,
                )
            )
            if ctx.limit and len(candidates) >= ctx.limit:
                break
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
        published, confidence = parse_date(parsed.published_at_raw or candidate.lastmod)
        body = parsed.body_markdown
        warnings = list(parsed.warnings)
        parse_confidence = parsed.parse_confidence
        needs_review = parsed.needs_review
        if not body:
            # Live check 2026-08-03: anduril.com is fully client-rendered
            # (Next.js + Sanity); static HTML carries title + JSON-LD but no
            # article body. Fall back to the JSON-LD description — a real
            # source field, flagged for review; full body needs a future
            # browser-rendering adapter (see docs/source_playbook.md).
            for node in parsed.json_ld:
                description = node.get("description") if isinstance(node, dict) else None
                if description:
                    body = str(description).strip()
                    warnings.append("body limited to JSON-LD description (client-rendered page)")
                    parse_confidence = min(parse_confidence, 0.5)
                    needs_review = True
                    break
        record_type = (
            RecordType.NEWS
            if candidate.extra.get("page_kind") == "news"
            else RecordType.OFFICIAL_PAGE
        )
        version_id = normalized_hash(body or title)
        return NormalizedRecord(
            record_id=make_record_id(source.source_id, None, canonical),
            version_id=version_id,
            record_type=record_type,
            source=SourceBlock(
                source_id=source.source_id,
                publisher=source.publisher,
                authority_tier=AuthorityTier.P0,
                provider_record_id=None,
            ),
            title=title,
            canonical_url=canonical,
            authors=parsed.authors,
            published_at=published,
            published_at_raw=parsed.published_at_raw or candidate.lastmod,
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
                parse_confidence=parse_confidence,
                needs_review=needs_review,
                warnings=warnings,
            ),
            extension={
                "page_kind": candidate.extra.get("page_kind"),
                "product_seed_hit": candidate.extra.get("product_seed_hit"),
                "json_ld_types": [n.get("@type") for n in parsed.json_ld if isinstance(n, dict)],
                "sitemap_lastmod": candidate.lastmod,
            },
        )

    @staticmethod
    def _classify(url: str) -> str:
        path = url.rstrip("/").lower()
        if "/news/" in path:
            return "news"
        slug = path.rsplit("/", 1)[-1]
        if _SLUG_PRODUCT_RE.match(slug) and "/" not in slug:
            return "product_or_page"
        return "page"

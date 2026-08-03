"""OpenAlex scholarly metadata adapter (P0/P1).

Verified 2026-08-03, no key required (polite pool via ``mailto`` param).
Strategy: affiliation-first, not full-text-first — a naive full-text search
for "Anduril Industries" is dominated by the unrelated MATLAB toolbox named
ANDURIL, so the default query filters on raw affiliation strings and falls
back to curated title/abstract queries from config/queries.yaml. Metadata
only: no PDF is stored unless the OA license explicitly allows it
(brief §2.4).
"""

from __future__ import annotations

import urllib.parse
from typing import Any

from ..dedupe import make_record_id, normalized_hash
from ..models import (
    AuthorityTier,
    ContentBlock,
    DiscoveryCandidate,
    FetchResult,
    NormalizedRecord,
    PaperExtension,
    Provenance,
    Quality,
    RecordType,
    SourceBlock,
    SourceConfig,
    SourceState,
)
from ..normalize import parse_date
from ..parsers import ParsedContent
from .base import CollectionContext, make_candidate_id

WORKS_URL = "https://api.openalex.org/works"


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    if not inverted_index:
        return None
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for idx in idxs:
            positions.append((idx, word))
    return " ".join(word for _, word in sorted(positions)) or None


class OpenAlexAdapter:
    name = "openalex"
    version = "1.0.0"

    def discover(
        self,
        source: SourceConfig,
        state: SourceState,
        ctx: CollectionContext,
        fetcher: Any,
    ) -> list[DiscoveryCandidate]:
        affiliation_terms = source.entry.get("affiliation_terms") or ["Anduril Industries"]
        cursor = state.cursor.get("cursor") or "*"
        per_page = min(source.entry.get("per_page", 50), 200)
        mailto = source.entry.get("mailto")

        candidates: list[DiscoveryCandidate] = []
        for term in affiliation_terms:
            params = {
                "filter": f"raw_affiliation_strings.search:{term}",
                "per-page": str(per_page),
                "cursor": cursor,
            }
            if mailto:
                params["mailto"] = mailto
            url = WORKS_URL + "?" + urllib.parse.urlencode(params)
            result = fetcher.fetch(url)
            if not result.ok or result.raw_path is None:
                raise RuntimeError(f"openalex query failed: {result.error}")
            import json as _json

            with open(result.raw_path, encoding="utf-8") as fh:
                data = _json.load(fh)
            for work in data.get("results", []):
                work_id = work.get("id")
                if not work_id:
                    continue
                candidates.append(
                    DiscoveryCandidate(
                        candidate_id=make_candidate_id(source.source_id, None, work_id),
                        source_id=source.source_id,
                        url=work_id.replace("https://openalex.org/", WORKS_URL + "/"),
                        provider_record_id=work_id,
                        title=work.get("title"),
                        discovered_by="api",
                        discovery_query=f"raw_affiliation_strings.search:{term}",
                        lastmod=work.get("publication_date"),
                        extra={"work": work},
                    )
                )
                if ctx.limit and len(candidates) >= ctx.limit:
                    return candidates
            next_cursor = (data.get("meta") or {}).get("next_cursor")
            if not next_cursor or next_cursor == cursor or not data.get("results"):
                continue
        return candidates

    def build_record(
        self,
        source: SourceConfig,
        candidate: DiscoveryCandidate,
        fetch: FetchResult,
        parsed: ParsedContent,
        raw_payload: dict[str, Any] | None,
    ) -> NormalizedRecord:
        work = candidate.extra.get("work") or raw_payload or {}
        work_id = candidate.provider_record_id or work.get("id")
        authors: list[dict[str, Any]] = []
        affiliations: set[str] = set()
        for authorship in work.get("authorships", []) or []:
            author = authorship.get("author") or {}
            institutions = [
                (inst.get("display_name") or "") for inst in authorship.get("institutions") or []
            ]
            authors.append(
                {
                    "name": author.get("display_name"),
                    "openalex_id": author.get("id"),
                    "orcid": author.get("orcid"),
                    "institutions": [i for i in institutions if i],
                    "raw_affiliations": authorship.get("raw_affiliation_strings") or [],
                }
            )
            affiliations.update(i for i in institutions if i)

        abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
        published, confidence = parse_date(work.get("publication_date"))
        location = work.get("primary_location") or {}
        venue = ((location.get("source") or {}).get("display_name")) or None
        oa = work.get("open_access") or {}
        title = work.get("title") or candidate.title or work_id or "untitled"

        body_parts = []
        if abstract:
            body_parts += ["## Abstract", "", abstract]
        body = "\n".join(body_parts)
        version_id = normalized_hash((title or "") + (abstract or "") + (work_id or ""))
        return NormalizedRecord(
            record_id=make_record_id(source.source_id, work_id, None),
            version_id=version_id,
            record_type=RecordType.PAPER,
            source=SourceBlock(
                source_id=source.source_id,
                publisher=source.publisher,
                authority_tier=AuthorityTier.P0,
                provider_record_id=work_id,
            ),
            title=title,
            canonical_url=work.get("doi") or work_id,
            authors=[a["name"] for a in authors if a.get("name")],
            published_at=published,
            published_at_raw=work.get("publication_date"),
            published_at_confidence=confidence,  # type: ignore[arg-type]
            fetched_at=fetch.fetched_at,
            body_markdown=body,
            content=ContentBlock(
                raw_sha256=fetch.sha256 or "",
                normalized_sha256=version_id,
                raw_path=fetch.raw_path,
                mime_type="application/json",
                license=source.policy.content_license,
            ),
            paper=PaperExtension(
                doi=work.get("doi"),
                openalex_id=work_id,
                venue=venue,
                abstract=abstract,
                authors=authors,
                affiliations=sorted(affiliations),
                open_access_url=(location.get("landing_page_url") if oa.get("is_oa") else None),
                cited_by_count=work.get("cited_by_count"),
                fulltext_license=None,  # PDF stored only when license verified
            ),
            provenance=Provenance(
                discovered_by=candidate.discovered_by,
                discovery_query=candidate.discovery_query,
                adapter=self.name,
                adapter_version=self.version,
            ),
            quality=Quality(
                parse_confidence=parsed.parse_confidence,
                needs_review=parsed.needs_review,
                warnings=parsed.warnings,
            ),
            extension={"provider_work_keys": sorted(work.keys())},
        )

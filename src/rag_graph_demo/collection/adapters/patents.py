"""PatentsView (USPTO) patent adapter (P0).

Status verified 2026-08-03: the legacy api.patentsview.org query endpoint now
301-redirects to the USPTO Open Data Portal web page (decommissioned). The
current search API at https://search.patentsview.org/api/v1/patent/ requires a
free API key sent as ``X-Api-Key``. This adapter therefore:

- reads the key ONLY from the env var named in the source policy
  (``PATENTSVIEW_API_KEY``); keys never live in config, logs or reports;
- performs a *policy skip* when no key is configured (brief §12), returning
  zero candidates and recording the reason instead of failing or bypassing;
- maps each publication as its own record. Family merging (INPADOC/simple
  family) keeps per-publication records and only adds relations — different
  publications of one family must never be deleted as duplicates (brief §2.5,
  §10).

Legal status fields from any provider are stored as ``provider_assertion``
with ``status_checked_at``, never as legal advice.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

from ..dedupe import make_record_id, normalized_hash
from ..models import (
    AuthorityTier,
    ContentBlock,
    DiscoveryCandidate,
    FetchResult,
    NormalizedRecord,
    PatentExtension,
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

API_URL = "https://search.patentsview.org/api/v1/patent/"

PATENT_FIELDS = [
    "patent_id",
    "patent_title",
    "patent_abstract",
    "patent_date",
    "application.filing_date",
    "assignees.assignee_organization",
    "inventors.inventor_name_first",
    "inventors.inventor_name_last",
    "cpc_current.cpc_section",
    "cpc_current.cpc_subclass",
]


class PatentsViewAdapter:
    name = "patentsview"
    version = "1.0.0"

    def __init__(self, environ: dict[str, str] | None = None) -> None:
        self._environ = environ if environ is not None else os.environ

    def _api_key(self, source: SourceConfig) -> str | None:
        env_name = source.policy.api_key_env or "PATENTSVIEW_API_KEY"
        key = self._environ.get(env_name)
        return key.strip() if key and key.strip() else None

    def discover(
        self,
        source: SourceConfig,
        state: SourceState,
        ctx: CollectionContext,
        fetcher: Any,
    ) -> list[DiscoveryCandidate]:
        key = self._api_key(source)
        if key is None:
            # Policy skip, not an error and not a bypass attempt (brief §12).
            import json as _json

            skip = {
                "source_id": source.source_id,
                "reason": "missing API key env PATENTSVIEW_API_KEY; adapter skipped per policy",
                "checked_at": datetime.now(UTC).isoformat(),
            }
            state.health["policy_skip"] = skip
            state.health["policy_skip_json"] = _json.dumps(skip)
            return []

        assignee_terms = source.entry.get("assignee_terms") or ["Anduril Industries"]
        candidates: list[DiscoveryCandidate] = []
        seen_ids: set[str] = set()  # same patent may match several alias queries
        import json as _json
        import urllib.parse as _up

        for term in assignee_terms:
            query = {"_text_phrase": {"assignees.assignee_organization": term}}
            params = _up.urlencode(
                {
                    "q": _json.dumps(query),
                    "f": _json.dumps(PATENT_FIELDS),
                    "o": _json.dumps({"size": min(ctx.limit or 100, 100)}),
                }
            )
            result = fetcher.fetch(f"{API_URL}?{params}", headers={"X-Api-Key": key})
            if not result.ok or result.raw_path is None:
                raise RuntimeError(f"patentsview query failed: {result.error}")
            with open(result.raw_path, encoding="utf-8") as fh:
                data = _json.load(fh)
            for patent in data.get("patents", []) or []:
                patent_id = patent.get("patent_id")
                if not patent_id or patent_id in seen_ids:
                    continue
                seen_ids.add(patent_id)
                candidates.append(
                    DiscoveryCandidate(
                        candidate_id=make_candidate_id(source.source_id, None, f"US{patent_id}"),
                        source_id=source.source_id,
                        url=f"https://patents.google.com/patent/US{patent_id}",
                        provider_record_id=f"US{patent_id}",
                        title=patent.get("patent_title"),
                        discovered_by="api",
                        discovery_query=f"assignee_text_phrase:{term}",
                        lastmod=patent.get("patent_date"),
                        extra={"patent": patent},
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
        patent = candidate.extra.get("patent") or raw_payload or {}
        publication = candidate.provider_record_id or f"US{patent.get('patent_id', '')}"
        inventors = [
            " ".join(
                filter(None, [inv.get("inventor_name_first"), inv.get("inventor_name_last")])
            )
            for inv in patent.get("inventors", []) or []
        ]
        assignees = [
            a.get("assignee_organization")
            for a in patent.get("assignees", []) or []
            if a.get("assignee_organization")
        ]
        cpc = sorted(
            {
                f"{c.get('cpc_section', '')}{c.get('cpc_subclass', '')}".strip()
                for c in patent.get("cpc_current", []) or []
            }
            - {""}
        )
        grant_date, _ = parse_date(patent.get("patent_date"))
        filing_date, _ = parse_date(
            (patent.get("application") or {}).get("filing_date")
            if isinstance(patent.get("application"), dict)
            else patent.get("filing_date")
        )
        abstract = patent.get("patent_abstract") or ""
        title = patent.get("patent_title") or candidate.title or publication
        body = f"## Abstract\n\n{abstract}" if abstract else ""
        version_id = normalized_hash(title + abstract + publication)
        return NormalizedRecord(
            record_id=make_record_id(source.source_id, publication, None),
            version_id=version_id,
            record_type=RecordType.PATENT,
            source=SourceBlock(
                source_id=source.source_id,
                publisher=source.publisher,
                authority_tier=AuthorityTier.P0,
                provider_record_id=publication,
            ),
            title=title,
            canonical_url=candidate.url,
            authors=inventors,
            published_at=grant_date,
            published_at_raw=patent.get("patent_date"),
            published_at_confidence="full" if grant_date else "none",
            fetched_at=fetch.fetched_at,
            body_markdown=body,
            content=ContentBlock(
                raw_sha256=fetch.sha256 or "",
                normalized_sha256=version_id,
                raw_path=fetch.raw_path,
                mime_type="application/json",
                license=source.policy.content_license,
            ),
            patent=PatentExtension(
                publication_number=publication,
                application_number=None,
                family_id=None,  # family merge is a later relation pass
                filing_date=filing_date,
                grant_date=grant_date,
                assignees_current=assignees,
                inventors=[i for i in inventors if i],
                cpc=cpc,
                legal_status_assertion=None,  # provider_assertion only, when present
                status_checked_at=None,
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
                warnings=([*parsed.warnings] if parsed.warnings else []),
            ),
            extension={"provider_patent_keys": sorted(patent.keys())},
        )

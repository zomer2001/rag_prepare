"""USAspending API adapter (P0, contracts).

Verified 2026-08-03, no API key required:
- Discovery: POST https://api.usaspending.gov/api/v2/search/spending_by_award/
  with recipient_search_text alias list + time_period cursor. Pagination via
  ``page`` + ``page_metadata.hasNext``.
- Per-award detail: GET https://api.usaspending.gov/api/v2/awards/{internal_id}/
  which separates potential value (ceiling) from obligation and outlay.

Money semantics (brief §2.3) are kept in distinct MoneyAmount entries with the
exact provider field names; nothing is converted or merged. Parent award IDs
are parsed from ``generated_internal_id`` (CONT_AWD_<piid>_<agency>_<parent>_<agency>)
and stored as relations, never merged into one record.
"""

from __future__ import annotations

from typing import Any

from ..dedupe import make_record_id, normalized_hash
from ..models import (
    AuthorityTier,
    ContentBlock,
    ContractExtension,
    DiscoveryCandidate,
    FetchResult,
    MoneyAmount,
    NormalizedRecord,
    Provenance,
    Quality,
    RecordType,
    Relation,
    SourceBlock,
    SourceConfig,
    SourceState,
)
from ..normalize import money_string, parse_date
from ..parsers import ParsedContent
from .base import CollectionContext, make_candidate_id

SEARCH_URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
DETAIL_URL = "https://api.usaspending.gov/api/v2/awards/{internal_id}/"
CONTRACT_TYPES = ["A", "B", "C", "D"]  # definitive + IDV contracts

SEARCH_FIELDS = [
    "Award ID",
    "Recipient Name",
    "Start Date",
    "End Date",
    "Award Amount",
    "Total Outlays",
    "Awarding Agency",
    "Awarding Sub Agency",
    "Contract Award Type",
    "recipient_id",
    "prime_award_recipient_id",
]


class USASpendingAdapter:
    name = "usaspending"
    version = "1.0.0"

    # ------------------------------------------------------------- discover
    def discover(
        self,
        source: SourceConfig,
        state: SourceState,
        ctx: CollectionContext,
        fetcher: Any,
    ) -> list[DiscoveryCandidate]:
        aliases = source.entry.get("recipient_aliases") or ["Anduril Industries"]
        start = ctx.since or source.entry.get("default_since") or "2017-06-01"
        end = state.cursor.get("run_end") or "2100-01-01"
        page = 1
        candidates: list[DiscoveryCandidate] = []
        while True:
            payload = {
                "filters": {
                    "recipient_search_text": aliases,
                    "award_type_codes": CONTRACT_TYPES,
                    "time_period": [{"start_date": start, "end_date": end}],
                },
                "fields": SEARCH_FIELDS,
                "limit": 100,
                "page": page,
                "sort": "Award ID",
                "order": "desc",
            }
            result = fetcher.fetch(SEARCH_URL, method="POST", json_body=payload)
            if not result.ok or result.raw_path is None:
                raise RuntimeError(f"usaspending search failed: {result.error}")
            import json as _json

            with open(result.raw_path, encoding="utf-8") as fh:
                data = _json.load(fh)
            for row in data.get("results", []):
                award_key = row.get("generated_internal_id") or str(row.get("internal_id"))
                detail_url = DETAIL_URL.format(internal_id=row["internal_id"])
                candidates.append(
                    DiscoveryCandidate(
                        candidate_id=make_candidate_id(source.source_id, None, award_key),
                        source_id=source.source_id,
                        url=detail_url,
                        provider_record_id=award_key,
                        title=f"{row.get('Award ID')} — {row.get('Recipient Name')}",
                        discovered_by="api",
                        discovery_query=f"recipient_search_text={aliases}",
                        extra={"row": row, "detail_url": detail_url},
                    )
                )
                if ctx.limit and len(candidates) >= ctx.limit:
                    return candidates
            if not data.get("page_metadata", {}).get("hasNext"):
                break
            page += 1
            if page > 25:  # hard pagination ceiling; tune after quota experience
                break
        return candidates

    # ---------------------------------------------------------- build record
    def build_record(
        self,
        source: SourceConfig,
        candidate: DiscoveryCandidate,
        fetch: FetchResult,
        parsed: ParsedContent,
        raw_payload: dict[str, Any] | None,
    ) -> NormalizedRecord:
        row = candidate.extra.get("row", {})
        detail = raw_payload or {}
        award_key = candidate.provider_record_id or ""
        piid = row.get("Award ID") or detail.get("piid")
        parent_piid = self._parent_from_key(award_key)

        amounts: list[MoneyAmount] = []

        def add_amount(kind: str, value: Any, source_field: str) -> None:
            rendered = money_string(value)
            if rendered is not None:
                amounts.append(
                    MoneyAmount(
                        kind=kind, value=rendered, currency="USD", source_field=source_field
                    )
                )

        add_amount(
            "current_award_amount", row.get("Award Amount"), "spending_by_award.Award Amount"
        )
        add_amount("outlay", row.get("Total Outlays"), "spending_by_award.Total Outlays")
        add_amount(
            "award_ceiling", detail.get("potential_award_amount"), "awards.potential_award_amount"
        )
        add_amount("obligation", detail.get("total_obligation"), "awards.total_obligation")
        if not any(a.kind == "outlay" for a in amounts):
            add_amount("outlay", detail.get("total_outlay"), "awards.total_outlay")

        start_iso, _ = parse_date(
            row.get("Start Date") or detail.get("period_of_performance_start_date")
        )
        end_iso, _ = parse_date(
            row.get("End Date") or detail.get("period_of_performance_current_end_date")
        )
        awarding_agency = row.get("Awarding Agency") or (detail.get("awarding_agency") or {}).get(
            "toptier_agency", {}
        ).get("name")
        awarding_office = (detail.get("awarding_agency") or {}).get("office_agency_name") or (
            detail.get("awarding_office") or {}
        ).get("office_name")
        description = detail.get("description") or row.get("Description")
        recipient_uei = (detail.get("recipient") or {}).get("uei")

        relations: list[Relation] = []
        if parent_piid and parent_piid != piid:
            relations.append(
                Relation(
                    relation_type="parent_award",
                    target_record_id=make_record_id(source.source_id, parent_piid, None),
                    machine_inferred=False,
                    rationale=f"parsed from generated_internal_id {award_key}",
                )
            )

        title = candidate.title or f"Award {piid or award_key}"
        body = self._render_body(piid, award_key, amounts, description, awarding_agency)
        version_id = normalized_hash(body + award_key)
        return NormalizedRecord(
            record_id=make_record_id(source.source_id, award_key, None),
            version_id=version_id,
            record_type=RecordType.CONTRACT,
            source=SourceBlock(
                source_id=source.source_id,
                publisher=source.publisher,
                authority_tier=AuthorityTier.P0,
                provider_record_id=award_key,
            ),
            title=title,
            canonical_url=(
                f"https://www.usaspending.gov/award/"
                f"{detail.get('generated_unique_award_id', award_key)}"
            )
            if award_key
            else None,
            published_at=None,
            published_at_raw=None,
            published_at_confidence="none",
            fetched_at=fetch.fetched_at,
            body_markdown=body,
            content=ContentBlock(
                raw_sha256=fetch.sha256 or "",
                normalized_sha256=version_id,
                raw_path=fetch.raw_path,
                mime_type="application/json",
                license=source.policy.content_license,
            ),
            relations=relations,
            contract=ContractExtension(
                award_key=award_key,
                piid=piid,
                parent_award_id=parent_piid,
                awarding_agency=awarding_agency,
                awarding_office=awarding_office,
                recipient_name=row.get("Recipient Name")
                or (detail.get("recipient") or {}).get("recipient_name"),
                recipient_uei=recipient_uei,
                award_type=row.get("Contract Award Type") or detail.get("type_description"),
                amounts=amounts,
                award_date=start_iso,
                period_start=start_iso,
                period_end=end_iso,
                description=description,
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
            extension={
                "provider_row": row,  # unknown fields preserved, not dropped
                "provider_detail_keys": sorted(detail.keys()),
                "detail_url": candidate.extra.get("detail_url"),
            },
        )

    @staticmethod
    def _parent_from_key(award_key: str) -> str | None:
        parts = award_key.split("_")
        # CONT_AWD_<piid>_<agency>_<parent_piid>_<parent_agency>
        if len(parts) >= 6 and parts[0] == "CONT" and parts[1] == "AWD":
            parent = parts[4]
            return None if parent in ("NONE", "-none-", "") else parent
        return None

    @staticmethod
    def _render_body(
        piid: str | None,
        award_key: str,
        amounts: list[MoneyAmount],
        description: str | None,
        agency: str | None,
    ) -> str:
        lines = [f"Contract award `{piid or award_key}`"]
        if agency:
            lines.append(f"- awarding agency: {agency}")
        for amount in amounts:
            lines.append(
                f"- {amount.kind}: {amount.value} {amount.currency} "
                f"(source field: `{amount.source_field}`)"
            )
        if description:
            lines += ["", "## Award description", "", description]
        return "\n".join(lines)

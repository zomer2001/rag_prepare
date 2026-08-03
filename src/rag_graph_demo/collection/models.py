"""Unified record models for the collection pipeline.

Mirrors ANDURIL_DATA_PIPELINE_AGENT_BRIEF.md §6. All timestamps are ISO 8601
UTC; money is stored as decimal strings together with currency and the exact
source field name, never as floats.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"


def utc_now() -> datetime:
    return datetime.now(UTC)


class AuthorityTier(StrEnum):
    P0 = "P0"  # primary, structured, high evidence value
    P1 = "P1"  # primary supplement or high-quality secondary
    P2 = "P2"  # leads and sentiment
    P3 = "P3"  # discovery only — never a verified clean body


class RecordType(StrEnum):
    OFFICIAL_PAGE = "official_page"
    CONTRACT = "contract"
    PAPER = "paper"
    PATENT = "patent"
    NEWS = "news"
    THINK_TANK = "think_tank"
    SOCIAL_POST = "social_post"
    COMMENT = "comment"
    VIDEO = "video"


class CandidateStatus(StrEnum):
    NEW = "new"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class ProductLinkConfidence(StrEnum):
    EXPLICIT = "explicit"
    STRONG_INFERENCE = "strong_inference"
    UNRESOLVED = "unresolved"


# ---------------------------------------------------------------------------
# Source registry / policy configuration
# ---------------------------------------------------------------------------


class SourcePolicy(BaseModel):
    """Legal / access policy for one source (brief §4.2, §8)."""

    robots_url: str | None = None
    robots_checked_at: str | None = None  # ISO date of last manual check
    tos_url: str | None = None
    tos_checked_at: str | None = None
    allowed_methods: list[str] = Field(default_factory=lambda: ["GET"])
    rate_limit_seconds: float = 2.0  # minimum interval between requests per host
    max_concurrency: int = 1
    auth: Literal["none", "api_key_env", "oauth"] = "none"
    api_key_env: str | None = None  # name of env var holding the key, never the key
    content_license: str | None = None
    requires_api_key: bool = False
    max_bytes: int = 25 * 1024 * 1024
    notes: str = ""


class SourceConfig(BaseModel):
    source_id: str
    name: str
    publisher: str
    authority_tier: AuthorityTier
    record_type: RecordType
    adapter: str  # dotted adapter key, e.g. "anduril_site"
    adapter_version: str = "1.0.0"
    enabled: bool = True
    groups: list[str] = Field(default_factory=list)  # e.g. ["p0"]
    entry: dict[str, Any] = Field(default_factory=dict)  # adapter-specific entry points
    default_query: str | None = None
    schedule_hint: str | None = None
    fixture_dir: str | None = None  # offline fixture root, relative to --fixtures-dir
    policy: SourcePolicy = Field(default_factory=SourcePolicy)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class DiscoveryCandidate(BaseModel):
    """A discovered URL/provider ID. Search summaries live here and ONLY here;
    they must never be promoted to a verified clean body (brief §0.4, §3.2)."""

    candidate_id: str
    source_id: str
    url: str | None = None
    provider_record_id: str | None = None
    title: str | None = None
    summary: str | None = None  # discovery-only snippet
    rank: int | None = None
    discovery_query: str | None = None
    discovered_by: Literal["sitemap", "api", "rss", "search", "citation", "manual"]
    discovered_at: datetime = Field(default_factory=utc_now)
    lastmod: str | None = None
    status: CandidateStatus = CandidateStatus.NEW
    policy_skip_reason: str | None = None
    parent_candidate_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


class FetchResult(BaseModel):
    url: str
    final_url: str | None = None
    status: int | None = None
    ok: bool = False
    not_modified: bool = False
    mime_type: str | None = None
    size_bytes: int = 0
    sha256: str | None = None
    fetched_at: datetime = Field(default_factory=utc_now)
    etag: str | None = None
    last_modified: str | None = None
    raw_path: str | None = None
    adapter: str | None = None
    adapter_version: str | None = None
    error: str | None = None
    retryable: bool = False


# ---------------------------------------------------------------------------
# Normalized record (brief §6)
# ---------------------------------------------------------------------------


class SourceBlock(BaseModel):
    source_id: str
    publisher: str
    authority_tier: AuthorityTier
    provider_record_id: str | None = None


class ContentBlock(BaseModel):
    raw_sha256: str
    normalized_sha256: str | None = None
    raw_path: str | None = None
    clean_path: str | None = None
    mime_type: str
    license: str | None = None


class Provenance(BaseModel):
    discovered_by: str
    discovery_query: str | None = None
    adapter: str
    adapter_version: str


class Quality(BaseModel):
    parse_confidence: float = 1.0
    completeness: float = 1.0
    needs_review: bool = False
    warnings: list[str] = Field(default_factory=list)


class EntityLink(BaseModel):
    entity_id: str
    name: str
    role: str | None = None  # e.g. "recipient", "awarding_agency", "author"
    machine_inferred: bool = False
    evidence: str | None = None


class ProductLink(BaseModel):
    product: str
    confidence: ProductLinkConfidence = ProductLinkConfidence.UNRESOLVED
    rationale: str | None = None
    evidence_url: str | None = None
    machine_inferred: bool = False


class Relation(BaseModel):
    relation_type: str  # e.g. "duplicate_of", "syndicated_from", "parent_award", "modification_of"
    target_record_id: str
    machine_inferred: bool = False
    rationale: str | None = None


class MoneyAmount(BaseModel):
    """One monetary figure, kept in original currency with its exact source
    field name. Derived conversions are NOT stored here (brief §2.3)."""

    kind: str  # award_ceiling | current_award_amount | obligation | outlay
    value: str  # decimal string
    currency: str = "USD"
    source_field: str  # exact field name in the provider payload


class ContractExtension(BaseModel):
    award_key: str | None = None  # provider-unique award key (generated_internal_id)
    piid: str | None = None
    parent_award_id: str | None = None
    awarding_agency: str | None = None
    awarding_office: str | None = None
    recipient_name: str | None = None
    recipient_uei: str | None = None
    naics: str | None = None
    psc: str | None = None
    award_type: str | None = None  # base_award | task_order | delivery_order | modification
    amounts: list[MoneyAmount] = Field(default_factory=list)
    announcement_date: str | None = None
    award_date: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    place_of_performance: str | None = None
    description: str | None = None


class PaperExtension(BaseModel):
    doi: str | None = None
    openalex_id: str | None = None
    arxiv_id: str | None = None
    venue: str | None = None
    abstract: str | None = None
    authors: list[dict[str, Any]] = Field(default_factory=list)
    affiliations: list[str] = Field(default_factory=list)
    open_access_url: str | None = None
    cited_by_count: int | None = None
    fulltext_license: str | None = None  # PDF stored only when license allows


class PatentExtension(BaseModel):
    publication_number: str | None = None
    application_number: str | None = None
    family_id: str | None = None
    priority_date: str | None = None
    filing_date: str | None = None
    publication_date: str | None = None
    grant_date: str | None = None
    assignees_current: list[str] = Field(default_factory=list)
    assignees_original: list[str] = Field(default_factory=list)
    inventors: list[str] = Field(default_factory=list)
    cpc: list[str] = Field(default_factory=list)
    legal_status_assertion: str | None = None  # provider_assertion, not legal advice
    status_checked_at: str | None = None


class NormalizedRecord(BaseModel):
    schema_version: str = SCHEMA_VERSION
    record_id: str
    version_id: str  # sha256 of normalized content
    version_index: int = 1
    record_type: RecordType
    source: SourceBlock
    title: str
    canonical_url: str | None = None
    language: str = "en"
    authors: list[str] = Field(default_factory=list)
    published_at: str | None = None  # ISO 8601, possibly partial ("2024", "2024-05")
    published_at_raw: str | None = None
    published_at_confidence: Literal["full", "month", "year", "none"] = "none"
    updated_at: str | None = None
    fetched_at: datetime = Field(default_factory=utc_now)
    first_seen_at: datetime = Field(default_factory=utc_now)
    last_seen_at: datetime = Field(default_factory=utc_now)
    body_markdown: str = ""
    content: ContentBlock
    entities: list[EntityLink] = Field(default_factory=list)
    product_links: list[ProductLink] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    contract: ContractExtension | None = None
    paper: PaperExtension | None = None
    patent: PatentExtension | None = None
    provenance: Provenance
    quality: Quality = Field(default_factory=Quality)
    extension: dict[str, Any] = Field(default_factory=dict)  # unknown fields kept, not dropped


# ---------------------------------------------------------------------------
# Run manifest / metrics (brief §7)
# ---------------------------------------------------------------------------


class RunMetrics(BaseModel):
    discovered: int = 0
    accepted: int = 0
    policy_skipped: int = 0
    fetched: int = 0
    not_modified: int = 0
    parsed: int = 0
    published_clean: int = 0
    new_records: int = 0
    new_versions: int = 0
    exact_duplicates: int = 0
    near_duplicate_candidates: int = 0
    review_pending: int = 0
    retryable_failures: int = 0
    permanent_failures: int = 0
    bytes_downloaded: int = 0


class RunError(BaseModel):
    stage: str
    message: str
    url: str | None = None
    candidate_id: str | None = None
    retryable: bool = False
    at: datetime = Field(default_factory=utc_now)


class RunManifest(BaseModel):
    run_id: str
    mode: Literal["discover", "run", "retry", "validate"]
    live: bool = False
    source_ids: list[str] = Field(default_factory=list)
    code_version: str = "0.1.0"
    config_hash: str | None = None
    adapter_versions: dict[str, str] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    cursor_before: dict[str, Any] = Field(default_factory=dict)
    cursor_after: dict[str, Any] = Field(default_factory=dict)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    errors: list[RunError] = Field(default_factory=list)
    indexed: bool = False  # MUST always be False in this phase (brief §11.7)


class SourceState(BaseModel):
    """Per-source persistent state: cursors, validator headers, health."""

    source_id: str
    cursor: dict[str, Any] = Field(default_factory=dict)
    validators: dict[str, dict[str, str]] = Field(default_factory=dict)  # url -> etag/last_modified
    health: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=utc_now)

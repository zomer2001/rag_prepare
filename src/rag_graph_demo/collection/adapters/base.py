"""Adapter protocol for collection sources.

An adapter turns provider-specific entry points into DiscoveryCandidates and
provider payloads into NormalizedRecords. Everything else (policy gate,
fetch, storage, dedupe, quality) is shared pipeline machinery. New legal
public sources should be added as new adapters, not as edits to the core
(brief §0.3).
"""

from __future__ import annotations

from typing import Any, Protocol

from ..models import (
    DiscoveryCandidate,
    FetchResult,
    NormalizedRecord,
    SourceConfig,
    SourceState,
)
from ..parsers import ParsedContent


class CollectionContext(Protocol):
    """Runtime services handed to adapters by the runner."""

    live: bool
    limit: int | None
    collection_root: Any  # Path
    since: str | None


class SourceAdapter(Protocol):
    name: str
    version: str

    def discover(
        self,
        source: SourceConfig,
        state: SourceState,
        ctx: CollectionContext,
        fetcher: Any,
    ) -> list[DiscoveryCandidate]:
        """Yield candidates only — no body download, no verdicts (§4.1)."""
        ...

    def build_record(
        self,
        source: SourceConfig,
        candidate: DiscoveryCandidate,
        fetch: FetchResult,
        parsed: ParsedContent,
        raw_payload: dict[str, Any] | None,
    ) -> NormalizedRecord:
        """Map one fetched+parsed item into the unified record model."""
        ...


def make_candidate_id(source_id: str, url: str | None, provider_record_id: str | None) -> str:
    import hashlib

    key = provider_record_id or url or "unknown"
    return f"{source_id}:{hashlib.sha256(key.encode()).hexdigest()[:16]}"

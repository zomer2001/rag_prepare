from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx

from rag_graph_demo.collection.dedupe import make_record_id
from rag_graph_demo.collection.http import DomainRateLimiter, HttpFetcher
from rag_graph_demo.collection.models import (
    AuthorityTier,
    RecordType,
    SourceConfig,
    SourcePolicy,
)
from rag_graph_demo.collection.policy import canonicalize_url
from rag_graph_demo.collection.runner import CollectionRunner
from rag_graph_demo.config import Settings

from .discovery import TopicDiscovery, source_id_for_topic
from .models import TopicProfile, TopicRoundReport, TopicState
from .planner import TopicPlanner
from .store import TopicStateStore, profile_hash


class TopicSourceRegistry:
    """Small registry facade understood by CollectionRunner."""

    def __init__(self, profile: TopicProfile) -> None:
        self.profile = profile
        self.source = SourceConfig(
            source_id=source_id_for_topic(profile.topic_id),
            name=f"Topic crawl: {profile.topic_id}",
            publisher=f"Open web ({profile.topic_id})",
            authority_tier=AuthorityTier.P2,
            record_type=RecordType.OFFICIAL_PAGE,
            adapter="generic_web",
            adapter_version="1.0.0",
            enabled=True,
            groups=["topic", profile.topic_id],
            policy=SourcePolicy(
                rate_limit_seconds=profile.rate_limit_seconds,
                max_concurrency=1,
                max_bytes=profile.max_bytes,
                content_license="Source-specific terms apply; research copy with provenance",
            ),
        )

    def get(self, source_id: str) -> SourceConfig:
        if source_id != self.source.source_id:
            raise KeyError(source_id)
        return self.source

    def config_hash(self) -> str:
        return profile_hash(self.profile)


class TopicRunner:
    def __init__(
        self,
        profile: TopicProfile,
        *,
        data_dir: Path | str = "data",
        settings: Settings | None = None,
        live: bool = False,
        fixtures_dir: Path | str = "tests/fixtures/collection",
    ) -> None:
        self.profile = profile
        self.data_dir = Path(data_dir)
        self.collection_root = self.data_dir / "collection"
        self.osint_root = self.data_dir / "osint"
        self.settings = settings or Settings(data_dir=self.data_dir)
        self.live = live
        self.fixtures_dir = Path(fixtures_dir)
        self.state_store = TopicStateStore(self.osint_root, profile.topic_id)
        self.planner = TopicPlanner(self.settings)
        self.registry = TopicSourceRegistry(profile)
        self.collection = CollectionRunner(
            self.registry,  # type: ignore[arg-type]
            collection_root=self.collection_root,
            live=live,
            fixtures_dir=self.fixtures_dir,
            limit=profile.max_candidates_per_round,
        )

    def plan(self) -> TopicState:
        state = self.state_store.load(self.profile)
        asyncio.run(self._initialize_terms(state))
        return state

    def run(self, *, rounds: int | None = None) -> TopicState:
        if not self.live:
            raise ValueError("topic crawling requires --live; offline mode is test-only")
        state = self.state_store.load(self.profile)
        asyncio.run(self._initialize_terms(state))
        target_rounds = self.profile.max_rounds
        if rounds is not None:
            target_rounds = min(target_rounds, state.completed_rounds + max(0, rounds))

        client = httpx.Client(
            timeout=30,
            follow_redirects=True,
            headers={"User-Agent": "rag-prepare-osint-collector/0.1"},
        )
        discovery_fetcher = HttpFetcher(
            self.collection_root / "raw",
            client=client,
            rate_limiter=DomainRateLimiter(
                default_interval=self.profile.rate_limit_seconds
            ),
            max_bytes=self.profile.max_bytes,
            adapter="topic_discovery",
            adapter_version="1.0.0",
        )
        try:
            while state.completed_rounds < target_rounds:
                if state.fetched_count >= self.profile.max_fetches:
                    self._append_stop(state, "fetch budget exhausted")
                    break
                round_index = state.completed_rounds
                discovery = TopicDiscovery(discovery_fetcher)
                candidates = discovery.discover(self.profile, state.terms, round_index)
                seen = set(state.seen_urls)
                new_candidates = [item for item in candidates if item.url not in seen]
                remaining = self.profile.max_fetches - state.fetched_count
                new_candidates = new_candidates[:remaining]
                if not new_candidates:
                    state.reports.append(
                        TopicRoundReport(
                            round_index=round_index,
                            terms=list(state.terms),
                            discovered=len(candidates),
                            errors=discovery.errors,
                            stop_reason="no new candidates",
                        )
                    )
                    state.completed_rounds += 1
                    self.state_store.save(state)
                    break

                manifest = self.collection.run_candidates(
                    self.registry.source.source_id, new_candidates
                )
                retryable = {
                    error.url for error in manifest.errors if error.retryable and error.url
                }
                for candidate in new_candidates:
                    if candidate.url and candidate.url not in retryable:
                        state.seen_urls.append(candidate.url)
                state.fetched_count += manifest.metrics.fetched
                documents = self._load_documents(new_candidates)
                expansion_terms = asyncio.run(
                    self.planner.expand(self.profile, state.terms, documents)
                )
                state.terms.extend(expansion_terms)
                state.reports.append(
                    TopicRoundReport(
                        round_index=round_index,
                        terms=list(state.terms),
                        discovered=len(candidates),
                        new_candidates=len(new_candidates),
                        fetched=manifest.metrics.fetched,
                        published=manifest.metrics.published_clean,
                        new_records=manifest.metrics.new_records,
                        expansion_terms=expansion_terms,
                        errors=[*discovery.errors, *(e.message for e in manifest.errors)],
                        stop_reason=("no new expansion terms" if not expansion_terms else None),
                        run_id=manifest.run_id,
                    )
                )
                state.completed_rounds += 1
                self.state_store.save(state)
                if not expansion_terms:
                    break
        finally:
            client.close()
        return state

    async def _initialize_terms(self, state: TopicState) -> None:
        if not state.terms:
            state.terms = list(self.profile.keywords)
        if state.planner_output:
            return
        try:
            output = await self.planner.plan(self.profile)
        except Exception as exc:
            output = {"terms": [], "mode": "fallback", "error": str(exc)}
        state.planner_output = output
        existing = {item.casefold() for item in state.terms}
        for term in output.get("terms", []):
            if isinstance(term, str) and term.casefold() not in existing:
                state.terms.append(term)
                existing.add(term.casefold())
        self.state_store.save(state)

    def _load_documents(self, candidates: list[Any]) -> list[str]:
        documents: list[str] = []
        for candidate in candidates:
            if not candidate.url:
                continue
            record_id = make_record_id(
                self.registry.source.source_id,
                None,
                canonicalize_url(candidate.url),
            )
            record = self.collection.record_store.load_latest(record_id)
            if not record:
                continue
            title = record.get("title") or ""
            body = record.get("body_markdown") or ""
            if title or body:
                documents.append(f"# {title}\n\n{body}")
        return documents

    def _append_stop(self, state: TopicState, reason: str) -> None:
        state.reports.append(
            TopicRoundReport(
                round_index=state.completed_rounds,
                terms=list(state.terms),
                stop_reason=reason,
            )
        )
        self.state_store.save(state)


def state_summary(state: TopicState) -> dict[str, Any]:
    latest = state.reports[-1] if state.reports else None
    return {
        "topic_id": state.topic_id,
        "completed_rounds": state.completed_rounds,
        "fetched_count": state.fetched_count,
        "known_terms": len(state.terms),
        "seen_urls": len(state.seen_urls),
        "latest_report": latest.model_dump(mode="json") if latest else None,
    }

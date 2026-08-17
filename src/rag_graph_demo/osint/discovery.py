from __future__ import annotations

import json
import os
import urllib.parse
from typing import Any

from rag_graph_demo.collection.adapters.base import make_candidate_id
from rag_graph_demo.collection.adapters.rss_feed import RssFeedAdapter
from rag_graph_demo.collection.adapters.sitemap_archive import SitemapArchiveAdapter
from rag_graph_demo.collection.models import (
    AuthorityTier,
    DiscoveryCandidate,
    RecordType,
    SourceConfig,
    SourcePolicy,
    SourceState,
)
from rag_graph_demo.collection.policy import canonicalize_url

from .models import SearchApiConfig, TopicProfile


def source_id_for_topic(topic_id: str) -> str:
    return f"topic_{topic_id.replace('-', '_')}"


class _Context:
    live = True
    collection_root = None
    since = None

    def __init__(self, limit: int) -> None:
        self.limit = limit


class TopicDiscovery:
    def __init__(self, fetcher: Any) -> None:
        self.fetcher = fetcher
        self.errors: list[str] = []

    def discover(
        self,
        profile: TopicProfile,
        terms: list[str],
        round_index: int,
    ) -> list[DiscoveryCandidate]:
        source_id = source_id_for_topic(profile.topic_id)
        candidates: list[DiscoveryCandidate] = []
        if round_index == 0:
            candidates.extend(self._seed_candidates(profile, source_id))
        candidates.extend(self._feed_candidates(profile, source_id, terms))
        candidates.extend(self._sitemap_candidates(profile, source_id, terms))
        candidates.extend(self._search_candidates(profile, source_id, terms))

        result: list[DiscoveryCandidate] = []
        seen: set[str] = set()
        excludes = [item.casefold() for item in profile.exclude_keywords]
        for candidate in candidates:
            if not candidate.url:
                continue
            canonical = canonicalize_url(candidate.url)
            haystack = " ".join(
                [canonical, candidate.title or "", candidate.summary or ""]
            ).casefold()
            if any(term in haystack for term in excludes):
                continue
            if canonical in seen:
                continue
            seen.add(canonical)
            candidate.url = canonical
            candidate.extra.update(
                {
                    "topic_id": profile.topic_id,
                    "topic_round": round_index,
                    "matched_terms": _matched_terms(haystack, terms),
                }
            )
            result.append(candidate)
            if len(result) >= profile.max_candidates_per_round:
                break
        return result

    def _seed_candidates(
        self, profile: TopicProfile, source_id: str
    ) -> list[DiscoveryCandidate]:
        return [
            DiscoveryCandidate(
                candidate_id=make_candidate_id(source_id, url, None),
                source_id=source_id,
                url=url,
                discovered_by="manual",
                discovery_query="topic seed URL",
                extra={"discovery_source": "seed_url"},
            )
            for url in profile.seed_urls
        ]

    def _feed_candidates(
        self, profile: TopicProfile, source_id: str, terms: list[str]
    ) -> list[DiscoveryCandidate]:
        if not profile.feeds:
            return []
        source = SourceConfig(
            source_id=source_id,
            name=f"{profile.topic_id} dynamic feeds",
            publisher="Dynamic RSS/Atom sources",
            authority_tier=AuthorityTier.P2,
            record_type=RecordType.NEWS,
            adapter="rss_feed",
            entry={"feeds": profile.feeds, "keywords": terms},
            policy=SourcePolicy(rate_limit_seconds=profile.rate_limit_seconds),
        )
        try:
            candidates = RssFeedAdapter().discover(
                source,
                SourceState(source_id=source_id),
                _Context(profile.max_candidates_per_round),
                self.fetcher,
            )
        except Exception as exc:
            self.errors.append(f"feed discovery: {exc}")
            return []
        for item in candidates:
            item.extra["discovery_source"] = item.extra.get("feed_url")
        return candidates

    def _sitemap_candidates(
        self, profile: TopicProfile, source_id: str, terms: list[str]
    ) -> list[DiscoveryCandidate]:
        if not profile.sitemaps:
            return []
        source = SourceConfig(
            source_id=source_id,
            name=f"{profile.topic_id} dynamic sitemaps",
            publisher="Dynamic sitemap sources",
            authority_tier=AuthorityTier.P2,
            record_type=RecordType.OFFICIAL_PAGE,
            adapter="sitemap_archive",
            entry={
                "sitemaps": profile.sitemaps,
                "url_keywords": terms,
                "max_sitemaps": 100,
            },
            policy=SourcePolicy(rate_limit_seconds=profile.rate_limit_seconds),
        )
        try:
            candidates = SitemapArchiveAdapter().discover(
                source,
                SourceState(source_id=source_id),
                _Context(profile.max_candidates_per_round),
                self.fetcher,
            )
        except Exception as exc:
            self.errors.append(f"sitemap discovery: {exc}")
            return []
        for item in candidates:
            item.extra["discovery_source"] = item.extra.get("sitemap_url")
        return candidates

    def _search_candidates(
        self, profile: TopicProfile, source_id: str, terms: list[str]
    ) -> list[DiscoveryCandidate]:
        config = profile.search
        if config is None:
            return []
        candidates: list[DiscoveryCandidate] = []
        queries = _queries(profile, terms)
        for query in queries[: profile.max_queries_per_round]:
            for page in range(config.page_start, config.page_start + config.max_pages):
                try:
                    rows = self._search_page(config, query, page)
                except Exception as exc:
                    self.errors.append(f"search query {query!r}: {exc}")
                    break
                if not rows:
                    break
                for rank, row in enumerate(rows, start=1):
                    if not isinstance(row, dict):
                        continue
                    url = _dig(row, config.url_field)
                    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                        continue
                    title = _dig(row, config.title_field)
                    summary = _dig(row, config.summary_field)
                    candidates.append(
                        DiscoveryCandidate(
                            candidate_id=make_candidate_id(source_id, url, None),
                            source_id=source_id,
                            url=url,
                            title=str(title) if title else None,
                            summary=str(summary) if summary else None,
                            rank=rank,
                            discovery_query=query,
                            discovered_by="search",
                            extra={"discovery_source": config.endpoint, "search_page": page},
                        )
                    )
                    if len(candidates) >= profile.max_candidates_per_round:
                        return candidates
        return candidates

    def _search_page(
        self, config: SearchApiConfig, query: str, page: int
    ) -> list[Any]:
        params = dict(config.extra_params)
        params[config.query_param] = query
        if config.page_param:
            params[config.page_param] = str(page)
        separator = "&" if urllib.parse.urlsplit(config.endpoint).query else "?"
        url = config.endpoint + separator + urllib.parse.urlencode(params)
        headers = dict(config.headers)
        if config.api_key_env:
            key = os.getenv(config.api_key_env)
            if not key:
                raise ValueError(f"missing search API key env {config.api_key_env}")
            headers[config.api_key_header] = f"{config.api_key_prefix}{key}"
        result = self.fetcher.fetch(url, headers=headers)
        if not result.ok or not result.raw_path:
            raise RuntimeError(result.error or f"http {result.status}")
        payload = json.loads(open(result.raw_path, encoding="utf-8").read())  # noqa: SIM115
        rows = _dig(payload, config.result_path)
        return rows if isinstance(rows, list) else []


def _queries(profile: TopicProfile, terms: list[str]) -> list[str]:
    values = [profile.question, *terms]
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(value.split())
        key = cleaned.casefold()
        if cleaned and key not in seen:
            result.append(cleaned)
            seen.add(key)
    return result


def _matched_terms(haystack: str, terms: list[str]) -> list[str]:
    return [term for term in terms if term.casefold() in haystack]


def _dig(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current

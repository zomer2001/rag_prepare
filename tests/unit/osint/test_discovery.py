from __future__ import annotations

import json
from pathlib import Path

from rag_graph_demo.collection.models import FetchResult
from rag_graph_demo.osint.discovery import TopicDiscovery
from rag_graph_demo.osint.models import SearchApiConfig, TopicProfile


class StubFetcher:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.urls: list[str] = []

    def fetch(self, url: str, **_kwargs):
        self.urls.append(url)
        return FetchResult(
            url=url,
            ok=True,
            status=200,
            raw_path=str(self.path),
            mime_type="application/json",
        )


def test_provider_neutral_search_discovery(tmp_path: Path) -> None:
    payload = tmp_path / "search.json"
    payload.write_text(
        json.dumps(
            {
                "web": {
                    "items": [
                        {
                            "link": "https://example.com/counter-uas?utm_source=search",
                            "name": "Counter-UAS contract",
                            "snippet": "A new contract",
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    profile = TopicProfile(
        topic_id="counter_uas",
        question="counter UAS contracts",
        keywords=["counter-UAS"],
        search=SearchApiConfig(
            endpoint="https://search.example/api",
            result_path="web.items",
            url_field="link",
            title_field="name",
            summary_field="snippet",
        ),
        use_llm=False,
    )
    fetcher = StubFetcher(payload)
    candidates = TopicDiscovery(fetcher).discover(profile, profile.keywords, 0)
    assert len(candidates) == 1
    assert candidates[0].url == "https://example.com/counter-uas"
    assert candidates[0].discovered_by == "search"
    assert fetcher.urls


def test_seed_url_exclusion() -> None:
    profile = TopicProfile(
        topic_id="seed_test",
        question="test",
        keywords=["Alpha"],
        exclude_keywords=["ignore"],
        seed_urls=["https://example.com/ignore-this"],
        use_llm=False,
    )
    assert TopicDiscovery(None).discover(profile, profile.keywords, 0) == []

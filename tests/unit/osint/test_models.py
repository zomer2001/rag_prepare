from __future__ import annotations

from pathlib import Path

import pytest

from rag_graph_demo.osint.models import TopicProfile
from rag_graph_demo.osint.store import TopicStateStore


def test_topic_profile_loads(tmp_path: Path) -> None:
    path = tmp_path / "topic.yaml"
    path.write_text(
        """topic_id: test_topic
question: Test a topic
keywords: [Alpha, alpha, Beta]
seed_urls: [https://example.com/topic]
use_llm: false
""",
        encoding="utf-8",
    )
    profile = TopicProfile.load(path)
    assert profile.keywords == ["Alpha", "Beta"]
    assert profile.max_rounds == 2


def test_topic_requires_discovery_entry() -> None:
    with pytest.raises(ValueError, match="at least one"):
        TopicProfile(topic_id="empty", question="nothing", keywords=["x"])


def test_profile_change_replans_but_preserves_frontier(tmp_path: Path) -> None:
    first = TopicProfile(
        topic_id="topic_one",
        question="first",
        keywords=["Alpha"],
        seed_urls=["https://example.com/a"],
        use_llm=False,
    )
    store = TopicStateStore(tmp_path, first.topic_id)
    state = store.load(first)
    state.completed_rounds = 2
    state.seen_urls = ["https://example.com/a"]
    store.save(state)

    changed = first.model_copy(update={"question": "changed"})
    loaded = store.load(changed)
    assert loaded.completed_rounds == 0
    assert loaded.seen_urls == ["https://example.com/a"]

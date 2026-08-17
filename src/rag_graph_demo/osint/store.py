from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from .models import TopicProfile, TopicState


def profile_hash(profile: TopicProfile) -> str:
    payload = profile.model_dump_json(exclude_none=False)
    return hashlib.sha256(payload.encode()).hexdigest()


class TopicStateStore:
    def __init__(self, root: Path | str, topic_id: str) -> None:
        self.root = Path(root) / topic_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "state.json"

    def load(self, profile: TopicProfile) -> TopicState:
        digest = profile_hash(profile)
        if not self.path.exists():
            return TopicState(topic_id=profile.topic_id, profile_hash=digest)
        state = TopicState.model_validate(json.loads(self.path.read_text(encoding="utf-8")))
        if state.profile_hash != digest:
            # Preserve the crawl frontier while recording that planning inputs changed.
            state.profile_hash = digest
            state.completed_rounds = 0
        return state

    def save(self, state: TopicState) -> None:
        state.updated_at = datetime.now(UTC)
        temp = self.path.with_suffix(".json.tmp")
        temp.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        temp.replace(self.path)

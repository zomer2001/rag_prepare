from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path

from rag_graph_demo.domain.events import (
    EventCandidate,
    EventConflict,
    EventMention,
    EventRecord,
    EventRelation,
    EventStatus,
)
from rag_graph_demo.storage.atomic_json import read_json, write_json_atomic


def _normal(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value.casefold())


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def _time_bucket(value: str | None) -> str:
    return (value or "unknown")[:10]


def _sets_overlap(left: set[str], right: set[str]) -> bool:
    return any(
        first == second
        or (min(len(first), len(second)) >= 3 and (first in second or second in first))
        for first in left
        for second in right
    )


def canonical_key(candidate: EventCandidate) -> str:
    actors = ",".join(sorted(_normal(item) for item in candidate.actors))
    objects = ",".join(sorted(_normal(item) for item in candidate.objects))
    return "|".join(
        (_normal(candidate.event_type), actors, objects, _time_bucket(candidate.start_time))
    )


class EventRegistry:
    """Single-process JSON event store for the demo.

    Event records are canonical assertions. Mentions retain per-document evidence,
    so merging never destroys provenance and a document can be rebuilt idempotently.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()

    async def replace_document_events(
        self,
        document_id: str,
        document_title: str,
        candidates: list[EventCandidate],
    ) -> list[EventRecord]:
        async with self._lock:
            events, mentions, _, _ = self._read()
            mentions = {
                key: value
                for key, value in mentions.items()
                if value.document_id != document_id
            }
            referenced = {mention.event_id for mention in mentions.values()}
            events = {key: value for key, value in events.items() if key in referenced}

            indexed_ids: list[str] = []
            for candidate in candidates:
                key = canonical_key(candidate)
                event_id = self._find_merge_target(events, candidate) or _stable_id(
                    "evt", key
                )
                if event_id not in events:
                    events[event_id] = EventRecord(
                        event_id=event_id,
                        canonical_key=key,
                        event_type=candidate.event_type,
                        summary=candidate.summary,
                        action=candidate.action,
                        actors=candidate.actors,
                        objects=candidate.objects,
                        start_time=candidate.start_time,
                        end_time=candidate.end_time,
                        time_precision=candidate.time_precision,
                        location=candidate.location,
                        result=candidate.result,
                    )
                mention_key = "|".join(
                    (document_id, event_id, _normal(candidate.evidence)[:500])
                )
                mention_id = _stable_id("men", mention_key)
                mentions[mention_id] = EventMention(
                    mention_id=mention_id,
                    event_id=event_id,
                    document_id=document_id,
                    document_title=document_title,
                    trigger=candidate.trigger,
                    evidence=candidate.evidence,
                    source_segment_id=candidate.source_segment_id,
                    source_segment_start=candidate.source_segment_start,
                    source_segment_end=candidate.source_segment_end,
                    evidence_start=candidate.evidence_start,
                    evidence_end=candidate.evidence_end,
                    stance=candidate.stance,
                    confidence=candidate.confidence,
                )
                indexed_ids.append(event_id)

            self._refresh_events(events, mentions)
            relations = self._build_relations(events)
            conflicts = self._build_conflicts(events, mentions)
            self._write(events, mentions, relations, conflicts)
            return [events[event_id] for event_id in dict.fromkeys(indexed_ids)]

    def _find_merge_target(
        self,
        events: dict[str, EventRecord],
        candidate: EventCandidate,
    ) -> str | None:
        candidate_actors = {_normal(item) for item in candidate.actors if _normal(item)}
        candidate_objects = {_normal(item) for item in candidate.objects if _normal(item)}
        if not candidate_actors and not candidate_objects:
            return None
        for event in events.values():
            if self._equivalent(
                event,
                event_type=candidate.event_type,
                actors=candidate_actors,
                objects=candidate_objects,
                start_time=candidate.start_time,
            ):
                return event.event_id
        return None

    @staticmethod
    def _equivalent(
        event: EventRecord,
        *,
        event_type: str,
        actors: set[str],
        objects: set[str],
        start_time: str | None,
    ) -> bool:
        if _normal(event.event_type) != _normal(event_type):
            return False
        event_actors = {_normal(item) for item in event.actors if _normal(item)}
        event_objects = {_normal(item) for item in event.objects if _normal(item)}
        if actors and event_actors and not _sets_overlap(actors, event_actors):
            return False
        if objects and event_objects and not _sets_overlap(objects, event_objects):
            return False
        if not (
            _sets_overlap(actors, event_actors)
            or _sets_overlap(objects, event_objects)
        ):
            return False
        if start_time and event.start_time:
            common_precision = min(len(start_time), len(event.start_time), 7)
            if start_time[:common_precision] != event.start_time[:common_precision]:
                return False
        return True

    async def delete_document(self, document_id: str) -> None:
        await self.replace_document_events(document_id, "", [])

    async def list_events(self) -> list[EventRecord]:
        async with self._lock:
            events, _, _, _ = self._read()
        return sorted(events.values(), key=lambda item: item.start_time or "", reverse=True)

    async def get_event(self, event_id: str) -> EventRecord | None:
        async with self._lock:
            events, _, _, _ = self._read()
        return events.get(event_id)

    async def snapshot(
        self,
    ) -> tuple[
        list[EventRecord], list[EventMention], list[EventRelation], list[EventConflict]
    ]:
        async with self._lock:
            events, mentions, relations, conflicts = self._read()
        return (
            list(events.values()),
            list(mentions.values()),
            list(relations.values()),
            list(conflicts.values()),
        )

    async def rebuild_derived(self) -> dict[str, int]:
        """Recompute confidence, timeline edges, and conflicts without an LLM call."""
        async with self._lock:
            events, mentions, _, _ = self._read()
            merged = self._merge_equivalent_events(events, mentions)
            self._refresh_events(events, mentions)
            relations = self._build_relations(events)
            conflicts = self._build_conflicts(events, mentions)
            self._write(events, mentions, relations, conflicts)
        return {
            "events": len(events),
            "mentions": len(mentions),
            "relations": len(relations),
            "conflicts": len(conflicts),
            "merged_events": merged,
        }

    def _merge_equivalent_events(
        self,
        events: dict[str, EventRecord],
        mentions: dict[str, EventMention],
    ) -> int:
        merged = 0
        survivors: dict[str, EventRecord] = {}
        for event in sorted(events.values(), key=lambda item: (item.created_at, item.event_id)):
            actors = {_normal(item) for item in event.actors if _normal(item)}
            objects = {_normal(item) for item in event.objects if _normal(item)}
            target = next(
                (
                    item
                    for item in survivors.values()
                    if self._equivalent(
                        item,
                        event_type=event.event_type,
                        actors=actors,
                        objects=objects,
                        start_time=event.start_time,
                    )
                ),
                None,
            )
            if target is None:
                survivors[event.event_id] = event
                continue
            target.actors = list(dict.fromkeys(target.actors + event.actors))
            target.objects = list(dict.fromkeys(target.objects + event.objects))
            for mention in mentions.values():
                if mention.event_id == event.event_id:
                    mention.event_id = target.event_id
            merged += 1
        if merged:
            events.clear()
            events.update(survivors)
        return merged

    def _refresh_events(
        self,
        events: dict[str, EventRecord],
        mentions: dict[str, EventMention],
    ) -> None:
        grouped: dict[str, list[EventMention]] = {event_id: [] for event_id in events}
        for mention in mentions.values():
            grouped.setdefault(mention.event_id, []).append(mention)
        now = datetime.now(UTC)
        for event_id, event in events.items():
            event_mentions = grouped[event_id]
            sources = sorted({item.document_id for item in event_mentions})
            affirmed = [item for item in event_mentions if item.stance == "affirmed"]
            denied = [item for item in event_mentions if item.stance == "denied"]
            base = sum(item.confidence for item in event_mentions) / max(len(event_mentions), 1)
            bonus = min(0.16, max(0, len(sources) - 1) * 0.08)
            event.confidence = round(max(0, min(1, base + bonus - (0.2 if denied else 0))), 3)
            if affirmed and denied:
                event.status = EventStatus.DISPUTED
            elif len(sources) >= 2:
                event.status = EventStatus.MULTI_SOURCE
            elif sources:
                event.status = EventStatus.SINGLE_SOURCE
            else:
                event.status = EventStatus.CANDIDATE
            event.source_document_ids = sources
            event.mention_ids = sorted(item.mention_id for item in event_mentions)
            event.updated_at = now

    def _build_relations(
        self, events: dict[str, EventRecord]
    ) -> dict[str, EventRelation]:
        entity_events: dict[str, list[tuple[EventRecord, str]]] = {}
        for event in events.values():
            for entity in dict.fromkeys(event.actors + event.objects):
                normalized = _normal(entity)
                if normalized:
                    entity_events.setdefault(normalized, []).append((event, entity))

        # A chain needs adjacent transitions, not a complete graph between every
        # pair that mentions Anduril. Accumulate shared entities for each adjacent
        # event pair, then emit one explainable edge per pair.
        pair_entities: dict[tuple[str, str], list[str]] = {}
        for linked in entity_events.values():
            ordered = sorted(
                linked,
                key=lambda item: (item[0].start_time or "9999", item[0].event_id),
            )
            unique: list[tuple[EventRecord, str]] = []
            seen: set[str] = set()
            for item in ordered:
                if item[0].event_id not in seen:
                    unique.append(item)
                    seen.add(item[0].event_id)
            for (source, source_name), (target, _) in zip(unique, unique[1:], strict=False):
                pair = (source.event_id, target.event_id)
                pair_entities.setdefault(pair, []).append(source_name)

        relations: dict[str, EventRelation] = {}
        for (source_id, target_id), shared in pair_entities.items():
            source = events[source_id]
            target = events[target_id]
            if source.start_time and target.start_time:
                relation_type = (
                    "same_time" if source.start_time == target.start_time else "precedes"
                )
            else:
                relation_type = "related"
            relation_id = _stable_id(
                "rel", f"{source_id}|{target_id}|{relation_type}"
            )
            relations[relation_id] = EventRelation(
                relation_id=relation_id,
                source_event_id=source_id,
                target_event_id=target_id,
                relation_type=relation_type,
                shared_entities=list(dict.fromkeys(shared)),
                confidence=0.8 if relation_type == "precedes" else 0.65,
            )
        return relations

    def _build_conflicts(
        self,
        events: dict[str, EventRecord],
        mentions: dict[str, EventMention],
    ) -> dict[str, EventConflict]:
        conflicts: dict[str, EventConflict] = {}
        for event in events.values():
            event_mentions = [
                item for item in mentions.values() if item.event_id == event.event_id
            ]
            stances = {item.stance for item in event_mentions}
            if "affirmed" not in stances or "denied" not in stances:
                continue
            conflict_id = _stable_id("con", f"{event.event_id}|stance")
            conflicts[conflict_id] = EventConflict(
                conflict_id=conflict_id,
                event_id=event.event_id,
                description="不同来源对该事件是否发生存在肯定与否定冲突。",
                mention_ids=sorted(item.mention_id for item in event_mentions),
            )
        return conflicts

    def _read(
        self,
    ) -> tuple[
        dict[str, EventRecord],
        dict[str, EventMention],
        dict[str, EventRelation],
        dict[str, EventConflict],
    ]:
        raw = read_json(self._path, default={})
        return (
            {
                key: EventRecord.model_validate(value)
                for key, value in raw.get("events", {}).items()
            },
            {
                key: EventMention.model_validate(value)
                for key, value in raw.get("mentions", {}).items()
            },
            {
                key: EventRelation.model_validate(value)
                for key, value in raw.get("relations", {}).items()
            },
            {
                key: EventConflict.model_validate(value)
                for key, value in raw.get("conflicts", {}).items()
            },
        )

    def _write(
        self,
        events: dict[str, EventRecord],
        mentions: dict[str, EventMention],
        relations: dict[str, EventRelation],
        conflicts: dict[str, EventConflict],
    ) -> None:
        write_json_atomic(
            self._path,
            {
                "events": {key: value.model_dump(mode="json") for key, value in events.items()},
                "mentions": {
                    key: value.model_dump(mode="json") for key, value in mentions.items()
                },
                "relations": {
                    key: value.model_dump(mode="json") for key, value in relations.items()
                },
                "conflicts": {
                    key: value.model_dump(mode="json") for key, value in conflicts.items()
                },
            },
        )

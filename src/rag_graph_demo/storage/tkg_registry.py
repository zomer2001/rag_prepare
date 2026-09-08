from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from rag_graph_demo.domain.tkg import (
    AssertionMention,
    AssertionRecord,
    AssertionStatus,
    TKGQueryRequest,
    TKGQueryResponse,
    TKGSnapshot,
    interval_is_inverted,
)
from rag_graph_demo.storage.atomic_json import read_json, write_json_atomic

T = TypeVar("T")
logger = logging.getLogger(__name__)


class TKGRegistry:
    """Atomic local persistence for the statement-level TKG.

    The service deliberately exposes a document transaction instead of low-level
    per-node writes. Reindexing one document therefore replaces only its provenance
    records while keeping canonical objects supported by other sources.
    """

    def __init__(
        self,
        path: Path,
        *,
        ontology_version: str,
        validate_snapshot: Callable[[TKGSnapshot], None] | None = None,
    ) -> None:
        self._path = path
        self._ontology_version = ontology_version
        self._validate_snapshot = validate_snapshot
        self._lock = asyncio.Lock()

    async def snapshot(self) -> TKGSnapshot:
        async with self._lock:
            return self._read()

    async def mutate_document(
        self,
        document_id: str,
        mutation: Callable[[TKGSnapshot], T],
    ) -> T:
        async with self._lock:
            snapshot = self._read()
            self._purge_document(snapshot, document_id)
            result = mutation(snapshot)
            self._refresh(snapshot)
            if self._validate_snapshot is not None:
                self._validate_snapshot(snapshot)
            self._write(snapshot)
            return result

    async def mutate(self, mutation: Callable[[TKGSnapshot], T]) -> T:
        async with self._lock:
            snapshot = self._read()
            result = mutation(snapshot)
            self._refresh(snapshot)
            if self._validate_snapshot is not None:
                self._validate_snapshot(snapshot)
            self._write(snapshot)
            return result

    async def delete_document(self, document_id: str) -> None:
        await self.mutate_document(document_id, lambda _: None)

    async def query(self, request: TKGQueryRequest) -> TKGQueryResponse:
        snapshot = await self.snapshot()
        entities = snapshot.entities
        entity_ids: set[str] | None = None
        if request.entity_id:
            entity_ids = {request.entity_id}
        elif request.entity_name:
            needle = _normal(request.entity_name)
            entity_ids = {
                item.entity_id
                for item in entities
                if item.status == "active"
                and (
                    needle == _normal(item.canonical_name)
                or any(needle == _normal(alias) for alias in item.aliases)
                )
            }
        assertions = []
        for item in snapshot.assertions:
            if entity_ids is not None and not {
                item.subject_id,
                item.object_id,
            } & entity_ids:
                continue
            if request.predicate_id and item.predicate_id != request.predicate_id:
                continue
            if not request.include_inactive and item.status in {
                AssertionStatus.INACTIVE,
                AssertionStatus.SUPERSEDED,
                AssertionStatus.RETRACTED,
            }:
                continue
            if not request.include_disputed and item.status == AssertionStatus.DISPUTED:
                continue
            if request.at_time and (
                (item.valid_from and request.at_time < item.valid_from)
                or (item.valid_to and request.at_time > item.valid_to)
            ):
                continue
            assertions.append(item)
        assertions = assertions[: request.limit]
        assertion_ids = {item.assertion_id for item in assertions}
        linked_entity_ids = {
            value
            for item in assertions
            for value in (item.subject_id, item.object_id)
        }
        return TKGQueryResponse(
            entities=[item for item in entities if item.entity_id in linked_entity_ids],
            assertions=assertions,
            mentions=[
                item
                for item in snapshot.assertion_mentions
                if item.assertion_id in assertion_ids
            ],
            effects=[
                item for item in snapshot.event_effects if item.assertion_id in assertion_ids
            ],
            conflicts=[
                item
                for item in snapshot.conflicts
                if assertion_ids & set(item.assertion_ids)
            ],
        )

    def _read(self) -> TKGSnapshot:
        raw = read_json(self._path, default=None)
        if raw is None:
            return TKGSnapshot(ontology_version=self._ontology_version)
        repaired = 0
        for assertion in raw.get("assertions", []):
            valid_from = assertion.get("valid_from")
            valid_to = assertion.get("valid_to")
            if interval_is_inverted(valid_from, valid_to):
                stated_valid_to = assertion.get("stated_valid_to")
                assertion["valid_to"] = (
                    stated_valid_to
                    if stated_valid_to and stated_valid_to >= valid_from
                    else None
                )
                assertion["invalidated_at"] = None
                repaired += 1
        if repaired:
            logger.warning(
                "Repaired %d legacy assertion interval(s) before validation: %s",
                repaired,
                self._path,
            )
        snapshot = TKGSnapshot.model_validate(raw)
        if snapshot.ontology_version != self._ontology_version:
            raise ValueError(
                "TKG ontology version mismatch: "
                f"store={snapshot.ontology_version}, runtime={self._ontology_version}. "
                "Run an explicit ontology migration before indexing."
            )
        return snapshot

    def _write(self, snapshot: TKGSnapshot) -> None:
        write_json_atomic(self._path, snapshot.model_dump(mode="json"))

    @staticmethod
    def _purge_document(snapshot: TKGSnapshot, document_id: str) -> None:
        snapshot.entity_mentions = [
            item for item in snapshot.entity_mentions if item.document_id != document_id
        ]
        snapshot.assertion_mentions = [
            item for item in snapshot.assertion_mentions if item.document_id != document_id
        ]
        snapshot.event_entity_links = [
            item for item in snapshot.event_entity_links if item.document_id != document_id
        ]
        snapshot.event_effects = [
            item for item in snapshot.event_effects if item.document_id != document_id
        ]
        snapshot.resolution_decisions = [
            item
            for item in snapshot.resolution_decisions
            if item.document_id != document_id
        ]
        TKGRegistry._refresh(snapshot)

    @staticmethod
    def _refresh(snapshot: TKGSnapshot) -> None:
        assertion_mentions: dict[str, list[AssertionMention]] = {}
        for mention in snapshot.assertion_mentions:
            assertion_mentions.setdefault(mention.assertion_id, []).append(mention)
        effects_by_assertion: dict[str, list] = {}
        for effect in snapshot.event_effects:
            effects_by_assertion.setdefault(effect.assertion_id, []).append(effect)
        retained_assertions: list[AssertionRecord] = []
        for assertion in snapshot.assertions:
            mentions = assertion_mentions.get(assertion.assertion_id, [])
            effects = effects_by_assertion.get(assertion.assertion_id, [])
            if not mentions and not effects:
                continue
            assertion.mention_ids = sorted(item.mention_id for item in mentions)
            assertion.event_effect_ids = sorted(item.effect_id for item in effects)
            stances = {item.stance.value for item in mentions}
            if "asserted" in stances and "denied" in stances:
                assertion.status = AssertionStatus.DISPUTED
            elif "asserted" in stances:
                assertion.status = AssertionStatus.ACTIVE
            elif "denied" in stances:
                assertion.status = AssertionStatus.DISPUTED
            else:
                assertion.status = AssertionStatus.CANDIDATE
            assertion.valid_to = assertion.stated_valid_to
            assertion.invalidated_at = None
            lifecycle_effects = sorted(
                effects,
                key=lambda item: (item.effective_time or "", item.created_at, item.effect_id),
            )
            for effect in lifecycle_effects:
                if effect.stance == "uncertain":
                    continue
                if effect.effect_type.value in {"creates", "confirms", "updates"}:
                    assertion.status = AssertionStatus.ACTIVE
                    if (
                        assertion.stated_valid_to is None
                        and assertion.valid_to
                        and effect.effective_time
                        and effect.effective_time > assertion.valid_to
                    ):
                        assertion.valid_to = None
                        assertion.invalidated_at = None
                elif effect.effect_type.value == "denies":
                    assertion.status = AssertionStatus.DISPUTED
                elif effect.effect_type.value == "terminates":
                    assertion.status = AssertionStatus.INACTIVE
                    assertion.valid_to = effect.effective_time
                    assertion.invalidated_at = effect.created_at
                elif effect.effect_type.value == "supersedes":
                    assertion.status = AssertionStatus.SUPERSEDED
                    assertion.valid_to = effect.effective_time
                    assertion.invalidated_at = effect.created_at
            if interval_is_inverted(assertion.valid_from, assertion.valid_to):
                logger.warning(
                    "Discarding inverted derived interval for assertion %s: %s > %s",
                    assertion.assertion_id,
                    assertion.valid_from,
                    assertion.valid_to,
                )
                assertion.valid_to = None
                assertion.invalidated_at = None
                if assertion.status in {
                    AssertionStatus.INACTIVE,
                    AssertionStatus.SUPERSEDED,
                }:
                    assertion.status = AssertionStatus.DISPUTED
            if mentions:
                assertion.confidence = round(
                    sum(item.extractor_confidence for item in mentions) / len(mentions), 3
                )
            retained_assertions.append(assertion)
        snapshot.assertions = retained_assertions

        assertion_ids = {item.assertion_id for item in snapshot.assertions}
        snapshot.assertion_mentions = [
            item for item in snapshot.assertion_mentions if item.assertion_id in assertion_ids
        ]
        snapshot.event_effects = [
            item for item in snapshot.event_effects if item.assertion_id in assertion_ids
        ]
        retained_mention_ids = {
            item.mention_id for item in snapshot.assertion_mentions
        }
        snapshot.conflicts = [
            item
            for item in snapshot.conflicts
            if set(item.assertion_ids) <= assertion_ids
            and set(item.mention_ids) <= retained_mention_ids
        ]

        referenced_entities = {
            value
            for assertion in snapshot.assertions
            for value in (assertion.subject_id, assertion.object_id)
        }
        referenced_entities.update(item.entity_id for item in snapshot.entity_mentions)
        referenced_entities.update(item.entity_id for item in snapshot.event_entity_links)
        referenced_entities.update(
            item.source_entity.entity_id for item in snapshot.merge_records
        )
        referenced_entities.update(
            entity_id
            for item in snapshot.review_cases
            for entity_id in [
                item.provisional_entity_id,
                *(candidate.entity_id for candidate in item.candidates),
            ]
        )
        snapshot.entities = [
            item for item in snapshot.entities if item.entity_id in referenced_entities
        ]
        entity_mentions: dict[str, list] = {}
        for mention in snapshot.entity_mentions:
            entity_mentions.setdefault(mention.entity_id, []).append(mention)
        for entity in snapshot.entities:
            mentions = entity_mentions.get(entity.entity_id, [])
            entity.mention_ids = sorted(item.mention_id for item in mentions)
            entity.source_document_ids = sorted({item.document_id for item in mentions})


def _normal(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())

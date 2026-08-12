from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from rag_graph_demo.domain.tkg import (
    AssertionCoalescence,
    EntityMergeRecord,
    EntityReviewCase,
    GovernanceStatus,
    ResolutionAction,
    TKGSnapshot,
)
from rag_graph_demo.storage.event_registry import EventRegistry
from rag_graph_demo.storage.neo4j_tkg_store import Neo4jTKGStore
from rag_graph_demo.storage.tkg_registry import TKGRegistry


def _stable_id(prefix: str, *values: str) -> str:
    digest = hashlib.sha256("|".join(values).encode()).hexdigest()[:20]
    return f"{prefix}_{digest}"


class EntityGovernanceService:
    def __init__(
        self,
        registry: TKGRegistry,
        *,
        neo4j_store: Neo4jTKGStore | None = None,
        event_registry: EventRegistry | None = None,
    ) -> None:
        self._registry = registry
        self._neo4j_store = neo4j_store
        self._event_registry = event_registry

    async def list_reviews(
        self,
        status: GovernanceStatus | None = GovernanceStatus.PENDING,
    ) -> list[EntityReviewCase]:
        snapshot = await self._registry.snapshot()
        return [
            item
            for item in snapshot.review_cases
            if status is None or item.status == status
        ]

    async def resolve_review(
        self,
        review_id: str,
        *,
        action: ResolutionAction,
        selected_entity_id: str | None,
        corrected_entity_type: str | None = None,
        reviewer: str,
        reason: str,
    ) -> EntityReviewCase:
        if action == ResolutionAction.MATCH_EXISTING and not selected_entity_id:
            raise ValueError("MATCH_EXISTING requires selected_entity_id")

        def mutate(snapshot: TKGSnapshot) -> EntityReviewCase:
            review = next(
                (item for item in snapshot.review_cases if item.review_id == review_id),
                None,
            )
            if review is None:
                raise KeyError(f"Review case not found: {review_id}")
            if review.status != GovernanceStatus.PENDING:
                raise ValueError(f"Review case is already {review.status.value}")
            if action == ResolutionAction.MATCH_EXISTING:
                allowed = {item.entity_id for item in review.candidates}
                if selected_entity_id not in allowed:
                    raise ValueError("Selected entity is outside the reviewed shortlist")
                self._merge(
                    snapshot,
                    source_entity_id=review.provisional_entity_id,
                    target_entity_id=selected_entity_id,
                    actor=reviewer,
                    reason=reason,
                    review_id=review_id,
                )
                review.status = GovernanceStatus.APPROVED
            elif action == ResolutionAction.CREATE_NEW:
                if corrected_entity_type:
                    allowed_types = {
                        "Person",
                        "Organization",
                        "Location",
                        "ResearchObject",
                        "Method",
                        "Model",
                        "Dataset",
                        "Material",
                        "Device",
                        "Concept",
                    }
                    if corrected_entity_type not in allowed_types:
                        raise ValueError("Corrected entity type is outside the ontology")
                    entity = next(
                        (
                            item
                            for item in snapshot.entities
                            if item.entity_id == review.provisional_entity_id
                        ),
                        None,
                    )
                    if entity is None:
                        raise ValueError("Provisional entity no longer exists")
                    entity.entity_type = corrected_entity_type
                    review.candidate_type = corrected_entity_type
                review.status = GovernanceStatus.REJECTED
            elif action == ResolutionAction.AMBIGUOUS:
                review.status = GovernanceStatus.DEFERRED
            else:
                review.status = GovernanceStatus.REJECTED
            review.resolution_action = action
            review.selected_entity_id = selected_entity_id
            review.reviewer = reviewer
            review.reviewed_at = datetime.now(UTC)
            return review

        result = await self._registry.mutate(mutate)
        await self._sync()
        return result

    async def rollback_merge(
        self,
        merge_id: str,
        *,
        actor: str,
    ) -> EntityMergeRecord:
        def mutate(snapshot: TKGSnapshot) -> EntityMergeRecord:
            merge = next(
                (item for item in snapshot.merge_records if item.merge_id == merge_id),
                None,
            )
            if merge is None:
                raise KeyError(f"Merge record not found: {merge_id}")
            if merge.status != GovernanceStatus.APPROVED:
                raise ValueError(f"Merge is already {merge.status.value}")
            source = next(
                (
                    item
                    for item in snapshot.entities
                    if item.entity_id == merge.source_entity.entity_id
                ),
                None,
            )
            target = next(
                (item for item in snapshot.entities if item.entity_id == merge.target_entity_id),
                None,
            )
            if source is None or target is None:
                raise ValueError("Merged entities are no longer available")
            if source.status != "merged" or source.merged_into_id != target.entity_id:
                raise ValueError("Source entity has changed since this merge")
            if any(
                item.status == GovernanceStatus.APPROVED
                and item.merge_id != merge.merge_id
                and (
                    item.source_entity.entity_id == source.entity_id
                    or item.target_entity_id == source.entity_id
                )
                for item in snapshot.merge_records
            ):
                raise ValueError("A later active merge depends on this entity")
            source.status = "active"
            source.merged_into_id = None
            source_names = {source.canonical_name, *source.aliases}
            original_target_names = {
                merge.target_entity_before.canonical_name,
                *merge.target_entity_before.aliases,
            }
            target.aliases = [
                name
                for name in target.aliases
                if name not in source_names or name in original_target_names
            ]
            for mention in snapshot.entity_mentions:
                if mention.mention_id in merge.moved_entity_mention_ids:
                    mention.entity_id = source.entity_id
            for assertion in snapshot.assertions:
                if assertion.assertion_id in merge.moved_assertion_subject_ids:
                    assertion.subject_id = source.entity_id
                if assertion.assertion_id in merge.moved_assertion_object_ids:
                    assertion.object_id = source.entity_id
            for link in snapshot.event_entity_links:
                if link.link_id in merge.moved_event_link_ids:
                    link.entity_id = source.entity_id
            for coalescence in merge.assertion_coalescences:
                snapshot.assertions.append(
                    coalescence.removed_assertion.model_copy(deep=True)
                )
                for mention in snapshot.assertion_mentions:
                    if mention.mention_id in coalescence.moved_mention_ids:
                        mention.assertion_id = coalescence.removed_assertion.assertion_id
                for effect in snapshot.event_effects:
                    if effect.effect_id in coalescence.moved_effect_ids:
                        effect.assertion_id = coalescence.removed_assertion.assertion_id
            snapshot.conflicts = [
                item
                for item in snapshot.conflicts
                if not any(
                    conflict.conflict_id == item.conflict_id
                    for conflict in merge.conflict_snapshots
                )
            ]
            snapshot.conflicts.extend(
                item.model_copy(deep=True) for item in merge.conflict_snapshots
            )
            merge.status = GovernanceStatus.ROLLED_BACK
            merge.rolled_back_at = datetime.now(UTC)
            merge.reason = f"{merge.reason} | rolled back by {actor}"
            if merge.review_id:
                review = next(
                    (
                        item
                        for item in snapshot.review_cases
                        if item.review_id == merge.review_id
                    ),
                    None,
                )
                if review:
                    review.status = GovernanceStatus.ROLLED_BACK
            return merge

        result = await self._registry.mutate(mutate)
        await self._sync()
        return result

    @staticmethod
    def _merge(
        snapshot: TKGSnapshot,
        *,
        source_entity_id: str,
        target_entity_id: str,
        actor: str,
        reason: str,
        review_id: str | None,
    ) -> EntityMergeRecord:
        if source_entity_id == target_entity_id:
            raise ValueError("Source and target entity must differ")
        source = next(
            (item for item in snapshot.entities if item.entity_id == source_entity_id),
            None,
        )
        target = next(
            (item for item in snapshot.entities if item.entity_id == target_entity_id),
            None,
        )
        if source is None or target is None:
            raise ValueError("Source or target entity does not exist")
        if source.status != "active" or target.status != "active":
            raise ValueError("Only active entities can be merged")
        if source.entity_type != target.entity_type:
            raise ValueError("Cross-type entity merge is forbidden by the ontology")
        merge_id = _stable_id(
            "mrg",
            source.entity_id,
            target.entity_id,
            datetime.now(UTC).isoformat(),
        )
        moved_mentions = [
            item.mention_id
            for item in snapshot.entity_mentions
            if item.entity_id == source.entity_id
        ]
        moved_subjects = [
            item.assertion_id
            for item in snapshot.assertions
            if item.subject_id == source.entity_id
        ]
        moved_objects = [
            item.assertion_id
            for item in snapshot.assertions
            if item.object_id == source.entity_id
        ]
        moved_links = [
            item.link_id
            for item in snapshot.event_entity_links
            if item.entity_id == source.entity_id
        ]
        record = EntityMergeRecord(
            merge_id=merge_id,
            source_entity=source.model_copy(deep=True),
            target_entity_before=target.model_copy(deep=True),
            target_entity_id=target.entity_id,
            moved_entity_mention_ids=moved_mentions,
            moved_assertion_subject_ids=moved_subjects,
            moved_assertion_object_ids=moved_objects,
            moved_event_link_ids=moved_links,
            reason=reason,
            actor=actor,
            review_id=review_id,
            conflict_snapshots=[
                item.model_copy(deep=True)
                for item in snapshot.conflicts
                if set(item.assertion_ids) & set(moved_subjects + moved_objects)
            ],
        )
        source.status = "merged"
        source.merged_into_id = target.entity_id
        for name in [source.canonical_name, *source.aliases]:
            if name != target.canonical_name and name not in target.aliases:
                target.aliases.append(name)
        for mention in snapshot.entity_mentions:
            if mention.mention_id in moved_mentions:
                mention.entity_id = target.entity_id
        for assertion in snapshot.assertions:
            if assertion.assertion_id in moved_subjects:
                assertion.subject_id = target.entity_id
            if assertion.assertion_id in moved_objects:
                assertion.object_id = target.entity_id
        for link in snapshot.event_entity_links:
            if link.link_id in moved_links:
                link.entity_id = target.entity_id
        EntityGovernanceService._coalesce_assertions(snapshot, record)
        snapshot.merge_records.append(record)
        return record

    @staticmethod
    def _coalesce_assertions(
        snapshot: TKGSnapshot,
        record: EntityMergeRecord,
    ) -> None:
        grouped: dict[tuple[str, str, str, str, str], list] = {}
        for assertion in snapshot.assertions:
            key = (
                assertion.subject_id,
                assertion.predicate_id,
                assertion.object_id,
                assertion.valid_from or "",
                assertion.stated_valid_to or "",
            )
            grouped.setdefault(key, []).append(assertion)
        removed_ids: set[str] = set()
        for duplicates in grouped.values():
            if len(duplicates) < 2:
                continue
            survivor = sorted(duplicates, key=lambda item: item.assertion_id)[0]
            for duplicate in duplicates:
                if duplicate.assertion_id == survivor.assertion_id:
                    continue
                mention_ids = [
                    item.mention_id
                    for item in snapshot.assertion_mentions
                    if item.assertion_id == duplicate.assertion_id
                ]
                effect_ids = [
                    item.effect_id
                    for item in snapshot.event_effects
                    if item.assertion_id == duplicate.assertion_id
                ]
                record.assertion_coalescences.append(
                    AssertionCoalescence(
                        removed_assertion=duplicate.model_copy(deep=True),
                        survivor_assertion_id=survivor.assertion_id,
                        moved_mention_ids=mention_ids,
                        moved_effect_ids=effect_ids,
                    )
                )
                for mention in snapshot.assertion_mentions:
                    if mention.mention_id in mention_ids:
                        mention.assertion_id = survivor.assertion_id
                for effect in snapshot.event_effects:
                    if effect.effect_id in effect_ids:
                        effect.assertion_id = survivor.assertion_id
                for conflict in snapshot.conflicts:
                    conflict.assertion_ids = list(
                        dict.fromkeys(
                            survivor.assertion_id
                            if item == duplicate.assertion_id
                            else item
                            for item in conflict.assertion_ids
                        )
                    )
                removed_ids.add(duplicate.assertion_id)
        snapshot.assertions = [
            item for item in snapshot.assertions if item.assertion_id not in removed_ids
        ]

    async def _sync(self) -> None:
        if self._neo4j_store is not None:
            events = []
            mentions = []
            if self._event_registry is not None:
                events, mentions, _, _ = await self._event_registry.snapshot()
            await self._neo4j_store.replace_snapshot(
                await self._registry.snapshot(),
                events=events,
                event_mentions=mentions,
            )

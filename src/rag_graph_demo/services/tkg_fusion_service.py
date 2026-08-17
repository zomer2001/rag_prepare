from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime

from rag_graph_demo.domain.events import EventRecord
from rag_graph_demo.domain.tkg import (
    AssertionMention,
    AssertionRecord,
    AssertionStatus,
    ConflictCase,
    ConflictType,
    EntityMention,
    EntityRecord,
    EntityResolutionProposal,
    EntityReviewCase,
    EventEffect,
    EventEffectType,
    EventEntityLink,
    MentionStance,
    ResolutionAction,
    ResolutionDecision,
    SemanticEntityCandidate,
    SemanticExtractionBatch,
    SemanticRelationshipCandidate,
    TKGIndexResult,
    TKGSnapshot,
)
from rag_graph_demo.services.entity_resolution_service import (
    EntityResolutionService,
    normalize_name,
)
from rag_graph_demo.storage.event_registry import EventRegistry
from rag_graph_demo.storage.neo4j_tkg_store import Neo4jTKGStore
from rag_graph_demo.storage.tkg_registry import TKGRegistry

ENTITY_TYPE_MAP = {
    "person": "Person",
    "organization": "Organization",
    "location": "Location",
    "researchobject": "ResearchObject",
    "method": "Method",
    "model": "Model",
    "dataset": "Dataset",
    "material": "Material",
    "device": "Device",
    "concept": "Concept",
}
PREDICATE_HINTS = {
    "develop": "develops",
    "produc": "produces",
    "use": "uses",
    "utiliz": "uses",
    "evaluat": "evaluates",
    "benchmark": "evaluates",
    "train": "trainedOn",
    "support": "supports",
    "contradict": "contradicts",
    "correct": "replaces",
    "supersed": "replaces",
    "replac": "replaces",
    "locat": "locatedAt",
    "affiliat": "participatesIn",
    "partner": "partnersWith",
}


def _normal(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value.casefold())


def _stable_id(prefix: str, *values: str) -> str:
    digest = hashlib.sha256("|".join(values).encode()).hexdigest()[:20]
    return f"{prefix}_{digest}"


def _entity_type(value: str) -> str:
    normalized = _normal(value)
    return ENTITY_TYPE_MAP.get(normalized, value.strip().title() or "Concept")


def _predicate(value: str) -> str:
    normalized = value.casefold()
    for hint, predicate in PREDICATE_HINTS.items():
        if hint in normalized:
            return predicate
    tokens = re.findall(r"[a-z][a-z0-9_-]*", normalized)
    if tokens:
        return tokens[0][:100]
    chinese = re.sub(r"[^\u4e00-\u9fff]", "", value)
    return chinese[:100] or "relatedFact"


class TKGFusionService:
    def __init__(
        self,
        registry: TKGRegistry,
        *,
        ontology_version: str,
        entity_resolver: EntityResolutionService | None = None,
        neo4j_store: Neo4jTKGStore | None = None,
        event_registry: EventRegistry | None = None,
    ) -> None:
        self._registry = registry
        self._ontology_version = ontology_version
        self._entity_resolver = entity_resolver
        self._neo4j_store = neo4j_store
        self._event_registry = event_registry

    async def index_document(
        self,
        document_id: str,
        document_title: str,
        semantic: SemanticExtractionBatch,
        events: list[EventRecord],
    ) -> TKGIndexResult:
        # Lifecycle effects must be applied in effective-time order. Source text can
        # mention a future recurrence before an earlier termination (for example,
        # daylight-saving changes), which would otherwise create an inverted interval.
        ordered_events = sorted(
            events,
            key=lambda item: (
                item.start_time or item.end_time or "9999-12-31",
                item.created_at,
                item.event_id,
            ),
        )
        proposals: dict[str, EntityResolutionProposal] = {}
        resolution_candidates = list(semantic.entities)
        if self._entity_resolver is not None:
            snapshot = await self._registry.snapshot()
            resolution_candidates.extend(
                self._event_resolution_candidates(snapshot, semantic, ordered_events)
            )
            resolved = await self._entity_resolver.resolve_batch(
                snapshot, resolution_candidates
            )
            proposals = {item.candidate_key: item for item in resolved}

        def fuse(snapshot: TKGSnapshot) -> TKGIndexResult:
            before_conflicts = len(snapshot.conflicts)
            names: dict[str, str] = {}
            for candidate in semantic.entities:
                entity = self._resolve_entity(
                    snapshot,
                    document_id,
                    document_title,
                    candidate,
                    proposals.get(
                        "|".join(
                            (
                                normalize_name(candidate.name),
                                normalize_name(candidate.entity_type),
                            )
                        )
                    ),
                )
                names[_normal(candidate.name)] = entity.entity_id
            for candidate in resolution_candidates[len(semantic.entities) :]:
                entity = self._resolve_entity(
                    snapshot,
                    document_id,
                    document_title,
                    candidate,
                    proposals.get(
                        "|".join(
                            (
                                normalize_name(candidate.name),
                                normalize_name(candidate.entity_type),
                            )
                        )
                    ),
                )
                names[_normal(candidate.name)] = entity.entity_id
            for relationship in semantic.relationships:
                self._fuse_relationship(
                    snapshot,
                    document_id,
                    document_title,
                    relationship,
                    names,
                )
            link_count = 0
            effect_count = 0
            for event in ordered_events:
                for role, values in (("actor", event.actors), ("object", event.objects)):
                    for value in values:
                        entity = self._resolve_name(
                            snapshot,
                            document_id,
                            document_title,
                            value,
                            names,
                            evidence=event.summary,
                        )
                        link_id = _stable_id("eln", event.event_id, entity.entity_id, role)
                        if not any(
                            item.link_id == link_id for item in snapshot.event_entity_links
                        ):
                            snapshot.event_entity_links.append(
                                EventEntityLink(
                                    link_id=link_id,
                                    event_id=event.event_id,
                                    entity_id=entity.entity_id,
                                    role=role,
                                    surface_form=value,
                                    document_id=document_id,
                                )
                            )
                            link_count += 1
                for fact in event.fact_effects:
                    effect_count += self._apply_event_fact(
                        snapshot,
                        document_id,
                        document_title,
                        event,
                        fact,
                        names,
                    )
            self._build_stance_conflicts(snapshot)
            document_assertions = {
                item.assertion_id
                for item in snapshot.assertion_mentions
                if item.document_id == document_id
            }
            document_entities = {
                item.entity_id
                for item in snapshot.entity_mentions
                if item.document_id == document_id
            }
            return TKGIndexResult(
                document_id=document_id,
                entity_count=len(document_entities),
                assertion_count=len(document_assertions),
                event_link_count=link_count,
                event_effect_count=effect_count,
                conflict_count=len(snapshot.conflicts) - before_conflicts,
            )

        result = await self._registry.mutate_document(document_id, fuse)
        await self.sync_neo4j()
        return result

    async def delete_document(self, document_id: str) -> None:
        await self._registry.delete_document(document_id)
        await self.sync_neo4j()

    async def sync_neo4j(self) -> None:
        if self._neo4j_store is None:
            return
        snapshot = await self._registry.snapshot()
        events = []
        mentions = []
        relations = []
        if self._event_registry is not None:
            events, mentions, relations, _ = await self._event_registry.snapshot()
        await self._neo4j_store.replace_snapshot(
            snapshot,
            events=events,
            event_mentions=mentions,
            event_relations=relations,
        )

    @staticmethod
    def _event_resolution_candidates(
        snapshot: TKGSnapshot,
        semantic: SemanticExtractionBatch,
        events: list[EventRecord],
    ) -> list[SemanticEntityCandidate]:
        semantic_names = {
            normalize_name(name): item
            for item in semantic.entities
            for name in [item.name, *item.aliases]
        }
        existing_names: dict[str, list[EntityRecord]] = {}
        for entity in snapshot.entities:
            if entity.status != "active":
                continue
            for name in [entity.canonical_name, *entity.aliases]:
                existing_names.setdefault(normalize_name(name), []).append(entity)
        candidates: list[SemanticEntityCandidate] = []
        seen = {normalize_name(item.name) for item in semantic.entities}
        for event in events:
            values = [*event.actors, *event.objects]
            values.extend(
                value
                for fact in event.fact_effects
                for value in (fact.subject, fact.object)
            )
            for value in values:
                normalized = normalize_name(value)
                if not normalized or normalized in seen:
                    continue
                semantic_match = semantic_names.get(normalized)
                existing = existing_names.get(normalized, [])
                entity_type = (
                    semantic_match.entity_type
                    if semantic_match is not None
                    else existing[0].entity_type
                    if len(existing) == 1
                    else "Concept"
                )
                candidates.append(
                    SemanticEntityCandidate(
                        name=value,
                        entity_type=entity_type,
                        description=event.summary,
                    )
                )
                seen.add(normalized)
        return candidates

    def _resolve_entity(
        self,
        snapshot: TKGSnapshot,
        document_id: str,
        document_title: str,
        candidate: SemanticEntityCandidate,
        proposal: EntityResolutionProposal | None = None,
    ) -> EntityRecord:
        normalized = _normal(candidate.name)
        entity_type = proposal.normalized_type if proposal else _entity_type(candidate.entity_type)
        matches = [
            item
            for item in snapshot.entities
            if entity_type == item.entity_type
            and normalized
            in {_normal(item.canonical_name), *(_normal(alias) for alias in item.aliases)}
        ]
        selected = (
            next(
                (
                    item
                    for item in snapshot.entities
                    if proposal
                    and item.entity_id == proposal.selected_entity_id
                    and item.status == "active"
                ),
                None,
            )
            if proposal and proposal.action == ResolutionAction.MATCH_EXISTING
            else None
        )
        if selected is not None:
            entity = selected
            action = ResolutionAction.MATCH_EXISTING
            reason = proposal.reason
        elif len(matches) == 1 and proposal is None:
            entity = matches[0]
            action = ResolutionAction.MATCH_EXISTING
            reason = "规范名或别名精确匹配，且本体类型一致。"
        else:
            is_ambiguous = bool(
                proposal
                and proposal.action
                in {ResolutionAction.AMBIGUOUS, ResolutionAction.REJECT}
            ) or len(matches) > 1
            entity_id = (
                _stable_id("entamb", document_id, entity_type, normalized)
                if is_ambiguous
                else _stable_id("ent", entity_type, normalized)
            )
            entity = next(
                (item for item in snapshot.entities if item.entity_id == entity_id), None
            ) or EntityRecord(
                entity_id=entity_id,
                canonical_name=candidate.name,
                entity_type=entity_type,
                aliases=candidate.aliases,
                description=candidate.description,
                ontology_version=self._ontology_version,
            )
            if not any(item.entity_id == entity.entity_id for item in snapshot.entities):
                snapshot.entities.append(entity)
            action = proposal.action if proposal else (
                ResolutionAction.AMBIGUOUS
                if len(matches) > 1
                else ResolutionAction.CREATE_NEW
            )
            reason = proposal.reason if proposal else (
                "存在多个同类型精确候选，未自动合并，创建隔离身份等待处理。"
                if matches
                else "未找到同类型的规范名或别名精确匹配。"
            )
            if is_ambiguous:
                review_id = _stable_id("rev", document_id, entity.entity_id)
                if not any(item.review_id == review_id for item in snapshot.review_cases):
                    snapshot.review_cases.append(
                        EntityReviewCase(
                            review_id=review_id,
                            document_id=document_id,
                            provisional_entity_id=entity.entity_id,
                            candidate_name=candidate.name,
                            candidate_type=entity_type,
                            description=candidate.description,
                            candidates=proposal.candidates if proposal else [],
                            reason=reason,
                        )
                    )
        for alias in candidate.aliases:
            if _normal(alias) != _normal(entity.canonical_name) and alias not in entity.aliases:
                entity.aliases.append(alias)
        if candidate.description and not entity.description:
            entity.description = candidate.description
        mention_id = _stable_id("emn", document_id, entity.entity_id, candidate.name)
        if not any(item.mention_id == mention_id for item in snapshot.entity_mentions):
            snapshot.entity_mentions.append(
                EntityMention(
                    mention_id=mention_id,
                    entity_id=entity.entity_id,
                    document_id=document_id,
                    document_title=document_title,
                    surface_form=candidate.name,
                    evidence_text=candidate.description or candidate.name,
                    source_segment_id=candidate.source_segment_id,
                )
            )
        decision_id = _stable_id("dec", document_id, entity_type, normalized)
        snapshot.resolution_decisions.append(
            ResolutionDecision(
                decision_id=decision_id,
                document_id=document_id,
                candidate_name=candidate.name,
                candidate_type=entity_type,
                action=action,
                selected_entity_id=entity.entity_id,
                candidate_entity_ids=(
                    [item.entity_id for item in proposal.candidates]
                    if proposal
                    else [item.entity_id for item in matches]
                ),
                reason=reason,
                similarity=(
                    proposal.candidates[0].embedding_score
                    if proposal and proposal.candidates
                    else None
                ),
                decided_by=proposal.decided_by if proposal else "rule",
                ontology_version=self._ontology_version,
            )
        )
        entity.updated_at = datetime.now(UTC)
        return entity

    def _resolve_name(
        self,
        snapshot: TKGSnapshot,
        document_id: str,
        document_title: str,
        name: str,
        names: dict[str, str],
        *,
        evidence: str,
    ) -> EntityRecord:
        entity_id = names.get(_normal(name))
        if entity_id:
            return next(item for item in snapshot.entities if item.entity_id == entity_id)
        matches = [
            item
            for item in snapshot.entities
            if _normal(name)
            in {_normal(item.canonical_name), *(_normal(alias) for alias in item.aliases)}
        ]
        candidate_type = matches[0].entity_type if len(matches) == 1 else "Concept"
        entity = self._resolve_entity(
            snapshot,
            document_id,
            document_title,
            SemanticEntityCandidate(
                name=name,
                entity_type=candidate_type,
                description=evidence,
            ),
        )
        names[_normal(name)] = entity.entity_id
        return entity

    def _fuse_relationship(
        self,
        snapshot: TKGSnapshot,
        document_id: str,
        document_title: str,
        relationship: SemanticRelationshipCandidate,
        names: dict[str, str],
    ) -> AssertionRecord:
        subject = self._resolve_name(
            snapshot,
            document_id,
            document_title,
            relationship.source,
            names,
            evidence=relationship.description,
        )
        obj = self._resolve_name(
            snapshot,
            document_id,
            document_title,
            relationship.target,
            names,
            evidence=relationship.description,
        )
        return self._upsert_assertion(
            snapshot,
            document_id,
            document_title,
            subject.entity_id,
            _predicate(relationship.predicate),
            obj.entity_id,
            evidence=relationship.description,
            confidence=relationship.confidence,
            source_segment_id=relationship.source_segment_id,
        )

    def _upsert_assertion(
        self,
        snapshot: TKGSnapshot,
        document_id: str,
        document_title: str,
        subject_id: str,
        predicate_id: str,
        object_id: str,
        *,
        evidence: str,
        confidence: float,
        stance: MentionStance = MentionStance.ASSERTED,
        valid_from: str | None = None,
        valid_to: str | None = None,
        source_segment_id: str | None = None,
    ) -> AssertionRecord:
        key = "|".join(
            (subject_id, predicate_id, object_id, valid_from or "", valid_to or "")
        )
        assertion_id = _stable_id("ast", key)
        assertion = next(
            (
                item
                for item in snapshot.assertions
                if item.assertion_id == assertion_id
                or (
                    item.subject_id == subject_id
                    and item.predicate_id == predicate_id
                    and item.object_id == object_id
                    and item.valid_from == valid_from
                    and item.stated_valid_to == valid_to
                )
            ),
            None,
        )
        if assertion is None:
            assertion = AssertionRecord(
                assertion_id=assertion_id,
                subject_id=subject_id,
                predicate_id=predicate_id,
                object_id=object_id,
                valid_from=valid_from,
                valid_to=valid_to,
                stated_valid_to=valid_to,
                status=AssertionStatus.CANDIDATE,
                ontology_version=self._ontology_version,
            )
            snapshot.assertions.append(assertion)
        mention_id = _stable_id("amn", document_id, assertion_id, stance.value, evidence)
        if not any(item.mention_id == mention_id for item in snapshot.assertion_mentions):
            snapshot.assertion_mentions.append(
                AssertionMention(
                    mention_id=mention_id,
                    assertion_id=assertion_id,
                    document_id=document_id,
                    document_title=document_title,
                    evidence_text=evidence,
                    source_segment_id=source_segment_id,
                    stance=stance,
                    extractor_confidence=confidence,
                )
            )
        return assertion

    def _apply_event_fact(
        self,
        snapshot: TKGSnapshot,
        document_id: str,
        document_title: str,
        event: EventRecord,
        fact,
        names: dict[str, str],
    ) -> int:
        subject = self._resolve_name(
            snapshot, document_id, document_title, fact.subject, names, evidence=event.summary
        )
        obj = self._resolve_name(
            snapshot, document_id, document_title, fact.object, names, evidence=event.summary
        )
        predicate = _predicate(fact.predicate)
        excluded_statuses = {
            AssertionStatus.RETRACTED,
            AssertionStatus.SUPERSEDED,
        }
        if fact.effect_type in {EventEffectType.CREATES, EventEffectType.UPDATES}:
            excluded_statuses.add(AssertionStatus.INACTIVE)
        same_fact = [
            item
            for item in snapshot.assertions
            if item.subject_id == subject.entity_id
            and item.predicate_id == predicate
            and item.object_id == obj.entity_id
            and item.status not in excluded_statuses
        ]
        replaced_facts = [
            item
            for item in snapshot.assertions
            if item.subject_id == subject.entity_id
            and item.predicate_id == predicate
            and item.object_id != obj.entity_id
            and item.status
            not in {
                AssertionStatus.INACTIVE,
                AssertionStatus.RETRACTED,
                AssertionStatus.SUPERSEDED,
            }
        ]
        if fact.effect_type == EventEffectType.DENIES or event.stance == "denied":
            stance = MentionStance.DENIED
        elif event.stance == "uncertain":
            stance = MentionStance.UNCERTAIN
        else:
            stance = MentionStance.ASSERTED
        assertion = same_fact[-1] if same_fact else self._upsert_assertion(
            snapshot,
            document_id,
            document_title,
            subject.entity_id,
            predicate,
            obj.entity_id,
            evidence=fact.reason or event.summary,
            confidence=event.confidence,
            stance=stance,
            valid_from=fact.valid_from or event.start_time,
            valid_to=fact.valid_to,
        )
        if same_fact:
            self._upsert_assertion(
                snapshot,
                document_id,
                document_title,
                assertion.subject_id,
                assertion.predicate_id,
                assertion.object_id,
                evidence=fact.reason or event.summary,
                confidence=event.confidence,
                stance=stance,
                valid_from=assertion.valid_from,
                valid_to=assertion.valid_to,
            )
        created = self._append_event_effect(
            snapshot,
            document_id,
            event,
            assertion,
            fact.effect_type,
            stance,
            fact.reason,
            fact.valid_from,
        )
        if fact.effect_type in {EventEffectType.UPDATES, EventEffectType.SUPERSEDES}:
            for replaced in replaced_facts:
                created += self._append_event_effect(
                    snapshot,
                    document_id,
                    event,
                    replaced,
                    EventEffectType.SUPERSEDES,
                    stance,
                    fact.reason or f"Replaced by {assertion.assertion_id}",
                    fact.valid_from,
                )
        return created

    @staticmethod
    def _append_event_effect(
        snapshot: TKGSnapshot,
        document_id: str,
        event: EventRecord,
        assertion: AssertionRecord,
        effect_type: EventEffectType,
        stance: MentionStance,
        reason: str,
        valid_from: str | None,
    ) -> int:
        effect_id = _stable_id(
            "eff",
            document_id,
            event.event_id,
            assertion.assertion_id,
            effect_type.value,
        )
        if any(item.effect_id == effect_id for item in snapshot.event_effects):
            return 0
        effect = EventEffect(
            effect_id=effect_id,
            event_id=event.event_id,
            assertion_id=assertion.assertion_id,
            effect_type=effect_type,
            stance=stance,
            document_id=document_id,
            effective_time=valid_from or event.start_time,
            reason=reason or event.summary,
        )
        snapshot.event_effects.append(effect)
        # Keep the in-memory lifecycle current while multiple events from the same
        # document are fused in one transaction. The registry recomputes the same
        # state from provenance before persistence.
        if stance != MentionStance.UNCERTAIN:
            if effect_type in {
                EventEffectType.CREATES,
                EventEffectType.CONFIRMS,
                EventEffectType.UPDATES,
            }:
                assertion.status = AssertionStatus.ACTIVE
                if (
                    assertion.stated_valid_to is None
                    and assertion.valid_to
                    and effect.effective_time
                    and effect.effective_time > assertion.valid_to
                ):
                    assertion.valid_to = None
                    assertion.invalidated_at = None
            elif effect_type == EventEffectType.DENIES:
                assertion.status = AssertionStatus.DISPUTED
            elif effect_type == EventEffectType.TERMINATES:
                assertion.status = AssertionStatus.INACTIVE
                assertion.valid_to = effect.effective_time
                assertion.invalidated_at = effect.created_at
            elif effect_type == EventEffectType.SUPERSEDES:
                assertion.status = AssertionStatus.SUPERSEDED
                assertion.valid_to = effect.effective_time
                assertion.invalidated_at = effect.created_at
        return 1

    @staticmethod
    def _build_stance_conflicts(snapshot: TKGSnapshot) -> None:
        existing = {item.conflict_id for item in snapshot.conflicts}
        for assertion in snapshot.assertions:
            mentions = [
                item
                for item in snapshot.assertion_mentions
                if item.assertion_id == assertion.assertion_id
            ]
            stances = {item.stance for item in mentions}
            if not {MentionStance.ASSERTED, MentionStance.DENIED} <= stances:
                continue
            conflict_id = _stable_id("cnf", assertion.assertion_id, "stance")
            if conflict_id in existing:
                continue
            snapshot.conflicts.append(
                ConflictCase(
                    conflict_id=conflict_id,
                    conflict_type=ConflictType.STANCE,
                    assertion_ids=[assertion.assertion_id],
                    mention_ids=sorted(item.mention_id for item in mentions),
                    description="来源对同一声明同时存在肯定与否定立场。",
                )
            )

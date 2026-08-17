from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from rag_graph_demo.domain.tkg import (
    TKGExpansionRequest,
    TKGExpansionResponse,
    TKGQueryRequest,
    TKGQueryResponse,
)
from rag_graph_demo.retrieval.models import EvidenceCandidate, QueryPlan


class CoordinationGraphStore(Protocol):
    async def query(self, request: TKGQueryRequest) -> TKGQueryResponse: ...

    async def expand(self, request: TKGExpansionRequest) -> TKGExpansionResponse: ...


@dataclass(frozen=True)
class EvoLinkResult:
    graph_candidates: tuple[EvidenceCandidate, ...] = ()
    event_candidates: tuple[EvidenceCandidate, ...] = ()
    evidence_graph: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: {"nodes": [], "edges": []}
    )
    metadata: dict[str, Any] = field(default_factory=dict)


class EvoLinkCoordinator:
    """Connect semantic assertions and events without replacing the three routes."""

    def __init__(self, graph_store: CoordinationGraphStore) -> None:
        self._graph_store = graph_store

    async def coordinate(
        self,
        plan: QueryPlan,
        *,
        entity_names: Iterable[str] = (),
        event_candidates: Iterable[EvidenceCandidate] = (),
        limit: int,
    ) -> EvoLinkResult:
        names = self._unique([*plan.entities, *entity_names], limit=8)
        seed_queries = await asyncio.gather(
            *(
                self._graph_store.query(
                    TKGQueryRequest(
                        entity_name=name,
                        include_inactive=True,
                        include_disputed=True,
                        limit=limit,
                    )
                )
                for name in names
            )
        )
        entity_ids = self._unique(
            [item.entity_id for result in seed_queries for item in result.entities],
            limit=limit,
        )
        assertion_ids = self._unique(
            [
                item.assertion_id
                for result in seed_queries
                for item in result.assertions
            ],
            limit=limit,
        )
        event_ids = self._unique(
            [
                str(item.metadata.get("event_id") or "")
                for item in event_candidates
            ],
            limit=limit,
        )
        if not entity_ids and not assertion_ids and not event_ids:
            return EvoLinkResult(
                metadata={
                    "entity_names": names,
                    "seed_entity_count": 0,
                    "seed_assertion_count": 0,
                    "seed_event_count": 0,
                }
            )

        expansion = await self._graph_store.expand(
            TKGExpansionRequest(
                entity_ids=entity_ids,
                assertion_ids=assertion_ids,
                event_ids=event_ids,
                start_time=plan.start_time,
                end_time=plan.end_time,
                include_inactive=True,
                include_disputed=True,
                expand_event_chain=plan.expand_event_chain,
                limit=limit,
            )
        )
        graph_candidates = self._assertion_candidates(expansion)
        linked_event_candidates = self._event_candidates(expansion)
        evidence_graph = self._evidence_graph(expansion)
        return EvoLinkResult(
            graph_candidates=tuple(graph_candidates),
            event_candidates=tuple(linked_event_candidates),
            evidence_graph=evidence_graph,
            metadata={
                "entity_names": names,
                "seed_entity_count": len(entity_ids),
                "seed_assertion_count": len(assertion_ids),
                "seed_event_count": len(event_ids),
                "linked_assertion_count": len(expansion.assertions),
                "linked_event_count": len(expansion.events),
                "linked_effect_count": len(expansion.effects),
                "graph_node_count": len(evidence_graph["nodes"]),
                "graph_edge_count": len(evidence_graph["edges"]),
            },
        )

    @staticmethod
    def _assertion_candidates(
        expansion: TKGExpansionResponse,
    ) -> list[EvidenceCandidate]:
        assertions = {item.assertion_id: item for item in expansion.assertions}
        entities = {item.entity_id: item for item in expansion.entities}
        effects: dict[str, list[Any]] = defaultdict(list)
        for effect in expansion.effects:
            effects[effect.assertion_id].append(effect)
        conflicted = {
            assertion_id
            for conflict in expansion.conflicts
            for assertion_id in conflict.assertion_ids
        }
        result: list[EvidenceCandidate] = []
        for mention in expansion.mentions:
            assertion = assertions.get(mention.assertion_id)
            if assertion is None:
                continue
            subject = entities.get(assertion.subject_id)
            obj = entities.get(assertion.object_id)
            result.append(
                EvidenceCandidate(
                    evidence_id=f"graph:evolink:{mention.mention_id}",
                    channel="graph",
                    rank=len(result) + 1,
                    document_id=mention.document_id,
                    chunk_id=mention.source_segment_id,
                    text=mention.evidence_text,
                    score=assertion.confidence,
                    metadata={
                        "source": "statement_tkg",
                        "linked_by": "ours_evolink",
                        "assertion_id": assertion.assertion_id,
                        "statement": (
                            f"{subject.canonical_name if subject else assertion.subject_id} "
                            f"{assertion.predicate_id} "
                            f"{obj.canonical_name if obj else assertion.object_id}"
                        ),
                        "valid_from": assertion.valid_from,
                        "valid_to": assertion.valid_to,
                        "status": assertion.status.value,
                        "stance": mention.stance.value,
                        "event_effects": [
                            item.effect_type.value
                            for item in effects[assertion.assertion_id]
                        ],
                        "conflicted": assertion.assertion_id in conflicted,
                    },
                )
            )
        return result

    @staticmethod
    def _event_candidates(
        expansion: TKGExpansionResponse,
    ) -> list[EvidenceCandidate]:
        events = {
            str(item.get("event_id") or ""): item
            for item in expansion.events
            if item.get("event_id")
        }
        result: list[EvidenceCandidate] = []
        for mention in expansion.event_mentions:
            event_id = str(mention.get("event_id") or "")
            event = events.get(event_id, {})
            evidence = str(mention.get("evidence") or "").strip()
            if not event_id or not evidence:
                continue
            result.append(
                EvidenceCandidate(
                    evidence_id=(
                        "event:evolink:"
                        f"{mention.get('mention_id') or event_id}"
                    ),
                    channel="event",
                    rank=len(result) + 1,
                    document_id=mention.get("document_id"),
                    chunk_id=(
                        mention.get("source_segment_id")
                        or mention.get("mention_id")
                    ),
                    text=evidence,
                    score=float(event.get("confidence") or 0),
                    metadata={
                        "source": "event_tkg",
                        "linked_by": "ours_evolink",
                        "event_id": event_id,
                        "event_type": event.get("event_type"),
                        "event_summary": event.get("summary"),
                        "event_time": (
                            event.get("start_time") or event.get("end_time")
                        ),
                        "event_status": event.get("status"),
                        "event_actors": event.get("actors") or [],
                        "event_objects": event.get("objects") or [],
                    },
                )
            )
        return result

    @staticmethod
    def _evidence_graph(
        expansion: TKGExpansionResponse,
    ) -> dict[str, list[dict[str, Any]]]:
        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        documents_by_assertion: dict[str, set[str]] = defaultdict(set)
        documents_by_event: dict[str, set[str]] = defaultdict(set)
        for mention in expansion.mentions:
            documents_by_assertion[mention.assertion_id].add(mention.document_id)
        for mention in expansion.event_mentions:
            event_id = str(mention.get("event_id") or "")
            document_id = str(mention.get("document_id") or "")
            if event_id and document_id:
                documents_by_event[event_id].add(document_id)

        for entity in expansion.entities:
            nodes[entity.entity_id] = {
                "id": entity.entity_id,
                "type": "entity",
                "route": "semantic",
                "label": entity.canonical_name,
                "entity_type": entity.entity_type,
                "aliases": entity.aliases,
            }
        for assertion in expansion.assertions:
            nodes[assertion.assertion_id] = {
                "id": assertion.assertion_id,
                "type": "assertion",
                "route": "semantic",
                "label": assertion.predicate_id,
                "predicate": assertion.predicate_id,
                "status": assertion.status.value,
                "valid_from": assertion.valid_from,
                "valid_to": assertion.valid_to,
                "source_document_ids": sorted(
                    documents_by_assertion[assertion.assertion_id]
                ),
            }
            edges[(assertion.subject_id, assertion.assertion_id, "SUBJECT")] = {
                "source": assertion.subject_id,
                "target": assertion.assertion_id,
                "type": "SUBJECT",
                "route": "semantic",
            }
            edges[(assertion.assertion_id, assertion.object_id, "OBJECT")] = {
                "source": assertion.assertion_id,
                "target": assertion.object_id,
                "type": "OBJECT",
                "route": "semantic",
            }
        for event in expansion.events:
            event_id = str(event.get("event_id") or "")
            if not event_id:
                continue
            nodes[event_id] = {
                "id": event_id,
                "type": "event",
                "route": "event",
                "label": event.get("summary") or event.get("event_type") or event_id,
                "event_type": event.get("event_type"),
                "start_time": event.get("start_time"),
                "end_time": event.get("end_time"),
                "status": event.get("status"),
                "source_document_ids": sorted(documents_by_event[event_id]),
            }
        for link in expansion.event_links:
            event_id = str(link.get("event_id") or "")
            entity_id = str(link.get("entity_id") or "")
            role = str(link.get("role") or "participant").upper()
            if event_id in nodes and entity_id in nodes:
                edges[(event_id, entity_id, role)] = {
                    "source": event_id,
                    "target": entity_id,
                    "type": role,
                    "route": "cross",
                }
        for effect in expansion.effects:
            if effect.event_id not in nodes or effect.assertion_id not in nodes:
                continue
            effect_type = effect.effect_type.value.upper()
            edges[(effect.event_id, effect.assertion_id, effect_type)] = {
                "source": effect.event_id,
                "target": effect.assertion_id,
                "type": effect_type,
                "route": "cross",
                "effective_time": effect.effective_time,
                "document_id": effect.document_id,
            }
        return {"nodes": list(nodes.values()), "edges": list(edges.values())}

    @staticmethod
    def _unique(values: Iterable[str], *, limit: int) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = " ".join(str(value).split()).strip()
            key = cleaned.casefold()
            if cleaned and key not in seen:
                result.append(cleaned)
                seen.add(key)
            if len(result) >= limit:
                break
        return result

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from rag_graph_demo.domain.tkg import TKGExpansionRequest
from rag_graph_demo.evaluation.models import RetrievedChunk
from rag_graph_demo.retrieval.planner import QueryPlanner
from rag_graph_demo.retrieval.tkg_vector import TKGVectorIndex
from rag_graph_demo.storage.neo4j_tkg_store import Neo4jTKGStore


class UnifiedTKGRetriever:
    """Retrieve source evidence from one TKG via vector seeds and bounded expansion."""

    def __init__(
        self,
        planner: QueryPlanner,
        vector_index: TKGVectorIndex,
        graph_store: Neo4jTKGStore,
        *,
        rrf_k: int = 60,
        candidate_multiplier: int = 3,
        dynamic: bool = False,
    ) -> None:
        self._planner = planner
        self._vector_index = vector_index
        self._graph_store = graph_store
        self._rrf_k = rrf_k
        self._candidate_multiplier = max(1, candidate_multiplier)
        self._dynamic = dynamic

    async def retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]:
        plan = await self._planner.plan(question)
        limit = max(top_k, top_k * self._candidate_multiplier)
        entity_query = (
            " ".join(plan.entities)
            if self._dynamic and plan.entities
            else " ".join([*plan.entities, *plan.low_level_keywords])
        ) or question
        relation_query = " ".join([*plan.relations, *plan.high_level_keywords]) or question
        event_query = " ".join(
            [
                *(plan.entities if self._dynamic else []),
                *plan.event_types,
                *plan.high_level_keywords,
            ]
        ) or question
        enabled = self._enabled_routes(plan)

        async def search(kind: str, text: str) -> list[dict[str, Any]]:
            if not enabled[kind]:
                return []
            values = await self._vector_index.search(kind, text, limit=limit)
            return self._select_hits(kind, values, top_k)

        chunk_hits, entity_hits, predicate_hits, assertion_hits, event_hits = await asyncio.gather(
            search("chunk", question),
            search("entity", entity_query),
            search("predicate", relation_query),
            search("assertion", question),
            search("event", event_query),
        )
        expansion = await self._graph_store.expand(
            TKGExpansionRequest(
                entity_ids=self._ids(entity_hits),
                predicate_ids=self._ids(predicate_hits),
                assertion_ids=self._ids(assertion_hits),
                event_ids=self._ids(event_hits),
                entity_scores=self._score_map(entity_hits),
                predicate_scores=self._score_map(predicate_hits),
                assertion_scores=self._score_map(assertion_hits),
                event_scores=self._score_map(event_hits),
                rank_by_seed_score=self._dynamic,
                start_time=plan.start_time,
                end_time=plan.end_time,
                include_inactive=(
                    bool(
                        plan.start_time
                        or plan.end_time
                        or set(plan.state_intents)
                        & {"change", "replacement", "correction"}
                    )
                    if self._dynamic
                    else bool(
                        set(plan.state_intents) & {"latest", "change", "replacement"}
                    )
                ),
                include_disputed=True,
                expand_event_chain=plan.expand_event_chain,
                limit=limit,
            )
        )
        assertion_rank = self._rank_map(assertion_hits)
        entity_rank = self._rank_map(entity_hits)
        predicate_rank = self._rank_map(predicate_hits)
        event_rank = self._rank_map(event_hits)
        assertions = {item.assertion_id: item for item in expansion.assertions}
        entities = {item.entity_id: item for item in expansion.entities}
        effects_by_assertion: dict[str, list[Any]] = defaultdict(list)
        for effect in expansion.effects:
            effects_by_assertion[effect.assertion_id].append(effect)
        conflicted = {
            assertion_id
            for conflict in expansion.conflicts
            for assertion_id in conflict.assertion_ids
        }

        graph_evidence: list[dict[str, Any]] = []
        for mention in expansion.mentions:
            assertion = assertions.get(mention.assertion_id)
            if assertion is None:
                continue
            seed_ranks = [
                assertion_rank.get(assertion.assertion_id),
                entity_rank.get(assertion.subject_id),
                entity_rank.get(assertion.object_id),
                predicate_rank.get(assertion.predicate_id),
            ]
            seed_rank = min((value for value in seed_ranks if value), default=limit + 1)
            relevance = self._assertion_relevance(
                assertion,
                entity_hits=entity_hits,
                predicate_hits=predicate_hits,
                assertion_hits=assertion_hits,
                plan=plan,
            )
            graph_evidence.append(
                {
                    "document_id": mention.document_id,
                    "chunk_id": mention.source_segment_id or mention.mention_id,
                    "text": mention.evidence_text,
                    "rank": seed_rank,
                    "relevance": relevance,
                    "metadata": {
                        "source": "statement_tkg",
                        "assertion_id": assertion.assertion_id,
                        "statement": (
                            f"{self._entity_name(entities, assertion.subject_id)} "
                            f"{assertion.predicate_id} "
                            f"{self._entity_name(entities, assertion.object_id)}"
                        ),
                        "valid_from": assertion.valid_from,
                        "valid_to": assertion.valid_to,
                        "status": assertion.status.value,
                        "stance": mention.stance.value,
                        "conflicted": assertion.assertion_id in conflicted,
                        "event_effects": [
                            item.effect_type.value
                            for item in effects_by_assertion[assertion.assertion_id]
                        ],
                    },
                }
            )
        for mention in expansion.event_mentions:
            event_id = str(mention.get("event_id") or "")
            graph_evidence.append(
                {
                    "document_id": mention.get("document_id"),
                    "chunk_id": mention.get("source_segment_id") or mention.get("mention_id"),
                    "text": str(mention.get("evidence") or ""),
                    "rank": event_rank.get(event_id, limit + 1),
                    "relevance": self._hit_score(event_hits, event_id),
                    "metadata": {
                        "source": "event_tkg",
                        "event_id": event_id,
                        "stance": mention.get("stance"),
                    },
                }
            )

        routes = {
            "vector": [
                {
                    "document_id": item.get("document_id"),
                    "chunk_id": item.get("chunk_id") or item.get("node_id"),
                    "text": str(item.get("content") or ""),
                    "rank": rank,
                    "score": float(item.get("score") or 0),
                    "metadata": {"source": "source_chunk"},
                }
                for rank, item in enumerate(chunk_hits, start=1)
            ],
            "graph": sorted(
                [item for item in graph_evidence if item["metadata"]["source"] == "statement_tkg"],
                key=(
                    (lambda item: (-item["relevance"], item["rank"]))
                    if self._dynamic
                    else (lambda item: item["rank"])
                ),
            ),
            "event": sorted(
                [item for item in graph_evidence if item["metadata"]["source"] == "event_tkg"],
                key=(
                    (lambda item: (-item["relevance"], item["rank"]))
                    if self._dynamic
                    else (lambda item: item["rank"])
                ),
            ),
        }
        scores: dict[str, float] = defaultdict(float)
        selected: dict[str, dict[str, Any]] = {}
        channels: dict[str, set[str]] = defaultdict(set)
        passages: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for channel, values in routes.items():
            seen: set[str] = set()
            for rank, item in enumerate(values, start=1):
                document_id = str(item.get("document_id") or "")
                if not document_id or document_id in seen or not item.get("text"):
                    continue
                seen.add(document_id)
                if self._dynamic:
                    route_score = self._route_score(channel, item, rank)
                    previous = scores.get(document_id)
                    if previous is None:
                        scores[document_id] = route_score
                    else:
                        # Multiple representations of one source are correlated. Let the
                        # strongest route dominate and use the others only as support.
                        scores[document_id] = max(previous, route_score) + 0.15 * min(
                            previous, route_score
                        )
                else:
                    scores[document_id] += 1.0 / (self._rrf_k + rank)
                channels[document_id].add(channel)
                passages[document_id].append(item)
                selected.setdefault(document_id, item)
        ordered = sorted(scores, key=lambda key: (scores[key], key), reverse=True)[:top_k]
        return [
            RetrievedChunk(
                chunk_id=str(selected[key].get("chunk_id") or f"tkg-{key}"),
                document_id=key,
                text="\n\n".join(
                    dict.fromkeys(str(item["text"]) for item in passages[key])
                ),
                score=scores[key],
                metadata={
                    "fusion": (
                        "dynamic_tkg_coverage"
                        if self._dynamic
                        else "unified_tkg_rrf"
                    ),
                    "channels": sorted(channels[key]),
                    "passages": passages[key],
                    "query_plan": plan.model_dump(mode="json"),
                    "tkg_seed_counts": {
                        "entity": len(entity_hits),
                        "predicate": len(predicate_hits),
                        "assertion": len(assertion_hits),
                        "event": len(event_hits),
                    },
                    "tkg_dynamic_plan": {
                        "enabled": enabled,
                        "thresholds": {
                            kind: self._effective_threshold(values, kind)
                            for kind, values in {
                                "chunk": chunk_hits,
                                "entity": entity_hits,
                                "predicate": predicate_hits,
                                "assertion": assertion_hits,
                                "event": event_hits,
                            }.items()
                        },
                    },
                },
            )
            for key in ordered
        ]

    @staticmethod
    def _ids(values: list[dict[str, Any]]) -> list[str]:
        return [str(item["node_id"]) for item in values if item.get("node_id")]

    @staticmethod
    def _rank_map(values: list[dict[str, Any]]) -> dict[str, int]:
        return {
            str(item["node_id"]): rank
            for rank, item in enumerate(values, start=1)
            if item.get("node_id")
        }

    @staticmethod
    def _score_map(values: list[dict[str, Any]]) -> dict[str, float]:
        return {
            str(item["node_id"]): float(item.get("score") or 0)
            for item in values
            if item.get("node_id")
        }

    def _enabled_routes(self, plan: Any) -> dict[str, bool]:
        if not self._dynamic:
            return {kind: True for kind in ("chunk", "entity", "predicate", "assertion", "event")}
        event_intents = {"change", "conflict", "denial", "correction", "replacement"}
        return {
            "chunk": True,
            "entity": bool(plan.entities),
            "predicate": bool(plan.relations),
            # Fact search is the stable structured backbone. Entity, predicate and
            # event indexes are optional query-linking tools around it.
            "assertion": True,
            "event": bool(
                plan.event_types
                or plan.expand_event_chain
                or set(plan.state_intents) & event_intents
            ),
        }

    def _select_hits(
        self,
        kind: str,
        values: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not self._dynamic or not values:
            return values
        caps = {
            "chunk": max(top_k, 12),
            "entity": 6,
            "predicate": 4,
            "assertion": max(top_k, 12),
            "event": max(6, top_k),
        }
        floors = {
            "chunk": 0.30,
            "entity": 0.42,
            "predicate": 0.38,
            "assertion": 0.34,
            "event": 0.34,
        }
        margins = {
            "chunk": 0.14,
            "entity": 0.10,
            "predicate": 0.10,
            "assertion": 0.13,
            "event": 0.13,
        }
        scored = [item for item in values if item.get("score") is not None]
        if not scored:
            return values[: caps[kind]]
        top = max(float(item["score"]) for item in scored)
        threshold = max(floors[kind], top - margins[kind])
        selected = [item for item in values if float(item.get("score") or -1) >= threshold]
        # Keep at least one direct semantic seed for an enabled route.
        return (selected or values[:1])[: caps[kind]]

    def _effective_threshold(
        self,
        values: list[dict[str, Any]],
        kind: str,
    ) -> float | None:
        del kind
        if not self._dynamic or not values:
            return None
        return min(float(item.get("score") or 0) for item in values)

    @staticmethod
    def _hit_score(values: list[dict[str, Any]], node_id: str) -> float:
        for item in values:
            if str(item.get("node_id") or "") == node_id:
                return max(0.0, float(item.get("score") or 0))
        return 0.0

    def _assertion_relevance(
        self,
        assertion: Any,
        *,
        entity_hits: list[dict[str, Any]],
        predicate_hits: list[dict[str, Any]],
        assertion_hits: list[dict[str, Any]],
        plan: Any,
    ) -> float:
        if not self._dynamic:
            return 0.0
        subject = self._hit_score(entity_hits, assertion.subject_id)
        obj = self._hit_score(entity_hits, assertion.object_id)
        predicate = self._hit_score(predicate_hits, assertion.predicate_id)
        direct = self._hit_score(assertion_hits, assertion.assertion_id)
        entity_coverage = int(subject > 0) + int(obj > 0)
        score = 0.42 * direct + 0.18 * max(subject, obj) + 0.16 * predicate
        score += 0.12 * entity_coverage
        if plan.start_time or plan.end_time:
            overlaps = (
                not plan.start_time
                or not assertion.valid_to
                or assertion.valid_to >= plan.start_time
            ) and (
                not plan.end_time
                or not assertion.valid_from
                or assertion.valid_from <= plan.end_time
            )
            score += 0.08 if overlaps else -0.12
        if assertion.status.value == "active":
            score += 0.04
        return max(0.0, score)

    def _route_score(self, channel: str, item: dict[str, Any], rank: int) -> float:
        if channel == "vector":
            raw = float(item.get("score") or 0)
            return 0.55 * max(0.0, raw) + 0.45 / (1 + rank)
        raw = float(item.get("relevance") or 0)
        weight = 1.0 if channel == "graph" else 0.85
        return weight * raw + 0.10 / (1 + rank)

    @staticmethod
    def _entity_name(entities: dict[str, Any], entity_id: str) -> str:
        value = entities.get(entity_id)
        return value.canonical_name if value else entity_id

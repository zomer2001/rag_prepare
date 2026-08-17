from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

from rag_graph_demo.domain.events import EventQueryRequest, EventQueryResponse
from rag_graph_demo.domain.tkg import TKGQueryRequest
from rag_graph_demo.lightrag_engine.engine import LightRAGEngine
from rag_graph_demo.retrieval.event_vector import EventVectorIndex
from rag_graph_demo.retrieval.evolink import (
    CoordinationGraphStore,
    EvoLinkCoordinator,
    EvoLinkResult,
)
from rag_graph_demo.retrieval.fusion import fuse_evidence
from rag_graph_demo.retrieval.models import (
    EvidenceCandidate,
    QueryPlan,
    RetrievalChannel,
    UnifiedRetrievalResult,
)
from rag_graph_demo.retrieval.planner import QueryPlanner
from rag_graph_demo.services.event_service import EventService
from rag_graph_demo.storage.tkg_registry import TKGRegistry

DocumentIdResolver = Callable[[str, str], str | None]
SOURCE_DOCUMENT_PATTERN = re.compile(
    r"<!--\s*rag-source\s+document_id=(?P<document_id>[\w-]+)\s+page=\d+\s*-->"
)
logger = logging.getLogger(__name__)


def default_document_id_resolver(content: str, file_path: str) -> str | None:
    marker = SOURCE_DOCUMENT_PATTERN.search(content)
    if marker:
        return marker.group("document_id")
    name = Path(file_path).name
    return name.split("__", 1)[0] if "__" in name else None


class UnifiedRetrievalOrchestrator:
    """Run vector, semantic graph, and event retrieval with one query plan."""

    def __init__(
        self,
        engine: LightRAGEngine,
        planner: QueryPlanner,
        *,
        event_service: EventService | None = None,
        event_vector_index: EventVectorIndex | None = None,
        tkg_registry: TKGRegistry | None = None,
        coordination_store: CoordinationGraphStore | None = None,
        document_id_resolver: DocumentIdResolver = default_document_id_resolver,
        strict_event_failures: bool = False,
        strict_coordination_failures: bool = False,
        max_sources_per_event: int = 2,
    ) -> None:
        self._engine = engine
        self._planner = planner
        self._event_service = event_service
        self._event_vector_index = event_vector_index
        self._tkg_registry = tkg_registry
        self._evolink = (
            EvoLinkCoordinator(coordination_store) if coordination_store else None
        )
        self._document_id_resolver = document_id_resolver
        self._strict_event_failures = strict_event_failures
        self._strict_coordination_failures = strict_coordination_failures
        self._max_sources_per_event = max(1, max_sources_per_event)

    def bind_coordination_store(self, store: CoordinationGraphStore) -> None:
        """Attach the Neo4j coordination graph after an evaluation index is built."""
        self._evolink = EvoLinkCoordinator(store)

    async def retrieve(
        self,
        question: str,
        *,
        top_k: int,
        include_events: bool,
        link_evidence: bool = False,
        candidate_multiplier: int = 3,
        channels: set[RetrievalChannel] | None = None,
    ) -> UnifiedRetrievalResult:
        started = perf_counter()
        plan = await self._planner.plan(question)
        candidate_limit = max(top_k, top_k * candidate_multiplier)
        requested_channels = set(channels or {"vector", "graph"})
        if channels is None and include_events:
            requested_channels.add("event")
        unsupported = requested_channels - {"vector", "graph", "event"}
        if unsupported:
            raise ValueError(f"Unsupported retrieval channels: {sorted(unsupported)}")
        if "vector" in requested_channels:
            vector_task = self._engine.retrieve_data(
                question,
                mode="naive",
                chunk_top_k=candidate_limit,
                enable_rerank=False,
            )
        else:
            vector_task = asyncio.sleep(0, result={"data": {}})
        if "graph" in requested_channels:
            graph_task = self._engine.retrieve_data(
                question,
                mode="hybrid",
                top_k=candidate_limit,
                chunk_top_k=candidate_limit,
                enable_rerank=False,
                high_level_keywords=plan.high_level_keywords,
                low_level_keywords=plan.low_level_keywords,
            )
        else:
            graph_task = asyncio.sleep(0, result={"data": {}})
        if "event" in requested_channels:
            event_task = self._retrieve_events_safely(question, plan, candidate_limit)
        else:
            event_task = asyncio.sleep(0, result=([], None, None))
        if "graph" in requested_channels:
            tkg_task = self._retrieve_tkg_graph_candidates(plan, candidate_limit)
        else:
            tkg_task = asyncio.sleep(0, result=[])
        vector_raw, graph_raw, event_output, tkg_candidates = await asyncio.gather(
            vector_task,
            graph_task,
            event_task,
            tkg_task,
        )
        event_candidates, event_result, event_error = event_output
        vector_candidates = self._light_candidates(vector_raw, "vector")
        graph_candidates = self._merge_graph_candidates(
            self._light_candidates(graph_raw, "graph"), tkg_candidates
        )
        evolink_result, evolink_error = await self._coordinate_safely(
            plan,
            graph_raw,
            event_candidates,
            candidate_limit,
            enabled=link_evidence,
        )
        if link_evidence:
            graph_candidates = self._merge_channel_candidates(
                graph_candidates,
                list(evolink_result.graph_candidates),
            )
            event_candidates = self._merge_channel_candidates(
                event_candidates,
                list(evolink_result.event_candidates),
            )
        channel_candidates: dict[str, list[EvidenceCandidate]] = {}
        if "vector" in requested_channels:
            channel_candidates["vector"] = vector_candidates
        if "graph" in requested_channels or evolink_result.graph_candidates:
            channel_candidates["graph"] = graph_candidates
        if "event" in requested_channels or evolink_result.event_candidates:
            channel_candidates["event"] = event_candidates
        bundles = fuse_evidence(channel_candidates, top_k=top_k)
        return UnifiedRetrievalResult(
            plan=plan,
            bundles=tuple(bundles),
            candidates={
                channel: tuple(channel_candidates.get(channel, []))
                for channel in ("vector", "graph", "event")
            },
            graph_data=dict(graph_raw.get("data") or {}),
            evidence_graph=evolink_result.evidence_graph,
            event_result=event_result,
            metadata={
                "fusion": "document_rrf",
                "rrf_k": 60,
                "candidate_limit_per_path": candidate_limit,
                "include_events": "event" in requested_channels,
                "requested_channels": sorted(requested_channels),
                "event_error": event_error,
                "evolink_enabled": link_evidence,
                "evolink_error": evolink_error,
                "evolink": evolink_result.metadata,
                "tkg_assertion_candidates": len(tkg_candidates),
                "elapsed_ms": int((perf_counter() - started) * 1_000),
            },
        )

    async def _coordinate_safely(
        self,
        plan: QueryPlan,
        graph_raw: dict[str, Any],
        event_candidates: list[EvidenceCandidate],
        candidate_limit: int,
        *,
        enabled: bool,
    ) -> tuple[EvoLinkResult, str | None]:
        if not enabled:
            return EvoLinkResult(), None
        if self._evolink is None:
            error = "Neo4j coordination graph is not configured"
            if self._strict_coordination_failures:
                raise RuntimeError(error)
            return EvoLinkResult(), error
        try:
            return (
                await self._evolink.coordinate(
                    plan,
                    entity_names=self._graph_entity_names(graph_raw),
                    event_candidates=event_candidates,
                    limit=candidate_limit,
                ),
                None,
            )
        except Exception as exc:
            if self._strict_coordination_failures:
                raise
            logger.exception("EvoLink coordination unavailable; using base routes")
            return EvoLinkResult(), f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _graph_entity_names(raw: dict[str, Any]) -> list[str]:
        values = (raw.get("data") or {}).get("entities") or []
        names: list[str] = []
        for value in values:
            if not isinstance(value, dict):
                continue
            name = next(
                (
                    str(value.get(key) or "").strip()
                    for key in ("entity_name", "name", "entity")
                    if value.get(key)
                ),
                "",
            )
            if name and name.casefold() not in {item.casefold() for item in names}:
                names.append(name)
            if len(names) >= 8:
                break
        return names

    async def _retrieve_tkg_graph_candidates(
        self,
        plan: QueryPlan,
        candidate_limit: int,
    ) -> list[EvidenceCandidate]:
        if self._tkg_registry is None or not plan.entities:
            return []
        candidates: list[EvidenceCandidate] = []
        seen_mentions: set[str] = set()
        for entity_name in plan.entities:
            result = await self._tkg_registry.query(
                TKGQueryRequest(
                    entity_name=entity_name,
                    include_disputed=True,
                    limit=candidate_limit,
                )
            )
            assertions = {item.assertion_id: item for item in result.assertions}
            entities = {item.entity_id: item for item in result.entities}
            effects = {}
            for effect in result.effects:
                effects.setdefault(effect.assertion_id, []).append(effect)
            conflicts = {
                assertion_id
                for conflict in result.conflicts
                for assertion_id in conflict.assertion_ids
            }
            for mention in result.mentions:
                if mention.mention_id in seen_mentions:
                    continue
                assertion = assertions.get(mention.assertion_id)
                if assertion is None:
                    continue
                seen_mentions.add(mention.mention_id)
                subject = entities.get(assertion.subject_id)
                obj = entities.get(assertion.object_id)
                candidates.append(
                    EvidenceCandidate(
                        evidence_id=f"graph:tkg:{mention.mention_id}",
                        channel="graph",
                        rank=len(candidates) + 1,
                        document_id=mention.document_id,
                        chunk_id=mention.source_segment_id,
                        text=mention.evidence_text,
                        score=assertion.confidence,
                        metadata={
                            "source": "statement_tkg",
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
                                for item in effects.get(assertion.assertion_id, [])
                            ],
                            "conflicted": assertion.assertion_id in conflicts,
                        },
                    )
                )
                if len(candidates) >= candidate_limit:
                    return candidates
        return candidates

    @staticmethod
    def _merge_graph_candidates(
        base: list[EvidenceCandidate],
        tkg: list[EvidenceCandidate],
    ) -> list[EvidenceCandidate]:
        merged: list[EvidenceCandidate] = []
        seen: set[tuple[str | None, str]] = set()
        for item in [*tkg, *base]:
            key = (item.document_id, item.text.strip())
            if key in seen:
                continue
            seen.add(key)
            merged.append(
                EvidenceCandidate(
                    evidence_id=item.evidence_id,
                    channel=item.channel,
                    rank=len(merged) + 1,
                    document_id=item.document_id,
                    chunk_id=item.chunk_id,
                    text=item.text,
                    file_path=item.file_path,
                    score=item.score,
                    metadata=item.metadata,
                )
            )
        return merged

    @staticmethod
    def _merge_channel_candidates(
        *groups: list[EvidenceCandidate],
    ) -> list[EvidenceCandidate]:
        merged: list[EvidenceCandidate] = []
        seen: set[tuple[str | None, str]] = set()
        for item in (value for group in groups for value in group):
            key = (item.document_id, item.text.strip())
            if key in seen:
                continue
            seen.add(key)
            merged.append(
                EvidenceCandidate(
                    evidence_id=item.evidence_id,
                    channel=item.channel,
                    rank=len(merged) + 1,
                    document_id=item.document_id,
                    chunk_id=item.chunk_id,
                    text=item.text,
                    file_path=item.file_path,
                    score=item.score,
                    metadata=item.metadata,
                )
            )
        return merged

    async def _retrieve_events_safely(
        self,
        question: str,
        plan: QueryPlan,
        candidate_limit: int,
    ) -> tuple[list[EvidenceCandidate], EventQueryResponse | None, str | None]:
        try:
            candidates, result = await self._retrieve_events(
                question,
                plan,
                candidate_limit,
            )
            return candidates, result, None
        except Exception as exc:
            if self._strict_event_failures:
                raise
            logger.exception("Event pathway unavailable; continuing with vector and graph")
            return [], None, f"{type(exc).__name__}: {exc}"

    def _light_candidates(
        self,
        raw: dict[str, Any],
        channel: RetrievalChannel,
    ) -> list[EvidenceCandidate]:
        data = raw.get("data") or {}
        result: list[EvidenceCandidate] = []
        for rank, value in enumerate(data.get("chunks") or [], start=1):
            content = str(value.get("content") or "")
            file_path = str(value.get("file_path") or "")
            chunk_id = str(value.get("chunk_id") or f"{channel}-{rank}")
            result.append(
                EvidenceCandidate(
                    evidence_id=f"{channel}:{chunk_id}",
                    channel=channel,
                    rank=rank,
                    document_id=self._document_id_resolver(content, file_path),
                    chunk_id=chunk_id,
                    text=content,
                    file_path=file_path or None,
                    metadata={"reference_id": value.get("reference_id")},
                )
            )
        return result

    async def _retrieve_events(
        self,
        question: str,
        plan: QueryPlan,
        candidate_limit: int,
    ) -> tuple[list[EvidenceCandidate], EventQueryResponse | None]:
        if self._event_service is None or self._event_vector_index is None:
            return [], None
        events = await self._event_service.list_events()
        vector_hits = await self._event_vector_index.search(
            question,
            events,
            limit=candidate_limit,
        )
        vector_rank = {
            event_id: rank for rank, (event_id, _) in enumerate(vector_hits, start=1)
        }
        vector_score = dict(vector_hits)
        event_result = await self._event_service.search(
            EventQueryRequest(
                question=question,
                entities=plan.entities,
                event_types=plan.event_types,
                start_time=plan.start_time,
                end_time=plan.end_time,
                state_intents=plan.state_intents,
                expand_event_chain=plan.expand_event_chain,
                limit=candidate_limit,
            ),
            candidate_event_ids=set(vector_rank),
        )
        structured_rank = {item.event_id: item.rank for item in event_result.rankings}
        ranking_details = {item.event_id: item for item in event_result.rankings}
        event_details = {item.event_id: item for item in event_result.events}
        event_ids = {item.event_id for item in event_result.events}
        combined_scores = {
            event_id: (
                1.0 / (60 + vector_rank.get(event_id, candidate_limit + 1))
                + 1.0 / (60 + structured_rank.get(event_id, candidate_limit + 1))
            )
            for event_id in event_ids
        }
        ordered_event_ids = sorted(
            event_ids,
            key=lambda event_id: (combined_scores[event_id], event_id),
            reverse=True,
        )
        combined_rank = {
            event_id: rank for rank, event_id in enumerate(ordered_event_ids, start=1)
        }
        ranked_mentions = sorted(
            event_result.mentions,
            key=lambda item: (
                combined_rank.get(item.event_id, candidate_limit + 1),
                -item.confidence,
                item.mention_id,
            ),
        )
        mentions = []
        sources_per_event: dict[str, int] = {}
        seen_event_documents: set[tuple[str, str]] = set()
        for mention in ranked_mentions:
            key = (mention.event_id, mention.document_id)
            if key in seen_event_documents:
                continue
            if sources_per_event.get(mention.event_id, 0) >= self._max_sources_per_event:
                continue
            seen_event_documents.add(key)
            sources_per_event[mention.event_id] = (
                sources_per_event.get(mention.event_id, 0) + 1
            )
            mentions.append(mention)
        return (
            [
                EvidenceCandidate(
                    evidence_id=f"event:{mention.mention_id}",
                    channel="event",
                    rank=evidence_rank,
                    document_id=mention.document_id,
                    chunk_id=mention.source_segment_id,
                    text=mention.evidence,
                    score=combined_scores.get(mention.event_id),
                    metadata={
                        "event_id": mention.event_id,
                        "event_type": event_details[mention.event_id].event_type,
                        "event_summary": event_details[mention.event_id].summary,
                        "event_time": (
                            event_details[mention.event_id].start_time
                            or event_details[mention.event_id].end_time
                        ),
                        "event_status": event_details[mention.event_id].status.value,
                        "event_actors": event_details[mention.event_id].actors,
                        "event_objects": event_details[mention.event_id].objects,
                        "event_vector_score": vector_score.get(mention.event_id),
                        "event_vector_rank": vector_rank.get(mention.event_id),
                        "event_structured_rank": structured_rank.get(mention.event_id),
                        "event_combined_rank": combined_rank.get(mention.event_id),
                        "expanded": bool(
                            ranking_details.get(mention.event_id)
                            and ranking_details[mention.event_id].expanded
                        ),
                        "source_segment_start": mention.source_segment_start,
                        "source_segment_end": mention.source_segment_end,
                        "evidence_start": mention.evidence_start,
                        "evidence_end": mention.evidence_end,
                    },
                )
                for evidence_rank, mention in enumerate(mentions, start=1)
            ],
            event_result,
        )

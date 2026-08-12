from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pytest

from rag_graph_demo.domain.events import (
    EventMention,
    EventQueryResponse,
    EventRanking,
    EventRecord,
)
from rag_graph_demo.domain.models import DocumentRecord, QueryRequest
from rag_graph_demo.retrieval.event_vector import EventVectorIndex
from rag_graph_demo.retrieval.fusion import fuse_evidence
from rag_graph_demo.retrieval.models import (
    EvidenceBundle,
    EvidenceCandidate,
    QueryPlan,
    UnifiedRetrievalResult,
)
from rag_graph_demo.retrieval.orchestrator import UnifiedRetrievalOrchestrator
from rag_graph_demo.retrieval.planner import (
    LEGACY_QUERY_PLANNER_PROFILE_VERSION,
    QueryPlanner,
)
from rag_graph_demo.services.query_service import QueryService
from rag_graph_demo.storage.document_registry import DocumentRegistry


@pytest.mark.asyncio
async def test_query_planner_calls_llm_once_and_repairs_empty_keywords() -> None:
    calls = 0

    async def complete(_prompt: str, _system_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return """{"low_level_keywords": [], "high_level_keywords": [],
        "entities": ["Anduril"], "expand_event_chain": true}"""

    planner = QueryPlanner(complete)
    first = await planner.plan("Anduril 经历了哪些变化？")
    second = await planner.plan("Anduril 经历了哪些变化？")

    assert first == second
    assert calls == 1
    assert first.low_level_keywords == ["Anduril 经历了哪些变化？"]
    assert first.high_level_keywords == ["Anduril 经历了哪些变化？"]


@pytest.mark.asyncio
async def test_query_planner_cache_is_shared_across_process_instances(tmp_path) -> None:
    calls = 0

    async def complete(_prompt: str, _system_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return '{"low_level_keywords":["Anduril"]}'

    cache_path = tmp_path / "plans.json"
    first = QueryPlanner(complete, cache_path=cache_path)
    assert (await first.plan("Anduril 是什么？")).low_level_keywords == ["Anduril"]
    second = QueryPlanner(complete, cache_path=cache_path)
    assert (await second.plan("Anduril 是什么？")).low_level_keywords == ["Anduril"]
    assert calls == 1


@pytest.mark.asyncio
async def test_query_planner_migrates_legacy_plan_to_safe_soft_hints(tmp_path) -> None:
    question = "Did reports agree about Google?"
    legacy_key = QueryPlanner._cache_key(
        question,
        profile=LEGACY_QUERY_PLANNER_PROFILE_VERSION,
    )
    cache_path = tmp_path / "plans.json"
    cache_path.write_text(
        json.dumps(
            {
                legacy_key: {
                    "entities": ["Google", "The Verge", "TechCrunch", "The Age"],
                    "start_time": "2023-01-01",
                    "end_time": "2023-12-31",
                    "state_intents": ["consistency", "discrepancy"],
                }
            }
        ),
        encoding="utf-8",
    )

    async def should_not_run(_prompt: str, _system: str) -> str:
        raise AssertionError("legacy cache should migrate without another LLM call")

    plan = await QueryPlanner(should_not_run, cache_path=cache_path).plan(question)

    assert plan.entities == ["Google", "The Verge", "TechCrunch"]
    assert plan.start_time is None
    assert plan.end_time is None
    assert plan.state_intents == ["conflict"]


def test_three_path_fusion_preserves_one_traceable_passage_per_path() -> None:
    candidates = {
        channel: [
            EvidenceCandidate(
                evidence_id=f"{channel}:1",
                channel=channel,
                rank=1,
                document_id="doc-a",
                chunk_id=f"{channel}-chunk",
                text=f"{channel} evidence",
            )
        ]
        for channel in ("vector", "graph", "event")
    }

    result = fuse_evidence(candidates, top_k=1)

    assert result[0].channels == ("vector", "graph", "event")
    assert {item.channel for item in result[0].passages} == {
        "vector",
        "graph",
        "event",
    }
    assert result[0].score == pytest.approx(3 / 61)


@pytest.mark.asyncio
async def test_event_vector_index_embeds_canonical_events_and_reuses_cache(tmp_path) -> None:
    contexts: list[str] = []

    async def embed(texts: list[str], context: str) -> np.ndarray:
        contexts.append(context)
        return np.asarray(
            [
                [1.0, 0.0] if "合作" in text else [0.0, 1.0]
                for text in texts
            ],
            dtype=np.float32,
        )

    events = [
        EventRecord(
            event_id="evt-cooperate",
            canonical_key="cooperate",
            event_type="合作",
            summary="Anduril 与 Meta 合作。",
            actors=["Anduril", "Meta"],
            start_time="2025",
            updated_at=datetime(2025, 1, 1, tzinfo=UTC),
        ),
        EventRecord(
            event_id="evt-release",
            canonical_key="release",
            event_type="发布",
            summary="发布产品。",
            actors=["Anduril"],
            start_time="2024",
            updated_at=datetime(2025, 1, 1, tzinfo=UTC),
        ),
    ]
    index = EventVectorIndex(
        tmp_path / "events.npz",
        embed,
        embedding_identity="test/2d",
    )

    first = await index.search("合作", events, limit=2)
    second = await index.search("合作", events, limit=2)

    assert first[0][0] == second[0][0] == "evt-cooperate"
    assert contexts == ["document", "query", "query"]


@pytest.mark.asyncio
async def test_event_vector_index_batches_api_embedding_requests(tmp_path) -> None:
    document_batch_sizes: list[int] = []

    async def embed(texts: list[str], context: str) -> np.ndarray:
        if context == "document":
            document_batch_sizes.append(len(texts))
        return np.asarray([[1.0, float(index)] for index, _ in enumerate(texts)])

    events = [
        EventRecord(
            event_id=f"event-{index}",
            canonical_key=f"key-{index}",
            event_type="测试",
            summary=f"事件 {index}",
            updated_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        for index in range(5)
    ]
    index = EventVectorIndex(
        tmp_path / "batched-events.npz",
        embed,
        embedding_identity="test/2d",
        batch_size=2,
        max_concurrency=2,
    )

    await index.search("测试", events, limit=2)

    assert sorted(document_batch_sizes) == [1, 2, 2]


class FakePlanner:
    async def plan(self, _question: str) -> QueryPlan:
        return QueryPlan(
            low_level_keywords=["Anduril", "Meta"],
            high_level_keywords=["合作"],
            entities=["Anduril", "Meta"],
            event_types=["合作"],
            start_time="2025",
            end_time="2025-12-31",
            expand_event_chain=True,
        )


class FakeRetrievalEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], list[str]]] = []

    async def retrieve_data(
        self,
        _question: str,
        *,
        mode: str,
        high_level_keywords=None,
        low_level_keywords=None,
        **_kwargs,
    ):
        self.calls.append(
            (mode, high_level_keywords or [], low_level_keywords or [])
        )
        if mode == "naive":
            chunks = [
                {
                    "chunk_id": "dense-a",
                    "content": "dense evidence",
                    "file_path": "doc-a__dense.md",
                }
            ]
            return {"status": "success", "data": {"chunks": chunks}}
        chunks = [
            {
                "chunk_id": "graph-a",
                "content": "graph evidence",
                "file_path": "doc-a__graph.md",
            },
            {
                "chunk_id": "graph-b",
                "content": "graph B",
                "file_path": "doc-b__graph.md",
            },
        ]
        return {
            "status": "success",
            "data": {"chunks": chunks, "entities": [{"entity_name": "Anduril"}]},
        }


class FakeEventVector:
    async def search(self, _question, _events, *, limit):
        assert limit == 6
        return [("evt-1", 0.9)]


class FakeEventService:
    async def list_events(self):
        return [
            EventRecord(
                event_id="evt-1",
                canonical_key="evt-1",
                event_type="合作",
                summary="Anduril 与 Meta 合作。",
            )
        ]

    async def search(self, request, *, candidate_event_ids):
        assert request.entities == ["Anduril", "Meta"]
        assert request.expand_event_chain is True
        assert candidate_event_ids == {"evt-1"}
        return EventQueryResponse(
            events=await self.list_events(),
            rankings=[
                EventRanking(
                    event_id="evt-1",
                    rank=1,
                    score=1,
                    lexical_score=1,
                    entity_score=1,
                    action_score=1,
                    temporal_score=1,
                    state_score=0,
                    confidence_score=0.8,
                    source_score=0.5,
                )
            ],
            mentions=[
                EventMention(
                    mention_id="men-1",
                    event_id="evt-1",
                    document_id="doc-a",
                    document_title="A",
                    evidence="event evidence",
                    source_segment_id="doc-a::event-part-1",
                    confidence=0.8,
                )
            ],
        )


class FakeMultiSourceEventService(FakeEventService):
    async def search(self, request, *, candidate_event_ids):
        result = await super().search(
            request,
            candidate_event_ids=candidate_event_ids,
        )
        result.mentions = [
            EventMention(
                mention_id=f"men-{index}",
                event_id="evt-1",
                document_id=f"doc-{index}",
                document_title=f"Document {index}",
                evidence=f"event evidence {index}",
                confidence=0.9 - index * 0.1,
            )
            for index in range(3)
        ]
        return result


@pytest.mark.asyncio
async def test_orchestrator_runs_three_paths_with_shared_plan() -> None:
    engine = FakeRetrievalEngine()
    orchestrator = UnifiedRetrievalOrchestrator(
        engine,  # type: ignore[arg-type]
        FakePlanner(),  # type: ignore[arg-type]
        event_service=FakeEventService(),  # type: ignore[arg-type]
        event_vector_index=FakeEventVector(),  # type: ignore[arg-type]
    )

    result = await orchestrator.retrieve(
        "2025 年 Anduril 与 Meta 如何合作？",
        top_k=2,
        include_events=True,
    )

    assert engine.calls == [
        ("naive", [], []),
        ("hybrid", ["合作"], ["Anduril", "Meta"]),
    ]
    assert result.bundles[0].document_id == "doc-a"
    assert result.bundles[0].channels == ("vector", "graph", "event")
    assert result.candidates["event"][0].chunk_id == "doc-a::event-part-1"
    assert result.graph_data["entities"][0]["entity_name"] == "Anduril"


@pytest.mark.asyncio
async def test_event_path_caps_sources_from_one_event() -> None:
    orchestrator = UnifiedRetrievalOrchestrator(
        FakeRetrievalEngine(),  # type: ignore[arg-type]
        FakePlanner(),  # type: ignore[arg-type]
        event_service=FakeMultiSourceEventService(),  # type: ignore[arg-type]
        event_vector_index=FakeEventVector(),  # type: ignore[arg-type]
        max_sources_per_event=2,
    )

    result = await orchestrator.retrieve(
        "2025 年 Anduril 与 Meta 如何合作？",
        top_k=2,
        include_events=True,
    )

    assert len(result.candidates["event"]) == 2
    assert [item.rank for item in result.candidates["event"]] == [1, 2]


class FakeAnswerEngine:
    def __init__(self) -> None:
        self.prompt = ""

    async def complete_query(self, prompt, *, system_prompt, history_messages):
        assert "只能依据" in system_prompt
        assert history_messages == []
        self.prompt = prompt
        return "Anduril 与 Meta 在 2025 年合作。[1]"


class FakeUnifiedOrchestrator:
    async def retrieve(self, _question, *, top_k, include_events):
        assert top_k == 1
        assert include_events is True
        candidate = EvidenceCandidate(
            evidence_id="event:men-1",
            channel="event",
            rank=1,
            document_id="doc-a",
            chunk_id="doc-a::event-part-1",
            text="Anduril 与 Meta 在 2025 年合作。",
        )
        return UnifiedRetrievalResult(
            plan=QueryPlan(entities=["Anduril", "Meta"]),
            bundles=(
                EvidenceBundle(
                    document_id="doc-a",
                    score=1 / 61,
                    channels=("event",),
                    passages=(candidate,),
                ),
            ),
            candidates={"vector": (), "graph": (), "event": (candidate,)},
        )


@pytest.mark.asyncio
async def test_online_answer_uses_the_same_unified_top_k_evidence(tmp_path) -> None:
    documents = DocumentRegistry(tmp_path / "documents.json")
    await documents.upsert(
        DocumentRecord(
            document_id="doc-a",
            filename="a.md",
            title="合作公告",
            suffix=".md",
            size_bytes=10,
            checksum="checksum",
            raw_path=str(tmp_path / "a.md"),
        )
    )
    engine = FakeAnswerEngine()
    service = QueryService(
        engine,  # type: ignore[arg-type]
        documents=documents,
        orchestrator=FakeUnifiedOrchestrator(),  # type: ignore[arg-type]
    )

    response = await service.query(
        QueryRequest(
            question="2025 年 Anduril 与 Meta 如何合作？",
            mode="mix",
            chunk_top_k=1,
        )
    )

    assert "event｜chunk=doc-a::event-part-1" in engine.prompt
    assert response.citations[0].document_id == "doc-a"
    assert response.chunks[0]["channels"] == ["event"]

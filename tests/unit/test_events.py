from __future__ import annotations

import pytest

from rag_graph_demo.domain.events import EventCandidate, EventQueryRequest, EventStatus
from rag_graph_demo.domain.models import DocumentRecord, QueryRequest, QueryResponse
from rag_graph_demo.services.event_service import EventService, LLMEventExtractor
from rag_graph_demo.services.query_service import QueryService
from rag_graph_demo.storage.document_registry import DocumentRegistry
from rag_graph_demo.storage.event_registry import EventRegistry


def candidate(
    *,
    summary: str,
    start_time: str,
    event_type: str = "产品发布",
    stance: str = "affirmed",
) -> EventCandidate:
    return EventCandidate(
        event_type=event_type,
        summary=summary,
        action="发布",
        actors=["Anduril Industries"],
        objects=["Lattice"],
        start_time=start_time,
        time_precision="day" if len(start_time) == 10 else "month",
        evidence=summary,
        stance=stance,
        confidence=0.8,
    )


@pytest.mark.asyncio
async def test_event_registry_merges_sources_and_retains_conflict(tmp_path) -> None:
    registry = EventRegistry(tmp_path / "events.json")
    await registry.replace_document_events(
        "doc-1", "官方资料", [candidate(summary="Anduril 发布 Lattice。", start_time="2025-03")]
    )
    await registry.replace_document_events(
        "doc-2",
        "新闻报道",
        [candidate(summary="Lattice 于 3 月正式亮相。", start_time="2025-03-12")],
    )
    events = await registry.list_events()

    assert len(events) == 1
    assert events[0].status == EventStatus.MULTI_SOURCE
    assert set(events[0].source_document_ids) == {"doc-1", "doc-2"}

    await registry.replace_document_events(
        "doc-3",
        "否认报道",
        [
            candidate(
                summary="公司否认已经发布 Lattice。",
                start_time="2025-03",
                stance="denied",
            )
        ],
    )
    events, _, _, conflicts = await registry.snapshot()

    assert events[0].status == EventStatus.DISPUTED
    assert len(conflicts) == 1


@pytest.mark.asyncio
async def test_event_query_filters_time_and_builds_timeline_relations(tmp_path) -> None:
    registry = EventRegistry(tmp_path / "events.json")
    service = EventService(registry)
    await registry.replace_document_events(
        "doc-1", "发布", [candidate(summary="发布 Lattice。", start_time="2024-01")]
    )
    await registry.replace_document_events(
        "doc-2",
        "部署",
        [
            candidate(
                summary="部署 Lattice。",
                start_time="2025-05",
                event_type="产品部署",
            )
        ],
    )

    result = await service.search(
        EventQueryRequest(
            question="2024 到 2025 年 Lattice 有哪些变化？",
            entity="Lattice",
        )
    )

    assert [item.start_time for item in result.events] == ["2025-05", "2024-01"]
    assert result.inferred_start_time == "2024"
    assert result.inferred_end_time == "2025-12-31"
    assert len(result.relations) == 1
    assert result.relations[0].relation_type == "precedes"
    context, _ = await service.build_query_context("2024 到 2025 年 Lattice 有哪些变化？")
    assert "事件链（只表达时间或共享实体，不自动表示因果）" in context
    assert "--先于(" in context
    assert "Lattice)-->" in context


@pytest.mark.asyncio
async def test_planner_entities_and_event_types_are_soft_ranking_hints(tmp_path) -> None:
    registry = EventRegistry(tmp_path / "events.json")
    service = EventService(registry)
    await registry.replace_document_events(
        "doc-1",
        "发布",
        [candidate(summary="Anduril 发布 Lattice。", start_time="2025")],
    )

    result = await service.search(
        EventQueryRequest(
            question="What happened to Lattice?",
            entities=["Anduril Industries", "TechCrunch", "The Verge"],
            event_types=["contract"],
        )
    )
    hard_filtered = await service.search(
        EventQueryRequest(
            question="What happened to Lattice?",
            entity="Meta",
        )
    )

    assert len(result.events) == 1
    assert hard_filtered.events == []


@pytest.mark.asyncio
async def test_unknown_event_time_survives_planner_time_hint(tmp_path) -> None:
    registry = EventRegistry(tmp_path / "events.json")
    service = EventService(registry)
    unknown_time = candidate(
        summary="Anduril 发布 Lattice。",
        start_time="2025",
    ).model_copy(update={"start_time": None})
    await registry.replace_document_events("doc-1", "发布", [unknown_time])

    result = await service.search(
        EventQueryRequest(
            question="What happened to Lattice in 2025?",
            start_time="2025",
            end_time="2025-12-31",
        )
    )

    assert len(result.events) == 1
    assert result.rankings[0].temporal_score == 0


@pytest.mark.asyncio
async def test_event_search_keeps_relevance_rank_and_timeline_can_sort_by_time(
    tmp_path,
) -> None:
    registry = EventRegistry(tmp_path / "events.json")
    service = EventService(registry)
    await registry.replace_document_events(
        "doc-old",
        "早期发布",
        [candidate(summary="Anduril 发布 Lattice。", start_time="2024")],
    )
    await registry.replace_document_events(
        "doc-new",
        "雷达升级测试",
        [
            candidate(
                summary="Anduril 完成 Lattice 雷达升级测试并公布结果。",
                start_time="2025",
            )
        ],
    )
    question = "Lattice 雷达升级测试结果"

    ranked = await service.search(EventQueryRequest(question=question))
    timeline = await service.search(EventQueryRequest(question=question, sort_by="time"))
    context, _ = await service.build_query_context(question)

    assert [item.start_time for item in ranked.events] == ["2025", "2024"]
    assert [item.event_id for item in ranked.rankings] == [item.event_id for item in ranked.events]
    assert ranked.rankings[0].score > ranked.rankings[1].score
    assert ranked.rankings[0].action_score >= ranked.rankings[1].action_score
    assert [item.start_time for item in timeline.events] == ["2024", "2025"]
    assert [item.rank for item in timeline.rankings] == [1, 2]
    assert context.index("Anduril 发布 Lattice。") < context.index(
        "Anduril 完成 Lattice 雷达升级测试并公布结果。"
    )


@pytest.mark.asyncio
async def test_latest_event_intent_exposes_recency_as_a_ranking_signal(tmp_path) -> None:
    registry = EventRegistry(tmp_path / "events.json")
    service = EventService(registry)
    for year in ("2024", "2025"):
        await registry.replace_document_events(
            f"doc-{year}",
            f"Lattice {year}",
            [candidate(summary="Anduril 更新 Lattice 状态。", start_time=year)],
        )

    result = await service.search(EventQueryRequest(question="目前 Lattice 最新状态是什么？"))

    assert [item.start_time for item in result.events] == ["2025", "2024"]
    assert [item.temporal_score for item in result.rankings] == [1.0, 0.0]


@pytest.mark.asyncio
async def test_event_timeline_uses_adjacent_edges_not_complete_graph(tmp_path) -> None:
    registry = EventRegistry(tmp_path / "events.json")
    for index, year in enumerate(("2023", "2024", "2025"), start=1):
        await registry.replace_document_events(
            f"doc-{index}",
            f"事件 {index}",
            [candidate(summary=f"第 {index} 个事件。", start_time=year)],
        )

    _, _, relations, _ = await registry.snapshot()

    assert len(relations) == 2


@pytest.mark.asyncio
async def test_event_vector_seed_can_expand_one_timeline_hop(tmp_path) -> None:
    registry = EventRegistry(tmp_path / "events.json")
    service = EventService(registry)
    await registry.replace_document_events(
        "doc-old",
        "旧事件",
        [candidate(summary="Anduril 发布 Lattice。", start_time="2024")],
    )
    await registry.replace_document_events(
        "doc-new",
        "新事件",
        [candidate(summary="Anduril 更新 Lattice。", start_time="2025")],
    )
    events = await registry.list_events()
    seed = next(item for item in events if item.start_time == "2025")

    result = await service.search(
        EventQueryRequest(
            question="Lattice 经历了哪些变化？",
            expand_event_chain=True,
        ),
        candidate_event_ids={seed.event_id},
    )

    assert {item.start_time for item in result.events} == {"2024", "2025"}
    expanded = {item.event_id for item in result.rankings if item.expanded}
    assert expanded == {next(item.event_id for item in events if item.start_time == "2024")}


@pytest.mark.asyncio
async def test_question_entity_excludes_unrelated_events_in_same_year(tmp_path) -> None:
    registry = EventRegistry(tmp_path / "events.json")
    service = EventService(registry)
    await registry.replace_document_events(
        "doc-anduril",
        "Anduril 合同",
        [candidate(summary="Anduril 获得合同。", start_time="2025")],
    )
    await registry.replace_document_events(
        "doc-paper",
        "无关论文",
        [
            EventCandidate(
                event_type="论文发表",
                summary="OpenAlex 收录一篇论文。",
                actors=["OpenAlex"],
                objects=["机器学习论文"],
                start_time="2025",
                time_precision="year",
                evidence="OpenAlex 收录一篇论文。",
            )
        ],
    )

    result = await service.search(EventQueryRequest(question="2025 年 Anduril 有哪些关键合同？"))

    assert [item.summary for item in result.events] == ["Anduril 获得合同。"]


@pytest.mark.asyncio
async def test_derived_rebuild_merges_contained_object_names(tmp_path) -> None:
    registry = EventRegistry(tmp_path / "events.json")
    first = candidate(summary="联合开发集成 XR 产品。", start_time="2025-05-29")
    first.objects = ["集成XR产品"]
    first.actors = ["Anduril", "Meta"]
    second = candidate(summary="联合开发 XR 产品。", start_time="2025-05-29")
    second.objects = ["XR产品"]
    second.actors = ["Anduril", "Meta"]
    await registry.replace_document_events("doc-1", "英文", [first])
    await registry.replace_document_events("doc-2", "中文", [second])

    stats = await registry.rebuild_derived()
    events, mentions, _, _ = await registry.snapshot()

    assert stats["merged_events"] == 0  # online merge already caught the containment
    assert len(events) == 1
    assert len(mentions) == 2
    assert events[0].status == EventStatus.MULTI_SOURCE


class FakeExtractionEngine:
    async def complete_extraction(self, prompt: str, system_prompt: str) -> str:
        assert "文档标题：测试" in prompt
        assert "不得" in system_prompt
        return """```json
        {"events": [{
          "event_type": "合同授予",
          "summary": "美国政府向 Anduril 授予合同。",
          "actors": ["美国政府"],
          "objects": ["Anduril"],
          "start_time": "2026-02",
          "time_precision": "month",
          "evidence": "美国政府向 Anduril 授予合同。",
          "confidence": 0.9
        }]}
        ```"""


@pytest.mark.asyncio
async def test_llm_event_extractor_accepts_fenced_json() -> None:
    extractor = LLMEventExtractor(FakeExtractionEngine())

    events = await extractor.extract("测试", "美国政府向 Anduril 授予合同。")

    assert len(events) == 1
    assert events[0].event_type == "合同授予"
    assert events[0].objects == ["Anduril"]
    assert events[0].source_segment_id == "event-part-1"
    assert events[0].source_segment_start == 0
    assert events[0].source_segment_end is not None
    assert events[0].evidence_start is not None


class MissingEvidenceEngine:
    async def complete_extraction(self, prompt: str, system_prompt: str) -> str:
        return """{"events": [{
          "event_type": "合同授予",
          "summary": "Anduril 获得合同",
          "actors": ["Anduril"],
          "objects": ["合同"],
          "confidence": 0.8
        }]}"""


@pytest.mark.asyncio
async def test_llm_event_extractor_repairs_missing_evidence_from_source() -> None:
    extractor = LLMEventExtractor(MissingEvidenceEngine())
    source = "背景说明。\n\n美国政府向 Anduril 授予合同。"

    events = await extractor.extract("测试", source)

    assert events[0].evidence == "美国政府向 Anduril 授予合同。"


class UnsupportedTimePrecisionEngine:
    async def complete_extraction(self, prompt: str, system_prompt: str) -> str:
        del prompt, system_prompt
        return """{"events": [{
          "event_type": "发布",
          "summary": "公司本周发布产品。",
          "actors": ["公司"],
          "objects": ["产品"],
          "start_time": "2026-08-03",
          "time_precision": "week",
          "evidence": "公司本周发布产品。"
        }]}"""


@pytest.mark.asyncio
async def test_llm_event_extractor_downgrades_unsupported_time_precision() -> None:
    extractor = LLMEventExtractor(UnsupportedTimePrecisionEngine())

    events = await extractor.extract("测试", "公司本周发布产品。")

    assert len(events) == 1
    assert events[0].time_precision.value == "unknown"


class RepairableJsonEngine:
    async def complete_extraction(self, prompt: str, system_prompt: str) -> str:
        del prompt, system_prompt
        return """{"events": [{
          "event_type": "发布",
          "summary": "公司发布产品。",
          "actors": ["公司"],
          "objects": ["产品"]
          "evidence": "公司发布产品。"
        }]}"""


@pytest.mark.asyncio
async def test_llm_event_extractor_repairs_invalid_json_syntax() -> None:
    extractor = LLMEventExtractor(RepairableJsonEngine())

    events = await extractor.extract("测试", "公司发布产品。")

    assert len(events) == 1
    assert events[0].summary == "公司发布产品。"


class NullableFactEffectEngine:
    async def complete_extraction(self, prompt: str, system_prompt: str) -> str:
        del prompt, system_prompt
        return """{"events": [{
          "event_type": "发布",
          "summary": "公司发布产品。",
          "actors": ["公司"],
          "objects": ["产品"],
          "evidence": "公司发布产品。",
          "fact_effects": [{
            "subject": "公司",
            "predicate": "发布",
            "object": null,
            "effect_type": "creates"
          }]
        }]}"""


@pytest.mark.asyncio
async def test_llm_event_extractor_drops_invalid_effect_not_event() -> None:
    extractor = LLMEventExtractor(NullableFactEffectEngine())

    events = await extractor.extract("测试", "公司发布产品。")

    assert len(events) == 1
    assert events[0].fact_effects == []


class RetryMalformedResponseEngine:
    def __init__(self) -> None:
        self.calls = 0

    async def complete_extraction(self, prompt: str, system_prompt: str) -> str:
        del prompt, system_prompt
        self.calls += 1
        if self.calls == 1:
            return "not JSON at all"
        return """{"events": [{
          "event_type": "发布",
          "summary": "公司发布产品。",
          "actors": ["公司"],
          "objects": ["产品"],
          "evidence": "公司发布产品。"
        }]}"""


@pytest.mark.asyncio
async def test_llm_event_extractor_retries_unrepairable_response() -> None:
    engine = RetryMalformedResponseEngine()
    extractor = LLMEventExtractor(engine)

    events = await extractor.extract("测试", "公司发布产品。")

    assert len(events) == 1
    assert engine.calls == 2


class MinutePrecisionEngine:
    async def complete_extraction(self, prompt: str, system_prompt: str) -> str:
        del prompt, system_prompt
        return """{"events": [{
          "event_type": "发布",
          "summary": "公司发布产品。",
          "actors": ["公司"],
          "objects": ["产品"],
          "start_time": "2026-09-03T16:47",
          "time_precision": "minute",
          "evidence": "公司发布产品。"
        }]}"""


@pytest.mark.asyncio
async def test_llm_event_extractor_maps_minute_precision_to_day() -> None:
    extractor = LLMEventExtractor(MinutePrecisionEngine())

    events = await extractor.extract("测试", "公司发布产品。")

    assert events[0].time_precision.value == "day"


class ManyEventsEngine:
    async def complete_extraction(self, prompt: str, system_prompt: str) -> str:
        del prompt, system_prompt
        return """{"events": [
          {"event_type":"发布","summary":"事件一","actors":["A"],"objects":["X"],"evidence":"事件一"},
          {"event_type":"发布","summary":"事件二","actors":["A"],"objects":["Y"],"evidence":"事件二"},
          {"event_type":"发布","summary":"事件三","actors":["A"],"objects":["Z"],"evidence":"事件三"}
        ]}"""


@pytest.mark.asyncio
async def test_event_extractor_enforces_per_part_importance_budget() -> None:
    extractor = LLMEventExtractor(ManyEventsEngine(), max_events_per_part=2)

    events = await extractor.extract("测试", "事件一。事件二。事件三。")

    assert [item.summary for item in events] == ["事件一", "事件二"]


class FakeQueryEngine:
    def __init__(self) -> None:
        self.user_prompt = None

    async def query(self, request, *, user_prompt=None):
        self.user_prompt = user_prompt
        return QueryResponse(
            status="success",
            message="",
            mode=request.mode,
            answer="answer",
            elapsed_ms=1,
        )


@pytest.mark.asyncio
async def test_query_service_injects_event_context(tmp_path) -> None:
    registry = EventRegistry(tmp_path / "events.json")
    await registry.replace_document_events(
        "doc-1", "发布", [candidate(summary="发布 Lattice。", start_time="2025-03")]
    )
    engine = FakeQueryEngine()
    documents = DocumentRegistry(tmp_path / "documents.json")
    await documents.upsert(
        DocumentRecord(
            document_id="doc-1",
            filename="release.md",
            title="发布",
            suffix=".md",
            size_bytes=10,
            checksum="checksum",
            raw_path=str(tmp_path / "release.md"),
        )
    )
    service = QueryService(engine, EventService(registry), documents)

    response = await service.query(QueryRequest(question="Lattice 何时发布？"))

    assert "事件索引" in engine.user_prompt
    assert "[E1]" in engine.user_prompt
    assert len(response.events) == 1
    assert response.citations[0].reference_id == "E1"
    assert response.citations[0].excerpt == "发布 Lattice。"
    assert response.metadata["event_pathway"]["enabled"] is True


class BrokenEventService:
    async def build_query_context(self, question, limit):
        raise ValueError("corrupt event store")


@pytest.mark.asyncio
async def test_query_falls_back_when_event_pathway_is_unavailable() -> None:
    engine = FakeQueryEngine()
    service = QueryService(engine, BrokenEventService())

    response = await service.query(QueryRequest(question="基础 RAG 仍应回答"))

    assert response.answer == "answer"
    assert engine.user_prompt is None
    assert response.metadata["event_pathway"]["enabled"] is False

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

    assert [item.start_time for item in result.events] == ["2024-01", "2025-05"]
    assert result.inferred_start_time == "2024"
    assert result.inferred_end_time == "2025-12-31"
    assert len(result.relations) == 1
    assert result.relations[0].relation_type == "precedes"


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

    result = await service.search(
        EventQueryRequest(question="2025 年 Anduril 有哪些关键合同？")
    )

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

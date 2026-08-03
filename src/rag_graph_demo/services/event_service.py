from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from rag_graph_demo.domain.events import (
    EventCandidate,
    EventQueryRequest,
    EventQueryResponse,
    EventRecord,
    EventStatus,
)
from rag_graph_demo.lightrag_engine.engine import LightRAGEngine
from rag_graph_demo.storage.event_registry import EventRegistry

EVENT_SYSTEM_PROMPT = """你是情报资料的结构化事件抽取器。只抽取原文明确陈述的事件，
不得把背景知识或推测写成事实。相同文段内的同一事件只输出一次。证据必须是原文中的
短句或忠实截取。时间统一为 ISO 部分日期（YYYY、YYYY-MM 或 YYYY-MM-DD）；无法确定则
为 null。输出 JSON 对象 {\"events\": [...]}，不要输出解释。"""

EVENT_PROMPT_TEMPLATE = """从以下文档片段抽取对情报分析有意义的事件。重点关注产品发布、
技术测试、合同授予、采购、部署、合作、投资、人员变化、监管、诉讼、论文、专利及能力变化。

每个事件字段：
- event_type：简短中文类型
- summary：一条可独立理解的事件摘要
- action：核心动作
- actors：发起者/参与主体数组
- objects：产品、合同、技术、机构等受作用对象数组
- start_time、end_time：ISO 部分日期或 null
- time_precision：unknown/year/month/day
- location、result、trigger：字符串或 null
- evidence：原文证据短句
- stance：affirmed（肯定）、denied（明确否定）、uncertain（传闻或未确认）
- confidence：仅依据该片段的抽取置信度，0 到 1

文档标题：{title}
片段序号：{part_number}/{part_count}

<document>
{text}
</document>
"""


def _extract_json_payload(value: str) -> dict[str, Any]:
    cleaned = value.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Event extractor did not return a JSON object") from None
        payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Event extractor JSON root must be an object")
    return payload


class LLMEventExtractor:
    def __init__(
        self,
        engine: LightRAGEngine,
        *,
        chunk_chars: int = 12_000,
        max_events: int = 40,
    ) -> None:
        self._engine = engine
        self._chunk_chars = chunk_chars
        self._max_events = max_events

    async def extract(self, title: str, text: str) -> list[EventCandidate]:
        parts = self._split(text)
        candidates: list[EventCandidate] = []
        for index, part in enumerate(parts, start=1):
            raw = await self._engine.complete_extraction(
                EVENT_PROMPT_TEMPLATE.format(
                    title=title,
                    part_number=index,
                    part_count=len(parts),
                    text=part,
                ),
                EVENT_SYSTEM_PROMPT,
            )
            payload = _extract_json_payload(raw)
            event_values = payload.get("events") or []
            if not isinstance(event_values, list):
                raise ValueError("Event extractor 'events' must be an array")
            for value in event_values:
                if not isinstance(value, dict):
                    continue
                candidates.append(EventCandidate.model_validate(value))
                if len(candidates) >= self._max_events:
                    return candidates
        return candidates

    def _split(self, text: str) -> list[str]:
        if len(text) <= self._chunk_chars:
            return [text]
        paragraphs = text.split("\n\n")
        parts: list[str] = []
        current: list[str] = []
        length = 0
        for paragraph in paragraphs:
            if current and length + len(paragraph) + 2 > self._chunk_chars:
                parts.append("\n\n".join(current))
                current = []
                length = 0
            if len(paragraph) > self._chunk_chars:
                for offset in range(0, len(paragraph), self._chunk_chars):
                    if current:
                        parts.append("\n\n".join(current))
                        current = []
                        length = 0
                    parts.append(paragraph[offset : offset + self._chunk_chars])
                continue
            current.append(paragraph)
            length += len(paragraph) + 2
        if current:
            parts.append("\n\n".join(current))
        return [part for part in parts if part.strip()]


class EventService:
    def __init__(
        self,
        registry: EventRegistry,
        extractor: LLMEventExtractor | None = None,
        extract_func: Callable[[str, str], Awaitable[list[EventCandidate]]] | None = None,
    ) -> None:
        self._registry = registry
        self._extractor = extractor
        self._extract_func = extract_func

    async def index_document(
        self, document_id: str, document_title: str, text: str
    ) -> list[EventRecord]:
        if self._extract_func is not None:
            candidates = await self._extract_func(document_title, text)
        elif self._extractor is not None:
            candidates = await self._extractor.extract(document_title, text)
        else:
            candidates = []
        return await self._registry.replace_document_events(
            document_id, document_title, candidates
        )

    async def delete_document(self, document_id: str) -> None:
        await self._registry.delete_document(document_id)

    async def get(self, event_id: str) -> EventRecord | None:
        return await self._registry.get_event(event_id)

    async def search(self, request: EventQueryRequest) -> EventQueryResponse:
        events, mentions, relations, conflicts = await self._registry.snapshot()
        start_time, end_time = self._infer_time_range(
            request.question, request.start_time, request.end_time
        )
        query_terms = self._terms(" ".join((request.question, request.entity or "")))
        type_filters = {_normal(item) for item in request.event_types if item.strip()}
        scored: list[tuple[float, EventRecord]] = []
        for event in events:
            if event.confidence < request.min_confidence:
                continue
            if not request.include_disputed and event.status == EventStatus.DISPUTED:
                continue
            if type_filters and _normal(event.event_type) not in type_filters:
                continue
            event_time = event.start_time or event.end_time
            if (start_time or end_time) and not event_time:
                continue
            event_lower, event_upper = self._time_bounds(event_time)
            query_lower, _ = self._time_bounds(start_time)
            _, query_upper = self._time_bounds(end_time)
            if query_lower and event_upper and event_upper < query_lower:
                continue
            if query_upper and event_lower and event_lower > query_upper:
                continue
            if request.entity:
                entity = _normal(request.entity)
                entities = [_normal(item) for item in event.actors + event.objects]
                if not any(entity in item or item in entity for item in entities):
                    continue
            haystack = " ".join(
                [
                    event.event_type,
                    event.summary,
                    event.action,
                    *event.actors,
                    *event.objects,
                    event.result or "",
                ]
            )
            event_terms = self._terms(haystack)
            lexical = len(query_terms & event_terms) / max(len(query_terms), 1)
            has_structured_filter = bool(
                request.entity or type_filters or start_time or end_time
            )
            if query_terms and lexical == 0 and not has_structured_filter:
                continue
            score = lexical + event.confidence * 0.25
            if event_time:
                score += 0.05
            scored.append((score, event))
        selected = [
            item
            for _, item in sorted(
                scored,
                key=lambda pair: (pair[0], pair[1].start_time or ""),
                reverse=True,
            )[: request.limit]
        ]
        selected_ids = {item.event_id for item in selected}
        selected_mentions = [
            item for item in mentions if item.event_id in selected_ids
        ]
        selected_relations = [
            item
            for item in relations
            if item.source_event_id in selected_ids and item.target_event_id in selected_ids
        ]
        selected_conflicts = [
            item for item in conflicts if item.event_id in selected_ids
        ]
        selected.sort(key=lambda item: item.start_time or "9999")
        return EventQueryResponse(
            events=selected,
            mentions=selected_mentions,
            relations=selected_relations,
            conflicts=selected_conflicts,
            inferred_start_time=start_time,
            inferred_end_time=end_time,
        )

    async def build_query_context(
        self, question: str, limit: int = 12
    ) -> tuple[str, EventQueryResponse]:
        result = await self.search(EventQueryRequest(question=question, limit=limit))
        if not result.events:
            return "", result
        lines = [
            "以下是事件索引按问题、时间和置信度筛出的结构化上下文。",
            "它们是带来源的断言，不是无条件真理；争议事件必须明确标注，不能编造因果关系。",
        ]
        for event in result.events:
            time_value = event.start_time or "时间未知"
            lines.append(
                f"- [{event.event_id}] {time_value}｜{event.event_type}｜{event.summary}"
                f"｜主体:{','.join(event.actors) or '-'}｜对象:{','.join(event.objects) or '-'}"
                f"｜状态:{event.status.value}｜置信度:{event.confidence:.2f}"
                f"｜来源文档:{','.join(event.source_document_ids)}"
            )
        if result.conflicts:
            lines.append("冲突提示：" + "；".join(item.description for item in result.conflicts))
        lines.append("回答时仍须以 LightRAG 返回的原文引用为证据，并说明时间与不确定性。")
        return "\n".join(lines), result

    @staticmethod
    def _terms(value: str) -> set[str]:
        lowered = value.casefold()
        words = set(re.findall(r"[a-z0-9][a-z0-9_-]+", lowered))
        chinese_runs = re.findall(r"[\u4e00-\u9fff]+", lowered)
        for run in chinese_runs:
            if len(run) == 1:
                words.add(run)
            else:
                words.update(run[index : index + 2] for index in range(len(run) - 1))
        return words

    @staticmethod
    def _infer_time_range(
        question: str, start_time: str | None, end_time: str | None
    ) -> tuple[str | None, str | None]:
        if start_time or end_time:
            return start_time, end_time
        years = [int(item) for item in re.findall(r"(?<!\d)(20\d{2})(?!\d)", question)]
        if len(years) >= 2:
            return f"{min(years):04d}", f"{max(years):04d}-12-31"
        if len(years) == 1:
            return f"{years[0]:04d}", f"{years[0]:04d}-12-31"
        recent = re.search(r"(?:近|过去)([一二三四五六七八九十\d]+)年", question)
        if recent:
            chinese_numbers = {
                "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
            }
            token = recent.group(1)
            years_back = int(token) if token.isdigit() else chinese_numbers.get(token, 3)
            current_year = datetime.now(UTC).year
            return f"{current_year - years_back + 1:04d}", f"{current_year:04d}-12-31"
        return None, None

    @staticmethod
    def _time_bounds(value: str | None) -> tuple[str | None, str | None]:
        if not value:
            return None, None
        if re.fullmatch(r"\d{4}", value):
            return f"{value}-01-01", f"{value}-12-31"
        if re.fullmatch(r"\d{4}-\d{2}", value):
            return f"{value}-01", f"{value}-31"
        return value, value


def _normal(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value.casefold())

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from rag_graph_demo.domain.events import (
    EventCandidate,
    EventMention,
    EventQueryRequest,
    EventQueryResponse,
    EventRanking,
    EventRecord,
    EventStatus,
)
from rag_graph_demo.lightrag_engine.engine import LightRAGEngine
from rag_graph_demo.storage.event_registry import EventRegistry

logger = logging.getLogger(__name__)

VALID_TIME_PRECISIONS = frozenset({"unknown", "year", "month", "day"})


def normalize_time_precision(value: Any) -> str:
    """Keep event extraction resilient when an LLM ignores the enum contract."""
    if value is None:
        return "unknown"
    normalized = str(value).strip().casefold()
    if normalized in VALID_TIME_PRECISIONS:
        return normalized
    logger.warning(
        "Unsupported event time_precision %r; falling back to 'unknown'",
        value,
    )
    return "unknown"


def event_evidence_labels(
    result: EventQueryResponse,
) -> list[tuple[str, EventMention]]:
    """Assign stable query-local citation labels in selected event order."""
    mentions_by_event: dict[str, list[EventMention]] = {}
    for mention in result.mentions:
        mentions_by_event.setdefault(mention.event_id, []).append(mention)
    labeled: list[tuple[str, EventMention]] = []
    seen_documents: set[str] = set()
    for event in result.events:
        for mention in mentions_by_event.get(event.event_id, []):
            if mention.document_id in seen_documents:
                continue
            seen_documents.add(mention.document_id)
            labeled.append((f"E{len(labeled) + 1}", mention))
    return labeled


EVENT_EXTRACTION_PROFILE_VERSION = "event_v4_key_events"
EVENT_SYSTEM_PROMPT = """你是高精度的关键事件抽取器。目标不是穷举文本里的所有动作，
而是选择会改变核心对象状态、能够支持时间/变化/冲突检索的重要事件。只使用原文明示的
事实，不得把共现、静态属性、章节主题、背景知识或模型推测写成事件。相同事件只输出一次；
证据必须直接支持主体、动作以及对象或结果。时间统一为 ISO 部分日期；无法确定则为 null。
输出 JSON 对象 {\"events\": [...]}，不要输出解释。"""

EVENT_PROMPT_TEMPLATE = """从以下文档片段选择最多 5 个最关键事件，按重要性降序输出。

事件必须满足全部基础条件：
1. 有明确核心动作和至少一个命名参与者；
2. 有受作用对象、明确结果或状态变化中的至少一项；
3. 删除该事件会明显损害对“何时发生、发生了什么变化、当前状态、是否被纠正/否认”
   等问题的回答。

优先保留：关键发布/部署、合同/采购、重要合作/投资、产生明确结果的实验或观测、能力或
版本变化、监管/诉讼、关键人员变化、正式声明/否认、纠正/撤回/替代，以及作为本文核心
贡献的论文或专利发布。

必须省略：普通提及和共现、静态属性、背景历史、引用文献中的事件、没有结果的日常活动、
泛泛的“讨论/介绍/认为/指出”、重复改写、无命名参与者的动作，以及对核心对象状态没有
实际影响的边缘事项。不要为了达到数量而补充候选；没有关键事件时返回空数组。

规范化规则：
- action 使用最短、可比较的核心动词；actors 和 objects 使用原文中的规范名称。
- 日期、版本、指标值不是独立事件，应进入时间、对象、结果或证据字段。
- 计划、预测、传闻和未确认说法可以保留，但 stance 必须为 uncertain。
- 明确否认使用 denied；后续纠正、撤回或替代应形成自己的事件，保留相应证据。
- summary 必须明确谁在何时（若已知）对什么做了什么以及结果，不得使用“本文称”等代词。

每个事件字段：
- event_type：简短中文类型
- summary：可独立理解的事件摘要；不限制长度，保留关键条件、变化和结果
- action：核心动作
- actors：发起者/参与主体数组
- objects：产品、合同、技术、机构等受作用对象数组
- start_time、end_time：ISO 部分日期或 null
- time_precision：unknown/year/month/day
- location、result、trigger：字符串或 null
- evidence：直接支持事件的原文证据；不限制长度，必要条件和结果不得为节省 token 而删减
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
        max_events: int = 12,
        max_events_per_part: int = 5,
    ) -> None:
        self._engine = engine
        self._chunk_chars = chunk_chars
        self._max_events = max_events
        self._max_events_per_part = max_events_per_part

    async def extract(self, title: str, text: str) -> list[EventCandidate]:
        parts = self._split(text)
        candidates: list[EventCandidate] = []
        search_offset = 0
        for index, part in enumerate(parts, start=1):
            part_start = text.find(part, search_offset)
            if part_start < 0:
                part_start = search_offset
            search_offset = part_start + len(part)
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
            valid_before = len(candidates)
            for value in event_values[: self._max_events_per_part]:
                if not isinstance(value, dict):
                    continue
                repaired = dict(value)
                repaired["time_precision"] = normalize_time_precision(
                    repaired.get("time_precision")
                )
                if not str(repaired.get("evidence") or "").strip():
                    repaired["evidence"] = self._source_evidence(part, repaired)
                try:
                    candidate = EventCandidate.model_validate(repaired)
                except ValidationError as exc:
                    logger.warning("Skipping malformed event candidate: %s", exc)
                    continue
                evidence_offset = part.find(candidate.evidence)
                candidate.source_segment_id = f"event-part-{index}"
                candidate.source_segment_start = part_start
                candidate.source_segment_end = part_start + len(part)
                if evidence_offset >= 0:
                    candidate.evidence_start = part_start + evidence_offset
                    candidate.evidence_end = (
                        candidate.evidence_start + len(candidate.evidence)
                    )
                candidates.append(candidate)
                if len(candidates) >= self._max_events:
                    return candidates
            if event_values and len(candidates) == valid_before:
                raise ValueError("All event candidates in a document part were malformed")
        return candidates

    @staticmethod
    def _source_evidence(part: str, value: dict[str, Any]) -> str:
        """Select an exact source sentence when the model omits evidence."""
        visible = re.sub(r"<!--.*?-->", "", part, flags=re.DOTALL).strip()
        sentences = [
            item.strip()
            for item in re.split(r"(?<=[。！？.!?])\s+|\n+", visible)
            if item.strip()
        ]
        if not sentences:
            if not visible:
                raise ValueError("Document part has no source text for event evidence")
            return visible[:1_000]
        raw_terms = [
            value.get("trigger"),
            value.get("action"),
            *(value.get("actors") or []),
            *(value.get("objects") or []),
        ]
        terms = [_normal(str(item)) for item in raw_terms if item]

        def score(sentence: str) -> tuple[int, int]:
            normalized = _normal(sentence)
            return sum(term in normalized for term in terms if term), -len(sentence)

        best = max(sentences, key=score)
        return (best if score(best)[0] else sentences[0])[:1_000]

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
        candidates = [
            item.model_copy(
                update={
                    "source_segment_id": (
                        f"{document_id}::{item.source_segment_id}"
                        if item.source_segment_id
                        else None
                    )
                }
            )
            for item in candidates
        ]
        return await self._registry.replace_document_events(
            document_id, document_title, candidates
        )

    async def delete_document(self, document_id: str) -> None:
        await self._registry.delete_document(document_id)

    async def get(self, event_id: str) -> EventRecord | None:
        return await self._registry.get_event(event_id)

    async def list_events(self) -> list[EventRecord]:
        return await self._registry.list_events()

    async def search(
        self,
        request: EventQueryRequest,
        *,
        candidate_event_ids: set[str] | None = None,
    ) -> EventQueryResponse:
        events, mentions, relations, conflicts = await self._registry.snapshot()
        expanded_ids: set[str] = set()
        allowed_event_ids = candidate_event_ids
        if candidate_event_ids is not None and request.expand_event_chain:
            allowed_event_ids = set(candidate_event_ids)
            for relation in relations:
                if relation.source_event_id in candidate_event_ids:
                    allowed_event_ids.add(relation.target_event_id)
                    expanded_ids.add(relation.target_event_id)
                if relation.target_event_id in candidate_event_ids:
                    allowed_event_ids.add(relation.source_event_id)
                    expanded_ids.add(relation.source_event_id)
            expanded_ids.difference_update(candidate_event_ids)
        start_time, end_time = self._infer_time_range(
            request.question, request.start_time, request.end_time
        )
        query_terms = self._terms(" ".join((request.question, request.entity or "")))
        type_hints = {_normal(item) for item in request.event_types if item.strip()}
        inferred_entity_events = self._events_matching_question_entities(
            request.question, events
        )
        mentions_by_event: dict[str, list[EventMention]] = {}
        for mention in mentions:
            mentions_by_event.setdefault(mention.event_id, []).append(mention)
        latest_intent = bool(
            re.search(
                r"最新|最近|目前|当前|现状|截至|latest|current|recent",
                request.question,
                re.I,
            )
        )
        state_intent = bool(request.state_intents) or bool(
            re.search(
                r"变化|演变|进展|状态|冲突|争议|否认|纠正|更正|撤回|替代|更新|"
                r"change|evol|status|conflict|deni|correct|retract|replace|update",
                request.question,
                re.I,
            )
        )
        eligible: list[tuple[EventRecord, dict[str, float]]] = []
        for event in events:
            if allowed_event_ids is not None and event.event_id not in allowed_event_ids:
                continue
            if event.confidence < request.min_confidence:
                continue
            if not request.include_disputed and event.status == EventStatus.DISPUTED:
                continue
            normalized_event_type = _normal(event.event_type)
            if inferred_entity_events and event.event_id not in inferred_entity_events:
                continue
            event_time = event.start_time or event.end_time
            event_lower, event_upper = self._time_bounds(event_time)
            query_lower, _ = self._time_bounds(start_time)
            _, query_upper = self._time_bounds(end_time)
            if query_lower and event_upper and event_upper < query_lower:
                continue
            if query_upper and event_lower and event_lower > query_upper:
                continue
            event_entities = [_normal(item) for item in event.actors + event.objects]
            if request.entity:
                hard_entity = _normal(request.entity)
                if not any(
                    hard_entity in item or item in hard_entity for item in event_entities
                ):
                    continue
            entity_hints = [_normal(item) for item in request.entities if _normal(item)]
            matched_entity_hints = sum(
                any(hint in item or item in hint for item in event_entities)
                for hint in entity_hints
            )
            entity_hint_score = matched_entity_hints / max(len(entity_hints), 1)
            type_hint_score = float(
                bool(type_hints)
                and any(
                    item in normalized_event_type or normalized_event_type in item
                    for item in type_hints
                )
            )
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
            has_relevance_signal = bool(
                request.entity
                or entity_hint_score
                or type_hint_score
                or event.event_id in inferred_entity_events
                or start_time
                or end_time
                or candidate_event_ids is not None
            )
            if query_terms and lexical == 0 and not has_relevance_signal:
                continue
            entity_terms = self._terms(" ".join(event.actors + event.objects))
            action_terms = self._terms(" ".join((event.event_type, event.action)))
            event_mentions = mentions_by_event.get(event.event_id, [])
            has_state_signal = event.status == EventStatus.DISPUTED or any(
                item.stance != "affirmed" for item in event_mentions
            )
            eligible.append(
                (
                    event,
                    {
                        "lexical_score": lexical,
                        "entity_score": max(
                            len(query_terms & entity_terms) / max(len(query_terms), 1),
                            entity_hint_score,
                            float(event.event_id in inferred_entity_events),
                        ),
                        "action_score": max(
                            len(query_terms & action_terms) / max(len(query_terms), 1),
                            type_hint_score,
                        ),
                        "temporal_score": 1.0
                        if (start_time or end_time) and event_time
                        else (0.25 if event_time else 0.0),
                        "state_score": 1.0
                        if state_intent and has_state_signal
                        else 0.0,
                        "confidence_score": event.confidence,
                        "source_score": min(
                            1.0, len(event.source_document_ids) / 2
                        ),
                    },
                )
            )
        if latest_intent:
            dated = [
                self._time_rank_value(item.start_time or item.end_time)
                for item, _ in eligible
                if item.start_time or item.end_time
            ]
            lower = min(dated, default=0)
            upper = max(dated, default=0)
            for event, signals in eligible:
                value = self._time_rank_value(event.start_time or event.end_time)
                signals["temporal_score"] = (
                    (value - lower) / (upper - lower)
                    if value and upper > lower
                    else float(bool(value))
                )
        scored = [
            (
                signals["lexical_score"]
                + signals["entity_score"] * 0.15
                + signals["action_score"] * 0.15
                + signals["temporal_score"] * 0.12
                + signals["state_score"] * 0.15
                + signals["confidence_score"] * 0.2
                + signals["source_score"] * 0.05,
                event,
                signals,
            )
            for event, signals in eligible
        ]
        ranked = sorted(
            scored,
            key=lambda item: (item[0], item[1].start_time or "", item[1].event_id),
            reverse=True,
        )[: request.limit]
        selected = [event for _, event, _ in ranked]
        rankings = [
            EventRanking(
                event_id=event.event_id,
                rank=rank,
                score=round(score, 6),
                expanded=event.event_id in expanded_ids,
                **{key: round(value, 6) for key, value in signals.items()},
            )
            for rank, (score, event, signals) in enumerate(ranked, start=1)
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
        if request.sort_by == "time":
            selected.sort(key=lambda item: (item.start_time or "9999", item.event_id))
        return EventQueryResponse(
            events=selected,
            rankings=rankings,
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
        evidence = event_evidence_labels(result)
        labels_by_document = {
            mention.document_id: label for label, mention in evidence
        }
        mentions_by_event: dict[str, list[EventMention]] = {}
        for mention in result.mentions:
            mentions_by_event.setdefault(mention.event_id, []).append(mention)
        lines = [
            "以下是事件索引按问题、时间和置信度筛出的结构化上下文。",
            "它们是带来源的断言，不是无条件真理；争议事件必须明确标注，不能编造因果关系。",
        ]
        timeline_events = sorted(
            result.events,
            key=lambda item: (item.start_time or "9999", item.event_id),
        )
        for event in timeline_events:
            time_value = event.start_time or "时间未知"
            event_mentions = mentions_by_event.get(event.event_id, [])
            references = [
                f"[{labels_by_document[item.document_id]}]"
                for item in event_mentions
                if item.document_id in labels_by_document
            ]
            excerpt = event_mentions[0].evidence[:300] if event_mentions else ""
            lines.append(
                f"- [{event.event_id}] {time_value}｜{event.event_type}｜{event.summary}"
                f"｜主体:{','.join(event.actors) or '-'}｜对象:{','.join(event.objects) or '-'}"
                f"｜状态:{event.status.value}｜置信度:{event.confidence:.2f}"
                f"｜证据:{','.join(references) or '-'} {excerpt}"
            )
        if result.conflicts:
            lines.append("冲突提示：" + "；".join(item.description for item in result.conflicts))
        if result.relations:
            events_by_id = {item.event_id: item for item in result.events}
            relation_labels = {
                "precedes": "先于",
                "same_time": "同期",
                "related": "相关",
            }
            lines.append("事件链（只表达时间或共享实体，不自动表示因果）：")
            for relation in result.relations:
                source = events_by_id.get(relation.source_event_id)
                target = events_by_id.get(relation.target_event_id)
                if source is None or target is None:
                    continue
                relation_label = relation_labels[relation.relation_type]
                shared = ",".join(relation.shared_entities) or "共享实体未知"
                lines.append(
                    f"- {source.start_time or '时间未知'} {source.summary} "
                    f"--{relation_label}({shared})--> "
                    f"{target.start_time or '时间未知'} {target.summary}"
                )
        lines.append(
            "回答时仍须以原文为证据；引用上述事件时必须保留对应的 [E1]、[E2] 标签，"
            "并说明时间与不确定性。"
        )
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
    def _events_matching_question_entities(
        question: str, events: list[EventRecord]
    ) -> set[str]:
        question_normal = _normal(question)
        question_tokens = set(re.findall(r"[a-z][a-z0-9_-]+", question.casefold()))
        generic_entity_tokens = {
            "company",
            "corporation",
            "group",
            "industries",
            "news",
            "report",
            "university",
        }
        matched: set[str] = set()
        for event in events:
            for entity in event.actors + event.objects:
                normalized = _normal(entity)
                if len(normalized) < 3:
                    continue
                entity_tokens = {
                    item
                    for item in re.findall(r"[a-z][a-z0-9_-]+", entity.casefold())
                    if len(item) >= 4 and item not in generic_entity_tokens
                }
                if normalized in question_normal or bool(entity_tokens & question_tokens):
                    matched.add(event.event_id)
                    break
        return matched

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

    @staticmethod
    def _time_rank_value(value: str | None) -> int:
        if not value:
            return 0
        parts = value.split("-")
        try:
            year = int(parts[0])
            month = int(parts[1]) if len(parts) > 1 else 1
            day = int(parts[2]) if len(parts) > 2 else 1
        except ValueError:
            return 0
        return year * 10_000 + month * 100 + day


def _normal(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value.casefold())

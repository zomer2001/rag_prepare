from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

from rag_graph_demo.retrieval.models import QueryPlan
from rag_graph_demo.storage.atomic_json import read_json, write_json_atomic

logger = logging.getLogger(__name__)

PlannerCompletion = Callable[[str, str], Awaitable[str]]
QUERY_PLANNER_PROFILE_VERSION = "query_planner_v2_soft_event_hints"
LEGACY_QUERY_PLANNER_PROFILE_VERSION = "query_planner_v1"

PLANNER_SYSTEM_PROMPT = """你是检索系统的 Query Planner。只分析用户问题，不回答问题，
不得补充问题中没有的实体或事实。输出一个 JSON 对象，不要输出 Markdown 或解释。"""

PLANNER_PROMPT = """分析下面的问题，为文本向量、实体关系图和事件图生成同一个查询计划。

字段要求：
- low_level_keywords：具体实体、产品、机构、编号、地点等检索词。
- high_level_keywords：关系、主题、能力、动作等抽象检索词。
- entities：最多3个问题明确提到的事件参与者或受作用对象，只作为排序提示。报道来源、媒体、
  作者不能因为“according to/reported by/报道”而成为事件实体，除非问题询问的事件确实发生在它们身上。
- event_types：问题明确要求的事件类型，如合作、合同、发布、部署、纠正；只作为排序提示，
  没有则为空。
- start_time、end_time：只有问题原文明确出现日期或年份时才输出ISO部分日期；不得从背景、
  新闻日期或模型知识猜测。没有显式时间约束则为 null。当前日期为 {today}。
- state_intents：从 latest、change、conflict、denial、correction、replacement 中选择。
- expand_event_chain：问题询问先后、变化、演化、原因、影响、纠正或替代时为 true。

问题：{question}
"""


class QueryPlanner:
    def __init__(
        self,
        complete: PlannerCompletion,
        *,
        cache_path: Path | None = None,
        strict_failures: bool = False,
    ) -> None:
        self._complete = complete
        self._cache_path = cache_path
        self._strict_failures = strict_failures
        self._lock = asyncio.Lock()
        raw_cache = read_json(cache_path, default={}) if cache_path else {}
        self._cache: dict[str, QueryPlan] = {}
        for key, value in raw_cache.items():
            if not isinstance(value, dict):
                continue
            try:
                self._cache[key] = QueryPlan.model_validate(value)
            except ValueError:
                logger.warning("Ignoring malformed cached query plan: %s", key)

    async def plan(self, question: str) -> QueryPlan:
        cache_key = self._cache_key(question)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return self._sanitize(question, cached)
        legacy = self._cache.get(
            self._cache_key(question, profile=LEGACY_QUERY_PLANNER_PROFILE_VERSION)
        )
        if legacy is not None:
            plan = self._sanitize(question, legacy)
            self._cache[cache_key] = plan
            self._persist_cache()
            return plan
        async with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return self._sanitize(question, cached)
            try:
                raw = await self._complete(
                    PLANNER_PROMPT.format(
                        question=question,
                        today=datetime.now(UTC).date().isoformat(),
                    ),
                    PLANNER_SYSTEM_PROMPT,
                )
                plan = QueryPlan.model_validate(self._json_payload(raw))
                if not plan.low_level_keywords and not plan.high_level_keywords:
                    plan.low_level_keywords = [question]
                    plan.high_level_keywords = [question]
            except Exception as exc:
                if self._strict_failures:
                    raise
                logger.warning("Query planner fallback for %r: %s", question, exc)
                plan = self._fallback(question)
            plan = self._sanitize(question, plan)
            self._cache[cache_key] = plan
            self._persist_cache()
            return plan

    @staticmethod
    def _cache_key(
        question: str, *, profile: str = QUERY_PLANNER_PROFILE_VERSION
    ) -> str:
        today = datetime.now(UTC).date().isoformat()
        value = f"{profile}\x1f{today}\x1f{question}"
        return hashlib.sha256(value.encode()).hexdigest()

    def _persist_cache(self) -> None:
        if self._cache_path is None:
            return
        write_json_atomic(
            self._cache_path,
            {
                key: value.model_dump(mode="json")
                for key, value in self._cache.items()
            },
        )

    @staticmethod
    def _sanitize(question: str, plan: QueryPlan) -> QueryPlan:
        """Turn fallible planner fields into safe hints and deterministic dates."""

        def unique(values: list[str], limit: int) -> list[str]:
            result: list[str] = []
            seen: set[str] = set()
            for value in values:
                cleaned = " ".join(value.split()).strip()
                key = cleaned.casefold()
                if cleaned and key not in seen:
                    result.append(cleaned)
                    seen.add(key)
                if len(result) >= limit:
                    break
            return result

        state_aliases = {
            "latest": "latest",
            "recent": "latest",
            "current": "latest",
            "change": "change",
            "evolution": "change",
            "conflict": "conflict",
            "consistency": "conflict",
            "discrepancy": "conflict",
            "denial": "denial",
            "correction": "correction",
            "replacement": "replacement",
        }
        states = unique(
            [
                state_aliases[item.casefold()]
                for item in plan.state_intents
                if item.casefold() in state_aliases
            ],
            6,
        )
        years = sorted(
            {int(item) for item in re.findall(r"(?<!\d)(20\d{2})(?!\d)", question)}
        )
        start_time = f"{years[0]:04d}" if years else None
        end_time = f"{years[-1]:04d}-12-31" if years else None
        return plan.model_copy(
            update={
                "low_level_keywords": unique(plan.low_level_keywords, 12),
                "high_level_keywords": unique(plan.high_level_keywords, 12),
                "entities": unique(plan.entities, 3),
                "event_types": unique(plan.event_types, 4),
                "start_time": start_time,
                "end_time": end_time,
                "state_intents": states,
            }
        )

    @staticmethod
    def _json_payload(raw: str) -> dict[str, object]:
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Query planner did not return a JSON object")
        value = json.loads(cleaned[start : end + 1])
        if not isinstance(value, dict):
            raise ValueError("Query planner JSON root must be an object")
        return value

    @staticmethod
    def _fallback(question: str) -> QueryPlan:
        years = [int(item) for item in re.findall(r"(?<!\d)(20\d{2})(?!\d)", question)]
        start_time = f"{min(years):04d}" if years else None
        end_time = f"{max(years):04d}-12-31" if years else None
        state_intents: list[str] = []
        patterns = {
            "latest": r"最新|最近|目前|当前|截至",
            "change": r"变化|演变|进展|更新|升级",
            "conflict": r"冲突|争议|矛盾",
            "denial": r"否认|否定",
            "correction": r"纠正|更正|修正",
            "replacement": r"替代|取代|撤回",
        }
        for name, pattern in patterns.items():
            if re.search(pattern, question):
                state_intents.append(name)
        return QueryPlan(
            low_level_keywords=[question],
            high_level_keywords=[question],
            start_time=start_time,
            end_time=end_time,
            state_intents=state_intents,
            expand_event_chain=bool(state_intents),
        )

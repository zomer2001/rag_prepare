from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path

from pydantic import ValidationError

from rag_graph_demo.domain.tkg import (
    SemanticEntityCandidate,
    SemanticExtractionBatch,
    SemanticRelationshipCandidate,
)
from rag_graph_demo.services.entity_resolution_service import normalize_name

CompletionCallable = Callable[[str, str], Awaitable[str]]

SEMANTIC_EXTRACTION_PROFILE_VERSION = "tkg_semantic_v1"
SEMANTIC_SYSTEM_PROMPT = """你是受本体约束的知识抽取器。只抽取原文明示、对检索与多跳推理
必要的稳定实体和事实关系。事件发生本身由独立事件抽取器处理；这里不要把事件、日期、数值
或宽泛主题当成实体。只能输出 JSON 对象，不要输出 Markdown 或解释。"""
SEMANTIC_PROMPT = """根据规则从文档片段抽取实体与关系。

<rules>
{rules}
</rules>

输出格式：
{{
  "entities": [
    {{"name":"规范名称","type":"本体类型","description":"原文支持的描述","aliases":[]}}
  ],
  "relationships": [
    {{"source":"实体名","target":"实体名","predicate":"稳定关系词",
      "description":"直接证据或忠实事实描述","confidence":0到1}}
  ]
}}

限制：每个片段最多 {max_entities} 个实体和 {max_relationships} 条关系；关系两端必须出现在
本次实体数组中；不确定时宁缺毋滥。

文档标题：{title}
片段ID：{segment_id}

<document>
{text}
</document>
"""


class LLMSemanticExtractor:
    """Extract controlled semantic facts without creating a LightRAG graph store."""

    def __init__(
        self,
        complete: CompletionCallable,
        *,
        prompt_file: Path,
        max_entities: int = 4,
        max_relationships: int = 4,
    ) -> None:
        self._complete = complete
        self._rules = prompt_file.read_text(encoding="utf-8")
        self._max_entities = max(1, max_entities)
        self._max_relationships = max(0, max_relationships)

    async def extract(
        self,
        title: str,
        segments: Iterable[tuple[str, str]],
    ) -> SemanticExtractionBatch:
        entities: dict[str, SemanticEntityCandidate] = {}
        relationships: dict[tuple[str, str, str, str], SemanticRelationshipCandidate] = {}
        for segment_id, text in segments:
            raw = await self._complete(
                SEMANTIC_PROMPT.format(
                    rules=self._rules,
                    max_entities=self._max_entities,
                    max_relationships=self._max_relationships,
                    title=title,
                    segment_id=segment_id,
                    text=text,
                ),
                SEMANTIC_SYSTEM_PROMPT,
            )
            payload = self._json_payload(raw)
            current_entities: list[SemanticEntityCandidate] = []
            for value in (payload.get("entities") or [])[: self._max_entities]:
                if not isinstance(value, dict):
                    continue
                repaired = {
                    **value,
                    "entity_type": value.get("entity_type") or value.get("type") or "Concept",
                    "source_segment_id": segment_id,
                }
                try:
                    candidate = SemanticEntityCandidate.model_validate(repaired)
                except ValidationError:
                    continue
                key = normalize_name(candidate.name)
                if not key:
                    continue
                previous = entities.get(key)
                if previous is None or len(candidate.description) > len(previous.description):
                    entities[key] = candidate
                current_entities.append(candidate)
            allowed = {normalize_name(item.name) for item in current_entities}
            retained = 0
            for value in payload.get("relationships") or []:
                if retained >= self._max_relationships or not isinstance(value, dict):
                    break
                repaired = {
                    **value,
                    "predicate": value.get("predicate") or value.get("keywords") or "",
                    "source_segment_id": segment_id,
                }
                try:
                    candidate = SemanticRelationshipCandidate.model_validate(repaired)
                except ValidationError:
                    continue
                if (
                    normalize_name(candidate.source) not in allowed
                    or normalize_name(candidate.target) not in allowed
                ):
                    continue
                key = (
                    normalize_name(candidate.source),
                    normalize_name(candidate.predicate),
                    normalize_name(candidate.target),
                    segment_id,
                )
                relationships[key] = candidate
                retained += 1
        return SemanticExtractionBatch(
            entities=list(entities.values()),
            relationships=list(relationships.values()),
        )

    @staticmethod
    def _json_payload(raw: str) -> dict[str, object]:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Semantic extractor did not return a JSON object")
        value = json.loads(cleaned[start : end + 1])
        if not isinstance(value, dict):
            raise ValueError("Semantic extractor JSON root must be an object")
        return value

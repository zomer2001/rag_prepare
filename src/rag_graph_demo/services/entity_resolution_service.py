from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np

from rag_graph_demo.domain.tkg import (
    EntityCandidateScore,
    EntityRecord,
    EntityResolutionProposal,
    ResolutionAction,
    SemanticEntityCandidate,
    TKGSnapshot,
)

EmbedFunc = Callable[[list[str], str], Awaitable[Any]]
CompleteFunc = Callable[[str, str], Awaitable[str]]

RESOLUTION_SYSTEM_PROMPT = """你是受本体约束的实体消歧器。判断新提及是否指向候选实体。
不得仅因名称相似就合并；类型、描述、上下文或唯一标识冲突时必须选择 CREATE_NEW 或
AMBIGUOUS。只能输出 JSON，不得输出解释文字。"""
OPEN_RESOLUTION_SYSTEM_PROMPT = """你是知识库实体消歧器。判断新提及是否指向候选实体。
实体类型是可修正的软标签，类型名称不同不能单独作为拒绝匹配的理由；名称、别名、描述、
上下文或唯一标识不一致时选择 CREATE_NEW 或 AMBIGUOUS。只能输出 JSON。"""
RESOLUTION_PROMPT = """待解析实体：
name: {name}
type: {entity_type}
description: {description}

候选实体：
{candidates}

返回：
{{"action":"MATCH_EXISTING|CREATE_NEW|AMBIGUOUS|REJECT",
  "selected_entity_id": null或候选ID,
  "confidence": 0到1,
  "reason":"必须引用名称、类型或上下文证据的简短理由"}}
MATCH_EXISTING 的 selected_entity_id 必须来自候选列表。"""


def normalize_name(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value.casefold())


def identity_signature(value: str, entity_type: str) -> str:
    normalized = normalize_name(value)
    if entity_type == "Organization":
        for suffix in (
            "incorporated",
            "corporation",
            "limited",
            "company",
            "集团有限公司",
            "股份有限公司",
            "有限公司",
            "集团",
            "公司",
            "inc",
            "corp",
            "ltd",
        ):
            if normalized.endswith(suffix) and len(normalized) > len(suffix) + 2:
                normalized = normalized[: -len(suffix)]
                break
    return normalized


class OntologyTypeMapper:
    def __init__(self, path: Path) -> None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        self._mapping = {normalize_name(key): str(value) for key, value in raw.items()}
        self.allowed_types = frozenset(self._mapping.values())

    def normalize(self, value: str) -> str | None:
        return self._mapping.get(normalize_name(value))


class EntityResolutionService:
    def __init__(
        self,
        type_mapper: OntologyTypeMapper,
        embed: EmbedFunc,
        complete: CompleteFunc,
        *,
        candidate_limit: int = 5,
        embedding_threshold: float = 0.72,
        auto_match_threshold: float = 0.94,
        llm_threshold: float = 0.78,
        enforce_type: bool = True,
    ) -> None:
        self._type_mapper = type_mapper
        self._embed = embed
        self._complete = complete
        self._candidate_limit = candidate_limit
        self._embedding_threshold = embedding_threshold
        self._auto_match_threshold = auto_match_threshold
        self._llm_threshold = llm_threshold
        self._enforce_type = enforce_type

    async def resolve_batch(
        self,
        snapshot: TKGSnapshot,
        candidates: list[SemanticEntityCandidate],
    ) -> list[EntityResolutionProposal]:
        proposals: list[EntityResolutionProposal | None] = [None] * len(candidates)
        unresolved: list[tuple[int, SemanticEntityCandidate, str, list[EntityRecord]]] = []
        for index, candidate in enumerate(candidates):
            entity_type = self._type_mapper.normalize(candidate.entity_type)
            if not self._enforce_type:
                entity_type = self._open_type(candidate.entity_type)
            elif entity_type is None:
                exact_any_type = [
                    item
                    for item in snapshot.entities
                    if item.status == "active"
                    and normalize_name(candidate.name)
                    in {
                        normalize_name(item.canonical_name),
                        *(normalize_name(alias) for alias in item.aliases),
                    }
                ]
                if len(exact_any_type) == 1:
                    entity_type = exact_any_type[0].entity_type
                else:
                    proposals[index] = EntityResolutionProposal(
                        candidate_key=self._key(candidate),
                        candidate_name=candidate.name,
                        normalized_type="Concept",
                        action=ResolutionAction.AMBIGUOUS,
                        reason=(
                            f"抽取类型 {candidate.entity_type!r} 不在本体映射中，"
                            "需要治理后才能确定类型。"
                        ),
                        decided_by="ontology",
                    )
                    continue
            same_type = [
                item
                for item in snapshot.entities
                if item.status == "active"
                and (
                    not self._enforce_type
                    or item.entity_type == entity_type
                )
            ]
            exact = [
                item
                for item in same_type
                if identity_signature(candidate.name, entity_type)
                in {
                    identity_signature(item.canonical_name, item.entity_type),
                    *(identity_signature(alias, item.entity_type) for alias in item.aliases),
                }
            ]
            if len(exact) == 1:
                proposals[index] = EntityResolutionProposal(
                    candidate_key=self._key(candidate),
                    candidate_name=candidate.name,
                    normalized_type=entity_type,
                    action=ResolutionAction.MATCH_EXISTING,
                    selected_entity_id=exact[0].entity_id,
                    candidates=[
                        self._score(
                            candidate,
                            exact[0],
                            1.0,
                            None,
                            ["规范名/别名签名一致"],
                        )
                    ],
                    reason="同类型规范名或别名签名唯一匹配。",
                    decided_by="rule",
                    confidence=1,
                )
            elif len(exact) > 1:
                proposals[index] = EntityResolutionProposal(
                    candidate_key=self._key(candidate),
                    candidate_name=candidate.name,
                    normalized_type=entity_type,
                    action=ResolutionAction.AMBIGUOUS,
                    candidates=[
                        self._score(candidate, item, 1.0, None, ["多个精确签名候选"])
                        for item in exact[: self._candidate_limit]
                    ],
                    reason="同类型下存在多个精确候选，禁止自动合并。",
                    decided_by="rule",
                )
            elif not same_type:
                proposals[index] = EntityResolutionProposal(
                    candidate_key=self._key(candidate),
                    candidate_name=candidate.name,
                    normalized_type=entity_type,
                    action=ResolutionAction.CREATE_NEW,
                    reason="知识库中不存在同类型实体。",
                    decided_by="rule",
                    confidence=1,
                )
            else:
                unresolved.append((index, candidate, entity_type, same_type))

        for index, candidate, entity_type, same_type in unresolved:
            proposals[index] = await self._resolve_semantic(candidate, entity_type, same_type)
        return [item for item in proposals if item is not None]

    @staticmethod
    def _open_type(value: str) -> str:
        cleaned = " ".join(value.split()).strip(" ,，;；")
        return cleaned[:100] or "Entity"

    async def _resolve_semantic(
        self,
        candidate: SemanticEntityCandidate,
        entity_type: str,
        same_type: list[EntityRecord],
    ) -> EntityResolutionProposal:
        lexical = [
            (
                max(
                    SequenceMatcher(
                        None,
                        identity_signature(candidate.name, entity_type),
                        identity_signature(name, entity_type),
                    ).ratio()
                    for name in [item.canonical_name, *item.aliases]
                ),
                item,
            )
            for item in same_type
        ]
        lexical.sort(key=lambda item: (item[0], item[1].entity_id), reverse=True)
        pool = [item for _, item in lexical[: max(self._candidate_limit * 3, 10)]]
        texts = [self._text(candidate)] + [self._entity_text(item) for item in pool]
        vectors = np.asarray(await self._embed(texts, "document"), dtype=np.float32)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        similarities = vectors[1:] @ vectors[0]
        lexical_by_id = {item.entity_id: score for score, item in lexical}
        scores = sorted(
            [
                self._score(
                    candidate,
                    item,
                    lexical_by_id[item.entity_id],
                    float(similarity),
                    [
                        "类型兼容" if item.entity_type == entity_type else "跨类型候选",
                        "名称与描述 Embedding 候选",
                    ],
                )
                for item, similarity in zip(pool, similarities, strict=True)
                if similarity >= self._embedding_threshold
            ],
            key=lambda item: (item.embedding_score or -1, item.lexical_score),
            reverse=True,
        )[: self._candidate_limit]
        if not scores:
            return EntityResolutionProposal(
                candidate_key=self._key(candidate),
                candidate_name=candidate.name,
                normalized_type=entity_type,
                action=ResolutionAction.CREATE_NEW,
                reason="同类型实体的名称与描述相似度均低于候选阈值。",
                decided_by="embedding",
                confidence=1,
            )
        top = scores[0]
        second = scores[1].embedding_score if len(scores) > 1 else -1
        if (
            (top.embedding_score or 0) >= self._auto_match_threshold
            and (top.embedding_score or 0) - (second or 0) >= 0.08
            and top.lexical_score >= 0.6
            and (
                not self._enforce_type
                or top.entity_type == entity_type
            )
        ):
            return EntityResolutionProposal(
                candidate_key=self._key(candidate),
                candidate_name=candidate.name,
                normalized_type=entity_type,
                action=ResolutionAction.MATCH_EXISTING,
                selected_entity_id=top.entity_id,
                candidates=scores,
                reason="同类型候选在名称与描述上高相似，且与次候选有明确间隔。",
                decided_by="embedding",
                confidence=top.embedding_score,
            )
        if (top.embedding_score or 0) < self._llm_threshold:
            return EntityResolutionProposal(
                candidate_key=self._key(candidate),
                candidate_name=candidate.name,
                normalized_type=entity_type,
                action=ResolutionAction.AMBIGUOUS,
                candidates=scores,
                reason="存在相似候选，但证据不足以自动匹配或调用LLM裁决。",
                decided_by="embedding",
                confidence=top.embedding_score,
            )
        return await self._llm_decide(candidate, entity_type, scores)

    async def _llm_decide(
        self,
        candidate: SemanticEntityCandidate,
        entity_type: str,
        scores: list[EntityCandidateScore],
    ) -> EntityResolutionProposal:
        raw = await self._complete(
            RESOLUTION_PROMPT.format(
                name=candidate.name,
                entity_type=entity_type,
                description=candidate.description,
                candidates="\n".join(
                    f"- {item.model_dump_json()}" for item in scores
                ),
            ),
            (
                RESOLUTION_SYSTEM_PROMPT
                if self._enforce_type
                else OPEN_RESOLUTION_SYSTEM_PROMPT
            ),
        )
        try:
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
            value = json.loads(cleaned)
            action = ResolutionAction(str(value.get("action", "")).casefold())
            selected = value.get("selected_entity_id")
            allowed = {item.entity_id for item in scores}
            if action == ResolutionAction.MATCH_EXISTING and selected not in allowed:
                raise ValueError("LLM selected an entity outside the shortlist")
            return EntityResolutionProposal(
                candidate_key=self._key(candidate),
                candidate_name=candidate.name,
                normalized_type=entity_type,
                action=action,
                selected_entity_id=selected,
                candidates=scores,
                reason=str(value.get("reason") or "LLM未提供理由"),
                decided_by="llm",
                confidence=float(value.get("confidence") or 0),
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            return EntityResolutionProposal(
                candidate_key=self._key(candidate),
                candidate_name=candidate.name,
                normalized_type=entity_type,
                action=ResolutionAction.AMBIGUOUS,
                candidates=scores,
                reason="LLM裁决格式无效或越权选择，已降级到人工治理。",
                decided_by="llm_invalid",
            )

    @staticmethod
    def _key(candidate: SemanticEntityCandidate) -> str:
        return "|".join((normalize_name(candidate.name), normalize_name(candidate.entity_type)))

    @staticmethod
    def _text(candidate: SemanticEntityCandidate) -> str:
        return f"name: {candidate.name}\ntype: {candidate.entity_type}\n{candidate.description}"

    @staticmethod
    def _entity_text(entity: EntityRecord) -> str:
        return (
            f"name: {entity.canonical_name}\ntype: {entity.entity_type}\n"
            f"aliases: {', '.join(entity.aliases)}\n{entity.description}"
        )

    @staticmethod
    def _score(
        candidate: SemanticEntityCandidate,
        entity: EntityRecord,
        lexical: float,
        embedding: float | None,
        reasons: list[str],
    ) -> EntityCandidateScore:
        del candidate
        return EntityCandidateScore(
            entity_id=entity.entity_id,
            canonical_name=entity.canonical_name,
            entity_type=entity.entity_type,
            lexical_score=round(lexical, 6),
            embedding_score=round(embedding, 6) if embedding is not None else None,
            reasons=reasons,
        )

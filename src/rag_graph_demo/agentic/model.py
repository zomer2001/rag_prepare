from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Protocol

from openai import AsyncOpenAI
from pydantic import BaseModel

from rag_graph_demo.agentic.models import (
    EvidenceAssessment,
    GroundedAnswer,
    KnowledgeNeedState,
    ModelCall,
    QueryRewrite,
)
from rag_graph_demo.agentic.prompts import (
    EVIDENCE_ASSESSMENT_SYSTEM,
    EVONEED_ASSESS_SYSTEM,
    EVONEED_PLAN_SYSTEM,
    EVONEED_PROMPT_PROFILE_VERSION,
    GROUNDED_ANSWER_SYSTEM,
    QUERY_REWRITE_SYSTEM,
    evidence_context,
)
from rag_graph_demo.config import Settings
from rag_graph_demo.evaluation.models import RetrievedChunk
from rag_graph_demo.storage.atomic_json import read_json, write_json_atomic

logger = logging.getLogger(__name__)


class AgentModel(Protocol):
    async def assess(
        self,
        question: str,
        evidence: list[RetrievedChunk],
        *,
        retrieval_round: int,
    ) -> ModelCall[EvidenceAssessment]: ...

    async def rewrite(
        self,
        question: str,
        current_query: str,
        evidence: list[RetrievedChunk],
        assessment: EvidenceAssessment,
    ) -> ModelCall[QueryRewrite]: ...

    async def answer(
        self,
        question: str,
        evidence: list[RetrievedChunk],
        assessment: EvidenceAssessment | None,
    ) -> ModelCall[GroundedAnswer]: ...


class DeepSeekAgentModel:
    """Typed DeepSeek facade used only by the LangGraph controller."""

    def __init__(
        self,
        settings: Settings,
        *,
        plan_cache_path: Path | None = None,
    ) -> None:
        if not settings.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for agentic retrieval")
        self.settings = settings
        self.plan_cache_path = plan_cache_path
        raw_cache = read_json(plan_cache_path, default={}) if plan_cache_path else {}
        self._plan_cache = raw_cache if isinstance(raw_cache, dict) else {}
        self.client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            timeout=settings.llm_request_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    async def assess(
        self,
        question: str,
        evidence: list[RetrievedChunk],
        *,
        retrieval_round: int,
    ) -> ModelCall[EvidenceAssessment]:
        user = (
            f"Question: {question}\nRetrieval round: {retrieval_round}\n\n"
            f"Evidence:\n{evidence_context(evidence)}\n\nReturn JSON only."
        )
        return await self._json_call(
            EvidenceAssessment,
            EVIDENCE_ASSESSMENT_SYSTEM,
            user,
            model=self.settings.deepseek_extraction_model,
        )

    async def plan_needs(
        self,
        question: str,
        *,
        total_budget: int,
    ) -> ModelCall[KnowledgeNeedState]:
        cache_key = hashlib.sha256(
            (
                f"{EVONEED_PROMPT_PROFILE_VERSION}\x1f{total_budget}\x1f{question}"
            ).encode()
        ).hexdigest()
        cached = self._plan_cache.get(cache_key)
        if isinstance(cached, dict):
            return ModelCall(value=KnowledgeNeedState.model_validate(cached))
        user = (
            f"Question: {question}\nTotal retrieval candidate budget: {total_budget}\n\n"
            "Allocate that total budget across the proposed needs. Return JSON only."
        )
        call = await self._json_call(
            KnowledgeNeedState,
            EVONEED_PLAN_SYSTEM,
            user,
            model=self.settings.deepseek_extraction_model,
        )
        self._plan_cache[cache_key] = call.value.model_dump(mode="json")
        if self.plan_cache_path is not None:
            write_json_atomic(self.plan_cache_path, self._plan_cache)
        return call

    async def assess_needs(
        self,
        question: str,
        need_state: KnowledgeNeedState,
        evidence: list[RetrievedChunk],
        *,
        retrieval_round: int,
    ) -> ModelCall[KnowledgeNeedState]:
        user = (
            f"Question: {question}\nRetrieval round: {retrieval_round}\n\n"
            "Previous KnowledgeNeedState:\n"
            f"{need_state.model_dump_json()}\n\n"
            f"Evidence:\n{evidence_context(evidence)}\n\nReturn JSON only."
        )
        return await self._json_call(
            KnowledgeNeedState,
            EVONEED_ASSESS_SYSTEM,
            user,
            model=self.settings.deepseek_extraction_model,
        )

    async def rewrite(
        self,
        question: str,
        current_query: str,
        evidence: list[RetrievedChunk],
        assessment: EvidenceAssessment,
    ) -> ModelCall[QueryRewrite]:
        user = (
            f"Original question: {question}\nPrevious query: {current_query}\n"
            f"Missing information: {assessment.missing_information}\n"
            f"Assessment reason: {assessment.reason}\n"
            f"Evidence already found:\n{evidence_context(evidence, max_chars=8_000)}\n\n"
            "Return JSON only."
        )
        return await self._json_call(
            QueryRewrite,
            QUERY_REWRITE_SYSTEM,
            user,
            model=self.settings.deepseek_extraction_model,
        )

    async def answer(
        self,
        question: str,
        evidence: list[RetrievedChunk],
        assessment: EvidenceAssessment | None,
    ) -> ModelCall[GroundedAnswer]:
        diagnosis = assessment.reason if assessment else "No assessment available"
        user = (
            f"Question: {question}\nEvidence diagnosis: {diagnosis}\n\n"
            f"Evidence:\n{evidence_context(evidence)}\n\nReturn JSON only."
        )
        return await self._json_call(
            GroundedAnswer,
            GROUNDED_ANSWER_SYSTEM,
            user,
            model=self.settings.deepseek_chat_model,
        )

    async def _json_call(
        self,
        schema: type[BaseModel],
        system: str,
        user: str,
        *,
        model: str,
    ) -> ModelCall[Any]:
        request: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "extra_body": {
                "thinking": {
                    "type": (
                        "enabled" if self.settings.deepseek_thinking_enabled else "disabled"
                    )
                }
            },
        }
        if not self.settings.deepseek_thinking_enabled:
            request["temperature"] = 0
        attempts = max(1, self.settings.llm_max_retries)
        for attempt in range(1, attempts + 1):
            response = await self.client.chat.completions.create(**request)
            content = response.choices[0].message.content or ""
            try:
                value = schema.model_validate(extract_json_object(content))
            except (json.JSONDecodeError, ValueError) as exc:
                if attempt >= attempts:
                    raise
                finish_reason = response.choices[0].finish_reason or "unknown"
                delay = min(2 ** (attempt - 1), 4)
                logger.warning(
                    "Invalid DeepSeek JSON (%s, finish_reason=%s); retrying in %ss "
                    "(%s/%s)",
                    type(exc).__name__,
                    finish_reason,
                    delay,
                    attempt,
                    attempts,
                )
                request["messages"] = [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": (
                            f"{user}\n\nA previous response was invalid or truncated. "
                            "Regenerate a complete, concise JSON object and verify that all "
                            "strings, arrays, and braces are closed."
                        ),
                    },
                ]
                await asyncio.sleep(delay)
                continue
            usage = response.usage
            return ModelCall(
                value=value,
                input_tokens=int(usage.prompt_tokens) if usage else 0,
                output_tokens=int(usage.completion_tokens) if usage else 0,
            )

        raise RuntimeError("unreachable DeepSeek JSON retry state")


def extract_json_object(value: str) -> dict[str, Any]:
    stripped = value.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("DeepSeek did not return a JSON object") from None
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("DeepSeek JSON response must be an object")
    return parsed
